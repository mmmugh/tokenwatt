# tests/test_calibration.py
import pytest

from tokenwatt import calibration as cal


def _sample(cell, rail_total, wall, dt=300.0):
    return {"cell": cell, "dt_s": dt, "e_rail_marginal_j": {"gpu": rail_total},
            "e_wall_marginal_j": wall, "tok_in": 1, "tok_out": 1, "requests": 1}


def test_fit_scalar_recovers_known_coefficients():
    # wall = 1.8*rail + 0.5*dt exactly, across varied rail with dt=300
    samples = [_sample("c", r, 1.8 * r + 0.5 * 300.0) for r in (1000.0, 5000.0, 9000.0)]
    a, b, res = cal.fit_scalar(samples)
    assert a == pytest.approx(1.8, abs=1e-3)
    assert b == pytest.approx(0.5, abs=1e-3)
    assert res == pytest.approx(0.0, abs=1e-6)


def test_fit_scalar_recovers_slope_from_real_c1_data():
    # the actual 6 marginal samples measured on the M3 Ultra vs qwen3.6-27b
    # (rail_total_J, wall_J); dt ~300s each. The slope must land near ~1.8.
    real = [(13335.1, 25728.5), (25536.7, 44910.7), (24913.9, 44082.8),
            (13637.3, 26578.2), (25751.2, 45346.7), (24891.2, 44511.0)]
    samples = [_sample("c", r, w) for r, w in real]
    a, b, res = cal.fit_scalar(samples)
    assert 1.5 <= a <= 2.2          # fitted slope a ~1.57 (the ~1.8 figure is the wall/rail RATIO, not a)
    assert b >= 0.0                 # NNLS non-negativity holds
    assert res < 0.15               # tight relative residual on real data


def test_run_variance_rel_widens_with_disagreeing_passes():
    tight = [_sample("prefill", 100, 200), _sample("prefill", 100, 202)]   # ~1% apart
    loose = [_sample("prefill", 100, 200), _sample("prefill", 100, 300)]   # ~40% apart
    assert cal.run_variance_rel(tight) < cal.run_variance_rel(loose)
    assert cal.run_variance_rel(loose) > 0.2


def test_confidence_band_combines_in_quadrature():
    band = cal.confidence_band_pct(residual_rel=0.03, run_var_rel=0.04, meter_accuracy_pct=1.0)
    assert band == pytest.approx(100 * (0.03**2 + 0.04**2 + 0.01**2) ** 0.5, abs=1e-6)


def test_tier_is_plug_calibrated_only_when_band_beats_estimated_floor():
    assert cal.tier_label("smart_plug", 5.0).startswith("plug-calibrated")
    assert "±5.0%" in cal.tier_label("smart_plug", 5.0)
    assert cal.tier_label("smart_plug", 22.0).startswith("uncertified")   # no tighter than estimated
    assert cal.tier_label("manual", 5.0).startswith("uncertified")        # only smart_plug certifies in C2


def test_cal_scalar_is_a_pure_linear_prediction():
    assert cal.cal_scalar(1.8, 0.5, e_rail_total_j=1000.0, dt_s=300.0) == pytest.approx(1950.0)


def test_fit_composes_a_consistent_fitresult_from_a_campaign():
    samples = [_sample("prefill", 1000.0, 2000.0), _sample("prefill", 1000.0, 2010.0),
               _sample("decode", 5000.0, 9000.0), _sample("decode", 5000.0, 9100.0)]
    campaign = {"samples": samples, "passes": 2,
                "meter": {"name": "shelly-plug", "tier": "smart_plug", "accuracy_pct": 1.0}}
    r = cal.fit(campaign)
    # composition must match the standalone helpers on the same inputs (no key mixups)
    a, b, res = cal.fit_scalar(samples)
    rv = cal.run_variance_rel(samples)
    band = cal.confidence_band_pct(res, rv, 1.0)
    assert r.fit_type == "scalar"
    assert (r.a, r.b, r.residual_rel) == (a, b, res)
    assert r.run_variance_rel == rv
    assert r.band_pct == band
    assert r.tier == cal.tier_label("smart_plug", band)
    assert r.n_samples == 4 and r.n_passes == 2


def _campaign(samples, idle_batt=None):
    return {
        "meter": {"tier": "smart_plug", "accuracy_pct": 1.0},
        "passes": 2,
        "idle": {"rail_w": {}, "wall_w": 0.0, "dt_s": 1.0, "battery_abs_w": idle_batt},
        "samples": samples,
    }


def _s(wall, rail, dt, cell="prefill", batt=None):
    return {"cell": cell, "dt_s": dt, "e_rail_marginal_j": {"gpu": rail},
            "e_wall_marginal_j": wall, "battery_abs_w": batt}


def test_fit_excludes_battery_contaminated_cells():
    clean = [_s(200.0, 100.0, 100.0, batt=0.1), _s(400.0, 200.0, 200.0, cell="decode", batt=0.2)]
    dirty = [_s(9999.0, 100.0, 100.0, cell="prefill", batt=25.0)]      # charging mid-cell
    r = cal.fit(_campaign(clean + dirty))
    assert r.n_excluded == 1
    assert r.n_samples == 2                                            # only the clean cells fit


def test_fit_keeps_cells_with_no_battery_signal():
    # desktop / no-battery: battery_abs_w is None -> never excluded
    r = cal.fit(_campaign([_s(200.0, 100.0, 100.0, batt=None),
                           _s(400.0, 200.0, 200.0, cell="decode", batt=None)]))
    assert r.n_excluded == 0 and r.n_samples == 2


def test_fit_fails_loud_on_contaminated_idle_baseline():
    with pytest.raises(ValueError, match="idle baseline"):
        cal.fit(_campaign([_s(200.0, 100.0, 100.0, batt=0.1)], idle_batt=12.0))


def test_fit_fails_loud_when_no_clean_samples():
    with pytest.raises(ValueError, match="no clean"):
        cal.fit(_campaign([_s(9.0, 1.0, 1.0, batt=25.0)]))


# --- fit_combined: varying-duration fit with a correctly-scoped band ---

def _dur_A():
    # wall = 1.5*rail + 0.5*dt at dt=120, symmetric ±1 pass jitter
    return _campaign([_s(209, 100, 120, "prefill"), _s(211, 100, 120, "prefill"),
                      _s(659, 400, 120, "decode"),  _s(661, 400, 120, "decode")])


def _dur_B():
    # same law at dt=600 (non-collinear with A in (rail,dt) -> a,b identifiable)
    return _campaign([_s(449, 100, 600, "prefill"), _s(451, 100, 600, "prefill"),
                      _s(899, 400, 600, "decode"),  _s(901, 400, 600, "decode")])


def test_fit_combined_recovers_coefficients_across_durations():
    r = cal.fit_combined([_dur_A(), _dur_B()])
    assert r.a == pytest.approx(1.5, abs=0.02)     # varying Δt pins b; slope a recovered
    assert r.b == pytest.approx(0.5, abs=0.02)
    assert r.n_samples == 8 and r.n_excluded == 0


def test_fit_refuses_merged_durations_and_fit_combined_stays_tight():
    # fit() must not be silently misused on a hand-merged multi-duration campaign:
    # it groups pass-repeatability by cell NAME across durations, so a merged
    # 'prefill' spanning 120s→600s would balloon the band (~±150%). It now fails
    # loud and points at fit_combined, which measures repeatability WITHIN each
    # duration and keeps the band honest.
    A, B = _dur_A(), _dur_B()
    with pytest.raises(ValueError, match="fit_combined"):
        cal.fit({**A, "samples": A["samples"] + B["samples"]})
    combined = cal.fit_combined([A, B])
    assert combined.run_variance_rel < 0.05         # within-duration: tight, not inflated


def test_fit_combined_single_campaign_equals_fit():
    c = _dur_A()
    one, ref = cal.fit_combined([c]), cal.fit(c)
    assert (one.a, one.b) == (ref.a, ref.b)
    assert one.run_variance_rel == ref.run_variance_rel
    assert one.band_pct == ref.band_pct


def test_fit_combined_fails_loud_on_contaminated_idle():
    with pytest.raises(ValueError, match="idle baseline"):
        cal.fit_combined([_dur_A(), _campaign(_dur_B()["samples"], idle_batt=12.0)])
