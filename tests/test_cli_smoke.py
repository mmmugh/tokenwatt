import subprocess, sys, pathlib

def test_version_matches_version_file():
    root = pathlib.Path(__file__).resolve().parents[1]
    expected = (root / "VERSION").read_text().strip()
    out = subprocess.run(
        [sys.executable, "-m", "tokenwatt", "--version"],
        capture_output=True, text=True,
    )
    assert out.returncode == 0, out.stderr
    assert expected in out.stdout


from typer.testing import CliRunner

from tokenwatt.cli import app

runner = CliRunner()


def test_calibrate_probe_help_lists_meter_host():
    res = runner.invoke(app, ["calibrate", "probe", "--help"])
    assert res.exit_code == 0
    assert "--meter-host" in res.output


def test_calibrate_probe_without_host_fails_loud():
    # no --meter-host and no config -> a pointable error, non-zero exit
    res = runner.invoke(app, ["calibrate", "probe"])
    assert res.exit_code == 1
    assert "meter host" in res.output.lower()


def test_calibrate_probe_failure_reason_goes_to_stderr(monkeypatch):
    # run_probe's (None, reason) is a failure — it must land on stderr like every other
    # CLI error path (no-host / config-error), not stdout, so scripts piping stdout don't
    # silently swallow the diagnostic.
    import tokenwatt.campaign as campaign

    monkeypatch.setattr(campaign, "run_probe",
                        lambda **_k: (None, "meter unreachable: ConnectError"))
    res = runner.invoke(app, ["calibrate", "probe", "--meter-host", "h"])
    assert res.exit_code == 1
    assert "meter unreachable: ConnectError" in res.stderr
    assert "meter unreachable: ConnectError" not in res.stdout


def test_calibrate_campaign_help_lists_upstream_and_model():
    res = runner.invoke(app, ["calibrate", "campaign", "--help"])
    assert res.exit_code == 0
    assert "--upstream" in res.output and "--model" in res.output


def test_calibrate_campaign_requires_upstream_and_model():
    res = runner.invoke(app, ["calibrate", "campaign", "--meter-host", "h"])
    assert res.exit_code == 1
    assert "upstream" in res.output.lower() or "model" in res.output.lower()


def test_calibrate_campaign_reaches_run_campaign_and_fails_loud_on_unreachable(monkeypatch):
    # exercises the real CLI -> run_campaign(defaults) -> _default_shelly wiring;
    # would have caught the missing-defaults TypeError. Fake source => no network.
    from tokenwatt import metersource

    class _Unreachable:
        name, tier, accuracy_pct = "fake", "fake", 0.0
        def __init__(self, *a, **k): pass
        def reachable(self): return (False, "no route to host")

    monkeypatch.setattr(metersource, "ShellyMeterSource", _Unreachable)
    res = runner.invoke(app, ["calibrate", "campaign", "--meter-host", "h",
                              "--upstream", "http://up", "--model", "m"])
    assert res.exit_code == 1
    assert "unreachable" in res.output.lower()


import json as _json


def test_calibrate_fit_writes_a_profile_from_a_campaign(tmp_path, monkeypatch):
    # a minimal campaign whose slope is obviously ~2.0
    campaign = {
        "schema_version": 1, "timestamp": 0.0, "model": "qwen3.6-27b",
        "meter": {"name": "shelly-plug", "tier": "smart_plug", "accuracy_pct": 1.0},
        "cell_seconds": 300.0, "passes": 2,
        "idle": {"rail_w": {"gpu": 0.1}, "wall_w": 7.0, "dt_s": 300.0},
        "samples": [
            {"cell": "prefill", "model": "m", "dt_s": 300.0,
             "e_rail_marginal_j": {"gpu": 1000.0}, "e_wall_marginal_j": 2000.0,
             "tok_in": 1, "tok_out": 1, "requests": 5},
            {"cell": "prefill", "model": "m", "dt_s": 300.0,
             "e_rail_marginal_j": {"gpu": 1000.0}, "e_wall_marginal_j": 2020.0,
             "tok_in": 1, "tok_out": 1, "requests": 5},
            {"cell": "decode", "model": "m", "dt_s": 300.0,
             "e_rail_marginal_j": {"gpu": 5000.0}, "e_wall_marginal_j": 10000.0,
             "tok_in": 1, "tok_out": 1, "requests": 5},
            {"cell": "decode", "model": "m", "dt_s": 300.0,
             "e_rail_marginal_j": {"gpu": 5000.0}, "e_wall_marginal_j": 10050.0,
             "tok_in": 1, "tok_out": 1, "requests": 5},
        ],
    }
    cpath = tmp_path / "campaign.json"
    cpath.write_text(_json.dumps(campaign))
    # deterministic machine, no real sysctl
    from tokenwatt import machineid
    monkeypatch.setattr(machineid, "_real_sysctl",
                        lambda key: {"machdep.cpu.brand_string": "Apple M3 Ultra",
                                     "hw.model": "Mac15,14",
                                     "hw.memsize": str(96 * 1024**3)}[key])
    monkeypatch.setattr(machineid.platform, "mac_ver", lambda: ("26.5.1", ("", "", ""), ""))

    res = runner.invoke(app, ["calibrate", "fit", str(cpath), "--out", str(tmp_path / "profiles")])
    assert res.exit_code == 0, res.output
    assert "plug-calibrated" in res.output
    prof = _json.load(open(tmp_path / "profiles" / "mac15-14_apple-m3-ultra_96gb_macos26.json"))
    assert prof["fit_type"] == "scalar"
    assert 1.8 <= prof["coefficients"]["a"] <= 2.2      # slope recovered
    assert prof["model_calibrated_on"] == "qwen3.6-27b"


def test_calibrate_fit_missing_campaign_file_fails_loud(tmp_path):
    res = runner.invoke(app, ["calibrate", "fit", str(tmp_path / "nope.json")])
    assert res.exit_code == 1
    assert "not found" in res.output.lower() or "no such" in res.output.lower()
