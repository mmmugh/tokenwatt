# TokenWatt Calibration C2 — Scalar Fit + Profile Registry + Band/Tier — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn a C1 campaign dataset into a per-machine **calibration profile**: fit the scalar model `E_wall_marg ≈ a·E_rail_total + b·Δt` (a,b ≥ 0) with a pure-Python NNLS, derive a measured confidence band + honest tier, key it to this machine, and persist it as a human-readable JSON profile. No fabricated numbers, no runtime wiring (that's C3), no per-rail fit (that's C5).

**Architecture:** `nnls.py` is a tiny, dependency-free Lawson–Hanson non-negative least-squares solver (reused by C5's per-rail fit later). `calibration.py` builds the scalar design matrix from the campaign samples, solves it, and derives `residual ⊕ run-variance ⊕ meter-accuracy → band → tier` (the band is *measured*, never asserted). `machineid.py` detects a stable machine key from `sysctl`/`platform` (no sudo). `profiles.py` is the per-machine JSON registry. `cli.py` gets `tokenwatt calibrate fit`. Everything is dependency-injected and deterministic in CI.

**Tech Stack:** Python ≥3.10 stdlib only — **no numpy, no scipy** (deliberate: the solve is a one-shot offline fit on ≤30 points; runtime application in C3 is plain arithmetic on the stored coefficients, never a solver). `typer`/`pydantic`/`httpx` already present.

## Global Constraints

- **Python floor:** `requires-python>=3.10`; runs on 3.10–3.14 (CI matrix).
- **No new dependencies** — the NNLS solver is pure Python (project keeps its `py3-none-any` wheel). The problem is tiny (≤6 unknowns, ≤30 samples), so pure Python is both fast enough and fully testable.
- **The band is MEASURED, not asserted:** `band = √(residual_rel² + run_variance_rel² + (meter_accuracy_pct/100)²)`, as a `±%`. Non-repeatable passes widen it. A smart-plug fit's tier is `plug-calibrated (±x%)` and can **never** be the lab-grade `±2–5%`. If the band is no tighter than the `estimated` floor (≥15%), the fit is **not certified** as plug-calibrated (honest: don't claim calibration the data didn't earn).
- **Honesty contract:** never a fabricated number. Coefficients/band/tier come from the data; a missing/absent value is `None`/`—`, never `0`. Tokens in the dataset may be `null` (unknown) — the fit uses ENERGY, not tokens, so that never affects it.
- **Marginal convention (inherited from C1):** samples are already idle-subtracted marginal energy on both sides; the fit consumes them as-is.
- **Profile store:** one human-readable JSON per machine under `~/.tokenwatt/profiles/<machine_id>.json` (same transparent/shareable ethos as `cloud.py`).
- **No sudo** in machine detection (`sysctl`/`platform` only).
- **Deterministic CI:** NNLS, fit, machine-id, and registry all run on injected inputs / `tmp_path` — no hardware, no network, no `~/.tokenwatt` writes in tests. The one real-data check is a static fixture of the 6 C1 samples.
- **`VERSION` auto-bumps** per commit via `.githooks/pre-commit` — expected; never `--no-verify`.
- **Leak-safety:** no numeric/private IPs in code/tests (a pre-commit leak-scan enforces it).
- **Match existing style:** `from __future__ import annotations`; frozen dataclasses; DI seams like `run_probe`/`doctor.run`; tiny inline test fakes.

---

## File Structure

- **Create `src/tokenwatt/nnls.py`** — `nnls(A, b) -> (x, residual)` (Lawson–Hanson) + small linear-algebra helpers. Single responsibility: *the solver*. Zero deps.
- **Create `src/tokenwatt/calibration.py`** — `fit_scalar`, `run_variance_rel`, `confidence_band_pct`, `tier_label`, `FitResult`, and the pure `cal_scalar` predict fn. Single responsibility: *the fit + band/tier* (C3 imports `cal_scalar`; C5 adds `fit_per_rail`).
- **Create `src/tokenwatt/machineid.py`** — `MachineInfo` + `detect_machine(...)`. Single responsibility: *the machine key*.
- **Create `src/tokenwatt/profiles.py`** — `Profile` + registry (`save`/`load`/`active_for_current_machine`/`list_profiles`). Single responsibility: *the per-machine profile store*.
- **Modify `src/tokenwatt/cli.py`** — add `tokenwatt calibrate fit`. Modify `src/tokenwatt/metersource.py` — add best-effort `ShellyMeterSource.device_info()` for exact meter identity.
- **Create `tests/test_nnls.py`, `tests/test_calibration.py`, `tests/test_machineid.py`, `tests/test_profiles.py`**; extend `tests/test_cli_smoke.py`, `tests/test_metersource.py`.

**Deferred out of C2 (do not build here):** runtime `Cal` application + `calib_*` ledger columns + proxy boot loading + `calibration:` config block (C3); per-rail fit + condition-number/VIF gate (C5); the interactive `calibrate` wizard + `calibrate show` (C4); the WT310E `lab-calibrated` tier. The `held-out residual` + `condition number` profile fields the spec lists are **per-rail-gate concepts (C5)**; C2 stores in-sample `residual_rel` and omits condition number (scalar fit is always well-posed).

---

### Task 1: Pure-Python NNLS solver (`nnls.py`)

**Files:**
- Create: `src/tokenwatt/nnls.py`
- Test: `tests/test_nnls.py`

**Interfaces:**
- Produces: `nnls(A: list[list[float]], b: list[float], *, tol: float = 1e-10, max_iter: int = 100) -> tuple[list[float], float]` — returns `(x, residual)` where `x ≥ 0` minimizes `‖A·x − b‖₂`; `A` is `m×n` (m rows of n floats), `residual` is the Euclidean norm `‖A·x − b‖`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_nnls.py
import math

import pytest

from tokenwatt.nnls import nnls


def test_identity_recovers_exact_solution():
    x, res = nnls([[1.0, 0.0], [0.0, 1.0]], [3.0, 4.0])
    assert x == pytest.approx([3.0, 4.0])
    assert res == pytest.approx(0.0, abs=1e-9)


def test_overdetermined_exact_line_through_origin():
    # b_i = 2 * a_i -> slope 2, zero residual
    x, res = nnls([[1.0], [2.0], [3.0]], [2.0, 4.0, 6.0])
    assert x[0] == pytest.approx(2.0)
    assert res == pytest.approx(0.0, abs=1e-9)


def test_nonnegativity_clamps_a_negative_unconstrained_solution():
    # unconstrained best fit is negative; NNLS must clamp to 0
    x, res = nnls([[1.0], [1.0]], [-2.0, -2.0])
    assert x[0] == pytest.approx(0.0)
    assert res == pytest.approx(math.sqrt(8.0))     # ‖[-2,-2] - 0‖


def test_two_variable_recovers_known_coefficients():
    # b_i = 2*col0 + 0.5*col1 exactly -> [2.0, 0.5]
    A = [[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]]
    b = [2.5, 4.5, 6.5]
    x, res = nnls(A, b)
    assert x == pytest.approx([2.0, 0.5], abs=1e-6)
    assert res == pytest.approx(0.0, abs=1e-6)


def test_two_variable_with_one_coefficient_clamped():
    # data wants a negative second coefficient; NNLS zeros it and refits the first
    A = [[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]]
    b = [1.0, 2.0, 3.0]           # perfectly slope-1 through origin; the +1 col wants < 0
    x, res = nnls(A, b)
    assert x[1] == pytest.approx(0.0, abs=1e-6)      # second coefficient clamped
    assert x[0] == pytest.approx(1.0, abs=1e-6)
    assert res == pytest.approx(0.0, abs=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_nnls.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tokenwatt.nnls'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/tokenwatt/nnls.py
from __future__ import annotations

import math


def _matvec(A: list[list[float]], x: list[float]) -> list[float]:
    return [sum(row[j] * x[j] for j in range(len(x))) for row in A]


def _matTvec(A: list[list[float]], y: list[float]) -> list[float]:
    n = len(A[0])
    return [sum(A[i][j] * y[i] for i in range(len(A))) for j in range(n)]


def _residual_norm(A: list[list[float]], x: list[float], b: list[float]) -> float:
    ax = _matvec(A, x)
    return math.sqrt(sum((ax[i] - b[i]) ** 2 for i in range(len(b))))


def _solve_passive(A: list[list[float]], b: list[float], passive: list[int]) -> list[float]:
    """Unconstrained least squares over the passive columns via the normal
    equations (AᵀA)s = Aᵀb, solved with Gaussian elimination + partial pivoting.
    Returns a full-length vector with zeros outside `passive`."""
    p = passive
    k = len(p)
    # normal-equation matrix G = A_pᵀ A_p (k×k) and rhs c = A_pᵀ b
    G = [[sum(A[i][p[r]] * A[i][p[c]] for i in range(len(A))) for c in range(k)] for r in range(k)]
    c = [sum(A[i][p[r]] * b[i] for i in range(len(A))) for r in range(k)]
    # Gaussian elimination with partial pivoting
    for col in range(k):
        piv = max(range(col, k), key=lambda r: abs(G[r][col]))
        if abs(G[piv][col]) < 1e-15:
            continue
        G[col], G[piv] = G[piv], G[col]
        c[col], c[piv] = c[piv], c[col]
        for r in range(col + 1, k):
            f = G[r][col] / G[col][col]
            for cc in range(col, k):
                G[r][cc] -= f * G[col][cc]
            c[r] -= f * c[col]
    s_p = [0.0] * k
    for r in range(k - 1, -1, -1):
        if abs(G[r][r]) < 1e-15:
            continue
        s_p[r] = (c[r] - sum(G[r][cc] * s_p[cc] for cc in range(r + 1, k))) / G[r][r]
    n = len(A[0])
    s = [0.0] * n
    for idx, col in enumerate(p):
        s[col] = s_p[idx]
    return s


def nnls(A: list[list[float]], b: list[float], *, tol: float = 1e-10,
         max_iter: int = 100) -> tuple[list[float], float]:
    """Lawson–Hanson non-negative least squares: minimize ‖A·x − b‖ s.t. x ≥ 0.
    A is m×n (m rows of n floats). Returns (x, residual_norm)."""
    n = len(A[0])
    x = [0.0] * n
    passive: list[int] = []
    active = list(range(n))
    for _ in range(max_iter):
        w = _matTvec(A, [b[i] - _matvec(A, x)[i] for i in range(len(b))])
        if not active or max(w[j] for j in active) <= tol:
            break
        j = max(active, key=lambda j: w[j])
        active.remove(j)
        passive.append(j)
        while True:
            s = _solve_passive(A, b, passive)
            if all(s[j] > tol for j in passive):
                x = s
                break
            # some passive coefficient went non-positive: back off along x -> s
            alpha = min(x[j] / (x[j] - s[j]) for j in passive if s[j] <= tol and x[j] != s[j])
            x = [x[j] + alpha * (s[j] - x[j]) for j in range(n)]
            for j in list(passive):
                if x[j] <= tol:
                    passive.remove(j)
                    active.append(j)
            if not passive:
                break
    x = [max(v, 0.0) for v in x]
    return x, _residual_norm(A, x, b)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_nnls.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/tokenwatt/nnls.py tests/test_nnls.py
git commit -m "feat(calib): pure-Python NNLS solver (Lawson-Hanson)"
```

---

### Task 2: Scalar fit + band + tier (`calibration.py`)

**Files:**
- Create: `src/tokenwatt/calibration.py`
- Test: `tests/test_calibration.py`

**Interfaces:**
- Consumes: `nnls` (Task 1).
- Produces:
  - `cal_scalar(a: float, b: float, e_rail_total_j: float, dt_s: float) -> float` — the pure prediction `a·E_rail + b·Δt` (J). (C3 uses this at runtime.)
  - `fit_scalar(samples: list[dict]) -> tuple[float, float, float]` — returns `(a, b, residual_rel)` from marginal samples (`e_rail_marginal_j` summed → `E_rail_total`, `dt_s`, target `e_wall_marginal_j`). `residual_rel` = RMS of relative residuals.
  - `run_variance_rel(samples: list[dict]) -> float` — RMS across cells of the pass-to-pass relative spread of `e_wall_marginal_j` (0.0 when <2 passes of a cell).
  - `confidence_band_pct(residual_rel: float, run_var_rel: float, meter_accuracy_pct: float) -> float` — `100·√(residual_rel² + run_var_rel² + (meter_accuracy_pct/100)²)`.
  - `tier_label(meter_tier: str, band_pct: float) -> str` — `plug-calibrated (±x%)` for `smart_plug` when `band_pct < 15`, else `uncertified (±x%) — no tighter than estimated`; other meter tiers stay uncertified for now.
  - `FitResult(fit_type, a, b, residual_rel, run_variance_rel, band_pct, tier, n_samples, n_passes)` frozen dataclass + `fit(campaign: dict) -> FitResult` composing the above from a loaded campaign JSON.

- [ ] **Step 1: Write the failing test**

```python
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
    assert 1.5 <= a <= 2.2          # measured marginal slope ~1.8
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
    assert "±5%" in cal.tier_label("smart_plug", 5.0)
    assert cal.tier_label("smart_plug", 22.0).startswith("uncertified")   # no tighter than estimated
    assert cal.tier_label("manual", 5.0).startswith("uncertified")        # only smart_plug certifies in C2


def test_cal_scalar_is_a_pure_linear_prediction():
    assert cal.cal_scalar(1.8, 0.5, e_rail_total_j=1000.0, dt_s=300.0) == pytest.approx(1980.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_calibration.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tokenwatt.calibration'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/tokenwatt/calibration.py
from __future__ import annotations

import math
from dataclasses import dataclass

from tokenwatt.nnls import nnls

_ESTIMATED_FLOOR_PCT = 15.0   # a plug fit no tighter than this isn't worth a distinct tier


def cal_scalar(a: float, b: float, e_rail_total_j: float, dt_s: float) -> float:
    """Pure prediction of marginal wall energy (J) from the scalar profile.
    Used at runtime (C3) — no solver, just arithmetic."""
    return a * e_rail_total_j + b * dt_s


def _rail_total(sample: dict) -> float:
    return sum(sample["e_rail_marginal_j"].values())


def fit_scalar(samples: list[dict]) -> tuple[float, float, float]:
    """Fit E_wall_marg ≈ a·E_rail_total + b·Δt (a,b ≥ 0). Returns (a, b, residual_rel)
    where residual_rel is the RMS relative residual over the samples."""
    A = [[_rail_total(s), s["dt_s"]] for s in samples]
    b_vec = [s["e_wall_marginal_j"] for s in samples]
    (a, b), _ = nnls(A, b_vec)
    rels = []
    for s in samples:
        pred = cal_scalar(a, b, _rail_total(s), s["dt_s"])
        actual = s["e_wall_marginal_j"]
        if actual > 0:
            rels.append(((pred - actual) / actual) ** 2)
    residual_rel = math.sqrt(sum(rels) / len(rels)) if rels else 0.0
    return a, b, residual_rel


def run_variance_rel(samples: list[dict]) -> float:
    """RMS across cells of each cell's pass-to-pass relative spread of marginal wall
    energy — the repeatability term. 0.0 when no cell has ≥2 passes."""
    by_cell: dict[str, list[float]] = {}
    for s in samples:
        by_cell.setdefault(s["cell"], []).append(s["e_wall_marginal_j"])
    spreads = []
    for vals in by_cell.values():
        if len(vals) >= 2:
            mean = sum(vals) / len(vals)
            if mean > 0:
                spreads.append(((max(vals) - min(vals)) / mean) ** 2)
    return math.sqrt(sum(spreads) / len(spreads)) if spreads else 0.0


def confidence_band_pct(residual_rel: float, run_var_rel: float, meter_accuracy_pct: float) -> float:
    return 100.0 * math.sqrt(residual_rel ** 2 + run_var_rel ** 2 + (meter_accuracy_pct / 100.0) ** 2)


def tier_label(meter_tier: str, band_pct: float) -> str:
    if meter_tier == "smart_plug" and band_pct < _ESTIMATED_FLOOR_PCT:
        return f"plug-calibrated (±{band_pct:.0f}%)"
    return f"uncertified (±{band_pct:.0f}%) — no tighter than estimated"


@dataclass(frozen=True)
class FitResult:
    fit_type: str
    a: float
    b: float
    residual_rel: float
    run_variance_rel: float
    band_pct: float
    tier: str
    n_samples: int
    n_passes: int


def fit(campaign: dict) -> FitResult:
    samples = campaign["samples"]
    a, b, residual_rel = fit_scalar(samples)
    rv = run_variance_rel(samples)
    acc = campaign["meter"]["accuracy_pct"]
    band = confidence_band_pct(residual_rel, rv, acc)
    return FitResult(
        fit_type="scalar", a=a, b=b, residual_rel=residual_rel, run_variance_rel=rv,
        band_pct=band, tier=tier_label(campaign["meter"]["tier"], band),
        n_samples=len(samples), n_passes=campaign.get("passes", 1))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_calibration.py -v`
Expected: PASS (6 passed) — including the real-data slope check.

- [ ] **Step 5: Commit**

```bash
git add src/tokenwatt/calibration.py tests/test_calibration.py
git commit -m "feat(calib): scalar fit + measured confidence band + honest tier"
```

---

### Task 3: Machine identity (`machineid.py`)

**Files:**
- Create: `src/tokenwatt/machineid.py`
- Test: `tests/test_machineid.py`

**Interfaces:**
- Produces:
  - `MachineInfo(machine_id: str, label: str, soc: str, model: str, ram_gb: int, macos_major: str)` frozen dataclass.
  - `detect_machine(*, sysctl=<default>, mac_ver=<default>) -> MachineInfo` — `sysctl(key: str) -> str` reads `machdep.cpu.brand_string`/`hw.model`/`hw.memsize`; `mac_ver() -> str` returns the macOS version string. Defaults call real `subprocess`/`platform`; tests inject fakes. `machine_id` is a stable lowercase slug (no sudo).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_machineid.py
from tokenwatt.machineid import detect_machine, MachineInfo


def _fake_sysctl(mapping):
    return lambda key: mapping[key]


def test_detects_m3_ultra_and_builds_stable_slug():
    info = detect_machine(
        sysctl=_fake_sysctl({
            "machdep.cpu.brand_string": "Apple M3 Ultra",
            "hw.model": "Mac15,14",
            "hw.memsize": str(96 * 1024**3),
        }),
        mac_ver=lambda: "26.5.1")
    assert isinstance(info, MachineInfo)
    assert info.soc == "Apple M3 Ultra"
    assert info.model == "Mac15,14"
    assert info.ram_gb == 96
    assert info.macos_major == "26"
    # slug is stable, lowercase, filesystem-safe, and encodes the identity
    assert info.machine_id == "mac15-14_apple-m3-ultra_96gb_macos26"
    assert "M3 Ultra" in info.label and "96" in info.label


def test_slug_is_deterministic():
    kw = dict(sysctl=_fake_sysctl({"machdep.cpu.brand_string": "Apple M4",
                                   "hw.model": "Mac16,12", "hw.memsize": str(16 * 1024**3)}),
              mac_ver=lambda: "26.1")
    assert detect_machine(**kw).machine_id == detect_machine(**kw).machine_id
    assert detect_machine(**kw).machine_id == "mac16-12_apple-m4_16gb_macos26"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_machineid.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tokenwatt.machineid'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/tokenwatt/machineid.py
from __future__ import annotations

import platform
import re
import subprocess
from dataclasses import dataclass


def _real_sysctl(key: str) -> str:
    return subprocess.check_output(["sysctl", "-n", key], text=True).strip()


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


@dataclass(frozen=True)
class MachineInfo:
    machine_id: str
    label: str
    soc: str
    model: str
    ram_gb: int
    macos_major: str


def detect_machine(*, sysctl=_real_sysctl, mac_ver=lambda: platform.mac_ver()[0]) -> MachineInfo:
    soc = sysctl("machdep.cpu.brand_string")
    model = sysctl("hw.model")
    ram_gb = round(int(sysctl("hw.memsize")) / 1024**3)
    macos_major = (mac_ver() or "0").split(".")[0]
    machine_id = f"{_slug(model)}_{_slug(soc)}_{ram_gb}gb_macos{macos_major}"
    label = f"{soc} · {ram_gb} GB · {model} · macOS {macos_major}"
    return MachineInfo(machine_id=machine_id, label=label, soc=soc, model=model,
                       ram_gb=ram_gb, macos_major=macos_major)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_machineid.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/tokenwatt/machineid.py tests/test_machineid.py
git commit -m "feat(calib): machine-id detection (sysctl/platform, no sudo)"
```

---

### Task 4: Profile registry (`profiles.py`)

**Files:**
- Create: `src/tokenwatt/profiles.py`
- Test: `tests/test_profiles.py`

**Interfaces:**
- Consumes: `FitResult` (Task 2), `MachineInfo` (Task 3).
- Produces:
  - `Profile(machine_id, label, fit_type, coefficients: dict, residual_rel, run_variance_rel, band_pct, tier, meter: dict, n_samples, n_passes, model_calibrated_on: str, macos: str, created_at: float, schema_version: int)` frozen dataclass.
  - `profile_from(fit: FitResult, machine: MachineInfo, meter: dict, *, model_calibrated_on: str, created_at: float) -> Profile` — builds a Profile (scalar coefficients `{"a":…, "b":…}`).
  - `save(profile: Profile, *, root: str | None = None) -> str` (writes `<root>/<machine_id>.json`, dirs auto-created; default root `~/.tokenwatt/profiles/`), `load(machine_id: str, *, root=None) -> Profile | None`, `active_for_current_machine(machine_id: str, *, root=None) -> Profile | None`, `list_profiles(*, root=None) -> list[Profile]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_profiles.py
import os

from tokenwatt.profiles import Profile, profile_from, save, load, list_profiles
from tokenwatt.calibration import FitResult
from tokenwatt.machineid import MachineInfo


def _fit():
    return FitResult(fit_type="scalar", a=1.83, b=0.4, residual_rel=0.03,
                     run_variance_rel=0.02, band_pct=4.0, tier="plug-calibrated (±4%)",
                     n_samples=6, n_passes=2)


def _machine():
    return MachineInfo("mac15-14_apple-m3-ultra_96gb_macos26", "Apple M3 Ultra · 96 GB",
                       "Apple M3 Ultra", "Mac15,14", 96, "26")


def test_profile_from_carries_fit_machine_and_meter():
    p = profile_from(_fit(), _machine(),
                     meter={"name": "shelly-plug", "tier": "smart_plug", "accuracy_pct": 1.0},
                     model_calibrated_on="qwen3.6-27b", created_at=1_700_000_000.0)
    assert p.machine_id == "mac15-14_apple-m3-ultra_96gb_macos26"
    assert p.coefficients == {"a": 1.83, "b": 0.4}
    assert p.tier == "plug-calibrated (±4%)"
    assert p.meter["tier"] == "smart_plug"
    assert p.model_calibrated_on == "qwen3.6-27b"


def test_save_then_load_round_trips(tmp_path):
    p = profile_from(_fit(), _machine(),
                     meter={"name": "shelly-plug", "tier": "smart_plug", "accuracy_pct": 1.0},
                     model_calibrated_on="qwen3.6-27b", created_at=1_700_000_000.0)
    path = save(p, root=str(tmp_path))
    assert os.path.isfile(path) and path.endswith("mac15-14_apple-m3-ultra_96gb_macos26.json")
    got = load(p.machine_id, root=str(tmp_path))
    assert got == p                                   # frozen dataclass equality
    assert [x.machine_id for x in list_profiles(root=str(tmp_path))] == [p.machine_id]


def test_load_missing_machine_returns_none(tmp_path):
    assert load("no-such-machine", root=str(tmp_path)) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_profiles.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tokenwatt.profiles'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/tokenwatt/profiles.py
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

from tokenwatt.calibration import FitResult
from tokenwatt.machineid import MachineInfo

_PROFILE_SCHEMA = 1
_DEFAULT_ROOT = "~/.tokenwatt/profiles"


def _root(root: str | None) -> str:
    return os.path.expanduser(root if root is not None else _DEFAULT_ROOT)


@dataclass(frozen=True)
class Profile:
    machine_id: str
    label: str
    fit_type: str
    coefficients: dict
    residual_rel: float
    run_variance_rel: float
    band_pct: float
    tier: str
    meter: dict                      # {name, tier, accuracy_pct, model?, gen?, mac?}
    n_samples: int
    n_passes: int
    model_calibrated_on: str
    macos: str
    created_at: float
    schema_version: int = _PROFILE_SCHEMA


def profile_from(fit: FitResult, machine: MachineInfo, meter: dict, *,
                 model_calibrated_on: str, created_at: float) -> Profile:
    return Profile(
        machine_id=machine.machine_id, label=machine.label, fit_type=fit.fit_type,
        coefficients={"a": fit.a, "b": fit.b}, residual_rel=fit.residual_rel,
        run_variance_rel=fit.run_variance_rel, band_pct=fit.band_pct, tier=fit.tier,
        meter=meter, n_samples=fit.n_samples, n_passes=fit.n_passes,
        model_calibrated_on=model_calibrated_on, macos=machine.macos_major,
        created_at=created_at)


def save(profile: Profile, *, root: str | None = None) -> str:
    d = _root(root)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{profile.machine_id}.json")
    with open(path, "w") as f:
        json.dump(asdict(profile), f, indent=2)
    return path


def load(machine_id: str, *, root: str | None = None) -> Profile | None:
    path = os.path.join(_root(root), f"{machine_id}.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return Profile(**json.load(f))


def active_for_current_machine(machine_id: str, *, root: str | None = None) -> Profile | None:
    return load(machine_id, root=root)


def list_profiles(*, root: str | None = None) -> list[Profile]:
    d = _root(root)
    if not os.path.isdir(d):
        return []
    out = []
    for name in sorted(os.listdir(d)):
        if name.endswith(".json"):
            with open(os.path.join(d, name)) as f:
                out.append(Profile(**json.load(f)))
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_profiles.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/tokenwatt/profiles.py tests/test_profiles.py
git commit -m "feat(calib): per-machine JSON profile registry"
```

---

### Task 5: `calibrate fit` CLI + best-effort meter identity (`cli.py`, `metersource.py`)

**Files:**
- Modify: `src/tokenwatt/metersource.py` (add `ShellyMeterSource.device_info()`)
- Modify: `src/tokenwatt/cli.py` (add `calibrate fit`)
- Test: `tests/test_metersource.py`, `tests/test_cli_smoke.py`

**Interfaces:**
- Consumes: `fit` (Task 2), `detect_machine` (Task 3), `profile_from`/`save` (Task 4), `ShellyMeterSource` (C0).
- Produces:
  - `ShellyMeterSource.device_info() -> dict` — GET `/rpc/Shelly.GetDeviceInfo` → `{"model":…, "gen":…, "mac":…, "app":…}` (raises on HTTP error; caller wraps best-effort).
  - CLI `tokenwatt calibrate fit CAMPAIGN_JSON [--out PROFILES_DIR] [--meter-host HOST]` — read the campaign, fit, detect machine, enrich meter identity if `--meter-host` reachable, write the profile, print the result.

- [ ] **Step 1: Write the failing test (device_info)**

```python
# add to tests/test_metersource.py
def test_shelly_device_info_parses_model_gen_mac():
    body = {"model": "S4PL-00116US", "gen": 4, "mac": "AABBCCDDEEFF", "app": "PlugUSG4"}
    client, _ = _mock_shelly(body)      # existing helper returns (client, seen)
    info = ShellyMeterSource("h", client=client).device_info()
    assert info["model"] == "S4PL-00116US" and info["gen"] == 4 and info["app"] == "PlugUSG4"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_metersource.py -k device_info -v`
Expected: FAIL — `AttributeError: 'ShellyMeterSource' object has no attribute 'device_info'`.

- [ ] **Step 3: Write minimal implementation (device_info)**

```python
# add to src/tokenwatt/metersource.py, in class ShellyMeterSource (after read_status)
    def device_info(self) -> dict:
        base = self._host if "://" in self._host else f"http://{self._host}"
        r = self._client.get(f"{base}/rpc/Shelly.GetDeviceInfo", timeout=self._timeout)
        r.raise_for_status()
        b = r.json()
        return {"model": b.get("model"), "gen": b.get("gen"),
                "mac": b.get("mac"), "app": b.get("app")}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_metersource.py -k device_info -v`
Expected: PASS.

- [ ] **Step 5: Write the failing test (CLI)**

```python
# add to tests/test_cli_smoke.py
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
```

- [ ] **Step 6: Run test to verify it fails**

Run: `uv run pytest tests/test_cli_smoke.py -k calibrate_fit -v`
Expected: FAIL — no such command `fit` (usage error / exit 2).

- [ ] **Step 7: Write minimal implementation (CLI)**

```python
# add to src/tokenwatt/cli.py, after the existing `calibrate campaign` command
@calibrate_app.command("fit")
def calibrate_fit(
    campaign: str = typer.Argument(..., help="path to a `calibrate campaign` samples JSON"),
    out: Optional[str] = typer.Option(None, "--out", help="profiles dir (default ~/.tokenwatt/profiles)"),
    meter_host: Optional[str] = typer.Option(None, "--meter-host", help="enrich exact device identity via Shelly.GetDeviceInfo (best-effort)"),
):
    """Fit a campaign dataset into this machine's calibration profile (scalar).
    The confidence band is MEASURED (fit residual ⊕ pass repeatability ⊕ meter accuracy)."""
    import json
    import os
    import time as _time
    from tokenwatt import calibration, profiles
    from tokenwatt.machineid import detect_machine

    path = os.path.expanduser(campaign)
    if not os.path.isfile(path):
        typer.echo(f"campaign file not found: {campaign}", err=True)
        raise typer.Exit(1)
    with open(path) as f:
        camp = json.load(f)

    result = calibration.fit(camp)
    machine = detect_machine()
    meter = dict(camp["meter"])
    if meter_host:                       # best-effort exact device identity
        try:
            from tokenwatt.metersource import ShellyMeterSource
            src = ShellyMeterSource(meter_host)
            meter.update({k: v for k, v in src.device_info().items() if v is not None})
            src.close()
        except Exception as e:
            typer.echo(f"(note: could not read device identity from {meter_host}: {type(e).__name__})", err=True)

    profile = profiles.profile_from(result, machine, meter,
                                    model_calibrated_on=camp.get("model", "?"), created_at=_time.time())
    saved = profiles.save(profile, root=out)
    typer.echo(f"machine: {machine.label}")
    typer.echo(f"fit (scalar): wall_J ≈ {result.a:.3f}·rail_J + {result.b:.3f}·Δt   "
               f"residual {result.residual_rel*100:.1f}%   repeatability {result.run_variance_rel*100:.1f}%")
    typer.echo(f"tier: {result.tier}")
    typer.echo(f"wrote {saved}")
```

- [ ] **Step 8: Run test to verify it passes**

Run: `uv run pytest tests/test_cli_smoke.py -k calibrate_fit -v`
Expected: PASS (2 passed).

- [ ] **Step 9: Run the full suite**

Run: `uv run pytest -q`
Expected: PASS — all prior tests plus the new ones; no regressions.

- [ ] **Step 10: Commit**

```bash
git add src/tokenwatt/metersource.py src/tokenwatt/cli.py tests/test_metersource.py tests/test_cli_smoke.py
git commit -m "feat(calib): tokenwatt calibrate fit (dataset -> per-machine profile)"
```

---

## On-device acceptance (run on the M3 Ultra)

Not CI — the real check that closes C2, using the C1 dataset already on disk.

1. `tokenwatt calibrate fit ~/.tokenwatt/calibration/campaign-c1-full.json --meter-host <plug-ip>`
2. Confirm: the printed slope `a ≈ 1.8` (matches the ~1.7–1.9 marginal wall/rail ratios), `b ≥ 0`, a small residual, repeatability ~1–3% (from the 2 passes), and a `plug-calibrated (±x%)` tier with a single-digit band.
3. Confirm the profile landed at `~/.tokenwatt/profiles/mac15-14_apple-m3-ultra_96gb_macos26.json` and (with `--meter-host` reachable) carries the exact `S4PL-00116US`/gen-4 meter identity.
4. This profile is what C3 will load at proxy boot to stamp ledger rows `plug-calibrated`.

---

## Self-Review

**Spec coverage (§7 fit, §8 registry):**
- Scalar NNLS `E_wall_marg ≈ a·E_rail_total + b·Δt`, a,b ≥ 0 → Tasks 1–2. Pure-Python solver (no scipy) per the agreed decision.
- Band = residual ⊕ run-variance ⊕ meter accuracy, **measured not asserted**; a plug fit no tighter than the estimated floor is **uncertified**; smart-plug tier only, never lab → Task 2.
- `machine_id` = SoC + `hw.model` + RAM + macOS major, slugged, no sudo → Task 3.
- JSON-per-machine registry with `save`/`load`/`active_for_current_machine`/`list`, graceful miss → `None` → Task 4.
- Exact device identity (`model`/`gen`/`mac` from `Shelly.GetDeviceInfo`) recorded best-effort → Task 5 (the C1 dataset lacked it; enriched here when the plug is reachable, else the meter's `{name,tier,accuracy_pct}` from the dataset).
- Real-data acceptance (recover a ≈ 1.8) → Task 2 fixture test + on-device step.
- Deferred correctly: runtime `Cal`/ledger/config (C3), per-rail + condition-number/VIF gate + held-out residual (C5), wizard + `calibrate show` (C4), lab tier.

**Placeholder scan:** none — every step ships runnable code + exact commands.

**Type consistency:** `nnls(A,b) -> (x, residual)`; `fit_scalar(samples) -> (a,b,residual_rel)`; `cal_scalar(a,b,e_rail_total_j,dt_s) -> float`; `FitResult(...)`; `detect_machine(*, sysctl, mac_ver) -> MachineInfo`; `profile_from(fit, machine, meter, *, model_calibrated_on, created_at) -> Profile`; `save/load(... , root=None)` are used identically across Tasks 1→5. `campaign["meter"]["tier"/"accuracy_pct"]` and `sample["e_rail_marginal_j"]/["e_wall_marginal_j"]/["dt_s"]/["cell"]` match the C1 persisted schema exactly.

---

## Execution Handoff

Plan complete. Two execution options:

1. **Subagent-Driven (recommended)** — fresh subagent per task, spec+quality review between tasks, Opus whole-branch review at the end.
2. **Inline Execution** — tasks in this session via executing-plans, batched with checkpoints.
