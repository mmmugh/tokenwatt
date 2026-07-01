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
