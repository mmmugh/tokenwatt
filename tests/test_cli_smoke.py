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
