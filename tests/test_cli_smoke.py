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
