from typer.testing import CliRunner
from tokenwatt.cli import app
from tokenwatt.config import load_config

runner = CliRunner()


def test_init_writes_config_that_round_trips(tmp_path):
    cfg = tmp_path / "tw.yaml"
    res = runner.invoke(app, ["init", "--config", str(cfg)])
    assert res.exit_code == 0 and cfg.exists()
    # the scaffolded file must parse cleanly and contain the three example routes
    c = load_config(str(cfg))
    names = {r.name for r in c.routes}
    assert {"m1", "v1", "embeddings"} <= names


def test_serve_bad_config_exits_1(tmp_path):
    # a missing config path must fail loud at the CLI boundary, before any backend/hardware.
    res = runner.invoke(app, ["serve", "--config", str(tmp_path / "nope.yaml")])
    assert res.exit_code == 1


def test_init_refuses_overwrite_without_force(tmp_path):
    cfg = tmp_path / "tw.yaml"
    cfg.write_text("port: 1\n")
    res = runner.invoke(app, ["init", "--config", str(cfg)])
    assert res.exit_code != 0
    assert cfg.read_text() == "port: 1\n"   # untouched
