import os
from tokenwatt.cli import _effective_log
from tokenwatt.config import LoggingConfig


def test_log_precedence_cli_over_env_over_config(monkeypatch):
    cfg = LoggingConfig(level="WARNING", file="/c.jsonl")
    monkeypatch.setenv("TOKENWATT_LOG_LEVEL", "INFO")
    # CLI wins
    assert _effective_log("DEBUG", "/cli.jsonl", cfg) == ("DEBUG", "/cli.jsonl")
    # env wins over config when no CLI level
    assert _effective_log(None, None, cfg)[0] == "INFO"
    # config file used when no CLI file
    assert _effective_log(None, None, cfg)[1] == "/c.jsonl"


def test_log_precedence_config_when_no_cli_or_env(monkeypatch):
    monkeypatch.delenv("TOKENWATT_LOG_LEVEL", raising=False)
    cfg = LoggingConfig(level="ERROR", file="/c.jsonl")
    assert _effective_log(None, None, cfg) == ("ERROR", "/c.jsonl")
    monkeypatch.setenv("TOKENWATT_LOG_LEVEL", "")          # empty env is treated as unset
    assert _effective_log(None, None, cfg)[0] == "ERROR"
