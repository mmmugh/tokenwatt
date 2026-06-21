import tempfile, os
from tokenwatt.config import load_config, Config


def test_logging_defaults():
    cfg = Config()
    assert cfg.logging.level == "INFO"
    assert cfg.logging.file.endswith("proxy.jsonl")
    assert cfg.logging.console is True
    assert cfg.logging.max_bytes == 10_485_760 and cfg.logging.backup_count == 5


def test_logging_section_parsed(tmp_path):
    p = tmp_path / "tw.yaml"
    p.write_text("logging:\n  level: DEBUG\n  console: false\n  file: /tmp/x.jsonl\n")
    cfg = load_config(str(p))
    assert cfg.logging.level == "DEBUG" and cfg.logging.console is False and cfg.logging.file == "/tmp/x.jsonl"
