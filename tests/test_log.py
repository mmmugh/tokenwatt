import json
import logging
from tokenwatt.log import event, JsonLineFormatter, setup_logging


def test_event_wraps_fields():
    assert event(model="m1", in_flight=1) == {"tw": {"model": "m1", "in_flight": 1}}


def test_json_formatter_emits_one_parseable_line():
    rec = logging.LogRecord("tokenwatt.proxy", logging.INFO, "f", 1, "req.start", None, None)
    rec.tw = {"model": "m1", "body_bytes": 42}
    line = JsonLineFormatter().format(rec)
    obj = json.loads(line)                       # one valid JSON object
    assert obj["event"] == "req.start" and obj["level"] == "INFO"
    assert obj["logger"] == "tokenwatt.proxy" and obj["model"] == "m1" and obj["body_bytes"] == 42
    assert "ts" in obj


def test_setup_logging_writes_jsonl_and_is_idempotent(tmp_path):
    f = str(tmp_path / "logs" / "proxy.jsonl")
    setup_logging(level="INFO", file=f, console=False)
    setup_logging(level="INFO", file=f, console=False)   # second call must not duplicate handlers
    root = logging.getLogger("tokenwatt")
    assert len(root.handlers) == 1                        # idempotent
    logging.getLogger("tokenwatt.proxy").info("req.start", extra=event(model="m1"))
    for h in root.handlers:
        h.flush()
    lines = [l for l in open(f).read().splitlines() if l.strip()]
    assert len(lines) == 1
    assert json.loads(lines[0])["model"] == "m1"
