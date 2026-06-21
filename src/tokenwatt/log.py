from __future__ import annotations
import json
import logging
import os
import sys
from logging.handlers import RotatingFileHandler


def event(**fields) -> dict:
    """Build the `extra=` payload for a structured log call:
    logger.info("req.start", extra=event(model=..., in_flight=...))."""
    return {"tw": fields}


class JsonLineFormatter(logging.Formatter):
    """One JSON object per line: ts, level, logger, event (the message), + the `tw` fields."""
    def format(self, record: logging.LogRecord) -> str:
        obj = {
            "ts": round(record.created, 3),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        tw = getattr(record, "tw", None)
        if isinstance(tw, dict):
            for k, v in tw.items():
                if k not in obj:
                    obj[k] = v
        if record.exc_info:
            obj["exc"] = self.formatException(record.exc_info)
        return json.dumps(obj, default=str)


class ConsoleFormatter(logging.Formatter):
    """Concise human line for stderr: '<LEVEL> <event>  k=v k=v'."""
    def format(self, record: logging.LogRecord) -> str:
        tw = getattr(record, "tw", None)
        kv = " ".join(f"{k}={v}" for k, v in tw.items()) if isinstance(tw, dict) else ""
        return f"{record.levelname:7} {record.getMessage()}  {kv}".rstrip()


def setup_logging(*, level: str = "INFO", file: str | None = "~/.tokenwatt/logs/proxy.jsonl",
                  console: bool = True, max_bytes: int = 10_485_760, backup_count: int = 5) -> None:
    """Configure the `tokenwatt` logger tree. Idempotent: clears prior handlers first."""
    lvl = getattr(logging, str(level).upper(), logging.INFO)
    tw_logger = logging.getLogger("tokenwatt")   # the tokenwatt PACKAGE logger, not the process root
    tw_logger.setLevel(lvl)
    tw_logger.propagate = False                  # stop tokenwatt.* double-emitting via root; uvicorn loggers untouched
    for h in list(tw_logger.handlers):
        tw_logger.removeHandler(h)
        h.close()
    if file:
        path = os.path.expanduser(file)
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        fh = RotatingFileHandler(path, maxBytes=max_bytes, backupCount=backup_count)
        fh.setFormatter(JsonLineFormatter())
        tw_logger.addHandler(fh)
    if console:
        sh = logging.StreamHandler(sys.stderr)
        sh.setFormatter(ConsoleFormatter())
        tw_logger.addHandler(sh)
