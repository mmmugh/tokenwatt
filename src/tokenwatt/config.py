from __future__ import annotations
import os
from typing import Literal

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator


class RouteConfig(BaseModel):
    name: str
    type: Literal["text", "vision", "embeddings"] = "text"
    dialect: Literal["openai", "anthropic"] = "openai"   # accepted now; dialect-specific token extraction deferred to a later milestone
    upstream: str
    match: list[str]

    @field_validator("upstream")
    @classmethod
    def _check_url(cls, v: str) -> str:
        if not (v.startswith("http://") or v.startswith("https://")):
            raise ValueError(f"upstream must start with http:// or https:// (got {v!r})")
        return v.rstrip("/")

    @field_validator("match")
    @classmethod
    def _check_match(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("match must list at least one pattern")
        return v


class RateConfig(BaseModel):
    flat_usd_per_kwh: float | None = None


class LoggingConfig(BaseModel):
    level: str = "INFO"
    file: str | None = "~/.tokenwatt/logs/proxy.jsonl"
    console: bool = True
    max_bytes: int = 10_485_760
    backup_count: int = 5


class Config(BaseModel):
    port: int = 7000
    host: str = "127.0.0.1"
    ledger: str = "~/.tokenwatt/ledger.sqlite"
    rate: RateConfig = Field(default_factory=RateConfig)
    routes: list[RouteConfig] = Field(default_factory=list)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    serialize_inference: bool = False   # serialize requests so per-request energy windows can't overlap (accurate metering; lower throughput)

    @model_validator(mode="after")
    def _unique_names(self) -> "Config":
        names = [r.name for r in self.routes]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(f"duplicate route name(s): {dupes}")
        return self


class ConfigError(Exception):
    pass


def default_config() -> Config:
    return Config(routes=[
        RouteConfig(name="default", type="text",
                    upstream="http://127.0.0.1:8080", match=["*"])
    ])


def _format_validation_error(path: str, err: ValidationError) -> str:
    lines = [f"invalid config in {path}:"]
    for e in err.errors():
        loc = ".".join(str(x) for x in e["loc"]) or "<root>"
        lines.append(f"  {loc}: {e['msg']}")
    return "\n".join(lines)


def load_config(path: str | None) -> Config:
    if path is None:
        return default_config()
    expanded = os.path.expanduser(path)
    if not os.path.isfile(expanded):
        raise ConfigError(f"config file not found: {path}")
    try:
        with open(expanded) as f:
            data = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise ConfigError(f"invalid YAML in {path}: {e}")
    try:
        return Config(**data)
    except ValidationError as e:
        raise ConfigError(_format_validation_error(path, e))
