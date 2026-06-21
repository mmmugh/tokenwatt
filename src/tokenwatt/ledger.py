from __future__ import annotations
import sqlite3
from dataclasses import asdict, dataclass

_SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_start REAL, ts_end REAL, model TEXT,
    e_window_j REAL, e_idle_j REAL, e_marginal_j REAL,
    kwh_marginal REAL, rate_usd_kwh REAL, cost_marginal_usd REAL,
    tok_in INTEGER, tok_out INTEGER, tok_source TEXT, energy_confidence TEXT,
    req_type TEXT DEFAULT 'text', cold INTEGER DEFAULT 0, in_flight INTEGER DEFAULT 1,
    request_id TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS model_loads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL, model TEXT, upstream TEXT, load_energy_j REAL, duration_ms REAL, trigger TEXT
);
"""


@dataclass
class LedgerRow:
    ts_start: float
    ts_end: float
    model: str
    e_window_j: float
    e_idle_j: float
    e_marginal_j: float
    kwh_marginal: float
    rate_usd_kwh: float | None
    cost_marginal_usd: float | None
    tok_in: int | None
    tok_out: int | None
    tok_source: str
    energy_confidence: str
    req_type: str = "text"
    cold: bool = False
    in_flight: int = 1
    request_id: str = ""


class Ledger:
    def __init__(self, path: str) -> None:
        self._path = path
        with self._conn() as c:
            c.executescript(_SCHEMA)
            self._migrate(c)

    def _migrate(self, c: sqlite3.Connection) -> None:
        cols = {r["name"] for r in c.execute("PRAGMA table_info(requests)")}
        if "req_type" not in cols:        # forward-only: pre-M1b DBs gain the column
            c.execute("ALTER TABLE requests ADD COLUMN req_type TEXT DEFAULT 'text'")
        if "cold" not in cols:
            c.execute("ALTER TABLE requests ADD COLUMN cold INTEGER DEFAULT 0")
        if "in_flight" not in cols:
            c.execute("ALTER TABLE requests ADD COLUMN in_flight INTEGER DEFAULT 1")
        if "request_id" not in cols:
            c.execute("ALTER TABLE requests ADD COLUMN request_id TEXT DEFAULT ''")

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        return conn

    def insert(self, row: LedgerRow) -> None:
        d = asdict(row)
        cols = ", ".join(d)
        ph = ", ".join("?" for _ in d)
        with self._conn() as c:
            c.execute(f"INSERT INTO requests ({cols}) VALUES ({ph})", tuple(d.values()))

    def by_model(self) -> list[dict]:
        # total_usd is an un-COALESCE'd SUM by design: a group with NO priced rows surfaces as
        # None (honesty contract — never a fabricated $0). Caveat: SQLite SUM skips NULL-cost
        # rows while the token denominators COUNT every row, so a group that MIXES priced and
        # unpriced rows would understate $/Mtok. This is latent only because the rate is set
        # once per `serve` (a model's rows are all-priced or all-unpriced), not per request.
        sql = """
        SELECT model, req_type,
               COUNT(*)                            AS requests,
               COALESCE(SUM(kwh_marginal), 0)      AS total_kwh,
               SUM(cost_marginal_usd)              AS total_usd,
               COALESCE(SUM(tok_out), 0)           AS total_out,
               COALESCE(SUM(tok_in), 0)            AS total_in,
               COALESCE(SUM(e_marginal_j), 0)      AS total_marginal_j
        FROM requests GROUP BY model, req_type ORDER BY total_usd DESC
        """
        out = []
        with self._conn() as c:
            for r in c.execute(sql):
                d = dict(r)
                denom = d["total_in"] if d["req_type"] == "embedding" else d["total_out"]
                d["j_per_token"] = (d["total_marginal_j"] / denom) if denom else None
                d["usd_per_mtok"] = (d["total_usd"] / denom * 1e6) if (d["total_usd"] is not None and denom) else None
                out.append(d)
        return out

    def totals(self, since_epoch: float) -> dict:
        sql = """
        SELECT COUNT(*) AS requests,
               COALESCE(SUM(kwh_marginal), 0)      AS kwh,
               SUM(cost_marginal_usd)             AS usd
        FROM requests WHERE ts_start >= ?
        """
        with self._conn() as c:
            return dict(c.execute(sql, (since_epoch,)).fetchone())

    def insert_model_load(self, ts: float, model: str, upstream: str,
                          load_energy_j: float, duration_ms: float, trigger: str) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO model_loads (ts, model, upstream, load_energy_j, duration_ms, trigger) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (ts, model, upstream, load_energy_j, duration_ms, trigger),
            )

    def model_load_summary(self) -> dict:
        sql = "SELECT COUNT(*) AS count, COALESCE(SUM(load_energy_j), 0) AS total_load_j FROM model_loads"
        with self._conn() as c:
            return dict(c.execute(sql).fetchone())
