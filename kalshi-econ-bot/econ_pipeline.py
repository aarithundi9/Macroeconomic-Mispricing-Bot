"""Economic data ingestion and naive probability estimation.

Each source is implemented as a subclass of :class:`BaseEconSource`. Results
are cached in the shared SQLite database so we don't hammer the upstream APIs
on every loop iteration.
"""

from __future__ import annotations

import abc
import json
import re
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable

import requests

import config
from logger import get_logger

log = get_logger(__name__)


# ------------------------------------------------------------------ storage

def init_db(db_path: str | None = None) -> sqlite3.Connection:
    """Open (and initialize if needed) the SQLite database."""
    path = str(db_path or config.DB_PATH)
    conn = sqlite3.connect(path, isolation_level=None)
    conn.row_factory = sqlite3.Row

    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS econ_observations (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            source        TEXT    NOT NULL,
            series        TEXT    NOT NULL,
            period        TEXT    NOT NULL,
            value         REAL    NOT NULL,
            fetched_at    TEXT    NOT NULL,
            raw           TEXT,
            UNIQUE(source, series, period)
        );

        CREATE TABLE IF NOT EXISTS econ_estimates (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            category      TEXT    NOT NULL,
            probability   REAL    NOT NULL,
            confidence    REAL    NOT NULL,
            rationale     TEXT,
            updated_at    TEXT    NOT NULL,
            UNIQUE(category)
        );

        CREATE TABLE IF NOT EXISTS paper_trades (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT    NOT NULL,
            market_ticker   TEXT    NOT NULL,
            market_title    TEXT,
            direction       TEXT    NOT NULL,
            contracts       INTEGER NOT NULL,
            entry_price     REAL    NOT NULL,
            our_probability REAL    NOT NULL,
            edge_at_entry   REAL    NOT NULL,
            status          TEXT    NOT NULL DEFAULT 'open',
            exit_price      REAL,
            pnl             REAL,
            resolved_at     TEXT
        );

        CREATE TABLE IF NOT EXISTS portfolio_snapshots (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp           TEXT    NOT NULL,
            cash_balance        REAL    NOT NULL,
            open_position_value REAL    NOT NULL,
            realized_pnl        REAL    NOT NULL,
            unrealized_pnl      REAL    NOT NULL,
            total_pnl           REAL    NOT NULL,
            num_open_trades     INTEGER NOT NULL,
            num_closed_trades   INTEGER NOT NULL
        );
        """
    )
    return conn


# ------------------------------------------------------------------ models

@dataclass
class EconEstimate:
    """A naive probability estimate produced by an econ source."""

    category: str            # one of config.MARKET_KEYWORDS keys
    probability: float       # P(market resolves YES) estimated from history
    confidence: float        # 0..1, rough confidence in the estimate
    rationale: str           # human-readable explanation


# --------------------------------------------------------------- base class

class BaseEconSource(abc.ABC):
    """Abstract base for all economic data sources."""

    name: str = "base"

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    # --- subclass contract -------------------------------------------------

    @abc.abstractmethod
    def fetch(self) -> list[dict[str, Any]]:
        """Fetch observations from the upstream API. Returns a list of
        ``{"series", "period", "value", "raw"}`` dicts."""

    @abc.abstractmethod
    def estimate(self) -> list[EconEstimate]:
        """Compute probability estimates from stored observations."""

    # --- shared helpers ----------------------------------------------------

    def is_fresh(self, series: str, max_age_seconds: int = config.ECON_DATA_STALE_SECONDS) -> bool:
        """True if we have any observation for ``series`` fetched recently."""
        row = self.conn.execute(
            "SELECT fetched_at FROM econ_observations "
            "WHERE source=? AND series=? ORDER BY fetched_at DESC LIMIT 1",
            (self.name, series),
        ).fetchone()
        if not row:
            return False
        fetched_at = datetime.fromisoformat(row["fetched_at"])
        return (datetime.now(timezone.utc) - fetched_at).total_seconds() < max_age_seconds

    def store_observations(self, series: str, observations: Iterable[dict[str, Any]]) -> int:
        """Upsert observations. Returns count of rows written."""
        now = datetime.now(timezone.utc).isoformat()
        count = 0
        for obs in observations:
            try:
                self.conn.execute(
                    "INSERT OR REPLACE INTO econ_observations "
                    "(source, series, period, value, fetched_at, raw) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        self.name,
                        series,
                        obs["period"],
                        float(obs["value"]),
                        now,
                        json.dumps(obs.get("raw", {}))[:4000],
                    ),
                )
                count += 1
            except (sqlite3.Error, ValueError, KeyError) as exc:
                log.warning("Failed to store obs for %s/%s: %s", self.name, series, exc)
        return count

    def latest_values(self, series: str, limit: int = 12) -> list[float]:
        """Return the most recent ``limit`` values for ``series``, newest first."""
        rows = self.conn.execute(
            "SELECT value FROM econ_observations "
            "WHERE source=? AND series=? ORDER BY period DESC LIMIT ?",
            (self.name, series, limit),
        ).fetchall()
        return [r["value"] for r in rows]

    def save_estimate(self, est: EconEstimate) -> None:
        """Persist an estimate for later matching against Kalshi markets."""
        self.conn.execute(
            "INSERT OR REPLACE INTO econ_estimates "
            "(category, probability, confidence, rationale, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (est.category, est.probability, est.confidence, est.rationale,
             datetime.now(timezone.utc).isoformat()),
        )


# ------------------------------------------------------------------ BLS

class BLSSource(BaseEconSource):
    """Bureau of Labor Statistics source for CPI, unemployment, and NFP."""

    name = "bls"
    BASE_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"

    # Series IDs — public, well-known BLS identifiers.
    SERIES = {
        "cpi":          "CUUR0000SA0",  # CPI-U, all items, NSA
        "unemployment": "LNS14000000",  # civilian unemployment rate
        "payrolls":     "CES0000000001",  # total nonfarm employment, thousands
    }

    def fetch(self) -> list[dict[str, Any]]:
        """Pull the last ~3 years of monthly values for each series."""
        observations: list[dict[str, Any]] = []
        current_year = datetime.now(timezone.utc).year
        payload: dict[str, Any] = {
            "seriesid": list(self.SERIES.values()),
            "startyear": str(current_year - 2),
            "endyear": str(current_year),
        }
        if config.BLS_API_KEY:
            payload["registrationkey"] = config.BLS_API_KEY

        try:
            resp = requests.post(self.BASE_URL, json=payload, timeout=20)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            log.error("BLS fetch failed: %s", exc)
            return []

        series_list = data.get("Results", {}).get("series", [])
        for series in series_list:
            sid = series.get("seriesID")
            # Map BLS series ID back to our human-friendly name.
            label = next((k for k, v in self.SERIES.items() if v == sid), sid)
            for entry in series.get("data", []):
                try:
                    value = float(entry["value"])
                except (TypeError, ValueError):
                    continue
                # Convert payrolls from thousands (BLS units) to raw job counts
                # so Kalshi strike thresholds align with the distribution.
                if label == "payrolls":
                    value *= 1000
                period = f"{entry.get('year')}-{entry.get('period', '').lstrip('M')}"
                observations.append({
                    "series": label, "period": period,
                    "value": value, "raw": entry,
                })
            self.store_observations(label, [o for o in observations if o["series"] == label])

        log.info("BLS: fetched %d observations", len(observations))
        return observations

    def estimate(self) -> list[EconEstimate]:
        """Estimate P(next print above prior period) per category."""
        estimates: list[EconEstimate] = []
        for category in ("cpi", "unemployment", "payrolls"):
            values = self.latest_values(category, limit=13)
            if len(values) < 3:
                continue
            # values are newest-first; compute MoM direction excluding the newest
            # print (that's the one the market is asking about).
            history = values[1:]
            ups = sum(1 for a, b in zip(history[:-1], history[1:]) if a > b)
            total = len(history) - 1
            if total == 0:
                continue
            prob_up = ups / total
            confidence = min(1.0, total / 12.0)
            est = EconEstimate(
                category=category,
                probability=prob_up,
                confidence=confidence,
                rationale=(
                    f"BLS {category}: {ups}/{total} recent months printed above prior"
                ),
            )
            self.save_estimate(est)
            estimates.append(est)
        return estimates


# ------------------------------------------------------------------ FRED

class FREDSource(BaseEconSource):
    """FRED source for broader macro indicators (e.g. Fed funds rate)."""

    name = "fred"
    BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

    SERIES = {
        "fed_rate": "FEDFUNDS",         # effective federal funds rate, monthly
        "oil":      "DCOILWTICO",       # WTI crude oil spot, daily
        "gdp":      "A191RL1Q225SBEA",  # real GDP QoQ SAAR, quarterly
    }

    def fetch(self) -> list[dict[str, Any]]:
        """Fetch observations for each FRED series we care about."""
        if not config.FRED_API_KEY:
            log.warning("FRED_API_KEY missing; skipping FRED fetch.")
            return []

        observations: list[dict[str, Any]] = []
        for label, sid in self.SERIES.items():
            params = {
                "series_id": sid,
                "api_key": config.FRED_API_KEY,
                "file_type": "json",
                "sort_order": "desc",
                "limit": 36,
            }
            try:
                resp = requests.get(self.BASE_URL, params=params, timeout=15)
                resp.raise_for_status()
                data = resp.json()
            except requests.RequestException as exc:
                log.error("FRED fetch failed for %s: %s", label, exc)
                continue

            series_obs = []
            for entry in data.get("observations", []):
                try:
                    value = float(entry["value"])
                except (TypeError, ValueError):
                    continue
                series_obs.append({
                    "series": label, "period": entry["date"],
                    "value": value, "raw": entry,
                })
            self.store_observations(label, series_obs)
            observations.extend(series_obs)

        log.info("FRED: fetched %d observations", len(observations))
        return observations

    def estimate(self) -> list[EconEstimate]:
        """Estimate P(Fed holds) and P(WTI oil up) from recent history."""
        estimates: list[EconEstimate] = []

        values = self.latest_values("fed_rate", limit=12)
        if len(values) >= 2:
            # values newest-first; an unchanged rate recently suggests the Fed
            # is more likely to hold than to move.
            diffs = [a - b for a, b in zip(values[:-1], values[1:])]
            holds = sum(1 for d in diffs if abs(d) < 0.05)
            total = len(diffs)
            prob_hold = holds / total if total else 0.5
            est = EconEstimate(
                category="fed_rate",
                probability=prob_hold,
                confidence=min(1.0, total / 12.0),
                rationale=f"FRED fed funds: {holds}/{total} recent months unchanged",
            )
            self.save_estimate(est)
            estimates.append(est)

        oil_values = self.latest_values("oil", limit=30)
        if len(oil_values) >= 5:
            ups = sum(1 for a, b in zip(oil_values[:-1], oil_values[1:]) if a > b)
            total = len(oil_values) - 1
            prob_up = ups / total if total else 0.5
            est = EconEstimate(
                category="oil",
                probability=prob_up,
                confidence=min(1.0, total / 30.0),
                rationale=f"FRED WTI: {ups}/{total} recent days printed above prior",
            )
            self.save_estimate(est)
            estimates.append(est)

        return estimates


# ------------------------------------------------------------------ Truflation

class TruflationSource(BaseEconSource):
    """Truflation real-time US CPI nowcast.

    Truflation scrapes ~30 retailers daily to produce a live CPI-style index,
    which is then:

    - The settlement oracle for Kalshi's KXTRUFCPI markets (dispositive)
    - A nowcast for BLS KXCPI markets (~0.8 correlation)

    The source is a no-op when ``TRUFLATION_API_URL`` isn't set, so the bot
    keeps running while credentials are being sorted out. When it does fetch,
    it stores one observation under series ``cpi_yoy`` — the most recent
    reading Truflation has published.

    Response parsing is intentionally flexible. Truflation's API shape varies
    by endpoint and may change; we try common field names and fall back to
    skipping the reading if none are present, with the raw payload logged.
    """

    name = "truflation"
    SERIES = {"cpi_yoy": "us_cpi_yoy"}

    # Common field names seen across Truflation's public/widget endpoints.
    _VALUE_FIELDS = ("value", "rate", "yoy", "current_value", "inflation_rate",
                     "us_cpi_yoy", "currentValue", "index_value")
    _DATE_FIELDS = ("as_of", "date", "timestamp", "updated_at", "asOf")

    def fetch(self) -> list[dict[str, Any]]:
        url = config.TRUFLATION_API_URL
        if not url:
            log.debug("Truflation: TRUFLATION_API_URL unset; skipping fetch.")
            return []

        headers = {"User-Agent": "kalshi-econ-bot/0.1"}
        if config.TRUFLATION_API_KEY:
            headers["Authorization"] = f"Bearer {config.TRUFLATION_API_KEY}"

        try:
            resp = requests.get(url, headers=headers, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            log.error("Truflation fetch failed: %s", exc)
            return []
        except ValueError as exc:
            log.error("Truflation response was not JSON: %s", exc)
            return []

        # Some endpoints wrap the payload in {"data": {...}} — unwrap if so.
        if isinstance(data, dict) and "data" in data and isinstance(data["data"], dict):
            data = data["data"]

        value = None
        for field in self._VALUE_FIELDS:
            if isinstance(data, dict) and field in data and data[field] is not None:
                value = data[field]
                break
        if value is None:
            log.warning("Truflation response missing value field; raw=%s",
                        str(data)[:300])
            return []

        try:
            value = float(value)
        except (TypeError, ValueError):
            log.warning("Truflation value not numeric: %r", value)
            return []

        as_of = None
        for field in self._DATE_FIELDS:
            if isinstance(data, dict) and data.get(field):
                as_of = str(data[field])[:10]  # trim ISO datetime to date
                break
        if as_of is None:
            as_of = datetime.now(timezone.utc).date().isoformat()

        obs = [{"series": "cpi_yoy", "period": as_of,
                "value": value, "raw": data}]
        self.store_observations("cpi_yoy", obs)
        log.info("Truflation: US CPI nowcast = %.3f as of %s", value, as_of)
        return obs

    def estimate(self) -> list[EconEstimate]:
        """No direction-only estimate; the nowcast is consumed directly by
        the strategy via ``EconPipeline.truflation_nowcast()``."""
        return []


# ------------------------------------------------------------------ Cleveland Fed

class ClevelandFedNowcastSource(BaseEconSource):
    """Cleveland Fed daily inflation nowcast (CPI / Core CPI / PCE / Core PCE).

    Their model has ~0.9 correlation with the eventual BLS print, making it
    the strongest free real-time nowcast for KXCPI markets.

    We read from a CSV the user manually drops into ``data/`` (Cleveland
    Fed's site lacks a stable public download URL — the JSON endpoints power
    a default historical example chart, not the live data). Filename pattern:
    ``QuarterlyAnnualizedPercentChange-YYYY-qN.csv``. Columns:
    ``Label,CPI Inflation,Core CPI Inflation,PCE Inflation,Core PCE Inflation``.

    Values are stored in their native units (**quarterly annualized %**). The
    strategy converts to MoM when needed: MoM ≈ quarterly_annualized / 12.
    """

    name = "cleveland_fed"
    SERIES = {
        "cpi_q_annualized":      "CPI Inflation",
        "core_cpi_q_annualized": "Core CPI Inflation",
        "pce_q_annualized":      "PCE Inflation",
        "core_pce_q_annualized": "Core PCE Inflation",
    }

    def _latest_csv(self) -> str | None:
        """Find the most recently modified Cleveland Fed CSV in the data dir."""
        import os
        from pathlib import Path

        csv_dir = Path(config.CLEVELAND_FED_CSV_DIR)
        if not csv_dir.exists():
            log.debug("Cleveland Fed: CSV dir %s missing; skipping.", csv_dir)
            return None
        candidates = list(csv_dir.glob("*.csv"))
        if not candidates:
            log.debug("Cleveland Fed: no CSV files in %s; skipping.", csv_dir)
            return None
        # Newest file wins — user typically re-downloads to refresh.
        latest = max(candidates, key=lambda p: os.path.getmtime(p))
        return str(latest)

    def fetch(self) -> list[dict[str, Any]]:
        path = self._latest_csv()
        if path is None:
            return []

        import csv as _csv
        observations: list[dict[str, Any]] = []
        try:
            with open(path, "r", encoding="utf-8") as fh:
                reader = _csv.DictReader(fh)
                rows = list(reader)
        except (OSError, _csv.Error) as exc:
            log.error("Cleveland Fed CSV read failed (%s): %s", path, exc)
            return []

        if not rows:
            log.warning("Cleveland Fed CSV %s is empty.", path)
            return []

        # Use the most recent row in the file (CSV is chronological).
        last = rows[-1]
        label = last.get("Label", "").strip()
        # Label is "MM/DD" — combine with the year from the filename if present.
        year = datetime.now(timezone.utc).year
        try:
            import re as _re
            m = _re.search(r"-(\d{4})-", path)
            if m:
                year = int(m.group(1))
        except Exception:
            pass
        period = f"{year}-{label.replace('/', '-')}" if label else datetime.now(timezone.utc).date().isoformat()

        for series_label, csv_column in self.SERIES.items():
            raw = last.get(csv_column)
            if raw is None or raw == "":
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError):
                log.warning("Cleveland Fed: non-numeric %s = %r", csv_column, raw)
                continue
            obs = {
                "series": series_label, "period": period,
                "value": value, "raw": {"row": last, "source_file": path},
            }
            self.store_observations(series_label, [obs])
            observations.append(obs)

        log.info("Cleveland Fed: stored %d nowcasts as of %s (from %s)",
                 len(observations), period, path)
        return observations

    def estimate(self) -> list[EconEstimate]:
        """No direction-only estimate; nowcast consumed via ``cpi_nowcast()``."""
        return []


# ------------------------------------------------------------------ yfinance

class YFinanceSource(BaseEconSource):
    """Live market data via yfinance (Yahoo Finance scraper).

    Pulls daily close prices for tradeable assets. WTI front-month futures
    (``CL=F``) replaces FRED's ``DCOILWTICO`` since FRED has a ~1-day lag
    that anchors stale prices in the strike-history distribution.
    """

    name = "yfinance"
    SERIES = {
        "oil": "CL=F",  # WTI front-month futures, NYMEX
    }

    def fetch(self) -> list[dict[str, Any]]:
        try:
            import yfinance as yf
        except ImportError:
            log.warning("yfinance not installed; run: pip install yfinance")
            return []

        observations: list[dict[str, Any]] = []
        for label, ticker in self.SERIES.items():
            try:
                hist = yf.Ticker(ticker).history(period="90d", interval="1d")
            except Exception as exc:  # yfinance can raise many things
                log.error("yfinance fetch failed for %s: %s", ticker, exc)
                continue
            if hist is None or hist.empty:
                log.warning("yfinance returned empty history for %s", ticker)
                continue

            series_obs: list[dict[str, Any]] = []
            for date, row in hist.iterrows():
                try:
                    close = float(row["Close"])
                except (KeyError, TypeError, ValueError):
                    continue
                series_obs.append({
                    "series": label,
                    "period": date.strftime("%Y-%m-%d"),
                    "value": close,
                    "raw": {"close": close, "ticker": ticker},
                })
            self.store_observations(label, series_obs)
            observations.extend(series_obs)
            log.info("yfinance %s: stored %d daily closes (latest=%.2f)",
                     ticker, len(series_obs),
                     series_obs[-1]["value"] if series_obs else float("nan"))
        return observations

    def estimate(self) -> list[EconEstimate]:
        """Direction-only estimate for oil (parallel to FREDSource.oil)."""
        values = self.latest_values("oil", limit=30)
        if len(values) < 5:
            return []
        ups = sum(1 for a, b in zip(values[:-1], values[1:]) if a > b)
        total = len(values) - 1
        prob_up = ups / total if total else 0.5
        est = EconEstimate(
            category="oil",
            probability=prob_up,
            confidence=min(1.0, total / 30.0),
            rationale=f"yfinance WTI CL=F: {ups}/{total} recent days printed above prior",
        )
        self.save_estimate(est)
        return [est]


# ------------------------------------------------------------------ Fed calendar

class FedCalendarSource(BaseEconSource):
    """Scrape the Federal Reserve's public FOMC calendar page."""

    name = "fed_calendar"
    URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"

    def fetch(self) -> list[dict[str, Any]]:
        """Return upcoming FOMC meeting dates as observations."""
        try:
            resp = requests.get(self.URL, timeout=15,
                                headers={"User-Agent": "kalshi-econ-bot/0.1"})
            resp.raise_for_status()
        except requests.RequestException as exc:
            log.error("Fed calendar fetch failed: %s", exc)
            return []

        # Very light parse: pull yyyy and month/day patterns. Good enough to
        # know when the next FOMC window is — we don't trade on this directly.
        text = resp.text
        year_matches = re.findall(r"fomc.*?(\d{4})", text, flags=re.IGNORECASE)
        unique_years = sorted(set(year_matches))[-3:] if year_matches else []
        observations = [
            {"series": "fomc_year", "period": y, "value": float(y), "raw": {}}
            for y in unique_years
        ]
        self.store_observations("fomc_year", observations)
        log.info("Fed calendar: recorded %d recent FOMC year markers", len(observations))
        return observations

    def estimate(self) -> list[EconEstimate]:
        """No direct probability estimate — this is calendar data only."""
        return []


# --------------------------------------------------------------- orchestrator

class EconPipeline:
    """Wires the sources together and exposes cached estimates."""

    def __init__(self, conn: sqlite3.Connection | None = None) -> None:
        self.conn = conn or init_db()
        self.sources: list[BaseEconSource] = [
            BLSSource(self.conn),
            FREDSource(self.conn),
            FedCalendarSource(self.conn),
            TruflationSource(self.conn),
            ClevelandFedNowcastSource(self.conn),
            YFinanceSource(self.conn),
        ]

    def refresh(self, force: bool = False) -> None:
        """Refresh every source whose data is stale (or all if ``force``)."""
        for src in self.sources:
            try:
                any_series = next(iter(getattr(src, "SERIES", {"_": None}).keys()))
                if force or not src.is_fresh(any_series):
                    log.info("Refreshing econ source: %s", src.name)
                    src.fetch()
                    src.estimate()
                else:
                    log.debug("Econ source %s still fresh, skipping fetch", src.name)
            except Exception as exc:  # never let one source break the loop
                log.exception("Error refreshing %s: %s", src.name, exc)

    def current_estimates(self) -> dict[str, EconEstimate]:
        """Load all stored estimates, keyed by category."""
        rows = self.conn.execute(
            "SELECT category, probability, confidence, rationale FROM econ_estimates"
        ).fetchall()
        return {
            r["category"]: EconEstimate(
                category=r["category"],
                probability=r["probability"],
                confidence=r["confidence"],
                rationale=r["rationale"] or "",
            )
            for r in rows
        }

    # Map category → (source_name, series_label, transform).
    #
    # The ``transform`` produces a sample set drawn (roughly) from the
    # predictive distribution of the next strike-unit value. Stationary
    # change-series (CPI MoM %, payrolls MoM diff, GDP QoQ %) use the raw
    # change distribution. Persistent/trending levels (fed funds, WTI oil,
    # unemployment rate) use a one-step random walk ``current + Δ`` so the
    # model isn't anchored to long-run level history that has no bearing on
    # next period's draw.
    _STRIKE_SPEC: dict[str, tuple[str, str, str]] = {
        "cpi":            ("bls",        "cpi",          "pct_change"),
        "payrolls":       ("bls",        "payrolls",     "diff"),
        "unemployment":   ("bls",        "unemployment", "random_walk"),
        "fed_rate":       ("fred",       "fed_rate",     "random_walk"),
        "oil":            ("yfinance",   "oil",          "random_walk"),
        "gdp":            ("fred",       "gdp",          "as_is"),
        # KXTRUFCPI settles on the Truflation index itself; the single latest
        # reading *is* the strike unit. We keep an ``as_is`` spec here so the
        # strike_history machinery returns something, but strategy.py special-
        # cases this category to use the nowcast directly with a tight σ.
        "truflation_cpi": ("truflation", "cpi_yoy",      "as_is"),
    }

    def truflation_nowcast(self) -> tuple[float, str] | None:
        """Return ``(value, as_of_date)`` of the latest Truflation reading,
        or ``None`` if no observation has been stored yet."""
        row = self.conn.execute(
            "SELECT value, period FROM econ_observations "
            "WHERE source='truflation' AND series='cpi_yoy' "
            "ORDER BY fetched_at DESC LIMIT 1"
        ).fetchone()
        if not row:
            return None
        return (float(row["value"]), str(row["period"]))

    def cpi_nowcast_mom(self) -> tuple[float, str] | None:
        """Return ``(MoM_pct, as_of_date)`` of the latest CPI nowcast.

        Prefers Cleveland Fed (quarterly annualized → divided by 12 for MoM).
        Falls back to Truflation YoY (÷12, rough approximation) if Cleveland
        Fed has no data yet. Returns ``None`` if neither source has data.
        """
        row = self.conn.execute(
            "SELECT value, period FROM econ_observations "
            "WHERE source='cleveland_fed' AND series='cpi_q_annualized' "
            "ORDER BY fetched_at DESC LIMIT 1"
        ).fetchone()
        if row:
            # Quarterly annualized → monthly: divide by 12 (rough but works
            # because the quarterly figure is itself constructed from the
            # implied monthly run rate of inflation).
            return (float(row["value"]) / 12.0, str(row["period"]))

        tru = self.truflation_nowcast()
        if tru is not None:
            yoy, as_of = tru
            return (yoy / 12.0, as_of)
        return None

    def strike_history(self, category: str, limit: int = 60) -> list[float]:
        """Return ``limit`` samples from the next-period predictive distribution
        in Kalshi-strike units. The caller fits a normal and evaluates at the
        market's strike.
        """
        spec = self._STRIKE_SPEC.get(category)
        if spec is None:
            return []
        source, series, transform = spec
        rows = self.conn.execute(
            "SELECT value FROM econ_observations "
            "WHERE source=? AND series=? ORDER BY period DESC LIMIT ?",
            (source, series, limit),
        ).fetchall()
        levels = [r["value"] for r in rows]  # newest-first
        if len(levels) < 2:
            return levels

        if transform == "as_is":
            # Each stored value is already a draw in strike units (e.g. GDP
            # QoQ SAAR % change). Fit directly.
            return levels

        if transform == "diff":
            return [new - old for new, old in zip(levels[:-1], levels[1:])]

        if transform == "pct_change":
            out: list[float] = []
            for new, old in zip(levels[:-1], levels[1:]):
                if old > 0:
                    out.append((new - old) / old * 100.0)
            return out

        if transform == "random_walk":
            # Next level ≈ current + Δ where Δ is drawn from historical
            # period-over-period changes. Samples are ``current + d_i``.
            current = levels[0]
            diffs = [new - old for new, old in zip(levels[:-1], levels[1:])]
            return [current + d for d in diffs]

        return levels


def _throttle() -> None:
    """Tiny helper to avoid hammering free APIs on tight loops in tests."""
    time.sleep(0.1)
