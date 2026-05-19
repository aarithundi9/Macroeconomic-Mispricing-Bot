"""Central configuration for the Kalshi economic trading bot.

All tunable thresholds, environment-driven settings, and file paths live here
so they can be changed in one place without hunting through the code.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Kalshi ------------------------------------------------------------------

DEMO_MODE: bool = False
# Shadow mode: we read real prod prices but never place a real order. The
# place_order method raises if this is True, so it's impossible to send an
# accidental order while shadow-trading.
SHADOW_MODE: bool = True

KALSHI_DEMO_BASE_URL: str = "https://demo-api.kalshi.co/trade-api/v2"
KALSHI_PROD_BASE_URL: str = "https://api.elections.kalshi.com/trade-api/v2"
KALSHI_BASE_URL: str = KALSHI_DEMO_BASE_URL if DEMO_MODE else KALSHI_PROD_BASE_URL

# Kalshi econ series we want to scan on each loop. Queried directly via
# ``series_ticker=`` so we don't have to paginate through thousands of
# sports markets. Each prefix maps to one of our internal categories.
ECON_SERIES_TICKERS: list[str] = [
    "KXCPI", "KXFED", "KXFEDDECISION", "KXPAYROLLS",
    "KXUNEMPLOYMENT", "KXWTI", "KXGDP", "KXTRUFCPI",
]

SERIES_TO_CATEGORY: dict[str, str] = {
    "KXCPI":          "cpi",
    "KXFED":          "fed_rate",
    "KXFEDDECISION":  "fed_decision",   # discrete — handled separately
    "KXPAYROLLS":     "payrolls",
    "KXUNEMPLOYMENT": "unemployment",
    "KXWTI":          "oil",
    "KXGDP":          "gdp",
    "KXTRUFCPI":      "truflation_cpi",  # settles directly on Truflation index
}

KALSHI_API_KEY: str = os.getenv("KALSHI_API_KEY", "")
KALSHI_PRIVATE_KEY_PATH: str = os.getenv("KALSHI_PRIVATE_KEY_PATH", "./kalshi_private_key.pem")
KALSHI_PRIVATE_KEY_PASSWORD: str | None = os.getenv("KALSHI_PRIVATE_KEY_PASSWORD") or None

# --- Economic data APIs ------------------------------------------------------

FRED_API_KEY: str = os.getenv("FRED_API_KEY", "")
BLS_API_KEY: str = os.getenv("BLS_API_KEY", "")

# --- Truflation -------------------------------------------------------------

# Truflation publishes a real-time US CPI-like index updated daily. Kalshi's
# KXTRUFCPI markets settle directly on it, and it's a useful nowcast for BLS
# KXCPI markets. The source is a no-op if TRUFLATION_API_URL is unset, so the
# rest of the bot keeps running while you're still getting access set up.
#
# Sign up at https://truflation.com/ for API credentials, then set
# TRUFLATION_API_URL to the endpoint that returns the current US CPI figure.
TRUFLATION_API_URL: str = os.getenv("TRUFLATION_API_URL", "")
TRUFLATION_API_KEY: str = os.getenv("TRUFLATION_API_KEY", "")

# How much weight to put on the Truflation nowcast vs. the historical BLS CPI
# distribution when computing μ for KXCPI markets. Truflation's YoY has ~0.8
# correlation with the BLS print; start conservative and bump up once you've
# checked backtested accuracy.
TRUFLATION_NOWCAST_WEIGHT: float = 0.6

# --- Cleveland Fed nowcast --------------------------------------------------
#
# Cleveland Fed publishes daily CPI/PCE nowcasts ("inflation nowcasting").
# Their model has ~0.9 correlation with the eventual BLS print, making it the
# strongest free real-time nowcast for KXCPI markets. We read whatever CSV
# the user has dropped into ``data/`` (e.g. ``QuarterlyAnnualizedPercentChange
# -2026-q2.csv``). Format: ``Label,CPI Inflation,Core CPI,PCE,Core PCE``.
CLEVELAND_FED_CSV_DIR: Path = Path(__file__).resolve().parent.parent / "data"

# Weight on the Cleveland Fed nowcast vs. historical BLS CPI distribution
# when computing μ for KXCPI markets. Higher than Truflation's because the
# nowcast is specifically designed to predict the BLS print.
CPI_NOWCAST_WEIGHT: float = 0.7

# --- Strategy ----------------------------------------------------------------

EDGE_THRESHOLD: float = 0.10            # minimum absolute edge to flag a trade
MIN_CONFIDENCE: float = 0.55            # legacy; kept for back-compat
MIN_MODEL_CONFIDENCE: float = 0.30      # strike-aware fit needs enough history
SCAN_INTERVAL_SECONDS: int = 300        # 5 minutes between scans
ECON_DATA_STALE_SECONDS: int = 3600     # re-fetch econ data after 1 hour

# Keyword → econ source mapping used for naive market matching.
# Keep terms specific enough to avoid sports false-positives (NHL "Hurricanes",
# NBA player "Harris", etc.). Prefer multi-word or rare tokens.
MARKET_KEYWORDS: dict[str, list[str]] = {
    "cpi":           ["cpi", "inflation", "consumer price"],
    "unemployment":  ["unemployment rate", "jobless claims", "initial claims"],
    "payrolls":      ["nonfarm", "payroll", "jobs report", "nfp"],
    "fed_rate":      ["fomc", "rate decision", "federal funds", "fed hike",
                      "fed cut", "fed rate"],
    "oil":           ["wti", "crude oil", "oil price", "brent"],
}

# Minimum quoted liquidity for a market to be considered tradeable.
# Kalshi demo lists many econ markets with zero quotes; scanning them wastes
# compute and produces meaningless mid-prices.
MIN_QUOTE_DOLLARS: float = 0.01

# --- Paper trading -----------------------------------------------------------

STARTING_BANKROLL: float = 1000.00
MAX_POSITION_SIZE: float = 50.00        # dollars per trade
KELLY_FRACTION: float = 0.25            # quarter Kelly for safety
MAX_OPEN_TRADES: int = 10               # hard cap on concurrent positions
MAX_NEW_TRADES_PER_LOOP: int = 3        # only take the top-N fresh signals each loop

# --- Paths -------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parent
DB_PATH: Path = PROJECT_ROOT / "econ_data.db"
OPPORTUNITIES_LOG: Path = PROJECT_ROOT / "opportunities.log"
TRADES_HISTORY_CSV: Path = PROJECT_ROOT / "trades_history.csv"
APP_LOG_PATH: Path = PROJECT_ROOT / "bot.log"

# --- Logging -----------------------------------------------------------------

LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
