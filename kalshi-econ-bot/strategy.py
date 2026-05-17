"""Market matching, edge computation, and opportunity logging.

This module is deliberately pure: it takes Kalshi markets and econ estimates,
produces :class:`TradeSignal` objects, and writes opportunities to disk. It
performs no network I/O itself.
"""

from __future__ import annotations

import csv
import re
import statistics
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import config
from econ_pipeline import EconEstimate, EconPipeline
from logger import get_logger

log = get_logger(__name__)


@dataclass
class TradeSignal:
    """A flagged trading opportunity with enough context to paper-trade it."""

    timestamp: str
    market_ticker: str
    market_title: str
    market_price: float          # yes price in [0, 1]
    our_probability: float       # our estimated P(YES)
    edge: float                  # abs(our - market)
    confidence: float            # 0..1 from the econ source
    action: str                  # "buy_yes" or "buy_no"
    category: str                # which econ category this matches
    rationale: str               # why we think this is mispriced


# ----------------------------------------------------------------- matching

def _market_text(market: dict[str, Any]) -> str:
    """Flatten the text fields we match keywords against."""
    parts = [
        market.get("title", ""),
        market.get("subtitle", ""),
        market.get("ticker", ""),
        market.get("event_ticker", ""),
        market.get("category", ""),
    ]
    return " ".join(p for p in parts if p).lower()


def _compile_patterns() -> dict[str, re.Pattern[str]]:
    """Build word-boundary regexes from the keyword dict.

    Plain substring matching hits accidents like ``Brentford`` → ``brent``.
    Wrapping each keyword in ``\\b`` eliminates sports-team false positives.
    """
    patterns: dict[str, re.Pattern[str]] = {}
    for category, kws in config.MARKET_KEYWORDS.items():
        alts = "|".join(re.escape(k) for k in kws)
        patterns[category] = re.compile(rf"\b(?:{alts})\b", re.IGNORECASE)
    return patterns


_PATTERNS = _compile_patterns()


def match_category(market: dict[str, Any]) -> str | None:
    """Return the econ category this market belongs to, or ``None``.

    Prefers the market's ``series_ticker`` (authoritative) and falls back to
    keyword matching on the title so hand-off from legacy code keeps working.
    """
    st = market.get("series_ticker") or ""
    if st and st in config.SERIES_TO_CATEGORY:
        return config.SERIES_TO_CATEGORY[st]

    # event_ticker looks like ``KXCPI-26MAY`` — strip to series prefix.
    et = market.get("event_ticker") or market.get("ticker") or ""
    prefix = et.split("-", 1)[0] if "-" in et else et
    if prefix in config.SERIES_TO_CATEGORY:
        return config.SERIES_TO_CATEGORY[prefix]

    text = _market_text(market)
    for category, pattern in _PATTERNS.items():
        if pattern.search(text):
            return category
    return None


# ----------------------------------------------------------------- strikes

@dataclass
class Strike:
    """Parsed strike info from a Kalshi market payload."""

    type: str               # "greater" | "less" | "between"
    floor: float            # lower bound / single threshold
    cap: float | None       # upper bound for "between" markets


def parse_strike(market: dict[str, Any]) -> Strike | None:
    """Extract a :class:`Strike` from a Kalshi market, or ``None`` if absent."""
    stype = (market.get("strike_type") or "").lower()
    floor = _num(market, "floor_strike")
    cap = _num(market, "cap_strike")
    if floor is None and cap is None:
        return None
    if stype in ("greater", "less"):
        if floor is None:
            return None
        return Strike(type=stype, floor=floor, cap=None)
    if stype == "between":
        if floor is None or cap is None:
            return None
        return Strike(type="between", floor=floor, cap=cap)
    # Unknown strike types (``structured``, etc.) — skip.
    return None


def _normal_cdf(z: float) -> float:
    """Standard normal CDF via ``statistics.NormalDist`` (stdlib only)."""
    return statistics.NormalDist().cdf(z)


def strike_probability(
    history: list[float],
    strike: Strike,
    mu_override: float | None = None,
    sigma_scale: float = 1.0,
) -> tuple[float, int, float, float] | None:
    """Return ``(probability, sample_size, mu, sigma)`` that history fits the strike.

    Fits a normal distribution to ``history`` and evaluates the strike's
    implied event under that distribution. Returns ``None`` when we have too
    few points or zero variance to produce a meaningful probability.

    ``mu_override``: replaces the historical mean (used for Truflation-anchored
    CPI markets so the distribution is centered on the live nowcast).

    ``sigma_scale``: multiplier on the fitted σ (used for KXTRUFCPI, where the
    current reading is nearly dispositive so we tighten the distribution).
    """
    n = len(history)
    if n < 6:
        return None
    mu = statistics.fmean(history) if mu_override is None else mu_override
    sigma_hist = statistics.pstdev(history) if n < 2 else statistics.stdev(history)
    sigma = sigma_hist * sigma_scale
    if sigma <= 1e-9:
        return None

    if strike.type == "greater":
        prob = 1.0 - _normal_cdf((strike.floor - mu) / sigma)
    elif strike.type == "less":
        prob = _normal_cdf((strike.floor - mu) / sigma)
    elif strike.type == "between" and strike.cap is not None:
        prob = (_normal_cdf((strike.cap - mu) / sigma)
                - _normal_cdf((strike.floor - mu) / sigma))
    else:
        return None

    return max(0.0, min(1.0, prob)), n, mu, sigma


def _confidence_from_sample(n: int, max_n: int = 36) -> float:
    """Monotone saturating confidence: ~n/max_n, clipped to [0, 1]."""
    return max(0.0, min(1.0, n / max_n))


# ----------------------------------------------------------------- pricing

def _num(market: dict[str, Any], field: str) -> float | None:
    """Safe numeric access — returns None if field is missing or non-numeric."""
    v = market.get(field)
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def yes_price(market: dict[str, Any]) -> float | None:
    """Extract the best YES price from a Kalshi market payload, in [0, 1].

    Kalshi's current v2 payload uses ``*_dollars`` fields already in the [0, 1]
    range. Older payloads used cent-scale ``yes_bid`` / ``yes_ask`` / ``last_price``.
    We try dollar fields first, fall back to cents, and prefer the bid/ask
    midpoint when both quotes exist (more informative than a stale last price).
    A market with zero bid **and** zero ask has no liquidity — we return
    ``None`` so the caller skips it rather than trading against a phantom 0.50.
    """
    bid_d = _num(market, "yes_bid_dollars")
    ask_d = _num(market, "yes_ask_dollars")
    if bid_d is not None and ask_d is not None:
        if bid_d < config.MIN_QUOTE_DOLLARS and ask_d < config.MIN_QUOTE_DOLLARS:
            return None
        return max(0.0, min(1.0, (bid_d + ask_d) / 2.0))

    last_d = _num(market, "last_price_dollars")
    if last_d is not None and last_d > 0:
        return max(0.0, min(1.0, last_d))

    bid_c = _num(market, "yes_bid")
    ask_c = _num(market, "yes_ask")
    if bid_c is not None and ask_c is not None:
        if bid_c == 0 and ask_c == 0:
            return None
        return max(0.0, min(1.0, (bid_c + ask_c) / 200.0))

    last_c = _num(market, "last_price")
    if last_c is not None and last_c > 0:
        return max(0.0, min(1.0, last_c / 100.0))

    return None


# ------------------------------------------------------------------ signals

def evaluate_market(
    market: dict[str, Any],
    pipeline: EconPipeline,
    estimates: dict[str, EconEstimate],
) -> TradeSignal | None:
    """Produce a :class:`TradeSignal` for ``market`` if edge clears the bar.

    Tries a **strike-aware** probability first: fits a normal distribution to
    the category's history (in strike units) and evaluates P(event) at the
    market's strike. Falls back to the naive direction-only
    :class:`EconEstimate` only when the strike can't be parsed.
    """
    category = match_category(market)
    if category is None:
        return None

    price = yes_price(market)
    if price is None:
        return None

    our_p: float | None = None
    confidence = 0.0
    rationale = ""

    strike = parse_strike(market)
    if strike is not None:
        history = pipeline.strike_history(category)

        # Truflation nowcast integration:
        # - KXTRUFCPI settles on Truflation directly → center μ on the current
        #   reading and tighten σ since the value is nearly known.
        # - KXCPI (BLS) → blend historical μ toward the nowcast.
        # - Other categories → unaffected.
        mu_override: float | None = None
        sigma_scale = 1.0
        nowcast = pipeline.truflation_nowcast() if category in ("cpi", "truflation_cpi") else None
        if nowcast is not None:
            nc_value, _ = nowcast
            if category == "truflation_cpi":
                mu_override = nc_value
                sigma_scale = 0.3
            elif category == "cpi":
                w = config.TRUFLATION_NOWCAST_WEIGHT
                hist_mu = statistics.fmean(history) if history else nc_value
                mu_override = w * nc_value + (1.0 - w) * hist_mu

        result = strike_probability(history, strike,
                                    mu_override=mu_override,
                                    sigma_scale=sigma_scale)
        if result is not None:
            our_p, n, mu, sigma = result
            confidence = _confidence_from_sample(n)
            nc_tag = ""
            if nowcast is not None and category in ("cpi", "truflation_cpi"):
                nc_tag = f" [truflation={nowcast[0]:.3f} @ {nowcast[1]}]"
            rationale = (
                f"strike-model {category}: n={n}, mu={mu:.3f}, sigma={sigma:.3f}, "
                f"strike={strike.type} {strike.floor}"
                + (f"..{strike.cap}" if strike.cap is not None else "")
                + nc_tag
            )

    if our_p is None:
        # Fallback: legacy direction-only estimate. Much lower confidence.
        est = estimates.get(category)
        if est is None:
            return None
        our_p = est.probability
        confidence = est.confidence * 0.5  # penalize the non-strike fallback
        rationale = f"naive {category}: {est.rationale}"

    edge = abs(our_p - price)
    if edge < config.EDGE_THRESHOLD:
        return None
    if confidence < config.MIN_MODEL_CONFIDENCE:
        return None

    action = "buy_yes" if our_p > price else "buy_no"
    return TradeSignal(
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        market_ticker=market.get("ticker", ""),
        market_title=market.get("title", market.get("ticker", "")),
        market_price=price,
        our_probability=our_p,
        edge=edge,
        confidence=confidence,
        action=action,
        category=category,
        rationale=rationale,
    )


def scan_markets(
    markets: Iterable[dict[str, Any]],
    pipeline: EconPipeline,
    estimates: dict[str, EconEstimate],
) -> list[TradeSignal]:
    """Evaluate every market, returning flagged signals sorted by edge desc."""
    signals: list[TradeSignal] = []
    for market in markets:
        try:
            sig = evaluate_market(market, pipeline, estimates)
            if sig is not None:
                signals.append(sig)
        except Exception as exc:
            log.warning("Failed to evaluate market %s: %s",
                        market.get("ticker"), exc)
    signals.sort(key=lambda s: s.edge, reverse=True)
    return signals


# ----------------------------------------------------------------- logging

_CSV_FIELDS = [
    "timestamp", "market_ticker", "market_title", "market_price",
    "our_probability", "edge", "confidence", "action", "category", "rationale",
]


def log_opportunities(signals: Iterable[TradeSignal],
                      path: Path | None = None) -> int:
    """Append flagged opportunities to ``opportunities.log`` as CSV."""
    out = Path(path or config.OPPORTUNITIES_LOG)
    new_file = not out.exists()
    count = 0
    with out.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS)
        if new_file:
            writer.writeheader()
        for sig in signals:
            writer.writerow({k: v for k, v in asdict(sig).items() if k in _CSV_FIELDS})
            log.info(
                "OPPORTUNITY %s | %-6s | price=%.2f our=%.2f edge=%.2f | %s",
                sig.market_ticker, sig.action, sig.market_price,
                sig.our_probability, sig.edge, sig.market_title[:60],
            )
            count += 1
    return count
