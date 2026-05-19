"""Backtesting harness: replay snapshots against settled market outcomes.

Usage:
    python backtest.py [--snapshots PATH] [--edge-threshold 0.10]

Loads recorded market snapshots, queries Kalshi API for settled outcomes,
simulates Kelly-sized trades, and computes performance metrics.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import config
from kalshi_client import KalshiClient
from logger import get_logger

log = get_logger(__name__)


@dataclass
class Snapshot:
    """A market snapshot from daily collection."""

    timestamp: str
    market_ticker: str
    market_price: float
    our_probability: float
    edge: float
    action: str
    category: str


@dataclass
class Trade:
    """A simulated trade from backtest."""

    ticker: str
    entry_price: float
    our_probability: float
    edge_at_entry: float
    contracts: int  # Kelly-sized
    exit_price: float | None
    pnl: float | None


def _kelly_contracts(
    bankroll: float,
    our_p: float,
    market_p: float,
    edge: float,
    max_per_trade: float = config.MAX_POSITION_SIZE,
) -> int:
    """Return Kelly-optimal contract count.

    Kelly fraction = f / b where f = edge, b = odds payout.
    For binary markets at price p, winning a YES share pays 1-p.
    Quarter-Kelly for safety: multiply by 0.25.
    """
    if edge <= 0:
        return 0
    # For YES position at price p, upside/downside = (1-p) / p
    odds_ratio = (1.0 - market_p) / market_p if market_p > 0 else 1.0
    kelly_frac = (our_p * odds_ratio - (1.0 - our_p)) / odds_ratio
    kelly_frac = max(0.0, kelly_frac) * config.KELLY_FRACTION
    # Dollar position
    position_size = min(kelly_frac * bankroll, max_per_trade)
    # Convert to contracts (each contract costs the entry price)
    return max(0, int(position_size / market_p))


def load_snapshots(path: Path) -> list[Snapshot]:
    """Load market snapshots from CSV."""
    snapshots = []
    if not path.exists():
        log.warning("Snapshots file not found: %s", path)
        return snapshots

    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            snapshots.append(Snapshot(
                timestamp=row["timestamp"],
                market_ticker=row["market_ticker"],
                market_price=float(row["market_price"]),
                our_probability=float(row["our_probability"]),
                edge=float(row["edge"]),
                action=row["action"],
                category=row["category"],
            ))
    log.info("Loaded %d market snapshots", len(snapshots))
    return snapshots


def fetch_settled_outcomes(
    kalshi: KalshiClient,
    tickers: set[str],
) -> dict[str, dict[str, Any]]:
    """Fetch settled markets from Kalshi API, keyed by ticker."""
    outcomes = {}
    for ticker in tickers:
        try:
            market = kalshi.get_market(ticker)
            if market.get("status") == "settled":
                outcomes[ticker] = market
            else:
                log.debug("Market %s not yet settled (status=%s)",
                         ticker, market.get("status"))
        except Exception as exc:
            log.warning("Failed to fetch %s: %s", ticker, exc)
    log.info("Fetched outcomes for %d settled markets", len(outcomes))
    return outcomes


def extract_settlement_price(market: dict[str, Any]) -> float | None:
    """Extract the settlement price from a Kalshi market."""
    # Settled markets have a resolution field indicating YES or NO.
    resolution = market.get("resolution")
    if resolution is None:
        return None
    # YES settled to 1.0, NO settled to 0.0
    return 1.0 if resolution == "yes" else 0.0


def backtest(
    snapshots: list[Snapshot],
    outcomes: dict[str, dict[str, Any]],
    edge_threshold: float = config.EDGE_THRESHOLD,
    initial_bankroll: float = config.STARTING_BANKROLL,
) -> dict[str, Any]:
    """Simulate trades from snapshots against outcomes.

    Returns metrics: pnl, hit_rate, sharpe, max_drawdown, num_trades.
    """
    trades: list[Trade] = []
    bankroll = initial_bankroll
    cumulative_pnl = 0.0
    daily_pnls = []  # for Sharpe calc
    running_balance = [initial_bankroll]

    for snap in snapshots:
        # Only trade if edge clears threshold
        if snap.edge < edge_threshold:
            continue

        # Can't trade if outcome not yet available
        if snap.market_ticker not in outcomes:
            continue

        market = outcomes[snap.market_ticker]
        settlement_price = extract_settlement_price(market)
        if settlement_price is None:
            continue

        # Simulate entry
        contracts = _kelly_contracts(
            bankroll, snap.our_probability, snap.market_price, snap.edge
        )
        if contracts <= 0:
            continue

        # Determine if we win/lose
        we_bought_yes = snap.action == "buy_yes"
        settlement_was_yes = settlement_price == 1.0

        if we_bought_yes and settlement_was_yes:
            # Bought YES, YES won: profit
            pnl = contracts * (1.0 - snap.market_price)
        elif we_bought_yes and not settlement_was_yes:
            # Bought YES, NO won: loss
            pnl = -contracts * snap.market_price
        elif not we_bought_yes and settlement_was_yes:
            # Bought NO, YES won: loss
            pnl = -contracts * (1.0 - snap.market_price)
        else:
            # Bought NO, NO won: profit
            pnl = contracts * snap.market_price

        bankroll += pnl
        cumulative_pnl += pnl
        daily_pnls.append(pnl)
        running_balance.append(bankroll)

        trades.append(Trade(
            ticker=snap.market_ticker,
            entry_price=snap.market_price,
            our_probability=snap.our_probability,
            edge_at_entry=snap.edge,
            contracts=contracts,
            exit_price=settlement_price,
            pnl=pnl,
        ))

    # Compute metrics
    num_winning = sum(1 for t in trades if t.pnl is not None and t.pnl > 0)
    hit_rate = num_winning / len(trades) if trades else 0.0

    sharpe = 0.0
    if len(daily_pnls) > 1:
        mean_pnl = statistics.fmean(daily_pnls)
        stdev_pnl = statistics.stdev(daily_pnls)
        if stdev_pnl > 0:
            # Annualize: ~250 trading days
            sharpe = (mean_pnl / stdev_pnl) * (250 ** 0.5)

    max_drawdown = 0.0
    if running_balance:
        peak = running_balance[0]
        for balance in running_balance[1:]:
            if balance < peak:
                drawdown = (peak - balance) / peak
                max_drawdown = max(max_drawdown, drawdown)

    return {
        "num_trades": len(trades),
        "num_winning": num_winning,
        "hit_rate": hit_rate,
        "cumulative_pnl": cumulative_pnl,
        "final_bankroll": bankroll,
        "roi": (bankroll - initial_bankroll) / initial_bankroll,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "trades": trades,
    }


def main() -> int:
    """Run backtesting."""
    parser = argparse.ArgumentParser(description="Backtest strategy against Kalshi markets.")
    parser.add_argument(
        "--snapshots",
        type=Path,
        default=config.PROJECT_ROOT / "market_snapshots.csv",
        help="Path to market snapshots CSV",
    )
    parser.add_argument(
        "--edge-threshold",
        type=float,
        default=config.EDGE_THRESHOLD,
        help="Minimum edge to trade",
    )
    parser.add_argument(
        "--initial-bankroll",
        type=float,
        default=config.STARTING_BANKROLL,
        help="Starting cash",
    )
    args = parser.parse_args()

    print("\n=== BACKTEST ===\n")
    print(f"Edge threshold: {args.edge_threshold:.2f}")
    print(f"Initial bankroll: ${args.initial_bankroll:.2f}")
    print(f"Snapshots: {args.snapshots}")
    print()

    # Load snapshots
    snapshots = load_snapshots(args.snapshots)
    if not snapshots:
        print("ERROR: No snapshots loaded. Run the bot first to collect data.")
        return 1

    # Fetch settled outcomes from Kalshi
    kalshi = KalshiClient()
    tickers = {s.market_ticker for s in snapshots}
    outcomes = fetch_settled_outcomes(kalshi, tickers)

    if not outcomes:
        print("ERROR: No settled market outcomes found. Markets may not be settled yet.")
        return 1

    # Run backtest
    results = backtest(
        snapshots,
        outcomes,
        edge_threshold=args.edge_threshold,
        initial_bankroll=args.initial_bankroll,
    )

    # Print results
    print("=== RESULTS ===\n")
    print(f"Trades simulated:         {results['num_trades']}")
    print(f"Winning trades:           {results['num_winning']} ({results['hit_rate']:.1%})")
    print(f"Cumulative PnL:           ${results['cumulative_pnl']:.2f}")
    print(f"Final bankroll:           ${results['final_bankroll']:.2f}")
    print(f"ROI:                      {results['roi']:.2%}")
    print(f"Sharpe ratio (annualized): {results['sharpe_ratio']:.2f}")
    print(f"Max drawdown:             {results['max_drawdown']:.2%}")
    print()

    if results["trades"]:
        print("Top 5 trades:")
        for trade in sorted(
            results["trades"], key=lambda t: t.pnl or 0.0, reverse=True
        )[:5]:
            print(
                f"  {trade.ticker:12s} | "
                f"entry=${trade.entry_price:.3f} | "
                f"{trade.contracts:3d}x | "
                f"PnL=${trade.pnl:.2f}"
            )

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
