"""Stand-alone CLI dashboard.

Run ``python dashboard.py`` to print a snapshot of the paper portfolio,
open positions, recent closed trades, and today's flagged opportunities.

This script reads from the SQLite database and the CSV opportunities log —
it does not call Kalshi live except for optional mark-to-market pricing,
and it never places orders.
"""

from __future__ import annotations

import csv
import math
import sqlite3
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import config
from econ_pipeline import init_db
from kalshi_client import KalshiClient
from logger import get_logger
from paper_portfolio import PaperPortfolio

log = get_logger(__name__)


# -------------------------------------------------------------- statistics

def daily_returns(conn: sqlite3.Connection,
                  starting_bankroll: float) -> list[float]:
    """Derive a per-day return series from portfolio_snapshots."""
    rows = conn.execute(
        "SELECT timestamp, cash_balance, open_position_value, total_pnl "
        "FROM portfolio_snapshots ORDER BY timestamp ASC"
    ).fetchall()
    if not rows:
        return []

    # Bucket by UTC day and take each day's last equity point.
    eod: dict[str, float] = {}
    for r in rows:
        day = r["timestamp"][:10]
        eod[day] = r["cash_balance"] + r["open_position_value"]

    equities = [starting_bankroll] + [eod[k] for k in sorted(eod.keys())]
    returns: list[float] = []
    for prev, curr in zip(equities[:-1], equities[1:]):
        if prev <= 0:
            continue
        returns.append((curr - prev) / prev)
    return returns


def sharpe_ratio(returns: list[float]) -> float:
    """Annualized Sharpe with zero risk-free rate (daily → sqrt(252))."""
    if len(returns) < 2:
        return 0.0
    mean = sum(returns) / len(returns)
    var = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    std = math.sqrt(var)
    if std == 0:
        return 0.0
    return (mean / std) * math.sqrt(252)


def max_drawdown(conn: sqlite3.Connection) -> float:
    """Largest peak-to-trough decline in total equity (as a fraction)."""
    rows = conn.execute(
        "SELECT cash_balance + open_position_value AS equity "
        "FROM portfolio_snapshots ORDER BY timestamp ASC"
    ).fetchall()
    if not rows:
        return 0.0
    peak = float("-inf")
    dd = 0.0
    for r in rows:
        eq = float(r["equity"])
        peak = max(peak, eq)
        if peak > 0:
            dd = min(dd, (eq - peak) / peak)
    return dd


def win_loss_counts(conn: sqlite3.Connection) -> tuple[int, int]:
    """Return (wins, losses) across closed trades (expired trades excluded)."""
    wins = conn.execute(
        "SELECT COUNT(*) AS c FROM paper_trades WHERE status='closed' AND pnl > 0"
    ).fetchone()["c"]
    losses = conn.execute(
        "SELECT COUNT(*) AS c FROM paper_trades WHERE status='closed' AND pnl <= 0"
    ).fetchone()["c"]
    return int(wins), int(losses)


def avg_edge_at_entry(conn: sqlite3.Connection) -> float:
    """Average ``edge_at_entry`` across all paper trades ever opened."""
    row = conn.execute(
        "SELECT COALESCE(AVG(edge_at_entry), 0) AS e FROM paper_trades"
    ).fetchone()
    return float(row["e"])


# --------------------------------------------------------------- rendering

def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def _print_summary(portfolio: PaperPortfolio, conn: sqlite3.Connection) -> None:
    """Print the top-level portfolio block."""
    cash = portfolio.cash_balance
    open_val = portfolio.open_position_value()
    realized = portfolio.realized_pnl
    unrealized = portfolio.unrealized_pnl()
    total = cash + open_val
    ret_pct = ((total - config.STARTING_BANKROLL) / config.STARTING_BANKROLL) * 100.0

    wins, losses = win_loss_counts(conn)
    total_decided = wins + losses
    win_rate = (wins / total_decided * 100.0) if total_decided else 0.0

    returns = daily_returns(conn, config.STARTING_BANKROLL)
    sharpe = sharpe_ratio(returns)
    mdd = max_drawdown(conn) * 100.0

    print("\nPORTFOLIO SUMMARY")
    print("-----------------")
    print(f"Starting Balance:     ${config.STARTING_BANKROLL:,.2f}")
    print(f"Current Balance:      ${total:,.2f}")
    print(f"Realized P&L:         {'+' if realized >= 0 else ''}${realized:,.2f}")
    print(f"Unrealized P&L:       {'+' if unrealized >= 0 else ''}${unrealized:,.2f}")
    print(f"Total Return:         {'+' if ret_pct >= 0 else ''}{ret_pct:.1f}%")
    print(f"Win Rate:             {win_rate:.0f}% ({wins} wins / {losses} losses)")
    print(f"Avg Edge at Entry:    {avg_edge_at_entry(conn) * 100:.1f}%")
    print(f"Sharpe Ratio:         {sharpe:.2f}")
    print(f"Max Drawdown:         {mdd:.1f}%")


def _print_open_positions(portfolio: PaperPortfolio) -> None:
    open_trades = portfolio.open_trades()
    print(f"\nOPEN POSITIONS ({len(open_trades)})")
    print("------------------")
    if not open_trades:
        print("  (none)")
        return
    for t in open_trades:
        # Best-effort current price; fall back to entry if unavailable.
        try:
            from strategy import yes_price
            market = portfolio.kalshi.get_market(t.market_ticker).get("market", {})
            price = yes_price(market) or t.entry_price
        except Exception:
            price = t.entry_price
        mark = price if t.direction == "buy_yes" else 1.0 - price
        unreal = t.contracts * (mark - t.entry_price)
        sign = "+" if unreal >= 0 else "-"
        print(
            f"  [{t.market_ticker}] {_truncate(t.market_title, 40):<40} | "
            f"Entry: {t.entry_price:.2f} | Current: {mark:.2f} | "
            f"Unrealized: {sign}${abs(unreal):.2f} | "
            f"Edge was: {t.edge_at_entry * 100:.0f}%"
        )


def _print_recent_closed(portfolio: PaperPortfolio) -> None:
    closed = portfolio.closed_trades(limit=10)
    print("\nRECENT CLOSED TRADES (last 10)")
    print("--------------------------------")
    if not closed:
        print("  (none)")
        return
    for t in closed:
        pnl = t.pnl or 0.0
        tag = "WIN" if pnl > 0 else "LOSS"
        sign = "+" if pnl >= 0 else "-"
        exit_p = t.exit_price if t.exit_price is not None else 0.0
        print(
            f"  [{t.market_ticker}] | Entry: {t.entry_price:.2f} | "
            f"Exit: {exit_p:.2f} | PnL: {sign}${abs(pnl):.2f} | [{tag}]"
        )


def _print_top_opportunities() -> None:
    print("\nTOP OPPORTUNITIES TODAY")
    print("-----------------------")
    path = Path(config.OPPORTUNITIES_LOG)
    if not path.exists():
        print("  (opportunities.log not yet written)")
        return

    today = date.today().isoformat()
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row.get("timestamp", "").startswith(today):
                rows.append(row)
    if not rows:
        print("  (no opportunities flagged today yet)")
        return

    rows.sort(key=lambda r: float(r.get("edge", 0)), reverse=True)
    for r in rows[:10]:
        edge = float(r.get("edge", 0)) * 100
        print(
            f"  [{r.get('market_ticker')}] {_truncate(r.get('market_title', ''), 40):<40} "
            f"| {r.get('action'):<8} | edge: {edge:.0f}%"
        )


# ----------------------------------------------------------------- exports

def export_trades_history(conn: sqlite3.Connection,
                          path: Path | None = None) -> Path:
    """Dump closed paper trades to CSV for external analysis."""
    out = Path(path or config.TRADES_HISTORY_CSV)
    rows = conn.execute(
        "SELECT * FROM paper_trades WHERE status='closed' ORDER BY resolved_at ASC"
    ).fetchall()
    fields = [
        "id", "timestamp", "market_ticker", "market_title", "direction",
        "contracts", "entry_price", "our_probability", "edge_at_entry",
        "status", "exit_price", "pnl", "resolved_at",
    ]
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({f: r[f] for f in fields})
    return out


# ------------------------------------------------------------------ entry

def render_dashboard() -> None:
    """Top-level renderer called by ``python dashboard.py``."""
    conn = init_db()
    kalshi = KalshiClient()
    portfolio = PaperPortfolio(kalshi, conn=conn)

    _print_summary(portfolio, conn)
    _print_open_positions(portfolio)
    _print_recent_closed(portfolio)
    _print_top_opportunities()

    out = export_trades_history(conn)
    print(f"\n(Exported closed trades → {out})\n")


if __name__ == "__main__":
    try:
        render_dashboard()
    except KeyboardInterrupt:
        sys.exit(0)
