"""Paper-trading simulation layer.

Kept deliberately separate from live order placement so this module can be
swapped for a real execution layer later with minimal surface-area change.
The only things it touches are the SQLite DB and the Kalshi client (read-only
calls to check market status and prices).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable

import config
from econ_pipeline import init_db
from kalshi_client import KalshiAPIError, KalshiClient
from logger import get_logger
from strategy import TradeSignal, yes_price

log = get_logger(__name__)


# ------------------------------------------------------------------ sizing

def kelly_position_size(
    probability: float,
    price: float,
    bankroll: float,
    kelly_fraction: float = config.KELLY_FRACTION,
    max_position: float = config.MAX_POSITION_SIZE,
) -> float:
    """Return dollar size for a bet using fractional Kelly.

    For a binary contract priced at ``price`` with our estimated win
    probability ``probability``, the Kelly fraction is
    ``f = (p*(b+1) - 1) / b`` where ``b = (1/price - 1)`` are the decimal odds.
    We cap at ``max_position`` and refuse to size when the edge is negative.
    """
    if not (0.0 < price < 1.0):
        return 0.0
    b = (1.0 / price) - 1.0
    if b <= 0:
        return 0.0
    f = (probability * (b + 1.0) - 1.0) / b
    if f <= 0:
        return 0.0
    sized = bankroll * f * kelly_fraction
    return min(sized, max_position)


# ------------------------------------------------------------------ types

@dataclass
class PaperTrade:
    """In-memory view of a row in ``paper_trades``."""

    id: int
    timestamp: str
    market_ticker: str
    market_title: str
    direction: str           # "buy_yes" | "buy_no"
    contracts: int
    entry_price: float       # yes-side price in [0, 1]
    our_probability: float
    edge_at_entry: float
    status: str              # "open" | "closed" | "expired"
    exit_price: float | None
    pnl: float | None
    resolved_at: str | None


def _row_to_trade(row: sqlite3.Row) -> PaperTrade:
    return PaperTrade(
        id=row["id"],
        timestamp=row["timestamp"],
        market_ticker=row["market_ticker"],
        market_title=row["market_title"] or "",
        direction=row["direction"],
        contracts=row["contracts"],
        entry_price=row["entry_price"],
        our_probability=row["our_probability"],
        edge_at_entry=row["edge_at_entry"],
        status=row["status"],
        exit_price=row["exit_price"],
        pnl=row["pnl"],
        resolved_at=row["resolved_at"],
    )


# ------------------------------------------------------------------ portfolio

class PaperPortfolio:
    """Tracks a simulated bankroll, open positions, and realized P&L."""

    def __init__(
        self,
        kalshi: KalshiClient,
        conn: sqlite3.Connection | None = None,
        starting_bankroll: float = config.STARTING_BANKROLL,
    ) -> None:
        self.conn = conn or init_db()
        self.kalshi = kalshi
        self.starting_bankroll = starting_bankroll

    # --- bookkeeping -------------------------------------------------------

    @property
    def realized_pnl(self) -> float:
        """Sum of PnL across closed trades."""
        row = self.conn.execute(
            "SELECT COALESCE(SUM(pnl), 0) AS total FROM paper_trades WHERE status='closed'"
        ).fetchone()
        return float(row["total"])

    @property
    def cash_balance(self) -> float:
        """Bankroll minus capital tied up in open positions."""
        open_cost = self.conn.execute(
            "SELECT COALESCE(SUM(contracts * entry_price), 0) AS c "
            "FROM paper_trades WHERE status='open'"
        ).fetchone()["c"]
        # Contracts are in units; price is a fraction in [0,1]. Kalshi
        # contracts settle at $1.00, so cost per contract in dollars equals
        # entry_price * 1.00.
        return self.starting_bankroll + self.realized_pnl - float(open_cost)

    def open_trades(self) -> list[PaperTrade]:
        """All trades still in the ``open`` state."""
        rows = self.conn.execute(
            "SELECT * FROM paper_trades WHERE status='open' ORDER BY timestamp DESC"
        ).fetchall()
        return [_row_to_trade(r) for r in rows]

    def closed_trades(self, limit: int | None = None) -> list[PaperTrade]:
        """Closed trades, newest first, optionally capped at ``limit``."""
        sql = ("SELECT * FROM paper_trades WHERE status='closed' "
               "ORDER BY resolved_at DESC")
        args: tuple[Any, ...] = ()
        if limit is not None:
            sql += " LIMIT ?"
            args = (limit,)
        rows = self.conn.execute(sql, args).fetchall()
        return [_row_to_trade(r) for r in rows]

    # --- opening positions -------------------------------------------------

    def simulate_from_signal(self, signal: TradeSignal) -> PaperTrade | None:
        """Open a paper trade sized per Kelly from ``signal``. Idempotent by
        ticker — we won't stack duplicate positions on the same market.
        """
        existing = self.conn.execute(
            "SELECT 1 FROM paper_trades WHERE market_ticker=? AND status='open'",
            (signal.market_ticker,),
        ).fetchone()
        if existing:
            log.debug("Already holding paper position in %s, skipping",
                      signal.market_ticker)
            return None

        # Price we pay depends on which side we buy.
        side_price = (
            signal.market_price if signal.action == "buy_yes"
            else 1.0 - signal.market_price
        )
        p_win = (
            signal.our_probability if signal.action == "buy_yes"
            else 1.0 - signal.our_probability
        )
        dollar_size = kelly_position_size(
            probability=p_win,
            price=side_price,
            bankroll=max(self.cash_balance, 1.0),
        )
        if dollar_size <= 0:
            return None

        # Kalshi contracts settle at $1.00, so dollars / price → contracts.
        contracts = max(1, int(dollar_size / max(side_price, 0.01)))
        if contracts * side_price > self.cash_balance:
            log.warning("Not enough paper cash for %s (need $%.2f, have $%.2f)",
                        signal.market_ticker, contracts * side_price, self.cash_balance)
            return None

        cur = self.conn.execute(
            """
            INSERT INTO paper_trades
              (timestamp, market_ticker, market_title, direction, contracts,
               entry_price, our_probability, edge_at_entry, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'open')
            """,
            (
                signal.timestamp, signal.market_ticker, signal.market_title,
                signal.action, contracts, side_price,
                signal.our_probability, signal.edge,
            ),
        )
        trade_id = cur.lastrowid
        log.info(
            "OPENED paper trade #%s | %s %s | %d contracts @ %.2f | size=$%.2f",
            trade_id, signal.action, signal.market_ticker,
            contracts, side_price, contracts * side_price,
        )
        row = self.conn.execute(
            "SELECT * FROM paper_trades WHERE id=?", (trade_id,)
        ).fetchone()
        return _row_to_trade(row)

    def simulate_from_signals(self, signals: Iterable[TradeSignal]) -> list[PaperTrade]:
        """Open paper trades for top-N signals, honoring open-position caps."""
        opened: list[PaperTrade] = []
        current_open = len(self.open_trades())
        remaining_slots = max(0, config.MAX_OPEN_TRADES - current_open)
        per_loop_slots = min(remaining_slots, config.MAX_NEW_TRADES_PER_LOOP)
        if per_loop_slots <= 0:
            log.info("Open-position cap reached (%d); skipping new trades.",
                     current_open)
            return opened
        # Signals are already sorted by edge desc upstream.
        for sig in list(signals)[:per_loop_slots]:
            trade = self.simulate_from_signal(sig)
            if trade is not None:
                opened.append(trade)
        return opened

    # --- resolution --------------------------------------------------------

    def check_resolutions(self) -> list[PaperTrade]:
        """Poll Kalshi for each open trade and close any resolved markets."""
        closed: list[PaperTrade] = []
        for trade in self.open_trades():
            try:
                market = self.kalshi.get_market(trade.market_ticker).get("market", {})
            except KalshiAPIError as exc:
                log.warning("Could not check %s: %s", trade.market_ticker, exc)
                continue

            status = (market.get("status") or "").lower()
            if status not in ("finalized", "settled", "closed", "resolved"):
                continue

            # Kalshi marks the winning side via ``result`` == "yes"/"no".
            result = (market.get("result") or "").lower()
            if result not in ("yes", "no"):
                # Market closed without a clear result — mark expired, no PnL.
                self._finalize_trade(trade, exit_price=trade.entry_price,
                                     pnl=0.0, status="expired")
                continue

            won = (
                (trade.direction == "buy_yes" and result == "yes")
                or (trade.direction == "buy_no" and result == "no")
            )
            if won:
                pnl = trade.contracts * (1.00 - trade.entry_price) * 100
                exit_price = 1.0
                log.info("[WIN ✓] %s settled %s — PnL: +$%.2f",
                         trade.market_ticker, result.upper(), pnl / 100)
            else:
                pnl = trade.contracts * trade.entry_price * -100
                exit_price = 0.0
                log.info("[LOSS ✗] %s settled %s — PnL: -$%.2f",
                         trade.market_ticker, result.upper(), abs(pnl) / 100)

            # PnL is tracked in dollars, not cents. The brief specifies
            # ``contracts * (1 - entry) * 100`` which yields cents; convert.
            pnl_dollars = pnl / 100.0
            closed_trade = self._finalize_trade(
                trade, exit_price=exit_price, pnl=pnl_dollars, status="closed"
            )
            closed.append(closed_trade)
        return closed

    def _finalize_trade(
        self, trade: PaperTrade, exit_price: float, pnl: float, status: str
    ) -> PaperTrade:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.conn.execute(
            "UPDATE paper_trades SET status=?, exit_price=?, pnl=?, resolved_at=? "
            "WHERE id=?",
            (status, exit_price, pnl, now, trade.id),
        )
        row = self.conn.execute(
            "SELECT * FROM paper_trades WHERE id=?", (trade.id,)
        ).fetchone()
        return _row_to_trade(row)

    # --- valuation & snapshots --------------------------------------------

    def open_position_value(self) -> float:
        """Mark-to-market value of open positions using current Kalshi prices."""
        total = 0.0
        for trade in self.open_trades():
            try:
                market = self.kalshi.get_market(trade.market_ticker).get("market", {})
                price = yes_price(market)
            except KalshiAPIError:
                price = None
            if price is None:
                price = trade.entry_price
            mark = price if trade.direction == "buy_yes" else 1.0 - price
            total += trade.contracts * mark
        return total

    def unrealized_pnl(self) -> float:
        """Open positions: market value minus the cost we paid."""
        total = 0.0
        for trade in self.open_trades():
            try:
                market = self.kalshi.get_market(trade.market_ticker).get("market", {})
                price = yes_price(market)
            except KalshiAPIError:
                price = None
            if price is None:
                price = trade.entry_price
            mark = price if trade.direction == "buy_yes" else 1.0 - price
            total += trade.contracts * (mark - trade.entry_price)
        return total

    def snapshot(self) -> dict[str, float]:
        """Persist a portfolio snapshot row and return the values."""
        cash = self.cash_balance
        open_val = self.open_position_value()
        realized = self.realized_pnl
        unrealized = self.unrealized_pnl()
        num_open = len(self.open_trades())
        num_closed = self.conn.execute(
            "SELECT COUNT(*) AS c FROM paper_trades WHERE status='closed'"
        ).fetchone()["c"]

        snap = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "cash_balance": cash,
            "open_position_value": open_val,
            "realized_pnl": realized,
            "unrealized_pnl": unrealized,
            "total_pnl": realized + unrealized,
            "num_open_trades": num_open,
            "num_closed_trades": num_closed,
        }
        self.conn.execute(
            """
            INSERT INTO portfolio_snapshots
              (timestamp, cash_balance, open_position_value,
               realized_pnl, unrealized_pnl, total_pnl,
               num_open_trades, num_closed_trades)
            VALUES (:timestamp, :cash_balance, :open_position_value,
                    :realized_pnl, :unrealized_pnl, :total_pnl,
                    :num_open_trades, :num_closed_trades)
            """,
            snap,
        )
        return snap
