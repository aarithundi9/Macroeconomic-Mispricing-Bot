"""Main orchestrator for the Kalshi economic-data trading bot.

Loop structure (every ``SCAN_INTERVAL_SECONDS``):
  1. Refresh econ data if stale
  2. Pull open Kalshi markets
  3. Match markets against econ estimates and compute edge
  4. Log flagged opportunities to CSV
  5. Simulate paper trades via Kelly sizing
  6. Check open paper trades for resolutions
  7. Snapshot portfolio state

The loop is wrapped in broad ``try/except`` so a single bad API response
never kills the process.
"""

from __future__ import annotations

import signal
import sys
import time
from datetime import datetime, timezone

import config
from dashboard import export_trades_history
from econ_pipeline import EconPipeline, init_db
from kalshi_client import KalshiAPIError, KalshiClient
from logger import get_logger
from paper_portfolio import PaperPortfolio
from strategy import log_opportunities, scan_markets

log = get_logger(__name__)

_stop = False


def _handle_sigint(_signum, _frame) -> None:  # noqa: ANN001 - signal handler
    """Mark the loop for graceful shutdown on Ctrl-C."""
    global _stop
    _stop = True
    log.info("Shutdown requested — will exit after current iteration.")


def run_once(
    kalshi: KalshiClient,
    pipeline: EconPipeline,
    portfolio: PaperPortfolio,
) -> None:
    """One pass through the trading loop. Never raises."""
    start = time.monotonic()

    # 1) Refresh econ data if stale.
    try:
        pipeline.refresh()
    except Exception as exc:
        log.exception("Econ refresh failed: %s", exc)

    estimates = pipeline.current_estimates()
    log.info("Loaded %d econ estimates", len(estimates))

    # 2) Fetch open econ markets directly by series ticker — skips the
    # tens of thousands of sports markets Kalshi hosts on the same host.
    markets: list[dict] = []
    try:
        markets = kalshi.get_econ_markets()
    except KalshiAPIError as exc:
        log.error("Kalshi markets fetch failed: %s", exc)

    # 3) Scan for opportunities using the strike-aware model.
    signals = scan_markets(markets, pipeline, estimates)

    # 4) Persist opportunities to CSV/log.
    if signals:
        log_opportunities(signals)

    # 5) Paper-trade the fresh signals.
    opened = []
    try:
        opened = portfolio.simulate_from_signals(signals)
    except Exception as exc:
        log.exception("Paper trade simulation failed: %s", exc)

    # 6) Resolution check on any open paper trades.
    closed = []
    try:
        closed = portfolio.check_resolutions()
    except Exception as exc:
        log.exception("Resolution check failed: %s", exc)

    # 7) Snapshot + CSV export.
    try:
        snap = portfolio.snapshot()
        export_trades_history(portfolio.conn)
    except Exception as exc:
        log.exception("Snapshot/export failed: %s", exc)
        snap = None

    # Console summary for this iteration.
    top = signals[0] if signals else None
    elapsed = time.monotonic() - start
    print(
        "\n== LOOP {ts} =="
        "\n  Markets scanned:     {nmkt}"
        "\n  Opportunities found: {nops}  (opened {nopen}, closed {nclose})"
        "\n  Top opportunity:     {top}"
        "\n  Portfolio value:     ${pv:.2f}  (cash ${cash:.2f} + open ${op:.2f})"
        "\n  Iteration took:      {sec:.1f}s"
        .format(
            ts=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
            nmkt=len(markets),
            nops=len(signals),
            nopen=len(opened),
            nclose=len(closed),
            top=(f"{top.market_ticker} edge={top.edge:.2f}" if top else "none"),
            pv=(snap["cash_balance"] + snap["open_position_value"])
                if snap else 0.0,
            cash=snap["cash_balance"] if snap else 0.0,
            op=snap["open_position_value"] if snap else 0.0,
            sec=elapsed,
        )
    )


def main() -> int:
    """Run the scanning/paper-trading loop until interrupted."""
    signal.signal(signal.SIGINT, _handle_sigint)
    try:
        signal.signal(signal.SIGTERM, _handle_sigint)
    except (AttributeError, ValueError):
        pass  # SIGTERM unavailable on some Windows configurations.

    log.info(
        "Starting Kalshi econ bot | DEMO_MODE=%s SHADOW_MODE=%s | host=%s | interval=%ds",
        config.DEMO_MODE, config.SHADOW_MODE, config.KALSHI_BASE_URL,
        config.SCAN_INTERVAL_SECONDS,
    )

    conn = init_db()
    kalshi = KalshiClient()
    pipeline = EconPipeline(conn=conn)
    portfolio = PaperPortfolio(kalshi, conn=conn)

    while not _stop:
        try:
            run_once(kalshi, pipeline, portfolio)
        except Exception as exc:
            log.exception("Unhandled error in loop (continuing): %s", exc)

        # Sleep in small steps so Ctrl-C is responsive.
        for _ in range(config.SCAN_INTERVAL_SECONDS):
            if _stop:
                break
            time.sleep(1)

    log.info("Exited cleanly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
