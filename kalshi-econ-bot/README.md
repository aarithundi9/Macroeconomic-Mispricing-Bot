# kalshi-econ-bot

A paper-trading bot that scans Kalshi's **demo** environment for economic-data
markets (CPI, unemployment, nonfarm payrolls, Fed rate decisions), computes
a naive probability from recent BLS / FRED history, and flags mispricings
where the implied Kalshi price diverges from our estimate by more than 10%.

**It does not place real orders.** Every flagged trade is simulated via
fractional Kelly sizing against a virtual $1,000 bankroll. All results live
in a local SQLite database so you can inspect them after the fact.

## Project layout

```
kalshi-econ-bot/
├── main.py                # orchestrator loop
├── kalshi_client.py       # signed Kalshi v2 REST client
├── econ_pipeline.py       # BLS / FRED / Fed calendar ingestion + estimates
├── strategy.py            # market ↔ estimate matching & edge scoring
├── paper_portfolio.py     # Kelly sizing, positions, resolutions, snapshots
├── dashboard.py           # CLI portfolio dashboard
├── config.py              # thresholds and file paths
├── logger.py              # shared logging setup
├── .env.example           # template for API keys (copy to .env)
├── requirements.txt
└── README.md
```

## Setup

### 1. Clone and install dependencies

```bash
cd kalshi-econ-bot
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Get a Kalshi demo API key

1. Sign up for a demo account at https://demo.kalshi.co
2. In the demo dashboard, create an API key. Kalshi will give you:
   - an **API key ID** (a UUID-like string)
   - a **private key** in PEM format (download and save as a file)
3. Put the PEM file somewhere outside version control (e.g. project root)
   and reference its path in `.env`. It is already in `.gitignore`.

### 3. Get a FRED API key

FRED requires a free key: https://fred.stlouisfed.org/docs/api/api_key.html

BLS works without a key at basic rate limits. Register one at
https://www.bls.gov/developers/ if you want a higher quota.

### 4. Configure `.env`

```bash
cp .env.example .env
```

Then edit `.env`:

```
KALSHI_API_KEY=<your Kalshi demo API key ID>
KALSHI_PRIVATE_KEY_PATH=./kalshi_private_key.pem
KALSHI_PRIVATE_KEY_PASSWORD=        # leave blank unless you set one
FRED_API_KEY=<your FRED API key>
BLS_API_KEY=                        # optional
```

## Running the bot

```bash
python main.py
```

On each loop (default every 5 minutes) you'll see:

```
== LOOP 2026-04-19 14:00:00Z ==
  Markets scanned:     183
  Opportunities found: 3  (opened 2, closed 0)
  Top opportunity:     KXCPI-26MAR-T4.0 edge=0.18
  Portfolio value:     $1,000.00  (cash $984.20 + open $15.80)
  Iteration took:      4.1s
```

Stop with **Ctrl-C** — it will finish the current iteration and exit cleanly.

## Viewing the dashboard

In a second terminal (or any time after running `main.py`):

```bash
python dashboard.py
```

You'll see:

- portfolio summary (balance, realized/unrealized PnL, return %, win rate,
  Sharpe, max drawdown)
- open positions with mark-to-market
- the last 10 closed trades
- today's top flagged opportunities
- a side effect: writes `trades_history.csv` for external analysis

## Configuration

Everything in `config.py` is tunable:

| Setting                   | Default   | Meaning                                  |
| ------------------------- | --------- | ---------------------------------------- |
| `EDGE_THRESHOLD`          | `0.10`    | minimum probability edge to flag a trade |
| `SCAN_INTERVAL_SECONDS`   | `300`     | seconds between loop iterations          |
| `STARTING_BANKROLL`       | `1000.00` | virtual dollars                          |
| `MAX_POSITION_SIZE`       | `50`      | cap per paper trade                      |
| `KELLY_FRACTION`          | `0.25`    | quarter Kelly for safety                 |
| `DEMO_MODE`               | `True`    | `False` would hit prod Kalshi (don't)    |

## Going live later

`paper_portfolio.py` and `dashboard.py` are intentionally decoupled from
execution. To swap paper trading for real orders:

1. Replace `PaperPortfolio.simulate_from_signal` with a call to
   `KalshiClient.place_order`.
2. Keep the DB tables (`paper_trades`, `portfolio_snapshots`) — they already
   model real execution shape.
3. Keep `config.DEMO_MODE = True` until you are comfortable with both
   the sizing logic and your probability estimates.

## Data sources

- **BLS** (`api.bls.gov/publicAPI/v2`) — CPI, unemployment rate, nonfarm payrolls
- **FRED** (`api.stlouisfed.org/fred`) — effective fed funds rate
- **Federal Reserve** (`federalreserve.gov/monetarypolicy/fomccalendars.htm`)
  — FOMC meeting markers

The probability model is intentionally simple: for each monthly series, we
look at the last ~12 prints and count how often the MoM change was positive.
That gives you a baseline to replace with something smarter (e.g. a consensus
survey, a seasonal model, an LLM forecast).

## Caveats

- This is an educational project. The naive estimator is not a real trading
  signal — treat flagged "edge" with healthy skepticism.
- Kalshi demo markets don't always mirror prod tickers, so the keyword
  matcher may pick up fewer markets in demo than in real life. Tune the
  `MARKET_KEYWORDS` dict in `config.py` if you want more breadth.
- Never commit `.env` or your PEM file.
