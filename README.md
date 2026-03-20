# Crypto Trading System

Adaptive trend-following system for BTC/USDT and ETH/USDT. Outputs BUY/SELL/HOLD signals based on EMA trend filters with adaptive position sizing, drawdown control, and leveraged collateral management.

## Strategy

**Core logic: hold in bull markets, protect in bear markets.**

Both BTC and ETH use the same framework — EMA trend filter + volatility-adaptive sizing + drawdown control — with different parameters tuned to each asset's characteristics.

### Entry / Exit Rules

- **Entry**: Price > EMA × (1 + entry_buffer), EMA slope positive → BUY
- **Exit**: Price < EMA × (1 − exit_buffer), EMA slope negative → CLOSE
- **Cooldown**: 3 bars (12h) after exit before re-entry
- **Drawdown control**: Equity drops 12% from peak → reduce to 25% position. Recovery to 95% of peak → restore full position.

### Parameters

| | BTC | ETH |
|---|---|---|
| Trend Filter | EMA 200 | EMA 150 (shorter ETH cycles) |
| Entry Buffer | 0.5% | 0.5% |
| Exit Buffer | 1.0% | 1.5% (wider for ETH volatility) |
| Slope Check | 3 bars | 5 bars |
| Max Size | 100% | 100% |
| Min Size | 30% | 30% |
| Leverage | **1.3x** (collateral loan) | **1.0x** (no leverage) |
| Position Sizing | Volatility-adaptive: low vol → bigger, high vol → smaller |

### Leverage Model

BTC uses 1.3x leverage via collateral borrowing: buy with full cash → pledge assets → borrow cash → buy more. This is NOT futures/margin trading — it's spot + lending. ETH runs at 1.0x (no leverage) for lower drawdown.

## Backtest Results: 6.5 Years (2019.10 – 2026.03)

### Summary

| | BTC (1.3x) | ETH (1.0x) |
|---|---|---|
| **Total Return** | **+2,878%** | **+5,381%** |
| **Buy & Hold** | +662% | +1,134% |
| **Alpha** | **+2,216%** | **+4,247%** |
| **Sharpe Ratio** | 0.88 | 0.83 |
| **Max Drawdown** | -32.4% | -37.1% |
| **Trades** | 95 | 123 |

### Yearly Breakdown — BTC (1.3x leverage)

| Year | Strategy | B&H | Alpha | Sharpe | MaxDD |
|------|----------|-----|-------|--------|-------|
| 2019 | -15.3% | -21.7% | +6.4% | -2.94 | -23.0% |
| 2020 | +339.6% | +299.8% | +39.9% | 3.04 | -26.8% |
| 2021 | +105.2% | +57.8% | +47.3% | 1.55 | -30.1% |
| 2022 | -22.7% | -64.7% | +42.0% | -0.78 | -28.4% |
| 2023 | +169.5% | +155.8% | +13.7% | 2.49 | -20.6% |
| 2024 | +108.8% | +120.9% | -12.2% | 1.83 | -28.8% |
| 2025 | -1.3% | -6.4% | +5.1% | 0.10 | -24.6% |
| 2026 | -10.6% | -20.4% | +9.7% | -2.09 | -18.3% |

### Yearly Breakdown — ETH (1.0x no leverage)

| Year | Strategy | B&H | Alpha | Sharpe | MaxDD |
|------|----------|-----|-------|--------|-------|
| 2019 | -3.9% | -25.3% | +21.4% | -0.76 | -9.7% |
| 2020 | +411.0% | +465.6% | -54.6% | 2.93 | -31.5% |
| 2021 | +268.6% | +393.3% | -124.7% | 2.19 | -35.7% |
| 2022 | +5.3% | -67.9% | +73.2% | 0.32 | -25.3% |
| 2023 | +50.6% | +91.0% | -40.3% | 1.23 | -29.8% |
| 2024 | +42.5% | +46.5% | -4.0% | 1.03 | -37.1% |
| 2025 | +38.4% | -11.4% | +49.8% | 1.00 | -30.2% |
| 2026 | -4.0% | -28.4% | +24.3% | -0.54 | -16.4% |

### Key Observations

- **Bear market protection**: Both strategies generate strong alpha in bear markets (2022: BTC +42%, ETH +73% vs B&H)
- **Bull market capture**: Holds through corrections, captures most of the upside (2020: BTC +340%, ETH +411%)
- **ETH trails B&H in bull years**: The 1.0x position can't match buy-and-hold during explosive ETH rallies, but dramatically outperforms in downturns

## Usage

### Get Current Signals

```bash
python3 main.py signal
```

### Run Backtest

```bash
python3 main.py backtest                          # Default: 3 years
python3 main.py backtest --days 2400              # Full history
python3 main.py backtest --days 2400 --symbols BTC/USDT   # Single asset
python3 main.py backtest --plot                   # With chart
```

### Dashboard

```bash
python3 web/server.py    # http://localhost:8050
```

Features: K-line charts (4H/1D/1W/1M), equity curves with trade markers, trade history with position tracking, yearly performance breakdown.

### Tests

```bash
python3 -m unittest discover -s tests -p "test_*.py"   # 116 tests
```

Pre-commit and pre-push hooks run tests automatically. Setup: `git config core.hooksPath .githooks`

## Project Structure

```
├── main.py                      # CLI: signal & backtest commands
├── strategies/
│   ├── robust.py                # BTC: EMA200 trend + 1.3x leverage
│   ├── eth.py                   # ETH: EMA150 trend + no leverage
│   ├── portfolio.py             # Portfolio rotation (unused)
│   ├── ml.py                    # ML regime classifier (experimental)
│   ├── enhanced.py              # External data integration (experimental)
│   └── legacy.py                # Legacy multi-indicator strategy
├── core/
│   ├── config.py                # Tunable parameters
│   ├── data_fetcher.py          # OHLCV via ccxt (Binance, CSV cache)
│   ├── indicators.py            # Technical indicators (EMA, RSI, MACD, BB, ATR)
│   ├── backtester.py            # Backtesting engine
│   └── external_data.py         # Fear & Greed Index, Funding Rates
├── tests/
│   ├── test_data_rules.py       # Backend data rules + sentry invariants
│   └── test_frontend_rules.py   # Frontend data consistency tests
├── web/
│   ├── server.py                # FastAPI backend
│   ├── trades.py                # Trade timeline builder
│   └── static/index.html        # Plotly.js dashboard
├── .githooks/                   # Pre-commit/pre-push test hooks
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
git config core.hooksPath .githooks   # Enable test hooks
```
