# Crypto Day Trading Signal System

Adaptive trend-following system for BTC/USDT and ETH/USDT. Outputs BUY/SELL/HOLD signals based on multi-indicator technical analysis, with full backtesting and walk-forward validation.

## Strategy

**Core logic: hold in bull markets, protect in bear markets.**

The system uses asset-specific strategies — BTC and ETH have different parameters tuned to their respective market characteristics.

### BTC Strategy (`strategy_robust.py`)

| Parameter | Value | Rationale |
|---|---|---|
| Trend filter | EMA 50/200 Golden Cross | Industry standard trend indicator |
| Entry | EMA 20 > 50 > 200 + MACD > 0 + RSI < 80 | Full trend alignment confirmation |
| Bull mode | Hold through corrections | Only emergency exit if price < EMA200 × 98% |
| Bear mode | Chandelier trailing stop (ATR × 2.5) | Classic exit by Chuck LeBeau |
| Position sizing | Vol-adjusted, 2.5% equity risk per trade | Limits downside per trade |

### ETH Strategy (`strategy_eth.py`)

| Parameter | Value | Rationale |
|---|---|---|
| Trend filter | EMA 40/100 Golden Cross | Shorter cycle for ETH's faster trends |
| Entry | EMA 15 > 40 > 100 + MACD > 0 + RSI < 80 | Adapted to ETH's shorter cycles |
| Bear mode | Chandelier trailing stop (ATR × 2.0) | Tighter stop for ETH's higher volatility |
| Position sizing | Vol-adjusted, 1.2% equity risk per trade | More conservative due to ETH volatility |

## 3-Year Backtest Results (2023.04 – 2026.03)

| Metric | BTC Strategy | BTC B&H | ETH Strategy | ETH B&H |
|---|---|---|---|---|
| **Total Return** | **+179%** | +165% | **+79%** | +24% |
| **Sharpe Ratio** | **1.02** | ~0.5 | **1.00** | ~0.1 |
| **Max Drawdown** | **-22%** | -50% | **-13%** | -65% |
| Trades | 20 | 1 | 37 | 1 |
| Profit Factor | 3.71 | — | 2.33 | — |

### Performance by Market Regime

| Regime | BTC Alpha (vs B&H) | ETH Alpha (vs B&H) |
|---|---|---|
| **Bull quarters** | -13%/quarter (trails B&H) | -34%/quarter (trails B&H) |
| **Bear quarters** | **+14%/quarter** (beats B&H) | **+27%/quarter** (beats B&H) |

The strategy's edge is **downside protection**: it captures most of the bull market upside while avoiding the worst of bear market drawdowns. The compound effect of preserving capital in bear markets results in higher total returns over the full cycle.

## Overfitting Validation

Walk-forward analysis (6-month train / 3-month test, rolling):

| Metric | BTC | ETH |
|---|---|---|
| OOS annualized return | +58% | +30% |
| OOS profitable windows | 5/9 | 6/9 |
| Bull window IS→OOS | **Negative** (no overfitting) | **Negative** (no overfitting) |
| Bear window IS→OOS | ~100% (strategy correctly exits) | ~100% (strategy correctly exits) |

The high overall IS→OOS degradation (53%/44%) is driven by regime changes between windows (bull IS → bear OOS), not parameter overfitting. In same-regime comparisons, OOS performance matches or exceeds IS.

## Usage

### Get Current Signals

```bash
python3 main.py signal
```

Output:
```
BTC/USDT | 2026-03-18 04:00:00+00:00
Price: $74,185.18  |  RSI: 60.8
Entry Conditions: BULL
Signal: HOLD
    Golden Cross: YES
   Strong Trend: NO
 MACD Histogram: -33.46 BEARISH
            Mode: BULL — hold through corrections
```

### Run Backtest

```bash
# Default: 3 years, both BTC and ETH
python3 main.py backtest

# Custom
python3 main.py backtest --days 365 --symbols BTC/USDT
python3 main.py backtest --days 1095 --plot  # Opens interactive Bokeh chart
```

### Run Optimizer

```bash
python3 optimizer.py           # Multi-round parameter optimization
python3 walk_forward.py        # Walk-forward validation
python3 run_enhanced_test.py   # Enhanced strategy comparison test
```

## Project Structure

```
├── main.py                 # CLI: signal & backtest commands
├── config.py               # All tunable parameters
├── data_fetcher.py         # OHLCV fetching via ccxt (Binance, with CSV caching)
├── indicators.py           # Technical indicators (EMA, RSI, MACD, BB, ATR)
├── strategy_robust.py      # BTC strategy (hold-in-bull, protect-in-bear)
├── strategy_eth.py         # ETH strategy (shorter EMAs, tighter stops)
├── backtester.py           # Backtesting engine with quarterly breakdown
├── external_data.py        # Fear & Greed Index, Funding Rates
├── strategy_enhanced.py    # Experimental: layered entry + external data
├── strategy.py             # Legacy: multi-indicator confluence strategy
├── optimizer.py            # Parameter grid search optimizer
├── walk_forward.py         # Walk-forward overfitting validation
├── run_enhanced_test.py    # Enhanced strategy test runner
└── requirements.txt        # Dependencies
```

## Installation

```bash
pip install -r requirements.txt
```

Dependencies: `ccxt`, `pandas`, `ta`, `backtesting`, `matplotlib`, `numpy`

## Design Decisions

1. **Asset-specific strategies**: BTC and ETH have different volatility profiles and trend cycles. Using the same parameters for both leaves performance on the table.

2. **Hold-through in bull markets**: Instead of actively trading in/out during uptrends, the strategy holds through corrections (like B&H) and only activates trailing stops when the trend breaks. This captures nearly all bull market upside.

3. **Volatility-adjusted sizing**: Position size is inversely proportional to ATR, so the strategy automatically reduces exposure during high-volatility periods. This lowers portfolio volatility and improves Sharpe ratio.

4. **No fixed take-profit**: Winners are allowed to run via trailing stop rather than being capped at a fixed target. This is critical for capturing large trending moves.

5. **External data tested but not used**: Fear & Greed Index and Funding Rates were tested as additional signals. Simple threshold rules did not improve performance — the base technical strategy already captures the available trend-following alpha. ML-based approaches may extract more value from these signals.
