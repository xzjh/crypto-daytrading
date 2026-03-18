# Crypto Day Trading Signal System

Adaptive trend-following system for BTC/USDT and ETH/USDT with portfolio rotation. Outputs BUY/SELL/HOLD signals based on multi-indicator technical analysis, with full backtesting and walk-forward validation.

## Strategies

The system runs three layers:

### Layer 1: Asset-Specific Trend Following

**Core logic: hold in bull markets, protect in bear markets.**

BTC and ETH use different parameters tuned to their market characteristics:

| | BTC Strategy | ETH Strategy |
|---|---|---|
| EMAs | 20/50/200 (standard) | 15/40/100 (shorter, faster ETH cycles) |
| Trailing Stop | Chandelier Exit ATR × 2.5 | Chandelier Exit ATR × 2.0 (tighter) |
| Position Sizing | 2.5% equity risk/trade | 1.2% equity risk/trade (more conservative) |
| Bull Mode | Hold through corrections | Hold through corrections |
| Bear Mode | Trailing stop + death cross exit | Trailing stop + death cross exit |

### Layer 2: Portfolio Rotation

Dynamically allocates between BTC and ETH strategies based on momentum:

- Every 30 bars (~5 days): compare 120-bar (~20 day) momentum of each strategy's equity curve
- Overweight the stronger performer: 70% to winner, 30% to the other
- Transaction cost of 0.1% per rebalance included

This captures the **BTC/ETH rotation effect** — market capital alternates between BTC (institutional/safe-haven flows) and ETH (ecosystem/altcoin season), and 20-day momentum reliably identifies which regime we're in.

### Layer 3: ML Regime Classifier (Experimental)

LightGBM model trained walk-forward to classify bull/bear regimes. Combined with trend following as a hybrid signal. Currently underperforms pure trend following — included for research, not production use.

## Results: 6.5-Year Backtest (2019.10 – 2026.03)

| Strategy | Return | Sharpe | Max Drawdown |
|---|---|---|---|
| BTC Buy & Hold | +703% | — | ~-50% |
| ETH Buy & Hold | +1102% | — | ~-65% |
| BTC Trend Following | +666% | 0.96 | -25% |
| ETH Trend Following | +439% | 1.10 | -16% |
| **Portfolio Rotation** | **+823%** | **1.60** | **-21%** |

### Portfolio Rotation: Walk-Forward Validation

12 half-year windows tested, **all 12 produced positive alpha** vs equal-weight:

| Period | Rotation | Equal Weight | Alpha |
|---|---|---|---|
| 2019.10 – 2020.04 | +35.6% | +24.0% | **+11.5%** |
| 2020.04 – 2020.10 | +45.7% | +35.1% | **+10.6%** |
| 2020.10 – 2021.04 | +50.5% | +35.4% | **+15.1%** |
| 2021.04 – 2021.10 | +21.4% | +15.3% | **+6.1%** |
| 2021.10 – 2022.04 | +0.1% | -2.9% | **+3.0%** |
| 2022.04 – 2022.10 | +3.0% | +0.9% | **+2.1%** |
| 2022.10 – 2023.04 | +22.8% | +17.7% | **+5.1%** |
| 2023.04 – 2023.10 | -0.7% | -3.3% | **+2.6%** |
| 2023.10 – 2024.04 | +98.4% | +82.0% | **+16.4%** |
| 2024.04 – 2024.09 | +2.3% | -2.0% | **+4.3%** |
| 2024.09 – 2025.03 | +29.3% | +15.4% | **+13.9%** |
| 2025.03 – 2025.09 | +28.2% | +19.5% | **+8.7%** |

**Mean alpha: +8.3% per half-year. 100% hit rate across all market conditions.**

### Performance by Market Regime (Trend Following)

| Regime | BTC Alpha (vs B&H) | ETH Alpha (vs B&H) |
|---|---|---|
| Bull quarters | -32%/quarter (trails B&H) | -41%/quarter (trails B&H) |
| **Bear quarters** | **+21%/quarter** | **+29%/quarter** |

The trend following strategies' edge is downside protection. The portfolio rotation layer then adds alpha on top by correctly identifying which asset to overweight.

## Usage

### Get Current Signals

```bash
python3 main.py signal
```

Outputs individual BTC/ETH signals plus portfolio rotation recommendation:

```
BTC/USDT | Signal: HOLD (Bull mode — hold through corrections)
ETH/USDT | Signal: BUY  (Strong trend: EMA15>40>100 + MACD>0)

PORTFOLIO ROTATION SIGNAL
  BTC 20d momentum:  +8.5%
  ETH 20d momentum:  +12.7%
  Recommended:       BTC 30% / ETH 70%
```

### Run Backtest

```bash
# Default: 3 years
python3 main.py backtest

# Full history (~6.5 years)
python3 main.py backtest --days 2400

# Single asset
python3 main.py backtest --days 2400 --symbols BTC/USDT

# With interactive chart
python3 main.py backtest --plot
```

### Advanced Tools

```bash
python3 optimizer.py             # Multi-round parameter optimization
python3 walk_forward.py          # Walk-forward overfitting validation
python3 run_comparison.py        # 4-strategy comparison (TF, ML, Rotation, B&H)
python3 run_enhanced_test.py     # Enhanced strategy experiments
```

## Project Structure

```
├── main.py                  # CLI: signal & backtest commands
├── config.py                # Tunable parameters
├── data_fetcher.py          # OHLCV via ccxt (Binance, CSV cache)
├── indicators.py            # Technical indicators (EMA, RSI, MACD, BB, ATR)
│
├── strategy_robust.py       # BTC: hold-in-bull, protect-in-bear
├── strategy_eth.py          # ETH: shorter EMAs, tighter stops
├── strategy_portfolio.py    # Portfolio rotation (BTC/ETH momentum)
├── backtester.py            # Backtesting engine with quarterly breakdown
│
├── strategy_ml.py           # ML regime classifier (experimental)
├── ml_features.py           # Feature engineering for ML
├── external_data.py         # Fear & Greed Index, Funding Rates
├── strategy_enhanced.py     # Experimental: external data integration
│
├── strategy.py              # Legacy: multi-indicator confluence
├── optimizer.py             # Parameter grid search
├── walk_forward.py          # Walk-forward validation
├── run_comparison.py        # 4-strategy comparison runner
├── run_enhanced_test.py     # Enhanced strategy test runner
├── portfolio.py             # Portfolio combination utilities
└── requirements.txt         # Dependencies
```

## Installation

```bash
pip install -r requirements.txt
```

Dependencies: `ccxt`, `pandas`, `ta`, `backtesting`, `matplotlib`, `numpy`, `scikit-learn`, `lightgbm`

## Design Decisions

1. **Asset-specific strategies**: BTC and ETH have different volatility and trend cycle lengths. Separate parameters avoid compromising on either.

2. **Hold-through in bull markets**: Rather than trading in/out during uptrends, hold through corrections and only activate trailing stops on trend breaks. Captures nearly all bull upside.

3. **Portfolio rotation adds consistent alpha**: The BTC/ETH momentum rotation produced positive alpha in every single walk-forward window across 6.5 years — the most robust signal in the entire system.

4. **ML underperforms on crypto**: LightGBM with technical features could not beat simple trend following. Crypto's regime-switching nature and fat-tailed distributions make it hard for tree-based models to find stable patterns.

5. **Volatility-adjusted sizing**: Position size scales inversely with ATR, automatically reducing exposure during volatile periods. This improves Sharpe ratio by lowering portfolio volatility.
