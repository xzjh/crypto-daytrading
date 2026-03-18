"""ML strategy: LightGBM regime classifier + trend following.

Instead of predicting returns, the ML model classifies market regime:
- Class 0: BEAR/FLAT (stay defensive)
- Class 1: BULL (go aggressive)

This is combined with the trend following base:
- If ML says BULL + trend filter agrees → full position
- If ML says BULL but trend says BEAR → reduced/no position
- If ML says BEAR → exit or stay out
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from backtesting import Strategy

import config
from ml_features import build_features


def precompute_ml_signals(df: pd.DataFrame, train_bars=2160, retrain_every=540,
                          pred_horizon=24) -> pd.Series:
    """Walk-forward regime predictions.

    Args:
        df: OHLCV DataFrame with external data columns
        train_bars: training window (~1 year of 4H bars)
        retrain_every: retrain frequency (~3 months)
        pred_horizon: bars ahead to predict (24 bars = 4 days)
    """
    # Build features with custom target
    feat = build_features(df, include_target=False)

    # Target: is the market in a bull regime over the next pred_horizon bars?
    # Defined as: next N-bar return > 0 AND max drawdown within N bars < -3%
    close = df["Close"].reindex(feat.index)
    fwd_ret = close.pct_change(pred_horizon).shift(-pred_horizon)
    # Simple regime: up > 1% = bull, else bear/flat
    feat["target"] = (fwd_ret > 0.01).astype(int)

    feature_cols = [c for c in feat.columns if c != "target"]
    predictions = pd.Series(index=feat.index, dtype=float)
    predictions[:] = np.nan

    total = len(feat)
    current = train_bars

    while current < total:
        train_end = current
        train_start = max(0, train_end - train_bars)
        df_train = feat.iloc[train_start:train_end].dropna(subset=["target"])

        if len(df_train) < 200:
            current += retrain_every
            continue

        X_train = df_train[feature_cols].values
        y_train = df_train["target"].values

        # LightGBM with strong regularization to prevent overfitting
        model = lgb.LGBMClassifier(
            n_estimators=150,
            max_depth=3,          # Shallow trees
            learning_rate=0.03,
            num_leaves=8,         # Very few leaves
            min_child_samples=100, # Large min leaf size
            subsample=0.7,
            colsample_bytree=0.6,
            reg_alpha=1.0,        # Strong L1
            reg_lambda=1.0,       # Strong L2
            random_state=42,
            verbose=-1,
        )
        model.fit(X_train, y_train)

        pred_end = min(current + retrain_every, total)
        df_pred = feat.iloc[current:pred_end]
        X_pred = df_pred[feature_cols].values

        if len(X_pred) > 0:
            probs = model.predict_proba(X_pred)[:, 1]
            predictions.iloc[current:pred_end] = probs

        current += retrain_every

    return predictions


class MLStrategy(Strategy):
    """ML regime classifier + trend following hybrid."""

    atr_mult = 2.5
    risk_pct = 0.025
    ml_bull_threshold = 0.55   # ML says bull if prob > this
    ml_bear_threshold = 0.40   # ML says bear if prob < this
    emerg_pct = 0.02

    def init(self):
        self.atr_col = f"ATR_{config.ATR_PERIOD}"
        # Auto-detect EMA columns
        cols = self.data.df.columns
        if f"EMA_{config.EMA_MID}" in cols:
            self.ema_fast = f"EMA_{config.EMA_FAST}"
            self.ema_mid = f"EMA_{config.EMA_MID}"
            self.ema_slow = f"EMA_{config.EMA_SLOW}"
        else:
            self.ema_fast = "EMA_15"
            self.ema_mid = "EMA_40"
            self.ema_slow = "EMA_100"
        self.macd_hist = f"MACDh_{config.MACD_FAST}_{config.MACD_SLOW}_{config.MACD_SIGNAL}"

        self.trailing_stop = 0
        self.highest = 0
        self.bars_since_exit = 999

    def next(self):
        self.bars_since_exit += 1
        price = self.data.Close[-1]
        df = self.data.df
        atr = df[self.atr_col].iloc[-1]

        ml_prob = df["ML_Signal"].iloc[-1] if "ML_Signal" in df.columns else 0.5
        if np.isnan(ml_prob):
            ml_prob = 0.5

        ema_f = df[self.ema_fast].iloc[-1]
        ema_m = df[self.ema_mid].iloc[-1]
        ema_s = df[self.ema_slow].iloc[-1]
        macd_h = df[self.macd_hist].iloc[-1]
        golden_cross = ema_m > ema_s
        full_align = ema_f > ema_m > ema_s and macd_h > 0

        ml_bull = ml_prob > self.ml_bull_threshold
        ml_bear = ml_prob < self.ml_bear_threshold

        if self.position and self.position.is_long:
            if price > self.highest:
                self.highest = price

            if golden_cross and not ml_bear:
                # Bull + ML agrees: hold through
                if self.emerg_pct > 0 and price < ema_s * (1 - self.emerg_pct):
                    self.position.close()
                    self.bars_since_exit = 0
            else:
                # Bear mode or ML says bear: trailing stop
                mult = self.atr_mult
                if ml_bear:
                    mult *= 0.7  # Tighter stop when ML is bearish
                chandelier = self.highest - atr * mult
                if chandelier > self.trailing_stop:
                    self.trailing_stop = chandelier
                if price < self.trailing_stop:
                    self.position.close()
                    self.bars_since_exit = 0
                    return
                if macd_h < 0 and price < ema_m:
                    self.position.close()
                    self.bars_since_exit = 0

        else:
            if self.bars_since_exit < 2:
                return

            # Entry: need BOTH ML bull AND trend confirmation
            if ml_bull and golden_cross and full_align:
                sl_price = price - atr * self.atr_mult
                risk_per = price - sl_price
                if risk_per > 0:
                    size = min(self.equity * self.risk_pct / risk_per * price / self.equity, 0.95)
                    size = max(size, 0.05)
                else:
                    size = 0.95
                self.trailing_stop = sl_price
                self.highest = price
                self.buy(sl=sl_price, size=size)
                self.bars_since_exit = 0
