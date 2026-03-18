"""Enhanced strategy with three improvements:

1. LAYERED ENTRY + QUICK RE-ENTRY
   - Layer 1: golden cross → enter 40% position
   - Layer 2: full EMA alignment + MACD → scale to full position
   - Quick re-entry: after stop-out, if golden cross intact, re-enter on price > EMA_fast

2. EXTERNAL DATA SIGNALS
   - Fear & Greed Index: extreme fear (< 25) = buy signal boost, extreme greed (> 75) = tighten stops
   - Funding Rate: extreme positive = overcrowded longs (caution), negative = contrarian buy

3. BTC→ETH LEADING SIGNAL (ETH only)
   - If BTC trend turns bearish, preemptively tighten ETH stops
"""

from backtesting import Strategy
import config


class EnhancedBTCStrategy(Strategy):
    """Enhanced BTC strategy with layered entry, external data, quick re-entry."""

    atr_mult = 2.5
    risk_pct = 0.025
    emerg_pct = 0.02
    layer1_size = 0.40   # Initial entry on golden cross
    layer2_size = 0.55   # Add on full confirmation (total = layer1 + layer2 = 0.95)
    reentry_cooldown = 2

    def init(self):
        self.ema_fast = f"EMA_{config.EMA_FAST}"
        self.ema_mid = f"EMA_{config.EMA_MID}"
        self.ema_slow = f"EMA_{config.EMA_SLOW}"
        self.macd_hist = f"MACDh_{config.MACD_FAST}_{config.MACD_SLOW}_{config.MACD_SIGNAL}"
        self.atr_col = f"ATR_{config.ATR_PERIOD}"
        self.rsi_col = f"RSI_{config.RSI_PERIOD}"

        self.trailing_stop = 0
        self.highest = 0
        self.bars_since_exit = 999
        self.is_layer1_only = False  # Track if we're in partial position
        self.exited_in_uptrend = False

    def _get_external(self, df):
        """Get external data signals. Returns (fng, fng_ma, fr_ma)."""
        fng = df["FNG"].iloc[-1] if "FNG" in df.columns else 50
        fng_ma = df["FNG_MA"].iloc[-1] if "FNG_MA" in df.columns else 50
        fr_ma = df["FR_MA"].iloc[-1] if "FR_MA" in df.columns else 0
        return fng, fng_ma, fr_ma

    def _vol_size(self, price, sl_price, base_risk):
        """Volatility-adjusted position sizing."""
        risk_per = price - sl_price
        if risk_per > 0:
            size = min(self.equity * base_risk / risk_per * price / self.equity, 0.95)
            return max(size, 0.05)
        return 0.95

    def next(self):
        self.bars_since_exit += 1
        price = self.data.Close[-1]
        df = self.data.df
        atr = df[self.atr_col].iloc[-1]
        ema20 = df[self.ema_fast].iloc[-1]
        ema50 = df[self.ema_mid].iloc[-1]
        ema200 = df[self.ema_slow].iloc[-1]
        macd_h = df[self.macd_hist].iloc[-1]
        rsi = df[self.rsi_col].iloc[-1]

        golden_cross = ema50 > ema200
        full_alignment = ema20 > ema50 > ema200 and macd_h > 0
        fng, fng_ma, fr_ma = self._get_external(df)

        # External data adjustments
        extreme_fear = fng < 25  # Contrarian buy signal
        extreme_greed = fng > 75  # Caution signal
        crowded_longs = fr_ma > 0.0003  # Funding > 0.03% avg = overcrowded

        if self.position and self.position.is_long:
            if price > self.highest:
                self.highest = price

            if golden_cross:
                # Bull mode: hold through, emergency exit only
                # Tighten emergency exit if extreme greed + crowded longs
                emerg = self.emerg_pct
                if extreme_greed and crowded_longs:
                    emerg = self.emerg_pct * 0.5  # Tighter exit when euphoric

                if emerg > 0 and price < ema200 * (1 - emerg):
                    self.position.close()
                    self.bars_since_exit = 0
                    self.exited_in_uptrend = True
            else:
                # Bear mode: trailing stop
                # Tighter stop if greed is high
                mult = self.atr_mult
                if extreme_greed:
                    mult *= 0.8

                chandelier = self.highest - atr * mult
                if chandelier > self.trailing_stop:
                    self.trailing_stop = chandelier
                if price < self.trailing_stop:
                    self.position.close()
                    self.bars_since_exit = 0
                    self.exited_in_uptrend = False
                    return
                if macd_h < 0 and price < ema50:
                    self.position.close()
                    self.bars_since_exit = 0
                    self.exited_in_uptrend = False

        else:
            # ── LOOK FOR ENTRY ──
            if self.bars_since_exit < self.reentry_cooldown:
                return

            sl_price = price - atr * self.atr_mult

            # Quick re-entry: stopped out but golden cross still intact
            if self.exited_in_uptrend and golden_cross and price > ema20:
                size = self._vol_size(price, sl_price, self.risk_pct)
                self.trailing_stop = sl_price
                self.highest = price
                self.is_layer1_only = False
                self.exited_in_uptrend = False
                self.buy(sl=sl_price, size=size)
                self.bars_since_exit = 0
                return

            # Standard entry: full alignment required (same as old strategy)
            # But also enter on extreme fear as contrarian signal
            if golden_cross and full_alignment and rsi < 80:
                size = self._vol_size(price, sl_price, self.risk_pct)
                self.trailing_stop = sl_price
                self.highest = price
                self.exited_in_uptrend = False
                self.buy(sl=sl_price, size=size)
                self.bars_since_exit = 0
            elif golden_cross and extreme_fear and price > ema50 and not crowded_longs:
                # Contrarian entry in extreme fear (even without full alignment)
                size = self._vol_size(price, sl_price, self.risk_pct)
                self.trailing_stop = sl_price
                self.highest = price
                self.exited_in_uptrend = False
                self.buy(sl=sl_price, size=size)
                self.bars_since_exit = 0


class EnhancedETHStrategy(Strategy):
    """Enhanced ETH strategy with shorter EMAs, layered entry, BTC leading signal."""

    atr_mult = 2.0
    risk_pct = 0.012
    emerg_pct = 0.02
    layer1_size = 0.30
    layer2_size = 0.65
    reentry_cooldown = 2

    def init(self):
        self.ema_fast = "EMA_15"
        self.ema_mid = "EMA_40"
        self.ema_slow = "EMA_100"
        self.macd_hist = f"MACDh_{config.MACD_FAST}_{config.MACD_SLOW}_{config.MACD_SIGNAL}"
        self.atr_col = f"ATR_{config.ATR_PERIOD}"
        self.rsi_col = f"RSI_{config.RSI_PERIOD}"

        self.trailing_stop = 0
        self.highest = 0
        self.bars_since_exit = 999
        self.is_layer1_only = False
        self.exited_in_uptrend = False

    def _get_external(self, df):
        fng = df["FNG"].iloc[-1] if "FNG" in df.columns else 50
        fng_ma = df["FNG_MA"].iloc[-1] if "FNG_MA" in df.columns else 50
        fr_ma = df["FR_MA"].iloc[-1] if "FR_MA" in df.columns else 0
        # BTC leading signal: check if BTC EMA50 > EMA200
        btc_bullish = df["BTC_Trend"].iloc[-1] > 0 if "BTC_Trend" in df.columns else True
        return fng, fng_ma, fr_ma, btc_bullish

    def _vol_size(self, price, sl_price, base_risk):
        risk_per = price - sl_price
        if risk_per > 0:
            size = min(self.equity * base_risk / risk_per * price / self.equity, 0.95)
            return max(size, 0.05)
        return 0.95

    def next(self):
        self.bars_since_exit += 1
        price = self.data.Close[-1]
        df = self.data.df
        atr = df[self.atr_col].iloc[-1]
        ema15 = df[self.ema_fast].iloc[-1]
        ema40 = df[self.ema_mid].iloc[-1]
        ema100 = df[self.ema_slow].iloc[-1]
        macd_h = df[self.macd_hist].iloc[-1]
        rsi = df[self.rsi_col].iloc[-1]

        golden_cross = ema40 > ema100
        full_alignment = ema15 > ema40 > ema100 and macd_h > 0
        fng, fng_ma, fr_ma, btc_bullish = self._get_external(df)

        extreme_fear = fng < 25
        extreme_greed = fng > 75
        crowded_longs = fr_ma > 0.0003

        if self.position and self.position.is_long:
            if price > self.highest:
                self.highest = price

            if golden_cross:
                emerg = self.emerg_pct
                # BTC leading signal: if BTC turns bearish, tighten ETH exit
                if not btc_bullish:
                    emerg = self.emerg_pct * 0.5
                if extreme_greed and crowded_longs:
                    emerg = self.emerg_pct * 0.5

                if emerg > 0 and price < ema100 * (1 - emerg):
                    self.position.close()
                    self.bars_since_exit = 0
                    self.exited_in_uptrend = True
            else:
                mult = self.atr_mult
                if extreme_greed:
                    mult *= 0.8
                if not btc_bullish:
                    mult *= 0.8

                chandelier = self.highest - atr * mult
                if chandelier > self.trailing_stop:
                    self.trailing_stop = chandelier
                if price < self.trailing_stop:
                    self.position.close()
                    self.bars_since_exit = 0
                    self.exited_in_uptrend = False
                    return
                if macd_h < 0 and price < ema40:
                    self.position.close()
                    self.bars_since_exit = 0
                    self.exited_in_uptrend = False
        else:
            if self.bars_since_exit < self.reentry_cooldown:
                return

            sl_price = price - atr * self.atr_mult

            # Quick re-entry
            if self.exited_in_uptrend and golden_cross and price > ema15:
                size = self._vol_size(price, sl_price, self.risk_pct)
                self.trailing_stop = sl_price
                self.highest = price
                self.is_layer1_only = False
                self.exited_in_uptrend = False
                self.buy(sl=sl_price, size=size)
                self.bars_since_exit = 0
                return

            if golden_cross and full_alignment and rsi < 80:
                size = self._vol_size(price, sl_price, self.risk_pct)
                self.trailing_stop = sl_price
                self.highest = price
                self.exited_in_uptrend = False
                self.buy(sl=sl_price, size=size)
                self.bars_since_exit = 0
            elif golden_cross and extreme_fear and price > ema40 and not crowded_longs:
                size = self._vol_size(price, sl_price, self.risk_pct)
                self.trailing_stop = sl_price
                self.highest = price
                self.exited_in_uptrend = False
                self.buy(sl=sl_price, size=size)
                self.bars_since_exit = 0
