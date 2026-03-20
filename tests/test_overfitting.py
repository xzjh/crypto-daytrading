"""Overfitting detection tests.

Run on every commit to verify the strategy is not overfitted:
1. Walk-forward: train on first half, test on second half — both must be profitable
2. Parameter sensitivity: ±20% param changes must not destroy performance
3. Consistency: second half Sharpe must not collapse vs first half
"""

import unittest
import warnings
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.chdir(os.path.dirname(os.path.dirname(__file__)))
warnings.filterwarnings("ignore")

from backtesting import Backtest
from core.data_fetcher import fetch_ohlcv
from core.indicators import add_indicators
from strategies.eth import add_eth_indicators
from strategies.robust import RobustTrendStrategy
from strategies.eth import ETHTrendStrategy
from core import config

# ── Load data once ──
_btc_df = add_indicators(fetch_ohlcv("BTC/USDT", "4h", days=2400, use_cache=True))
_eth_df = add_eth_indicators(fetch_ohlcv("ETH/USDT", "4h", days=2400, use_cache=True))
_mid_btc = len(_btc_df) // 2
_mid_eth = len(_eth_df) // 2


def _run_btc(df=None, **overrides):
    if df is None:
        df = _btc_df
    return Backtest(df, RobustTrendStrategy, cash=config.BACKTEST_CASH,
                    commission=0.0018, exclusive_orders=False,
                    trade_on_close=True, margin=1/1.3).run(**overrides)


def _run_eth(df=None, **overrides):
    if df is None:
        df = _eth_df
    return Backtest(df, ETHTrendStrategy, cash=config.BACKTEST_CASH,
                    commission=0.0015, exclusive_orders=False,
                    trade_on_close=True).run(**overrides)


# ── Pre-compute results for reuse ──
_btc_full = _run_btc()
_eth_full = _run_eth()
_btc_first = _run_btc(_btc_df.iloc[:_mid_btc])
_btc_second = _run_btc(_btc_df.iloc[_mid_btc:])
_eth_first = _run_eth(_eth_df.iloc[:_mid_eth])
_eth_second = _run_eth(_eth_df.iloc[_mid_eth:])


class TestWalkForward(unittest.TestCase):
    """Both halves of the data must produce positive results."""

    def test_btc_first_half_positive_sharpe(self):
        self.assertGreater(_btc_first["Sharpe Ratio"], 0,
                           f"BTC first half Sharpe={_btc_first['Sharpe Ratio']:.2f}")

    def test_btc_second_half_positive_sharpe(self):
        self.assertGreater(_btc_second["Sharpe Ratio"], 0,
                           f"BTC second half Sharpe={_btc_second['Sharpe Ratio']:.2f}")

    def test_eth_first_half_positive_sharpe(self):
        self.assertGreater(_eth_first["Sharpe Ratio"], 0,
                           f"ETH first half Sharpe={_eth_first['Sharpe Ratio']:.2f}")

    def test_eth_second_half_positive_sharpe(self):
        self.assertGreater(_eth_second["Sharpe Ratio"], 0,
                           f"ETH second half Sharpe={_eth_second['Sharpe Ratio']:.2f}")

    def test_btc_second_half_not_collapsed(self):
        """Second half Sharpe must be at least 30% of first half."""
        ratio = _btc_second["Sharpe Ratio"] / _btc_first["Sharpe Ratio"]
        self.assertGreater(ratio, 0.3,
                           f"BTC Sharpe collapsed: {_btc_first['Sharpe Ratio']:.2f} → "
                           f"{_btc_second['Sharpe Ratio']:.2f} (ratio {ratio:.2f})")

    def test_eth_second_half_not_collapsed(self):
        """Second half Sharpe must be at least 30% of first half."""
        ratio = _eth_second["Sharpe Ratio"] / _eth_first["Sharpe Ratio"]
        self.assertGreater(ratio, 0.3,
                           f"ETH Sharpe collapsed: {_eth_first['Sharpe Ratio']:.2f} → "
                           f"{_eth_second['Sharpe Ratio']:.2f} (ratio {ratio:.2f})")

    def test_btc_both_halves_profitable(self):
        self.assertGreater(_btc_first["Return [%]"], 0)
        self.assertGreater(_btc_second["Return [%]"], 0)

    def test_eth_both_halves_profitable(self):
        self.assertGreater(_eth_first["Return [%]"], 0)
        self.assertGreater(_eth_second["Return [%]"], 0)


class TestParameterSensitivity(unittest.TestCase):
    """Changing key parameters ±20% must not destroy performance.
    Sharpe must stay positive for all variations."""

    def _check_btc_param(self, param, lo, hi):
        lo_s = _run_btc(**{param: lo})
        hi_s = _run_btc(**{param: hi})
        self.assertGreater(lo_s["Sharpe Ratio"], 0,
                           f"BTC {param}={lo}: Sharpe={lo_s['Sharpe Ratio']:.2f}")
        self.assertGreater(hi_s["Sharpe Ratio"], 0,
                           f"BTC {param}={hi}: Sharpe={hi_s['Sharpe Ratio']:.2f}")

    def _check_eth_param(self, param, lo, hi):
        lo_s = _run_eth(**{param: lo})
        hi_s = _run_eth(**{param: hi})
        self.assertGreater(lo_s["Sharpe Ratio"], 0,
                           f"ETH {param}={lo}: Sharpe={lo_s['Sharpe Ratio']:.2f}")
        self.assertGreater(hi_s["Sharpe Ratio"], 0,
                           f"ETH {param}={hi}: Sharpe={hi_s['Sharpe Ratio']:.2f}")

    def test_btc_exit_buffer(self):
        self._check_btc_param("exit_buffer", 0.008, 0.012)

    def test_btc_entry_buffer(self):
        self._check_btc_param("entry_buffer", 0.003, 0.008)

    def test_btc_dd_reduce(self):
        self._check_btc_param("dd_reduce", 0.08, 0.16)

    def test_btc_min_size(self):
        self._check_btc_param("min_size", 0.20, 0.40)

    def test_eth_exit_buffer(self):
        self._check_eth_param("exit_buffer", 0.010, 0.020)

    def test_eth_entry_buffer(self):
        self._check_eth_param("entry_buffer", 0.003, 0.008)

    def test_eth_dd_reduce(self):
        self._check_eth_param("dd_reduce", 0.08, 0.16)

    def test_eth_min_size(self):
        self._check_eth_param("min_size", 0.20, 0.40)


class TestStrategyRobustness(unittest.TestCase):
    """General robustness checks."""

    def test_btc_full_positive_sharpe(self):
        self.assertGreater(_btc_full["Sharpe Ratio"], 0.5,
                           f"BTC full Sharpe={_btc_full['Sharpe Ratio']:.2f} < 0.5")

    def test_eth_full_positive_sharpe(self):
        self.assertGreater(_eth_full["Sharpe Ratio"], 0.5,
                           f"ETH full Sharpe={_eth_full['Sharpe Ratio']:.2f} < 0.5")

    def test_btc_max_drawdown_acceptable(self):
        self.assertGreater(_btc_full["Max. Drawdown [%]"], -50,
                           f"BTC MaxDD={_btc_full['Max. Drawdown [%]']:.1f}% exceeds -50%")

    def test_eth_max_drawdown_acceptable(self):
        self.assertGreater(_eth_full["Max. Drawdown [%]"], -50,
                           f"ETH MaxDD={_eth_full['Max. Drawdown [%]']:.1f}% exceeds -50%")

    def test_btc_beats_buy_and_hold(self):
        alpha = _btc_full["Return [%]"] - _btc_full["Buy & Hold Return [%]"]
        self.assertGreater(alpha, 0,
                           f"BTC alpha={alpha:.1f}% — strategy underperforms B&H")

    def test_eth_beats_buy_and_hold(self):
        alpha = _eth_full["Return [%]"] - _eth_full["Buy & Hold Return [%]"]
        self.assertGreater(alpha, 0,
                           f"ETH alpha={alpha:.1f}% — strategy underperforms B&H")


if __name__ == "__main__":
    unittest.main()
