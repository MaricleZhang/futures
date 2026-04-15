"""
单元测试: FundingRateMonitor
"""
import os
import sys
import time
import unittest
from unittest.mock import MagicMock

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from arbitrage.funding_rate_monitor import FundingRateMonitor


def _mock_adapter(name, rate, interval_hours=8, mark=30000.0, next_ts=1700000000000):
    a = MagicMock()
    a.name = name
    a.normalize_symbol = lambda s: s
    a.fetch_funding_rate = MagicMock(return_value={
        'rate': rate, 'interval_hours': interval_hours,
        'mark_price': mark, 'next_funding_time': next_ts, 'raw': {},
    })
    return a


class TestFundingRateMonitor(unittest.TestCase):

    def test_spread_bps_positive(self):
        adapters = {
            'binance': _mock_adapter('binance', 0.0002),
            'gate':    _mock_adapter('gate',    0.00005),
        }
        monitor = FundingRateMonitor(adapters)
        diff = monitor.get_differential('BTC/USDT', 'binance', 'gate')
        # (0.0002 - 0.00005) * 10000 = 1.5
        self.assertAlmostEqual(diff['spread_bps'], 1.5, places=6)
        self.assertEqual(diff['sign'], 1)
        # 年化：0.0002*3*365 - 0.00005*3*365 = 0.21900 - 0.05475 = 0.16425
        self.assertAlmostEqual(diff['annualized_spread'], 0.16425, places=4)

    def test_spread_bps_negative(self):
        adapters = {
            'binance': _mock_adapter('binance', 0.00005),
            'gate':    _mock_adapter('gate',    0.0002),
        }
        monitor = FundingRateMonitor(adapters)
        diff = monitor.get_differential('BTC/USDT', 'binance', 'gate')
        self.assertAlmostEqual(diff['spread_bps'], -1.5, places=6)
        self.assertEqual(diff['sign'], -1)

    def test_zero_spread(self):
        adapters = {
            'binance': _mock_adapter('binance', 0.0001),
            'gate':    _mock_adapter('gate',    0.0001),
        }
        monitor = FundingRateMonitor(adapters)
        diff = monitor.get_differential('BTC/USDT', 'binance', 'gate')
        self.assertEqual(diff['sign'], 0)

    def test_interval_mismatch(self):
        # Gate 1h 结算；同样的单周期费率 0.0001，年化应显著不同
        adapters = {
            'binance': _mock_adapter('binance', 0.0001, interval_hours=8),
            'gate':    _mock_adapter('gate',    0.0001, interval_hours=1),
        }
        monitor = FundingRateMonitor(adapters)
        diff = monitor.get_differential('BTC/USDT', 'binance', 'gate')
        # 单周期相等 -> bps=0，但年化差应为负（Gate 年化更高）
        self.assertAlmostEqual(diff['spread_bps'], 0, places=6)
        self.assertLess(diff['annualized_spread'], 0)

    def test_cache_ttl(self):
        adapters = {
            'binance': _mock_adapter('binance', 0.0001),
            'gate':    _mock_adapter('gate',    0.0001),
        }
        monitor = FundingRateMonitor(adapters, cache_ttl=60)
        monitor.get_rate('binance', 'BTC/USDT')
        monitor.get_rate('binance', 'BTC/USDT')
        # 第二次应走缓存，只调用一次 fetch
        self.assertEqual(adapters['binance'].fetch_funding_rate.call_count, 1)

    def test_unknown_exchange(self):
        monitor = FundingRateMonitor({'binance': _mock_adapter('binance', 0.0001)})
        with self.assertRaises(ValueError):
            monitor.get_rate('okx', 'BTC/USDT')


if __name__ == '__main__':
    unittest.main()
