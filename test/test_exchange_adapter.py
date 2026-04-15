"""
单元测试: ExchangeAdapter (BinanceAdapter / GateAdapter)
重点验证：
- symbol 归一化
- 从不同 position 结构解析持仓数量
- fetch_funding_rate 统一结构
"""
import os
import sys
import unittest
from unittest.mock import MagicMock

# 允许从项目根 import
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.exchange_adapter import BinanceAdapter, GateAdapter, make_adapter


class TestBinanceAdapterSymbol(unittest.TestCase):
    def setUp(self):
        self.adapter = BinanceAdapter(MagicMock())

    def test_normalize_slash(self):
        self.assertEqual(self.adapter.normalize_symbol('BTC/USDT'), 'BTC/USDT')

    def test_normalize_no_slash(self):
        self.assertEqual(self.adapter.normalize_symbol('BTCUSDT'), 'BTC/USDT')

    def test_normalize_strips_settle(self):
        # Binance 不使用 ':USDT'，应剥离
        self.assertEqual(self.adapter.normalize_symbol('BTC/USDT:USDT'), 'BTC/USDT')


class TestGateAdapterSymbol(unittest.TestCase):
    def setUp(self):
        self.adapter = GateAdapter(MagicMock())

    def test_normalize_slash_adds_settle(self):
        self.assertEqual(self.adapter.normalize_symbol('BTC/USDT'), 'BTC/USDT:USDT')

    def test_normalize_no_slash_adds_settle(self):
        self.assertEqual(self.adapter.normalize_symbol('BTCUSDT'), 'BTC/USDT:USDT')

    def test_normalize_already_full_form(self):
        self.assertEqual(self.adapter.normalize_symbol('BTC/USDT:USDT'), 'BTC/USDT:USDT')


class TestBinancePositionParsing(unittest.TestCase):
    def setUp(self):
        self.adapter = BinanceAdapter(MagicMock())

    def test_long_position(self):
        pos = {'info': {'positionAmt': '0.5', 'entryPrice': '30000'}}
        self.assertEqual(self.adapter.get_position_amt(pos), 0.5)
        self.assertEqual(self.adapter.get_position_entry_price(pos), 30000)

    def test_short_position(self):
        pos = {'info': {'positionAmt': '-0.5', 'entryPrice': '30000'}}
        self.assertEqual(self.adapter.get_position_amt(pos), -0.5)

    def test_empty_position(self):
        self.assertEqual(self.adapter.get_position_amt({}), 0)
        self.assertEqual(self.adapter.get_position_amt(None), 0)


class TestGatePositionParsing(unittest.TestCase):
    def setUp(self):
        ex = MagicMock()
        ex.market.return_value = {'contractSize': 0.0001}  # Gate BTC 永续 1 张 = 0.0001 BTC
        self.adapter = GateAdapter(ex)

    def test_long_size(self):
        pos = {'symbol': 'BTC/USDT:USDT', 'info': {'size': '5000', 'entry_price': '30000'}}
        # 5000 张 * 0.0001 = 0.5 BTC
        self.assertAlmostEqual(self.adapter.get_position_amt(pos), 0.5, places=6)
        self.assertEqual(self.adapter.get_position_entry_price(pos), 30000)

    def test_short_size(self):
        pos = {'symbol': 'BTC/USDT:USDT', 'info': {'size': '-5000', 'entry_price': '30000'}}
        self.assertAlmostEqual(self.adapter.get_position_amt(pos), -0.5, places=6)


class TestFundingRateStructure(unittest.TestCase):

    def test_binance_funding_rate(self):
        ex = MagicMock()
        ex.fetch_funding_rate.return_value = {
            'fundingRate': 0.0001,
            'fundingTimestamp': 1700000000000,
            'markPrice': 30000,
            'info': {'fundingIntervalHours': '8'},
        }
        adapter = BinanceAdapter(ex)
        rate = adapter.fetch_funding_rate('BTC/USDT')
        self.assertEqual(rate['rate'], 0.0001)
        self.assertEqual(rate['interval_hours'], 8)
        self.assertEqual(rate['next_funding_time'], 1700000000000)
        self.assertEqual(rate['mark_price'], 30000)

    def test_gate_funding_rate_interval_seconds(self):
        ex = MagicMock()
        ex.fetch_funding_rate.return_value = {
            'fundingRate': 0.00005,
            'fundingTimestamp': 1700000000000,
            'markPrice': 30000,
            # Gate info funding_interval 单位是秒，28800=8h
            'info': {'funding_interval': '28800'},
        }
        adapter = GateAdapter(ex)
        rate = adapter.fetch_funding_rate('BTC/USDT:USDT')
        self.assertEqual(rate['rate'], 0.00005)
        self.assertEqual(rate['interval_hours'], 8)


class TestMakeAdapter(unittest.TestCase):
    def test_make_binance(self):
        adapter = make_adapter('binance', MagicMock())
        self.assertIsInstance(adapter, BinanceAdapter)

    def test_make_gate(self):
        adapter = make_adapter('gate', MagicMock())
        self.assertIsInstance(adapter, GateAdapter)

    def test_unknown_raises(self):
        with self.assertRaises(ValueError):
            make_adapter('okx', MagicMock())


if __name__ == '__main__':
    unittest.main()
