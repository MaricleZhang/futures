"""
单元测试: ArbExecutor
- 两腿成功 -> 记录 open
- 长腿成功 / 短腿失败 -> 触发 emergency_close 回滚
- 长腿失败 / 短腿成功 -> 触发 emergency_close 回滚
- 两腿都失败 -> 不调用平仓
- dry_run -> 不调用 create_order
- close_pair 正常路径
"""
import os
import sys
import unittest
import tempfile
from unittest.mock import MagicMock, patch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from arbitrage.arb_executor import ArbExecutor
from utils.arb_trade_recorder import ArbTradeRecorder


def _make_trader(name='binance', symbol='BTC/USDT'):
    t = MagicMock()
    t.exchange_name = name
    t.symbol = symbol
    t.adapter.normalize_amount = lambda s, a: round(a, 6)
    t.adapter.contract_size = lambda s: 1.0
    t.exchange.create_order = MagicMock(return_value={
        'id': f'{name}-oid', 'average': 30000.0, 'filled': 0.01, 'cost': 300,
    })
    t.close_position = MagicMock(return_value={'average': 30010.0})
    return t


def _recorder_with_tempdb():
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
    tmp.close()
    return ArbTradeRecorder(db_path=tmp.name), tmp.name


class TestArbExecutorOpen(unittest.TestCase):

    def test_open_both_success(self):
        recorder, path = _recorder_with_tempdb()
        long_tr = _make_trader('binance')
        short_tr = _make_trader('gate')
        ex = ArbExecutor({'binance': long_tr, 'gate': short_tr}, recorder=recorder)

        result = ex.open_pair(
            pair_id='P1', symbol='BTC/USDT',
            long_exchange='binance', short_exchange='gate',
            amount=0.01, notional_usdt=300, entry_spread_bps=5,
            entry_rate_long=0.00005, entry_rate_short=0.0002,
            ref_long_price=30000, ref_short_price=30000,
        )
        self.assertTrue(result['ok'])
        self.assertIsNotNone(result['trade_id'])
        long_tr.exchange.create_order.assert_called_once()
        short_tr.exchange.create_order.assert_called_once()
        # recorder 中应有一条 OPEN 记录
        row = recorder.get_open_trade('P1')
        self.assertIsNotNone(row)
        os.unlink(path)

    def test_open_short_fails_triggers_rollback(self):
        recorder, path = _recorder_with_tempdb()
        long_tr = _make_trader('binance')
        short_tr = _make_trader('gate')
        # 短腿抛异常
        short_tr.exchange.create_order = MagicMock(side_effect=RuntimeError('api error'))

        ex = ArbExecutor({'binance': long_tr, 'gate': short_tr}, recorder=recorder)

        result = ex.open_pair(
            pair_id='P2', symbol='BTC/USDT',
            long_exchange='binance', short_exchange='gate',
            amount=0.01, notional_usdt=300, entry_spread_bps=5,
            entry_rate_long=0.00005, entry_rate_short=0.0002,
            ref_long_price=30000, ref_short_price=30000,
        )
        self.assertFalse(result['ok'])
        # 长腿应被调用 2 次：初始 buy + emergency sell
        self.assertEqual(long_tr.exchange.create_order.call_count, 2)
        last_call = long_tr.exchange.create_order.call_args
        self.assertEqual(last_call.kwargs.get('side') or last_call[1].get('side'), 'sell')
        # 没有持仓记录
        self.assertIsNone(recorder.get_open_trade('P2'))
        os.unlink(path)

    def test_open_long_fails_triggers_rollback(self):
        recorder, path = _recorder_with_tempdb()
        long_tr = _make_trader('binance')
        short_tr = _make_trader('gate')
        long_tr.exchange.create_order = MagicMock(side_effect=RuntimeError('oops'))

        ex = ArbExecutor({'binance': long_tr, 'gate': short_tr}, recorder=recorder)

        result = ex.open_pair(
            pair_id='P3', symbol='BTC/USDT',
            long_exchange='binance', short_exchange='gate',
            amount=0.01, notional_usdt=300, entry_spread_bps=5,
            entry_rate_long=0.00005, entry_rate_short=0.0002,
            ref_long_price=30000, ref_short_price=30000,
        )
        self.assertFalse(result['ok'])
        # 短腿应被调用 2 次：初始 sell + emergency buy
        self.assertEqual(short_tr.exchange.create_order.call_count, 2)
        os.unlink(path)

    def test_open_both_fail_no_close(self):
        recorder, path = _recorder_with_tempdb()
        long_tr = _make_trader('binance')
        short_tr = _make_trader('gate')
        long_tr.exchange.create_order = MagicMock(side_effect=RuntimeError('l'))
        short_tr.exchange.create_order = MagicMock(side_effect=RuntimeError('s'))

        ex = ArbExecutor({'binance': long_tr, 'gate': short_tr}, recorder=recorder)
        result = ex.open_pair(
            pair_id='P4', symbol='BTC/USDT',
            long_exchange='binance', short_exchange='gate',
            amount=0.01, notional_usdt=300, entry_spread_bps=5,
            entry_rate_long=0.00005, entry_rate_short=0.0002,
            ref_long_price=30000, ref_short_price=30000,
        )
        self.assertFalse(result['ok'])
        # 都失败，不应触发回滚（各自只调用一次）
        self.assertEqual(long_tr.exchange.create_order.call_count, 1)
        self.assertEqual(short_tr.exchange.create_order.call_count, 1)
        os.unlink(path)

    def test_dry_run_does_not_place_orders(self):
        recorder, path = _recorder_with_tempdb()
        long_tr = _make_trader('binance')
        short_tr = _make_trader('gate')
        ex = ArbExecutor({'binance': long_tr, 'gate': short_tr}, recorder=recorder)

        result = ex.open_pair(
            pair_id='PDRY', symbol='BTC/USDT',
            long_exchange='binance', short_exchange='gate',
            amount=0.01, notional_usdt=300, entry_spread_bps=5,
            entry_rate_long=0.00005, entry_rate_short=0.0002,
            ref_long_price=30000, ref_short_price=30000,
            dry_run=True,
        )
        self.assertTrue(result['ok'])
        self.assertTrue(result.get('dry_run'))
        long_tr.exchange.create_order.assert_not_called()
        short_tr.exchange.create_order.assert_not_called()
        os.unlink(path)


class TestArbExecutorClose(unittest.TestCase):

    def test_close_success(self):
        recorder, path = _recorder_with_tempdb()
        long_tr = _make_trader('binance')
        short_tr = _make_trader('gate')
        ex = ArbExecutor({'binance': long_tr, 'gate': short_tr}, recorder=recorder)

        # 先开仓
        open_res = ex.open_pair(
            pair_id='PC', symbol='BTC/USDT',
            long_exchange='binance', short_exchange='gate',
            amount=0.01, notional_usdt=300, entry_spread_bps=5,
            entry_rate_long=0.00005, entry_rate_short=0.0002,
            ref_long_price=30000, ref_short_price=30000,
        )
        trade_id = open_res['trade_id']

        open_info = {
            'long_fill_price': 30000.0,
            'short_fill_price': 30000.0,
            'amount': 0.01,
        }
        close_res = ex.close_pair(
            pair_id='PC', symbol='BTC/USDT',
            long_exchange='binance', short_exchange='gate',
            amount=0.01, trade_id=trade_id,
            close_reason='SPREAD_CONVERGED',
            open_info=open_info, current_spread_bps=0.5,
        )
        self.assertTrue(close_res['ok'])
        long_tr.close_position.assert_called_once()
        short_tr.close_position.assert_called_once()
        # 记录中应变成 CLOSED（get_open_trade 应找不到该 pair）
        self.assertIsNone(recorder.get_open_trade('PC'))
        os.unlink(path)


if __name__ == '__main__':
    unittest.main()
