"""
套利执行引擎
- 两腿并行下单（ThreadPoolExecutor）
- 部分成交自动回滚（_emergency_close）
- clientOrderId 幂等：重试不会重复下单
- 三次回滚失败 -> 写 pending_unwind 并返回 fatal

订单模式：市价，不传 stop_loss/take_profit（风控交给策略层）。
"""
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional

from utils.arb_trade_recorder import ArbTradeRecorder


class ArbExecutor:
    def __init__(self, traders: Dict[str, "Trader"], recorder: Optional[ArbTradeRecorder] = None,
                 order_timeout_sec: int = 30):
        self.traders = traders
        self.recorder = recorder or ArbTradeRecorder()
        self.pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix='arb-exec')
        self.order_timeout = order_timeout_sec
        self.logger = logging.getLogger('arb.executor')

    # ------------------------------------------------------------------
    # 开仓
    # ------------------------------------------------------------------
    def open_pair(self, *, pair_id: str, symbol: str, long_exchange: str, short_exchange: str,
                  amount: float, notional_usdt: float, entry_spread_bps: float,
                  entry_rate_long: float, entry_rate_short: float,
                  ref_long_price: float, ref_short_price: float,
                  dry_run: bool = False) -> dict:

        long_trader = self.traders[long_exchange]
        short_trader = self.traders[short_exchange]

        if dry_run:
            trade_id = self.recorder.record_open(
                pair_id=pair_id, symbol=symbol,
                long_exchange=long_exchange, short_exchange=short_exchange,
                long_price=ref_long_price, short_price=ref_short_price,
                amount=amount, notional_usdt=notional_usdt,
                entry_spread_bps=entry_spread_bps,
                entry_rate_long=entry_rate_long, entry_rate_short=entry_rate_short,
                dry_run=True)
            return {
                'ok': True,
                'trade_id': trade_id,
                'long_fill_price': ref_long_price,
                'short_fill_price': ref_short_price,
                'filled_amount': amount,
                'dry_run': True,
            }

        # 规范两腿下单量（Gate 可能需要换算成张数）
        long_amount = long_trader.adapter.normalize_amount(long_trader.symbol, amount)
        short_amount = short_trader.adapter.normalize_amount(short_trader.symbol, amount)

        # 对齐两腿数量：取较小值重算，保证 delta 中性
        # （各交易所的 amount_to_precision 可能精度不同，以较小值为准）
        long_base = long_amount * (long_trader.adapter.contract_size(long_trader.symbol) or 1)
        short_base = short_amount * (short_trader.adapter.contract_size(short_trader.symbol) or 1)
        common_base = min(long_base, short_base)
        long_amount = long_trader.adapter.normalize_amount(long_trader.symbol, common_base)
        short_amount = short_trader.adapter.normalize_amount(short_trader.symbol, common_base)

        client_root = f"{pair_id}-{int(time.time())}"

        self.logger.info(
            f"[open] pair={pair_id} long={long_exchange}({long_amount}) "
            f"short={short_exchange}({short_amount}) client_root={client_root}")

        long_future = self.pool.submit(
            self._safe_market_order, long_trader, 'buy', long_amount,
            f"{client_root}-L")
        short_future = self.pool.submit(
            self._safe_market_order, short_trader, 'sell', short_amount,
            f"{client_root}-S")

        long_res = self._await(long_future, 'long')
        short_res = self._await(short_future, 'short')

        if long_res['ok'] and short_res['ok']:
            trade_id = self.recorder.record_open(
                pair_id=pair_id, symbol=symbol,
                long_exchange=long_exchange, short_exchange=short_exchange,
                long_price=long_res['avg_price'], short_price=short_res['avg_price'],
                amount=common_base, notional_usdt=notional_usdt,
                entry_spread_bps=entry_spread_bps,
                entry_rate_long=entry_rate_long, entry_rate_short=entry_rate_short,
                dry_run=False)
            return {
                'ok': True,
                'trade_id': trade_id,
                'long_fill_price': long_res['avg_price'],
                'short_fill_price': short_res['avg_price'],
                'filled_amount': common_base,
            }

        # 单腿失败 -> 回滚
        fatal = False
        if long_res['ok'] and not short_res['ok']:
            self.logger.error(f"短腿失败，回滚长腿。reason={short_res.get('reason')}")
            ok = self._emergency_close(long_trader, 'sell', long_amount, pair_id,
                                       'rollback_short_failed', {'short_error': short_res.get('reason')})
            fatal = not ok
        elif short_res['ok'] and not long_res['ok']:
            self.logger.error(f"长腿失败，回滚短腿。reason={long_res.get('reason')}")
            ok = self._emergency_close(short_trader, 'buy', short_amount, pair_id,
                                       'rollback_long_failed', {'long_error': long_res.get('reason')})
            fatal = not ok

        return {
            'ok': False,
            'reason': f"partial_or_failed long_ok={long_res['ok']} short_ok={short_res['ok']}",
            'fatal': fatal,
        }

    # ------------------------------------------------------------------
    # 平仓
    # ------------------------------------------------------------------
    def close_pair(self, *, pair_id: str, symbol: str, long_exchange: str, short_exchange: str,
                   amount: float, trade_id: int, close_reason: str, open_info: dict,
                   current_spread_bps: float, dry_run: bool = False) -> dict:

        long_trader = self.traders[long_exchange]
        short_trader = self.traders[short_exchange]

        if dry_run:
            self.recorder.record_close(
                trade_id,
                close_long_price=open_info['long_fill_price'],
                close_short_price=open_info['short_fill_price'],
                close_spread_bps=current_spread_bps,
                gross_pnl=0, fees=0, net_pnl=0,
                close_reason=close_reason)
            return {'ok': True, 'dry_run': True}

        self.logger.info(f"[close] pair={pair_id} reason={close_reason}")

        long_future = self.pool.submit(self._safe_close, long_trader, long_exchange)
        short_future = self.pool.submit(self._safe_close, short_trader, short_exchange)

        long_res = self._await(long_future, 'long-close')
        short_res = self._await(short_future, 'short-close')

        # 如果任一腿失败，走 emergency_close 兜底
        if not long_res['ok']:
            long_amt = long_trader.adapter.normalize_amount(long_trader.symbol, amount)
            long_ok = self._emergency_close(long_trader, 'sell', long_amt, pair_id,
                                            'close_pair_long_failed',
                                            {'err': long_res.get('reason')})
            long_res['ok'] = long_ok
        if not short_res['ok']:
            short_amt = short_trader.adapter.normalize_amount(short_trader.symbol, amount)
            short_ok = self._emergency_close(short_trader, 'buy', short_amt, pair_id,
                                             'close_pair_short_failed',
                                             {'err': short_res.get('reason')})
            short_res['ok'] = short_ok

        if not (long_res['ok'] and short_res['ok']):
            return {
                'ok': False,
                'reason': f"close_failed long_ok={long_res['ok']} short_ok={short_res['ok']}",
            }

        # 计算 PnL
        long_close_price = long_res.get('avg_price') or open_info['long_fill_price']
        short_close_price = short_res.get('avg_price') or open_info['short_fill_price']
        long_open_price = open_info['long_fill_price']
        short_open_price = open_info['short_fill_price']
        qty = open_info['amount']

        long_pnl = (long_close_price - long_open_price) * qty
        short_pnl = (short_open_price - short_close_price) * qty
        gross_pnl = long_pnl + short_pnl

        # 费用估算：4 次 taker（2 腿开 + 2 腿平）
        taker_fee_bps = 4
        fees = sum([long_open_price, long_close_price, short_open_price, short_close_price]) * qty \
               * taker_fee_bps / 10000
        net_pnl = gross_pnl - fees

        self.recorder.record_close(
            trade_id,
            close_long_price=long_close_price,
            close_short_price=short_close_price,
            close_spread_bps=current_spread_bps,
            gross_pnl=gross_pnl, fees=fees, net_pnl=net_pnl,
            close_reason=close_reason)

        return {'ok': True, 'net_pnl': net_pnl}

    # ------------------------------------------------------------------
    # 底层工具
    # ------------------------------------------------------------------
    def _await(self, future, tag: str) -> dict:
        try:
            return future.result(timeout=self.order_timeout)
        except Exception as e:
            self.logger.error(f"{tag} future 异常: {e}")
            return {'ok': False, 'reason': f"{type(e).__name__}: {e}"}

    def _safe_market_order(self, trader, side: str, amount: float, client_order_id: str) -> dict:
        """市价下单；成功返回 {ok, avg_price, filled, order_id}；失败返回 {ok:false, reason}"""
        try:
            params = {}
            if trader.exchange_name == 'binance':
                params['newClientOrderId'] = client_order_id
            else:
                # Gate 用 text 字段做幂等；必须以 't-' 前缀（Gate 要求）
                params['text'] = f"t-{client_order_id[-22:]}"

            order = trader.exchange.create_order(
                symbol=trader.symbol,
                type='market',
                side=side,
                amount=amount,
                params=params,
            )
            avg = order.get('average') or order.get('price')
            if avg is None and order.get('cost') and order.get('filled'):
                avg = float(order['cost']) / float(order['filled'])
            filled = order.get('filled', amount)

            self.logger.info(
                f"[order ok] {trader.exchange_name} {side} amount={amount} avg={avg} id={order.get('id')}")
            return {
                'ok': True,
                'avg_price': float(avg) if avg else None,
                'filled': float(filled) if filled else amount,
                'order_id': order.get('id'),
                'raw': order,
            }
        except Exception as e:
            self.logger.error(f"[order fail] {trader.exchange_name} {side} amount={amount}: {e}")
            return {'ok': False, 'reason': f"{type(e).__name__}: {e}"}

    def _safe_close(self, trader, exchange_name: str) -> dict:
        """调用 Trader.close_position 并尽量取回均价"""
        try:
            res = trader.close_position(trader.symbol)
            avg = None
            if isinstance(res, dict):
                avg = res.get('average') or res.get('price')
            return {'ok': True, 'avg_price': float(avg) if avg else None, 'raw': res}
        except Exception as e:
            self.logger.error(f"[close fail] {exchange_name}: {e}")
            return {'ok': False, 'reason': f"{type(e).__name__}: {e}"}

    def _emergency_close(self, trader, side: str, amount: float, pair_id: str,
                         reason: str, payload: dict) -> bool:
        """市价平仓重试 3 次；全部失败 -> 写 pending_unwind 返回 False（fatal）"""
        for attempt in range(3):
            try:
                order = trader.exchange.create_order(
                    symbol=trader.symbol,
                    type='market',
                    side=side,
                    amount=amount,
                    params={'reduceOnly': True},
                )
                self.logger.warning(
                    f"[emergency_close ok attempt={attempt+1}] {trader.exchange_name} {side} {amount} id={order.get('id')}")
                return True
            except Exception as e:
                self.logger.error(
                    f"[emergency_close fail attempt={attempt+1}] {trader.exchange_name} {side} {amount}: {e}")
                time.sleep(2)

        # 3 次失败 -> pending_unwind
        self.recorder.record_pending_unwind(
            pair_id=pair_id,
            exchange=trader.exchange_name,
            symbol=trader.symbol,
            side=side,
            amount=amount,
            reason=reason,
            payload=payload,
        )
        return False
