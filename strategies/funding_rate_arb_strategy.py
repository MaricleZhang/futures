"""
跨交易所资金费率套利策略

核心逻辑：
1. 每 tick 读取两所当前 funding rate + mark price，计算 spread_bps。
2. 开仓：|spread_bps| >= threshold_bps 且扣除 2 腿 taker 费用后预期净收益 >= min_net_bps_after_fee。
   - 高费率方向做空（付费方付给持空方）
   - 低费率方向做多
   - 单腿名义金额 = cfg['notional_usdt']
3. 持仓期间：
   - 差价收敛到 exit_bps 以下 -> SPREAD_CONVERGED 平仓
   - 差价方向反转 -> SPREAD_INVERTED
   - 持仓超过 max_holding_hours -> TIMEOUT
   - 两腿价差偏离超过 price_divergence_stop_pct -> PRICE_DIVERGENCE_STOP
   - 任一腿维持保证金率 < margin_stop_ratio -> MARGIN_STOP
4. 平仓：两腿并行市价平仓（ArbExecutor 负责）。
"""
import time
from typing import Optional, Dict

from strategies.arbitrage_base_strategy import ArbitrageStrategy


def _bps(value: float) -> float:
    return value * 10000


class FundingRateArbStrategy(ArbitrageStrategy):

    def __init__(self, traders, monitor, config_):
        super().__init__(traders, monitor, config_)
        self._ex_a, self._ex_b = config_['exchanges']
        self._symbol = config_['symbol']
        self._taker_fee_bps = float(config_.get('taker_fee_bps', 4))
        self._min_net_bps = float(config_.get('min_net_bps_after_fee', 2))
        self._threshold_bps = float(config_['threshold_bps'])
        self._exit_bps = float(config_['exit_bps'])
        self._max_holding_hours = float(config_['max_holding_hours'])
        self._price_divergence_stop_pct = float(config_['price_divergence_stop_pct'])
        self._margin_stop_ratio = float(config_['margin_stop_ratio'])
        self._max_slippage_bps = float(config_.get('max_slippage_bps', 30))

    # ---------------- 决策函数 ----------------
    def should_open(self, diff: dict) -> Optional[dict]:
        if self.state != self.STATE_FLAT:
            return None

        spread_bps = diff['spread_bps']
        if abs(spread_bps) < self._threshold_bps:
            return None

        # 手续费净收益校验：开仓 + 平仓共计 4 笔 taker
        net_bps = abs(spread_bps) - self._taker_fee_bps * 4
        if net_bps < self._min_net_bps:
            self.logger.debug(
                f"spread={spread_bps:.2f}bp 扣费后净={net_bps:.2f}bp < 最低 {self._min_net_bps}bp，不开仓")
            return None

        # 价格一致性校验：mark 偏离过大不开仓
        a_mark = diff.get('a_mark')
        b_mark = diff.get('b_mark')
        if a_mark and b_mark:
            price_gap_pct = abs(a_mark - b_mark) / ((a_mark + b_mark) / 2) * 100
            if price_gap_pct > self.cfg.get('entry_price_gap_max_pct', 0.2):
                self.logger.warning(f"两所 mark 偏离 {price_gap_pct:.3f}% > 0.2%，放弃本轮开仓")
                return None

        # 决定腿向：a_rate > b_rate  -> a 做空 + b 做多
        if diff['a_rate'] > diff['b_rate']:
            short_ex, long_ex = self._ex_a, self._ex_b
            short_rate, long_rate = diff['a_rate'], diff['b_rate']
            short_mark, long_mark = a_mark, b_mark
        else:
            short_ex, long_ex = self._ex_b, self._ex_a
            short_rate, long_rate = diff['b_rate'], diff['a_rate']
            short_mark, long_mark = b_mark, a_mark

        notional = float(self.cfg['notional_usdt'])
        ref_price = long_mark or short_mark
        if not ref_price or ref_price <= 0:
            self.logger.error("无法获取 mark price，跳过开仓")
            return None
        amount = notional / ref_price

        return {
            'long_exchange': long_ex,
            'short_exchange': short_ex,
            'long_rate': long_rate,
            'short_rate': short_rate,
            'long_mark': long_mark,
            'short_mark': short_mark,
            'amount': amount,
            'notional_usdt': notional,
            'entry_spread_bps': spread_bps,
        }

    def should_close(self, diff: dict, pos: dict) -> Optional[str]:
        if self.state != self.STATE_HEDGED:
            return None

        # 1. 持仓时长
        holding_hours = (time.time() - pos['open_ts']) / 3600
        if holding_hours >= self._max_holding_hours:
            return 'TIMEOUT'

        # 2. 差价收敛
        if abs(diff['spread_bps']) <= self._exit_bps:
            return 'SPREAD_CONVERGED'

        # 3. 方向反转（原本 short_ex 费率高，现在反过来）
        if pos['entry_spread_bps'] * diff['spread_bps'] < 0 and abs(diff['spread_bps']) >= self._threshold_bps:
            return 'SPREAD_INVERTED'

        # 4. 两腿价差偏离
        if pos.get('price_divergence_pct') is not None and \
                abs(pos['price_divergence_pct']) > self._price_divergence_stop_pct:
            return 'PRICE_DIVERGENCE_STOP'

        # 5. 保证金健康度
        if pos.get('min_margin_ratio') is not None and \
                pos['min_margin_ratio'] < self._margin_stop_ratio:
            return 'MARGIN_STOP'

        return None

    # ---------------- 状态机 ----------------
    def tick(self, executor) -> None:
        try:
            diff = self.monitor.get_differential(self._symbol, self._ex_a, self._ex_b)
        except Exception as e:
            self.logger.error(f"获取利差失败: {e}")
            return

        self.logger.info(
            f"state={self.state} rate_a={diff['a_rate']:.6f} rate_b={diff['b_rate']:.6f} "
            f"spread_bps={diff['spread_bps']:.2f} (ann={diff['annualized_spread']*100:.2f}%)")

        if self.state == self.STATE_CRITICAL:
            return  # 已入 pending_unwind，暂停新开仓，等待人工或下游处理

        if self.state == self.STATE_FLAT:
            plan = self.should_open(diff)
            if plan is None:
                return
            self._do_open(executor, plan)
            return

        if self.state == self.STATE_HEDGED:
            pos_snapshot = self._snapshot_position(diff)
            reason = self.should_close(diff, pos_snapshot)
            if reason:
                self._do_close(executor, reason, diff, pos_snapshot)
            return

    # ---------------- 内部动作 ----------------
    def _do_open(self, executor, plan: dict):
        dry_run = bool(self.cfg.get('dry_run', False))
        self.logger.info(
            f"准备开仓 long={plan['long_exchange']}@{plan['long_mark']} "
            f"short={plan['short_exchange']}@{plan['short_mark']} amount={plan['amount']:.6f} "
            f"spread_bps={plan['entry_spread_bps']:.2f} dry_run={dry_run}")

        result = executor.open_pair(
            pair_id=self.pair_id,
            symbol=self._symbol,
            long_exchange=plan['long_exchange'],
            short_exchange=plan['short_exchange'],
            amount=plan['amount'],
            notional_usdt=plan['notional_usdt'],
            entry_spread_bps=plan['entry_spread_bps'],
            entry_rate_long=plan['long_rate'],
            entry_rate_short=plan['short_rate'],
            ref_long_price=plan['long_mark'],
            ref_short_price=plan['short_mark'],
            dry_run=dry_run,
        )

        if not result.get('ok'):
            self.logger.error(f"开仓失败: {result.get('reason')}")
            if result.get('fatal'):
                self.state = self.STATE_CRITICAL
            return

        self.open_trade_id = result['trade_id']
        self.open_info = {
            'open_ts': time.time(),
            'long_exchange': plan['long_exchange'],
            'short_exchange': plan['short_exchange'],
            'long_fill_price': result.get('long_fill_price', plan['long_mark']),
            'short_fill_price': result.get('short_fill_price', plan['short_mark']),
            'amount': result.get('filled_amount', plan['amount']),
            'entry_spread_bps': plan['entry_spread_bps'],
        }
        self.state = self.STATE_HEDGED
        self.logger.info(f"开仓成功 trade_id={self.open_trade_id} 状态 -> HEDGED")

    def _do_close(self, executor, reason: str, diff: dict, pos_snapshot: dict):
        dry_run = bool(self.cfg.get('dry_run', False))
        self.logger.info(f"准备平仓 reason={reason} dry_run={dry_run}")
        self.state = self.STATE_CLOSING

        result = executor.close_pair(
            pair_id=self.pair_id,
            symbol=self._symbol,
            long_exchange=self.open_info['long_exchange'],
            short_exchange=self.open_info['short_exchange'],
            amount=self.open_info['amount'],
            trade_id=self.open_trade_id,
            close_reason=reason,
            open_info=self.open_info,
            current_spread_bps=diff['spread_bps'],
            dry_run=dry_run,
        )

        if not result.get('ok'):
            self.logger.error(f"平仓失败: {result.get('reason')}；状态 -> CRITICAL")
            self.state = self.STATE_CRITICAL
            return

        self.state = self.STATE_FLAT
        self.open_info = None
        self.open_trade_id = None
        self.logger.info(f"平仓成功 状态 -> FLAT reason={reason}")

    def _snapshot_position(self, diff: dict) -> dict:
        """读取当前持仓辅助信息（用于 should_close 判断）"""
        snap = {
            'open_ts': self.open_info['open_ts'] if self.open_info else time.time(),
            'entry_spread_bps': self.open_info.get('entry_spread_bps', 0) if self.open_info else 0,
        }
        # 两腿价格发散百分比
        a_mark, b_mark = diff.get('a_mark'), diff.get('b_mark')
        if a_mark and b_mark:
            mid = (a_mark + b_mark) / 2
            snap['price_divergence_pct'] = abs(a_mark - b_mark) / mid * 100

        # 各腿维持保证金率（通过 position 对象读取）
        margin_ratios = []
        for ex in (self._ex_a, self._ex_b):
            try:
                pos = self.traders[ex].get_position()
                if pos and isinstance(pos, dict):
                    mmr = pos.get('info', {}).get('maintMargin')
                    im = pos.get('info', {}).get('initialMargin')
                    margin_ratio = pos.get('marginRatio')
                    if margin_ratio is not None:
                        margin_ratios.append(float(margin_ratio))
                    elif mmr and im and float(im) > 0:
                        margin_ratios.append(1 - float(mmr) / float(im))
            except Exception as e:
                self.logger.debug(f"{ex} 持仓查询失败: {e}")
        if margin_ratios:
            snap['min_margin_ratio'] = min(margin_ratios)
        return snap
