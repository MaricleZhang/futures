"""
K线形态趋势策略 (15分钟)
KlineTrendStrategy - 纯趋势跟踪，震荡行情强制平仓

策略逻辑：
1. 使用 ADX 判断趋势/震荡状态
   - ADX > 22：趋势行情，允许开仓
   - ADX < 18：震荡行情，强制平仓
2. 使用 K线形态 + EMA + MACD 双重确认入场信号
3. ATR 动态止损止盈（盈亏比 3:1）
4. 追踪止损锁定利润

File: strategies/kline_trend_strategy.py
"""
import numpy as np
import pandas as pd
import pandas_ta_classic as ta
from datetime import datetime
import time
import logging
from strategies.base_strategy import BaseStrategy
from utils.enhanced_candlestick_patterns import EnhancedCandlestickPattern


class KlineTrendStrategy(BaseStrategy):
    """
    K线形态趋势策略

    核心机制：
    1. ADX 趋势过滤：只在趋势行情开仓，震荡行情立即平仓
    2. K线形态确认：使用 EnhancedCandlestickPattern 做入场确认
    3. EMA 趋势对齐：EMA21/EMA55 确认趋势方向
    4. MACD 动量确认：柱状图方向一致才入场
    5. ATR 动态止损止盈：盈亏比 3:1
    """

    def __init__(self, trader, interval='15m'):
        super().__init__(trader)
        self.logger = self.get_logger()

        # ==================== 时间配置 ====================
        self.kline_interval = interval
        interval_to_check = {
            '1m': 60, '5m': 300, '15m': 300,
            '30m': 600, '1h': 900, '4h': 3600,
        }
        self.check_interval = interval_to_check.get(interval, 300)
        self.lookback_period = 120

        # ==================== ADX 趋势判别参数 ====================
        self.adx_period = 14
        self.adx_trend_threshold = 22   # ADX > 22 才允许开仓
        self.adx_exit_threshold = 18    # ADX < 18 触发震荡平仓

        # ==================== EMA 趋势过滤 ====================
        self.ema_fast = 21
        self.ema_slow = 55

        # ==================== MACD 动量确认 ====================
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal_period = 9

        # ==================== ATR 止损止盈 ====================
        self.atr_period = 14
        self.stop_loss_atr_mult = 1.5    # 止损 = ATR × 1.5
        self.take_profit_atr_mult = 4.5  # 止盈 = ATR × 4.5（盈亏比 3:1）

        # ==================== 追踪止损 ====================
        self.trailing_activation = 0.020   # 盈利 2% 激活追踪
        self.trailing_distance = 0.010     # 追踪距离 1%

        # ==================== K线形态参数 ====================
        self.pattern_detector = EnhancedCandlestickPattern()
        self.min_long_prob = 0.45    # 做多最低形态概率
        self.min_short_prob = 0.45   # 做空最低形态概率

        # 强力形态白名单（即使概率略低也可触发）
        self.strong_bullish_patterns = {
            'morning_star', 'three_white_soldiers', 'engulfing'
        }
        self.strong_bearish_patterns = {
            'evening_star', 'three_black_crows', 'engulfing'
        }

        # ==================== 最大持仓时间 ====================
        self.max_hold_bars = 96   # 15m × 96 = 24 小时

        # ==================== 运行时状态 ====================
        self.position_entry_time = None
        self.position_entry_price = None
        self.max_profit_reached = 0.0
        self.trailing_stop_price = None
        self.current_stop_loss_pct = 0.02
        self.current_take_profit_pct = 0.06
        self.hold_bars = 0

        self.logger.info("=" * 65)
        self.logger.info("KlineTrendStrategy 初始化完成")
        self.logger.info(f"K线周期: {self.kline_interval} | ADX入场: {self.adx_trend_threshold} | ADX退出: {self.adx_exit_threshold}")
        self.logger.info(f"EMA: {self.ema_fast}/{self.ema_slow} | ATR止损×{self.stop_loss_atr_mult} 止盈×{self.take_profit_atr_mult}")
        self.logger.info(f"K线形态最低概率: 多头>{self.min_long_prob} 空头>{self.min_short_prob}")
        self.logger.info("=" * 65)

    # ================================================================
    #  指标计算
    # ================================================================

    def _calc_indicators(self, df: pd.DataFrame) -> dict:
        """计算所有技术指标，返回 dict（失败返回 None）"""
        try:
            high = df['high']
            low = df['low']
            close = df['close']
            volume = df['volume']

            ind = {}

            # ADX + DI
            adx_df = ta.adx(high, low, close, length=self.adx_period)
            ind['adx'] = adx_df[f'ADX_{self.adx_period}']
            ind['plus_di'] = adx_df[f'DMP_{self.adx_period}']
            ind['minus_di'] = adx_df[f'DMN_{self.adx_period}']

            # EMA
            ind['ema_fast'] = ta.ema(close, length=self.ema_fast)
            ind['ema_slow'] = ta.ema(close, length=self.ema_slow)

            # MACD
            macd_df = ta.macd(close, fast=self.macd_fast,
                               slow=self.macd_slow, signal=self.macd_signal_period)
            key_h = f'MACDh_{self.macd_fast}_{self.macd_slow}_{self.macd_signal_period}'
            ind['macd_hist'] = macd_df[key_h]

            # ATR
            ind['atr'] = ta.atr(high, low, close, length=self.atr_period)

            # raw price
            ind['close'] = close
            ind['open'] = df['open']
            ind['high'] = high
            ind['low'] = low
            ind['volume'] = volume

            return ind

        except Exception as e:
            self.logger.error(f"指标计算失败: {e}")
            return None

    # ================================================================
    #  趋势/震荡状态判断
    # ================================================================

    def _market_state(self, ind: dict, idx: int) -> str:
        """
        返回市场状态：'trend_up' / 'trend_down' / 'ranging'
        """
        adx = ind['adx'].iloc[idx]
        plus_di = ind['plus_di'].iloc[idx]
        minus_di = ind['minus_di'].iloc[idx]
        close = ind['close'].iloc[idx]
        ema_fast = ind['ema_fast'].iloc[idx]
        ema_slow = ind['ema_slow'].iloc[idx]

        if adx < self.adx_exit_threshold:
            return 'ranging'

        if adx >= self.adx_trend_threshold:
            if plus_di > minus_di and close > ema_fast and ema_fast > ema_slow:
                return 'trend_up'
            if minus_di > plus_di and close < ema_fast and ema_fast < ema_slow:
                return 'trend_down'

        return 'ranging'

    # ================================================================
    #  K线形态确认
    # ================================================================

    def _check_pattern(self, df: pd.DataFrame, idx: int, direction: int) -> bool:
        """
        direction: 1=多, -1=空
        返回：是否有足够强的K线形态支持入场
        """
        try:
            probs = self.pattern_detector.calculate_direction_probability(df, idx)
            patterns = self.pattern_detector.detect_all_patterns(df, idx)
            summary = self.pattern_detector.get_pattern_summary(patterns)

            if direction == 1:
                # 概率满足
                if probs['long_prob'] >= self.min_long_prob:
                    return True
                # 强力形态出现
                for name, result in patterns.items():
                    if result.detected and name in self.strong_bullish_patterns:
                        if result.direction.value == 'bullish' and result.confidence >= 0.5:
                            return True
            else:
                if probs['short_prob'] >= self.min_short_prob:
                    return True
                for name, result in patterns.items():
                    if result.detected and name in self.strong_bearish_patterns:
                        if result.direction.value == 'bearish' and result.confidence >= 0.5:
                            return True

            return False

        except Exception as e:
            self.logger.debug(f"K线形态检测异常: {e}")
            return False

    # ================================================================
    #  信号生成
    # ================================================================

    def generate_signal(self, klines=None) -> int:
        """
        返回: 1=做多, -1=做空, 0=观望
        """
        try:
            if klines is None:
                klines = self.trader.get_klines(
                    symbol=self.trader.symbol,
                    interval=self.kline_interval,
                    limit=self.lookback_period
                )

            if not klines or len(klines) < 60:
                return 0

            df = pd.DataFrame(klines,
                               columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            ind = self._calc_indicators(df)
            if ind is None:
                return 0

            # 使用 -2 避免未完成K线影响
            idx = -2
            state = self._market_state(ind, idx)

            if state == 'ranging':
                return 0

            adx = ind['adx'].iloc[idx]
            adx_prev = ind['adx'].iloc[idx - 1]
            macd_hist = ind['macd_hist'].iloc[idx]
            atr = ind['atr'].iloc[idx]
            close = ind['close'].iloc[idx]

            # ADX 必须上升（趋势加速）
            adx_rising = adx > adx_prev

            if state == 'trend_up':
                if not adx_rising:
                    return 0
                if macd_hist <= 0:
                    return 0
                if not self._check_pattern(df, idx, direction=1):
                    return 0
                # 计算动态止损止盈
                if atr > 0:
                    self.current_stop_loss_pct = min(atr * self.stop_loss_atr_mult / close, 0.04)
                    self.current_take_profit_pct = min(atr * self.take_profit_atr_mult / close, 0.15)
                self.logger.info(
                    f"[KlineTrend] 做多信号 | ADX={adx:.1f} | ATR={atr:.4f} | "
                    f"SL={self.current_stop_loss_pct:.2%} TP={self.current_take_profit_pct:.2%}"
                )
                return 1

            if state == 'trend_down':
                if not adx_rising:
                    return 0
                if macd_hist >= 0:
                    return 0
                if not self._check_pattern(df, idx, direction=-1):
                    return 0
                if atr > 0:
                    self.current_stop_loss_pct = min(atr * self.stop_loss_atr_mult / close, 0.04)
                    self.current_take_profit_pct = min(atr * self.take_profit_atr_mult / close, 0.15)
                self.logger.info(
                    f"[KlineTrend] 做空信号 | ADX={adx:.1f} | ATR={atr:.4f} | "
                    f"SL={self.current_stop_loss_pct:.2%} TP={self.current_take_profit_pct:.2%}"
                )
                return -1

            return 0

        except Exception as e:
            self.logger.error(f"generate_signal 异常: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return 0

    # ================================================================
    #  仓位监控
    # ================================================================

    def monitor_position(self):
        """监控仓位：无仓位时找入场，有仓位时管理出场"""
        try:
            position = self.trader.get_position()
            pos_amt = float(position['info'].get('positionAmt', 0)) if position else 0.0

            if pos_amt == 0:
                # ----- 无仓位：寻找入场信号 -----
                self.hold_bars = 0
                self.max_profit_reached = 0.0
                self.trailing_stop_price = None

                signal = self.generate_signal()
                if signal != 0:
                    balance = self.trader.get_balance()
                    available = float(balance['free'])
                    current_price = self.trader.get_market_price()

                    symbol_config = getattr(self.trader, 'symbol_config', {})
                    trade_pct = symbol_config.get('trade_amount_percent', 100)
                    trade_amount = (available * trade_pct / 100) / current_price

                    if signal == 1:
                        self.trader.open_long(amount=trade_amount)
                    else:
                        self.trader.open_short(amount=trade_amount)

                    self.position_entry_time = time.time()
                    self.position_entry_price = current_price

            else:
                # ----- 有仓位：执行出场检查 -----
                self.hold_bars += 1
                self._manage_position(position)

        except Exception as e:
            self.logger.error(f"monitor_position 异常: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())

    def _manage_position(self, position):
        """出场逻辑（优先级从高到低）"""
        try:
            pos_amt = float(position['info'].get('positionAmt', 0))
            entry_price = float(position['info'].get('entryPrice', 0))
            current_price = self.trader.get_market_price()
            side = "long" if pos_amt > 0 else "short"

            if entry_price == 0:
                return

            # 盈亏百分比
            if side == "long":
                pnl_pct = (current_price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - current_price) / entry_price

            # 1. 追踪止损
            if pnl_pct > self.max_profit_reached:
                self.max_profit_reached = pnl_pct
                if pnl_pct >= self.trailing_activation:
                    if side == "long":
                        new_ts = current_price * (1 - self.trailing_distance)
                        if self.trailing_stop_price is None or new_ts > self.trailing_stop_price:
                            self.trailing_stop_price = new_ts
                    else:
                        new_ts = current_price * (1 + self.trailing_distance)
                        if self.trailing_stop_price is None or new_ts < self.trailing_stop_price:
                            self.trailing_stop_price = new_ts

            if self.trailing_stop_price is not None:
                hit = (side == "long" and current_price <= self.trailing_stop_price) or \
                      (side == "short" and current_price >= self.trailing_stop_price)
                if hit:
                    self.logger.info(f"[KlineTrend] 追踪止损 | {side} | 盈亏={pnl_pct:.2%}")
                    self.trader.close_position()
                    return

            # 2. 固定止损
            sl = getattr(self, 'current_stop_loss_pct', 0.02)
            if pnl_pct <= -sl:
                self.logger.info(f"[KlineTrend] 止损 | {side} | 盈亏={pnl_pct:.2%}")
                self.trader.close_position()
                return

            # 3. 固定止盈
            tp = getattr(self, 'current_take_profit_pct', 0.06)
            if pnl_pct >= tp:
                self.logger.info(f"[KlineTrend] 止盈 | {side} | 盈亏={pnl_pct:.2%}")
                self.trader.close_position()
                return

            # 4. 震荡平仓：实时检查ADX是否跌破退出阈值
            klines = self.trader.get_klines(
                symbol=self.trader.symbol,
                interval=self.kline_interval,
                limit=self.lookback_period
            )
            if klines and len(klines) >= 60:
                df = pd.DataFrame(klines,
                                   columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                ind = self._calc_indicators(df)
                if ind is not None:
                    adx_now = ind['adx'].iloc[-2]
                    ema_f = ind['ema_fast'].iloc[-2]
                    ema_s = ind['ema_slow'].iloc[-2]
                    close_now = ind['close'].iloc[-2]

                    # ADX 跌破震荡阈值
                    if adx_now < self.adx_exit_threshold:
                        self.logger.info(f"[KlineTrend] 震荡平仓 ADX={adx_now:.1f}<{self.adx_exit_threshold} | 盈亏={pnl_pct:.2%}")
                        self.trader.close_position()
                        return

                    # EMA 排列破坏（趋势结构失效）
                    if side == "long" and ema_f < ema_s:
                        self.logger.info(f"[KlineTrend] EMA空头破坏 多仓平仓 | 盈亏={pnl_pct:.2%}")
                        self.trader.close_position()
                        return
                    if side == "short" and ema_f > ema_s:
                        self.logger.info(f"[KlineTrend] EMA多头破坏 空仓平仓 | 盈亏={pnl_pct:.2%}")
                        self.trader.close_position()
                        return

            # 5. 超时强平
            if self.hold_bars >= self.max_hold_bars:
                self.logger.info(f"[KlineTrend] 超时平仓 hold={self.hold_bars}bars | 盈亏={pnl_pct:.2%}")
                self.trader.close_position()
                return

        except Exception as e:
            self.logger.error(f"_manage_position 异常: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
