"""
高级短线趋势量化策略
Advanced Short-term Trend Trading Strategy

技术指标：
- EMA (指数移动平均线): 快速和慢速EMA用于识别趋势方向
- MACD (移动平均收敛发散): 确认动量和趋势转折
- ROC (变化率): 衡量价格变动速率
- ADX (平均趋向指数): 确认趋势强度
- +DI/-DI (方向指标): 确认趋势方向
- KAMA (考夫曼自适应移动平均): 动态支撑/阻力位
- MOM (动量指标): 价格动量确认

策略特点：
1. 多指标交叉验证，提高信号可靠性
2. 动态阈值调整
3. 趋势强度评分系统
4. 完整的风险管理
"""

import pandas as pd
import numpy as np
import talib
import time
from datetime import datetime
from .base_strategy import BaseStrategy


class AdvancedShortTermTrendStrategy(BaseStrategy):
    """
    高级短线趋势策略

    信号生成逻辑：
    1. 趋势识别：EMA快线 vs 慢线
    2. 动量确认：MACD、ROC、MOM
    3. 趋势强度：ADX >= 阈值
    4. 方向确认：+DI vs -DI
    5. 动态支撑：KAMA位置

    综合评分系统：
    - 每个指标贡献0-100分
    - 总分 >= 60: 强信号
    - 总分 40-60: 中等信号
    - 总分 < 40: 弱信号（观望）
    """

    def __init__(self, trader):
        """
        初始化策略参数

        Args:
            trader: 交易器实例
        """
        super().__init__(trader)

        # === 时间框架配置 ===
        self.kline_interval = '15m'         # K线周期：15分钟（短线）
        self.check_interval = 300           # 检查间隔：5分钟（300秒）
        self.lookback_period = 100          # 回看周期：100根K线
        self.training_lookback = self.lookback_period

        # === EMA 指标参数 ===
        self.ema_fast_period = 8            # 快速EMA周期
        self.ema_slow_period = 21           # 慢速EMA周期
        self.ema_trend_period = 50          # 趋势EMA周期

        # === MACD 指标参数 ===
        self.macd_fast_period = 12          # MACD快线周期
        self.macd_slow_period = 26          # MACD慢线周期
        self.macd_signal_period = 9         # MACD信号线周期

        # === ROC 和 MOM 参数 ===
        self.roc_period = 10                # ROC周期
        self.mom_period = 10                # 动量周期

        # === ADX 和 DI 参数 ===
        self.adx_period = 14                # ADX周期
        self.adx_threshold = 18             # ADX阈值（趋势强度）
        self.adx_strong = 30                # 强趋势阈值

        # === KAMA 参数 ===
        self.kama_period = 30               # KAMA周期

        # === 信号阈值 ===
        self.signal_threshold_strong = 70   # 强信号阈值
        self.signal_threshold_medium = 50   # 中等信号阈值
        self.di_diff_threshold = 5          # DI差值阈值
        self.macd_hist_threshold = 0        # MACD柱状图阈值

        # === 风险管理参数 ===
        self.stop_loss_pct = 0.015          # 止损百分比：1.5%
        self.take_profit_pct = 0.045        # 止盈百分比：4.5%（1:3风险回报比）
        self.trailing_stop_enabled = True   # 启用追踪止损
        self.trailing_stop_pct = 0.02       # 追踪止损百分比：2%
        self.max_holding_periods = 48       # 最大持仓周期（12小时）

        # === 持仓追踪 ===
        self.entry_time = None              # 开仓时间
        self.entry_price = None             # 开仓价格
        self.highest_price = None           # 最高价格（追踪止损用）
        self.lowest_price = None            # 最低价格（追踪止损用）

        # === 日志 ===
        self.logger.info("="*60)
        self.logger.info("高级短线趋势策略已初始化")
        self.logger.info(f"交易品种: {trader.symbol}")
        self.logger.info(f"K线周期: {self.kline_interval}")
        self.logger.info(f"指标配置: EMA({self.ema_fast_period}/{self.ema_slow_period}), "
                        f"MACD({self.macd_fast_period},{self.macd_slow_period},{self.macd_signal_period}), "
                        f"ADX({self.adx_period}), KAMA({self.kama_period})")
        self.logger.info(f"风控: 止损{self.stop_loss_pct*100}%, 止盈{self.take_profit_pct*100}%")
        self.logger.info("="*60)

    def calculate_indicators(self, df):
        """
        计算所有技术指标

        Args:
            df: DataFrame包含OHLCV数据

        Returns:
            dict: 包含所有计算好的指标
        """
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values

        indicators = {}

        try:
            # === 1. EMA 指标 ===
            indicators['ema_fast'] = talib.EMA(close, timeperiod=self.ema_fast_period)
            indicators['ema_slow'] = talib.EMA(close, timeperiod=self.ema_slow_period)
            indicators['ema_trend'] = talib.EMA(close, timeperiod=self.ema_trend_period)

            # === 2. MACD 指标 ===
            macd, macd_signal, macd_hist = talib.MACD(
                close,
                fastperiod=self.macd_fast_period,
                slowperiod=self.macd_slow_period,
                signalperiod=self.macd_signal_period
            )
            indicators['macd'] = macd
            indicators['macd_signal'] = macd_signal
            indicators['macd_hist'] = macd_hist

            # === 3. ROC 指标 ===
            indicators['roc'] = talib.ROC(close, timeperiod=self.roc_period)

            # === 4. ADX 和 DI 指标 ===
            indicators['adx'] = talib.ADX(high, low, close, timeperiod=self.adx_period)
            indicators['plus_di'] = talib.PLUS_DI(high, low, close, timeperiod=self.adx_period)
            indicators['minus_di'] = talib.MINUS_DI(high, low, close, timeperiod=self.adx_period)

            # === 5. KAMA 指标 ===
            indicators['kama'] = talib.KAMA(close, timeperiod=self.kama_period)

            # === 6. MOM 指标 ===
            indicators['mom'] = talib.MOM(close, timeperiod=self.mom_period)

            # === 计算衍生指标 ===
            # EMA差值百分比
            indicators['ema_diff_pct'] = ((indicators['ema_fast'] - indicators['ema_slow']) /
                                          indicators['ema_slow'] * 100)

            # DI差值
            indicators['di_diff'] = indicators['plus_di'] - indicators['minus_di']

            # KAMA距离百分比
            indicators['kama_diff_pct'] = ((close - indicators['kama']) /
                                           indicators['kama'] * 100)

            # 价格相对于趋势EMA的位置
            indicators['price_trend_position'] = ((close - indicators['ema_trend']) /
                                                  indicators['ema_trend'] * 100)

            return indicators

        except Exception as e:
            self.logger.error(f"计算指标时出错: {e}")
            return None

    def analyze_signal_strength(self, indicators, current_price):
        """
        分析信号强度（评分系统）

        Args:
            indicators: 技术指标字典
            current_price: 当前价格

        Returns:
            dict: {
                'direction': 1/-1/0 (多/空/观望),
                'strength': 0-100 (信号强度分数),
                'details': {} (详细评分)
            }
        """
        # 获取最新指标值
        ema_fast = indicators['ema_fast'][-1]
        ema_slow = indicators['ema_slow'][-1]
        ema_trend = indicators['ema_trend'][-1]
        ema_diff_pct = indicators['ema_diff_pct'][-1]

        macd = indicators['macd'][-1]
        macd_signal = indicators['macd_signal'][-1]
        macd_hist = indicators['macd_hist'][-1]
        macd_hist_prev = indicators['macd_hist'][-2]

        roc = indicators['roc'][-1]
        mom = indicators['mom'][-1]

        adx = indicators['adx'][-1]
        plus_di = indicators['plus_di'][-1]
        minus_di = indicators['minus_di'][-1]
        di_diff = indicators['di_diff'][-1]

        kama = indicators['kama'][-1]
        kama_diff_pct = indicators['kama_diff_pct'][-1]

        # 初始化评分
        bull_score = 0  # 多头评分
        bear_score = 0  # 空头评分
        details = {}

        # === 1. EMA 趋势评分 (0-15分) ===
        if ema_fast > ema_slow:
            ema_score = 15 if ema_diff_pct > 0.5 else 10
            bull_score += ema_score
            details['ema'] = f"多头+{ema_score}分 (快线>慢线, 差值{ema_diff_pct:.2f}%)"
        elif ema_fast < ema_slow:
            ema_score = 15 if ema_diff_pct < -0.5 else 10
            bear_score += ema_score
            details['ema'] = f"空头+{ema_score}分 (快线<慢线, 差值{ema_diff_pct:.2f}%)"
        else:
            details['ema'] = "中性"

        # === 2. MACD 动量评分 (0-20分) ===
        macd_strength = 0
        if macd > macd_signal and macd_hist > 0:
            # MACD金叉且柱状图向上
            macd_strength = 20 if macd_hist > macd_hist_prev else 15
            bull_score += macd_strength
            details['macd'] = f"多头+{macd_strength}分 (金叉, 柱状图向上)"
        elif macd < macd_signal and macd_hist < 0:
            # MACD死叉且柱状图向下
            macd_strength = 20 if macd_hist < macd_hist_prev else 15
            bear_score += macd_strength
            details['macd'] = f"空头+{macd_strength}分 (死叉, 柱状图向下)"
        else:
            details['macd'] = "中性"

        # === 3. ROC 变化率评分 (0-10分) ===
        if roc > 2:
            roc_score = 10 if roc > 5 else 7
            bull_score += roc_score
            details['roc'] = f"多头+{roc_score}分 (ROC={roc:.2f}%)"
        elif roc < -2:
            roc_score = 10 if roc < -5 else 7
            bear_score += roc_score
            details['roc'] = f"空头+{roc_score}分 (ROC={roc:.2f}%)"
        else:
            details['roc'] = f"中性 (ROC={roc:.2f}%)"

        # === 4. ADX 趋势强度评分 (0-20分) ===
        if adx >= self.adx_strong:
            adx_score = 20
            details['adx'] = f"强趋势+{adx_score}分 (ADX={adx:.2f})"
        elif adx >= self.adx_threshold:
            adx_score = 12
            details['adx'] = f"中等趋势+{adx_score}分 (ADX={adx:.2f})"
        else:
            adx_score = 0
            details['adx'] = f"弱趋势 (ADX={adx:.2f})"

        # ADX分数根据DI方向分配
        if plus_di > minus_di:
            bull_score += adx_score
        elif minus_di > plus_di:
            bear_score += adx_score

        # === 5. DI 方向评分 (0-15分) ===
        if di_diff > self.di_diff_threshold:
            di_score = 15 if di_diff > 15 else 10
            bull_score += di_score
            details['di'] = f"多头+{di_score}分 (+DI>{minus_di}, 差值{di_diff:.2f})"
        elif di_diff < -self.di_diff_threshold:
            di_score = 15 if di_diff < -15 else 10
            bear_score += di_score
            details['di'] = f"空头+{di_score}分 (-DI>+DI, 差值{di_diff:.2f})"
        else:
            details['di'] = f"中性 (差值{di_diff:.2f})"

        # === 6. KAMA 位置评分 (0-10分) ===
        if current_price > kama and kama_diff_pct > 0.3:
            kama_score = 10 if kama_diff_pct > 1 else 7
            bull_score += kama_score
            details['kama'] = f"多头+{kama_score}分 (价格高于KAMA {kama_diff_pct:.2f}%)"
        elif current_price < kama and kama_diff_pct < -0.3:
            kama_score = 10 if kama_diff_pct < -1 else 7
            bear_score += kama_score
            details['kama'] = f"空头+{kama_score}分 (价格低于KAMA {kama_diff_pct:.2f}%)"
        else:
            details['kama'] = f"中性 (差值{kama_diff_pct:.2f}%)"

        # === 7. MOM 动量评分 (0-10分) ===
        mom_pct = (mom / current_price) * 100
        if mom > 0 and mom_pct > 0.5:
            mom_score = 10 if mom_pct > 2 else 7
            bull_score += mom_score
            details['mom'] = f"多头+{mom_score}分 (动量={mom_pct:.2f}%)"
        elif mom < 0 and mom_pct < -0.5:
            mom_score = 10 if mom_pct < -2 else 7
            bear_score += mom_score
            details['mom'] = f"空头+{mom_score}分 (动量={mom_pct:.2f}%)"
        else:
            details['mom'] = f"中性 (动量={mom_pct:.2f}%)"

        # === 综合评估 ===
        if bull_score > bear_score:
            direction = 1
            strength = bull_score
            signal_type = "做多"
        elif bear_score > bull_score:
            direction = -1
            strength = bear_score
            signal_type = "做空"
        else:
            direction = 0
            strength = max(bull_score, bear_score)
            signal_type = "观望"

        # 判断信号等级
        if strength >= self.signal_threshold_strong:
            signal_level = "强"
        elif strength >= self.signal_threshold_medium:
            signal_level = "中"
        else:
            signal_level = "弱"

        return {
            'direction': direction,
            'strength': strength,
            'bull_score': bull_score,
            'bear_score': bear_score,
            'signal_type': signal_type,
            'signal_level': signal_level,
            'details': details,
            'adx_value': adx
        }

    def generate_signal(self, klines=None):
        """
        生成交易信号

        Args:
            klines: K线数据（可选）

        Returns:
            int: 1=买入, -1=卖出, 0=观望
        """
        try:
            # 获取K线数据
            if klines is None:
                klines = self.trader.get_klines(
                    symbol=self.trader.symbol,
                    interval=self.kline_interval,
                    limit=self.lookback_period
                )

            if not klines or len(klines) < self.lookback_period:
                self.logger.warning(f"K线数据不足: {len(klines) if klines else 0}/{self.lookback_period}")
                return 0

            # 转换为DataFrame
            row_len = len(klines[0]) if klines and len(klines) > 0 else 0
            if row_len >= 12:
                df = pd.DataFrame(klines, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                    'taker_buy_quote', 'ignore'
                ])
            else:
                df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

            # 转换数据类型
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # 计算指标
            indicators = self.calculate_indicators(df)
            if indicators is None:
                return 0

            # 获取当前价格
            current_price = float(df['close'].iloc[-1])

            # 分析信号强度
            signal_analysis = self.analyze_signal_strength(indicators, current_price)

            # 记录详细信息
            self.logger.info("="*60)
            self.logger.info(f"信号分析 | 价格: {current_price:.4f}")
            self.logger.info(f"方向: {signal_analysis['signal_type']} | "
                           f"强度: {signal_analysis['signal_level']}({signal_analysis['strength']:.0f}分)")
            self.logger.info(f"多头评分: {signal_analysis['bull_score']:.0f} | "
                           f"空头评分: {signal_analysis['bear_score']:.0f}")
            self.logger.info("指标详情:")
            for indicator, detail in signal_analysis['details'].items():
                self.logger.info(f"  {indicator.upper()}: {detail}")

            # 判断是否达到开仓阈值
            direction = signal_analysis['direction']
            strength = signal_analysis['strength']
            adx = signal_analysis['adx_value']

            # 条件1: 方向明确
            # 条件2: 信号强度达到中等以上
            # 条件3: ADX确认趋势存在
            if direction != 0 and strength >= self.signal_threshold_medium and adx >= self.adx_threshold:
                self.logger.info(f"✓ 信号有效: {signal_analysis['signal_type']} "
                               f"(强度{strength:.0f}, ADX{adx:.2f})")
                self.logger.info("="*60)
                return direction
            else:
                reasons = []
                if direction == 0:
                    reasons.append("方向不明确")
                if strength < self.signal_threshold_medium:
                    reasons.append(f"信号强度不足({strength:.0f}<{self.signal_threshold_medium})")
                if adx < self.adx_threshold:
                    reasons.append(f"趋势强度不足(ADX {adx:.2f}<{self.adx_threshold})")

                self.logger.info(f"✗ 信号无效: {', '.join(reasons)}")
                self.logger.info("="*60)
                return 0

        except Exception as e:
            self.logger.error(f"生成信号时出错: {e}", exc_info=True)
            return 0

    def monitor_position(self):
        """
        监控持仓并执行交易逻辑
        """
        try:
            # 获取当前持仓
            position = self.trader.get_position()
            current_price = self.trader.get_current_price()

            if position and abs(position['positionAmt']) > 0:
                # === 有持仓：监控止盈止损 ===
                self._manage_open_position(position, current_price)
            else:
                # === 无持仓：寻找开仓机会 ===
                self._look_for_entry(current_price)

        except Exception as e:
            self.logger.error(f"监控持仓时出错: {e}", exc_info=True)

    def _manage_open_position(self, position, current_price):
        """
        管理已开持仓

        Args:
            position: 持仓信息
            current_price: 当前价格
        """
        position_amt = float(position['positionAmt'])
        entry_price = float(position['entryPrice'])
        unrealized_pnl = float(position['unrealizedProfit'])

        # 计算持仓收益率
        if position_amt > 0:  # 多头
            pnl_pct = (current_price - entry_price) / entry_price
            position_side = "多头"
        else:  # 空头
            pnl_pct = (entry_price - current_price) / entry_price
            position_side = "空头"

        self.logger.info(f"持仓监控 | {position_side} | "
                        f"价格: {current_price:.4f} | "
                        f"开仓: {entry_price:.4f} | "
                        f"收益: {pnl_pct*100:.2f}% (${unrealized_pnl:.2f})")

        # 更新最高/最低价（追踪止损用）
        if self.entry_price is None:
            self.entry_price = entry_price
            self.entry_time = datetime.now()

        if position_amt > 0:  # 多头
            if self.highest_price is None or current_price > self.highest_price:
                self.highest_price = current_price
                self.logger.info(f"更新最高价: {self.highest_price:.4f}")
        else:  # 空头
            if self.lowest_price is None or current_price < self.lowest_price:
                self.lowest_price = current_price
                self.logger.info(f"更新最低价: {self.lowest_price:.4f}")

        # === 检查止盈止损条件 ===
        should_close = False
        close_reason = ""

        # 1. 止盈
        if pnl_pct >= self.take_profit_pct:
            should_close = True
            close_reason = f"止盈 ({pnl_pct*100:.2f}% >= {self.take_profit_pct*100}%)"

        # 2. 止损
        elif pnl_pct <= -self.stop_loss_pct:
            should_close = True
            close_reason = f"止损 ({pnl_pct*100:.2f}% <= -{self.stop_loss_pct*100}%)"

        # 3. 追踪止损
        elif self.trailing_stop_enabled:
            if position_amt > 0 and self.highest_price:
                # 多头：从最高点回撤
                drawdown = (self.highest_price - current_price) / self.highest_price
                if drawdown >= self.trailing_stop_pct:
                    should_close = True
                    close_reason = f"追踪止损 (回撤{drawdown*100:.2f}%)"

            elif position_amt < 0 and self.lowest_price:
                # 空头：从最低点反弹
                drawup = (current_price - self.lowest_price) / self.lowest_price
                if drawup >= self.trailing_stop_pct:
                    should_close = True
                    close_reason = f"追踪止损 (反弹{drawup*100:.2f}%)"

        # 4. 最大持仓时间
        if self.entry_time:
            holding_periods = (datetime.now() - self.entry_time).total_seconds() / (int(self.kline_interval[:-1]) * 60)
            if holding_periods >= self.max_holding_periods:
                should_close = True
                close_reason = f"超时平仓 (持仓{holding_periods:.0f}个周期)"

        # 5. 信号反转
        signal = self.generate_signal()
        if (position_amt > 0 and signal == -1) or (position_amt < 0 and signal == 1):
            should_close = True
            close_reason = "信号反转"

        # 执行平仓
        if should_close:
            self.logger.info(f"触发平仓条件: {close_reason}")
            success = self.trader.close_position()

            if success:
                self.logger.info(f"✓ 平仓成功 | 收益: {pnl_pct*100:.2f}% | 原因: {close_reason}")
                # 重置追踪变量
                self.entry_price = None
                self.entry_time = None
                self.highest_price = None
                self.lowest_price = None
            else:
                self.logger.error("✗ 平仓失败")

    def _look_for_entry(self, current_price):
        """
        寻找开仓机会

        Args:
            current_price: 当前价格
        """
        signal = self.generate_signal()

        if signal == 1:
            # 做多信号
            self.logger.info(f"检测到做多信号 @ {current_price:.4f}")
            success = self.trader.open_long()
            if success:
                self.logger.info("✓ 做多开仓成功")
                self.entry_price = current_price
                self.entry_time = datetime.now()
                self.highest_price = current_price
                self.lowest_price = None
            else:
                self.logger.error("✗ 做多开仓失败")

        elif signal == -1:
            # 做空信号
            self.logger.info(f"检测到做空信号 @ {current_price:.4f}")
            success = self.trader.open_short()
            if success:
                self.logger.info("✓ 做空开仓成功")
                self.entry_price = current_price
                self.entry_time = datetime.now()
                self.highest_price = None
                self.lowest_price = current_price
            else:
                self.logger.error("✗ 做空开仓失败")

        else:
            self.logger.debug(f"暂无交易信号 @ {current_price:.4f}")
