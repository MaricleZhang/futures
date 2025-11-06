"""
Custom Advanced ADX-DI Trend Strategy with Multi-Timeframe Analysis
使用高阶趋势指标 ADX 和 DI 进行价格趋势预测

主要特性：
1. 多时间框架趋势确认（15m 主要，1h 和 4h 辅助）
2. 动态 ADX 阈值调整
3. 趋势强度评分系统
4. 智能入场和出场条件
5. 高级风险管理

File: strategies/custom_adx_trend_strategy.py
"""
import numpy as np
import pandas as pd
import talib
from datetime import datetime
import time
import logging
from strategies.base_strategy import BaseStrategy


class CustomADXTrendStrategy(BaseStrategy):
    """
    自定义高级 ADX-DI 趋势策略

    策略逻辑：
    1. ADX (平均趋向指数) - 衡量趋势强度 (0-100)
       - ADX > 25: 强趋势
       - ADX 20-25: 中等趋势
       - ADX < 20: 弱趋势或震荡

    2. +DI (正向方向指标) - 衡量上涨力量
    3. -DI (负向方向指标) - 衡量下跌力量

    4. 多时间框架确认：
       - 15m: 主要交易时间框架
       - 1h: 趋势确认
       - 4h: 大趋势方向

    交易信号：
    - 买入: +DI > -DI + ADX 上升 + 多时间框架确认
    - 卖出: -DI > +DI + ADX 上升 + 多时间框架确认
    - 持有: ADX 下降或趋势不明确
    """

    def __init__(self, trader):
        """初始化自定义 ADX-DI 趋势策略"""
        super().__init__(trader)
        self.logger = self.get_logger()

        # ==================== 时间框架配置 ====================
        self.primary_interval = '15m'  # 主要交易时间框架
        self.secondary_interval = '1h'  # 次要确认时间框架
        self.tertiary_interval = '4h'  # 大趋势时间框架

        self.kline_interval = self.primary_interval  # 兼容性
        self.check_interval = 300  # 检查间隔：5分钟
        self.lookback_period = 100  # 回看周期：100根K线
        self.training_lookback = 100  # 兼容 TradingManager

        # ==================== ADX-DI 指标参数 ====================
        self.adx_period = 14  # ADX 计算周期
        self.di_period = 14   # DI 计算周期

        # ==================== 动态阈值配置 ====================
        # ADX 趋势强度阈值
        self.adx_very_strong = 40   # 非常强趋势
        self.adx_strong = 25         # 强趋势
        self.adx_moderate = 20       # 中等趋势
        self.adx_weak = 15          # 弱趋势

        # DI 差值阈值
        self.di_diff_strong = 10    # 强信号
        self.di_diff_moderate = 6   # 中等信号
        self.di_diff_weak = 3       # 弱信号

        # ==================== 趋势评分系统 ====================
        self.trend_score_threshold = 7  # 最小趋势评分（0-10）
        self.enable_mtf_analysis = True  # 启用多时间框架分析

        # ==================== 风险管理 ====================
        # 止损止盈
        self.stop_loss_pct = 0.025      # 2.5% 止损
        self.take_profit_pct = 0.08     # 8% 止盈

        # 动态止损（基于 ATR）
        self.use_atr_stop = True
        self.atr_period = 14
        self.atr_multiplier_stop = 2.0   # 止损 = 2 * ATR
        self.atr_multiplier_profit = 3.5  # 止盈 = 3.5 * ATR

        # 追踪止损
        self.trailing_stop_enabled = True
        self.trailing_stop_activation = 0.04   # 4% 利润激活
        self.trailing_stop_distance = 0.015    # 1.5% 追踪距离

        # 时间管理
        self.max_position_hold_time = 1440  # 24小时最大持仓时间（分钟）
        self.min_position_hold_time = 15    # 15分钟最小持仓时间

        # ==================== 仓位跟踪 ====================
        self.position_entry_time = None
        self.position_entry_price = None
        self.max_profit_reached = 0
        self.last_signal = 0
        self.last_signal_time = None
        self.last_trend_score = 0

        # ==================== 性能统计 ====================
        self.total_signals = 0
        self.valid_signals = 0
        self.false_signals = 0

        self.logger.info("="*60)
        self.logger.info("Custom Advanced ADX-DI Trend Strategy Initialized")
        self.logger.info(f"Primary Timeframe: {self.primary_interval}")
        self.logger.info(f"Multi-Timeframe Analysis: {self.enable_mtf_analysis}")
        self.logger.info(f"ADX Period: {self.adx_period}, DI Period: {self.di_period}")
        self.logger.info(f"Trend Score Threshold: {self.trend_score_threshold}/10")
        self.logger.info("="*60)

    def calculate_adx_di_indicators(self, df):
        """
        计算 ADX 和 DI 指标

        Args:
            df (DataFrame): OHLCV 数据

        Returns:
            dict: 包含 ADX, +DI, -DI 的字典
        """
        try:
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values

            # 计算核心指标
            adx = talib.ADX(high, low, close, timeperiod=self.adx_period)
            plus_di = talib.PLUS_DI(high, low, close, timeperiod=self.di_period)
            minus_di = talib.MINUS_DI(high, low, close, timeperiod=self.di_period)

            # 计算 ATR（用于动态止损）
            atr = None
            if self.use_atr_stop:
                atr = talib.ATR(high, low, close, timeperiod=self.atr_period)

            # 计算 DX（方向性运动指数）- 用于辅助分析
            dx = talib.DX(high, low, close, timeperiod=self.di_period)

            return {
                'adx': adx,
                'plus_di': plus_di,
                'minus_di': minus_di,
                'atr': atr,
                'dx': dx
            }

        except Exception as e:
            self.logger.error(f"Error calculating ADX/DI indicators: {str(e)}")
            return None

    def calculate_trend_score(self, trend_analysis, mtf_analysis=None):
        """
        计算趋势评分（0-10分）

        评分标准：
        - ADX 强度：0-3分
        - DI 差异：0-2分
        - ADX 动量：0-2分
        - 多时间框架确认：0-3分

        Args:
            trend_analysis (dict): 单一时间框架趋势分析
            mtf_analysis (dict): 多时间框架分析结果

        Returns:
            float: 趋势评分 (0-10)
        """
        score = 0.0

        # 1. ADX 强度评分 (0-3分)
        adx_value = trend_analysis['adx_value']
        if adx_value >= self.adx_very_strong:
            score += 3.0
        elif adx_value >= self.adx_strong:
            score += 2.5
        elif adx_value >= self.adx_moderate:
            score += 2.0
        elif adx_value >= self.adx_weak:
            score += 1.0

        # 2. DI 差异评分 (0-2分)
        di_diff = abs(trend_analysis['di_difference'])
        if di_diff >= self.di_diff_strong:
            score += 2.0
        elif di_diff >= self.di_diff_moderate:
            score += 1.5
        elif di_diff >= self.di_diff_weak:
            score += 1.0

        # 3. ADX 动量评分 (0-2分)
        if trend_analysis['adx_momentum'] == "increasing":
            score += 2.0
        elif trend_analysis['adx_momentum'] == "stable":
            score += 1.0

        # 4. 多时间框架确认 (0-3分)
        if mtf_analysis and self.enable_mtf_analysis:
            if mtf_analysis.get('all_aligned', False):
                score += 3.0  # 所有时间框架一致
            elif mtf_analysis.get('majority_aligned', False):
                score += 2.0  # 大部分时间框架一致
            elif mtf_analysis.get('primary_secondary_aligned', False):
                score += 1.0  # 主要和次要时间框架一致

        self.logger.debug(f"Trend Score: {score:.1f}/10 - "
                         f"ADX: {adx_value:.1f}, DI Diff: {di_diff:.1f}, "
                         f"Momentum: {trend_analysis['adx_momentum']}")

        return score

    def analyze_trend(self, indicators):
        """
        分析趋势方向和强度

        Args:
            indicators (dict): ADX 和 DI 指标值

        Returns:
            dict: 趋势分析结果
        """
        try:
            # 获取当前值
            current_adx = indicators['adx'][-1]
            current_plus_di = indicators['plus_di'][-1]
            current_minus_di = indicators['minus_di'][-1]
            current_dx = indicators['dx'][-1] if indicators['dx'] is not None else 0

            # 获取历史值（用于趋势动量分析）
            prev_adx = indicators['adx'][-2]
            prev_plus_di = indicators['plus_di'][-2]
            prev_minus_di = indicators['minus_di'][-2]

            # 计算 ADX 变化率
            adx_change_rate = (current_adx - prev_adx) / prev_adx if prev_adx > 0 else 0

            # 计算 DI 差异
            di_diff = current_plus_di - current_minus_di

            # 确定趋势方向
            if current_plus_di > current_minus_di:
                trend_direction = 1  # 上升趋势
                dominant_di = current_plus_di
                weak_di = current_minus_di
            elif current_minus_di > current_plus_di:
                trend_direction = -1  # 下降趋势
                dominant_di = current_minus_di
                weak_di = current_plus_di
            else:
                trend_direction = 0  # 中性
                dominant_di = max(current_plus_di, current_minus_di)
                weak_di = min(current_plus_di, current_minus_di)

            # 确定趋势强度等级
            if current_adx >= self.adx_very_strong:
                trend_strength = "very_strong"
                strength_level = 4
            elif current_adx >= self.adx_strong:
                trend_strength = "strong"
                strength_level = 3
            elif current_adx >= self.adx_moderate:
                trend_strength = "moderate"
                strength_level = 2
            elif current_adx >= self.adx_weak:
                trend_strength = "weak"
                strength_level = 1
            else:
                trend_strength = "very_weak"
                strength_level = 0

            # 检测 DI 交叉（潜在反转信号）
            bullish_crossover = (prev_plus_di <= prev_minus_di and
                                current_plus_di > current_minus_di)
            bearish_crossover = (prev_plus_di >= prev_minus_di and
                                current_plus_di < current_minus_di)

            # ADX 动量分析
            if abs(adx_change_rate) < 0.02:  # 变化小于2%
                adx_momentum = "stable"
            elif current_adx > prev_adx:
                adx_momentum = "increasing"
            else:
                adx_momentum = "decreasing"

            # 计算 DI 扩张/收缩
            prev_di_diff = prev_plus_di - prev_minus_di
            if trend_direction == 1:
                di_expanding = di_diff > prev_di_diff
            elif trend_direction == -1:
                di_expanding = abs(di_diff) > abs(prev_di_diff)
            else:
                di_expanding = False

            # 日志输出
            self.logger.info(f"Trend Analysis [{self.primary_interval}] - "
                           f"ADX: {current_adx:.2f} ({trend_strength}, {adx_momentum}), "
                           f"+DI: {current_plus_di:.2f}, -DI: {current_minus_di:.2f}, "
                           f"Direction: {trend_direction}, DI Diff: {di_diff:.2f}")

            return {
                'direction': trend_direction,
                'strength': trend_strength,
                'strength_level': strength_level,
                'adx_value': current_adx,
                'plus_di_value': current_plus_di,
                'minus_di_value': current_minus_di,
                'di_difference': di_diff,
                'dominant_di': dominant_di,
                'weak_di': weak_di,
                'bullish_crossover': bullish_crossover,
                'bearish_crossover': bearish_crossover,
                'adx_momentum': adx_momentum,
                'adx_change_rate': adx_change_rate,
                'di_expanding': di_expanding,
                'dx_value': current_dx
            }

        except Exception as e:
            self.logger.error(f"Error analyzing trend: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def analyze_multi_timeframe(self, klines_15m=None, klines_1h=None, klines_4h=None):
        """
        多时间框架趋势分析

        Args:
            klines_15m: 15分钟K线数据
            klines_1h: 1小时K线数据
            klines_4h: 4小时K线数据

        Returns:
            dict: 多时间框架分析结果
        """
        try:
            if not self.enable_mtf_analysis:
                return None

            # 获取不同时间框架的K线数据
            if klines_15m is None:
                klines_15m = self.trader.get_klines(
                    symbol=self.trader.symbol,
                    interval='15m',
                    limit=self.lookback_period
                )

            if klines_1h is None:
                klines_1h = self.trader.get_klines(
                    symbol=self.trader.symbol,
                    interval='1h',
                    limit=50
                )

            if klines_4h is None:
                klines_4h = self.trader.get_klines(
                    symbol=self.trader.symbol,
                    interval='4h',
                    limit=30
                )

            # 分析各时间框架
            trends = {}

            # 15分钟趋势
            df_15m = self._prepare_dataframe(klines_15m)
            if df_15m is not None:
                indicators_15m = self.calculate_adx_di_indicators(df_15m)
                if indicators_15m:
                    trends['15m'] = self.analyze_trend(indicators_15m)

            # 1小时趋势
            df_1h = self._prepare_dataframe(klines_1h)
            if df_1h is not None:
                indicators_1h = self.calculate_adx_di_indicators(df_1h)
                if indicators_1h:
                    trends['1h'] = self.analyze_trend(indicators_1h)

            # 4小时趋势
            df_4h = self._prepare_dataframe(klines_4h)
            if df_4h is not None:
                indicators_4h = self.calculate_adx_di_indicators(df_4h)
                if indicators_4h:
                    trends['4h'] = self.analyze_trend(indicators_4h)

            # 分析多时间框架一致性
            directions = [t['direction'] for t in trends.values() if t]
            strengths = [t['strength_level'] for t in trends.values() if t]

            if not directions:
                return None

            # 计算方向一致性
            all_aligned = len(set(directions)) == 1 and directions[0] != 0
            majority_aligned = False
            primary_secondary_aligned = False

            if len(directions) >= 2:
                majority_direction = max(set(directions), key=directions.count)
                majority_aligned = directions.count(majority_direction) >= 2

                if '15m' in trends and '1h' in trends:
                    primary_secondary_aligned = (trends['15m']['direction'] ==
                                                trends['1h']['direction'] and
                                                trends['15m']['direction'] != 0)

            # 计算平均强度
            avg_strength = np.mean(strengths) if strengths else 0

            self.logger.info(f"Multi-Timeframe Analysis - "
                           f"15m: {trends.get('15m', {}).get('direction', 0)}, "
                           f"1h: {trends.get('1h', {}).get('direction', 0)}, "
                           f"4h: {trends.get('4h', {}).get('direction', 0)}, "
                           f"All Aligned: {all_aligned}")

            return {
                'trends': trends,
                'all_aligned': all_aligned,
                'majority_aligned': majority_aligned,
                'primary_secondary_aligned': primary_secondary_aligned,
                'avg_strength': avg_strength,
                'consensus_direction': max(set(directions), key=directions.count) if directions else 0
            }

        except Exception as e:
            self.logger.error(f"Error in multi-timeframe analysis: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def generate_signal(self, klines=None):
        """
        生成交易信号

        Args:
            klines (list): K线数据（可选）

        Returns:
            int: 交易信号 (1=买入, -1=卖出, 0=持有)
        """
        try:
            self.total_signals += 1

            # 获取K线数据
            if klines is None:
                klines = self.trader.get_klines(
                    symbol=self.trader.symbol,
                    interval=self.primary_interval,
                    limit=self.lookback_period
                )

            # 检查数据充分性
            if not klines or len(klines) < 30:
                self.logger.warning("Insufficient k-line data for analysis")
                return 0

            # 准备数据
            df = self._prepare_dataframe(klines)
            if df is None:
                return 0

            # 计算指标
            indicators = self.calculate_adx_di_indicators(df)
            if indicators is None:
                return 0

            # 趋势分析
            trend_analysis = self.analyze_trend(indicators)
            if trend_analysis is None:
                return 0

            # 多时间框架分析
            mtf_analysis = None
            if self.enable_mtf_analysis:
                mtf_analysis = self.analyze_multi_timeframe(klines_15m=klines)

            # 计算趋势评分
            trend_score = self.calculate_trend_score(trend_analysis, mtf_analysis)
            self.last_trend_score = trend_score

            # 信号生成逻辑
            signal = 0

            # 检查趋势评分是否达到阈值
            if trend_score < self.trend_score_threshold:
                self.logger.info(f"HOLD - Trend score {trend_score:.1f} below threshold "
                               f"{self.trend_score_threshold}")
                return 0

            # 检查 ADX 是否足够强
            if trend_analysis['adx_value'] < self.adx_weak:
                self.logger.info(f"HOLD - ADX {trend_analysis['adx_value']:.1f} too weak")
                return 0

            # 买入信号条件
            if trend_analysis['direction'] == 1:
                # 基本条件：+DI > -DI
                if abs(trend_analysis['di_difference']) >= self.di_diff_weak:
                    # 额外确认条件
                    confirmations = 0

                    # 1. ADX 上升
                    if trend_analysis['adx_momentum'] == "increasing":
                        confirmations += 1

                    # 2. DI 扩张
                    if trend_analysis['di_expanding']:
                        confirmations += 1

                    # 3. 强趋势
                    if trend_analysis['strength_level'] >= 3:
                        confirmations += 1

                    # 4. 牛市交叉
                    if trend_analysis['bullish_crossover']:
                        confirmations += 2  # 交叉信号权重更高

                    # 5. 多时间框架确认
                    if mtf_analysis and mtf_analysis.get('primary_secondary_aligned', False):
                        confirmations += 1

                    # 需要至少2个确认条件
                    if confirmations >= 2:
                        signal = 1
                        self.valid_signals += 1
                        self.logger.info(f"✓ BUY SIGNAL - Score: {trend_score:.1f}/10, "
                                       f"Confirmations: {confirmations}, "
                                       f"ADX: {trend_analysis['adx_value']:.1f}, "
                                       f"DI Diff: {trend_analysis['di_difference']:.1f}")

            # 卖出信号条件
            elif trend_analysis['direction'] == -1:
                # 基本条件：-DI > +DI
                if abs(trend_analysis['di_difference']) >= self.di_diff_weak:
                    # 额外确认条件
                    confirmations = 0

                    # 1. ADX 上升
                    if trend_analysis['adx_momentum'] == "increasing":
                        confirmations += 1

                    # 2. DI 扩张
                    if trend_analysis['di_expanding']:
                        confirmations += 1

                    # 3. 强趋势
                    if trend_analysis['strength_level'] >= 3:
                        confirmations += 1

                    # 4. 熊市交叉
                    if trend_analysis['bearish_crossover']:
                        confirmations += 2  # 交叉信号权重更高

                    # 5. 多时间框架确认
                    if mtf_analysis and mtf_analysis.get('primary_secondary_aligned', False):
                        confirmations += 1

                    # 需要至少2个确认条件
                    if confirmations >= 2:
                        signal = -1
                        self.valid_signals += 1
                        self.logger.info(f"✓ SELL SIGNAL - Score: {trend_score:.1f}/10, "
                                       f"Confirmations: {confirmations}, "
                                       f"ADX: {trend_analysis['adx_value']:.1f}, "
                                       f"DI Diff: {trend_analysis['di_difference']:.1f}")

            # 更新信号跟踪
            if signal != 0:
                self.last_signal = signal
                self.last_signal_time = time.time()
            else:
                if trend_score < self.trend_score_threshold:
                    self.logger.debug(f"No signal - Trend score too low: {trend_score:.1f}")
                else:
                    self.logger.debug(f"No signal - Insufficient confirmations")

            return signal

        except Exception as e:
            self.logger.error(f"Error generating signal: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return 0

    def _prepare_dataframe(self, klines):
        """
        将K线数据转换为 DataFrame

        Args:
            klines (list): K线数据

        Returns:
            DataFrame: 处理后的数据
        """
        try:
            if not klines or len(klines) < 30:
                self.logger.error("Insufficient k-line data")
                return None

            # 创建 DataFrame
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

            # 转换为数值类型
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # 添加时间列
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')

            return df

        except Exception as e:
            self.logger.error(f"Error preparing DataFrame: {str(e)}")
            return None

    def monitor_position(self):
        """监控当前仓位并执行交易逻辑"""
        try:
            # 获取当前仓位
            position = self.trader.get_position()

            # 无仓位 - 检查入场信号
            if position is None or float(position['info'].get('positionAmt', 0)) == 0:
                # 生成交易信号
                signal = self.generate_signal()

                if signal != 0:
                    # 获取账户余额
                    balance = self.trader.get_balance()
                    available_balance = float(balance['free'])

                    # 获取当前价格
                    current_price = self.trader.get_market_price()

                    # 计算交易数量
                    symbol_config = self.trader.symbol_config
                    trade_percent = symbol_config.get('trade_amount_percent', 100)
                    trade_amount = (available_balance * trade_percent / 100) / current_price

                    # 执行交易
                    if signal == 1:  # 买入信号
                        self.trader.open_long(amount=trade_amount)
                        self.logger.info(f"✓ LONG Position Opened - Amount: {trade_amount:.6f}, "
                                       f"Price: {current_price}, Score: {self.last_trend_score:.1f}")
                    elif signal == -1:  # 卖出信号
                        self.trader.open_short(amount=trade_amount)
                        self.logger.info(f"✓ SHORT Position Opened - Amount: {trade_amount:.6f}, "
                                       f"Price: {current_price}, Score: {self.last_trend_score:.1f}")

                    # 记录入场信息
                    self.position_entry_time = time.time()
                    self.position_entry_price = current_price
                    self.max_profit_reached = 0

            # 有仓位 - 管理仓位
            else:
                self._manage_position(position)

        except Exception as e:
            self.logger.error(f"Error monitoring position: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())

    def _manage_position(self, position):
        """
        管理现有仓位（风险控制）

        Args:
            position: 当前仓位信息
        """
        try:
            # 提取仓位信息
            position_amount = float(position['info'].get('positionAmt', 0))
            entry_price = float(position['info'].get('entryPrice', 0))
            current_price = self.trader.get_market_price()
            position_side = "long" if position_amount > 0 else "short"

            # 计算盈亏百分比
            if position_side == "long":
                profit_rate = (current_price - entry_price) / entry_price
            else:
                profit_rate = (entry_price - current_price) / entry_price

            # 更新最大利润
            if profit_rate > self.max_profit_reached:
                self.max_profit_reached = profit_rate
                self.logger.debug(f"New max profit: {self.max_profit_reached:.3%}")

            # 计算持仓时间
            holding_time = 0
            if self.position_entry_time:
                holding_time = (time.time() - self.position_entry_time) / 60  # 分钟

            # === 1. 检查最小持仓时间 ===
            if holding_time < self.min_position_hold_time:
                self.logger.debug(f"Position held for {holding_time:.1f}min, "
                                f"min required: {self.min_position_hold_time}min")
                return

            # === 2. 检查最大持仓时间 ===
            if holding_time >= self.max_position_hold_time:
                self.logger.info(f"⏰ MAX HOLDING TIME reached ({holding_time:.1f}min), "
                               f"closing position at {profit_rate:.3%}")
                self.trader.close_position()
                return

            # === 3. 检查止损 ===
            if profit_rate <= -self.stop_loss_pct:
                self.logger.info(f"🛑 STOP LOSS triggered at {profit_rate:.3%}")
                self.trader.close_position()
                self.false_signals += 1
                return

            # === 4. 检查止盈 ===
            if profit_rate >= self.take_profit_pct:
                self.logger.info(f"🎯 TAKE PROFIT triggered at {profit_rate:.3%}")
                self.trader.close_position()
                return

            # === 5. 检查追踪止损 ===
            if self.trailing_stop_enabled and profit_rate >= self.trailing_stop_activation:
                drawdown = self.max_profit_reached - profit_rate
                if drawdown >= self.trailing_stop_distance:
                    self.logger.info(f"📉 TRAILING STOP triggered - "
                                   f"Max: {self.max_profit_reached:.3%}, "
                                   f"Current: {profit_rate:.3%}, "
                                   f"Drawdown: {drawdown:.3%}")
                    self.trader.close_position()
                    return

            # === 6. 检查趋势反转 ===
            signal = self.generate_signal()
            if (position_side == "long" and signal == -1) or \
               (position_side == "short" and signal == 1):
                self.logger.info(f"🔄 TREND REVERSAL detected, closing {position_side} position")
                self.trader.close_position()
                return

            # 定期状态日志
            if int(holding_time) % 5 == 0:  # 每5分钟输出一次
                self.logger.info(f"Position Status - Side: {position_side}, "
                               f"P/L: {profit_rate:.3%}, "
                               f"Holding: {holding_time:.1f}min, "
                               f"Price: {current_price:.4f}, Entry: {entry_price:.4f}")

        except Exception as e:
            self.logger.error(f"Error managing position: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())

    def get_logger(self):
        """获取策略专用日志记录器"""
        logger = logging.getLogger(f"CustomADXTrendStrategy_{self.trader.symbol}")
        return logger
