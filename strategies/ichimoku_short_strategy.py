#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ichimoku云图短线交易策略
一目均衡表（Ichimoku Kinko Hyo）是一种综合技术分析指标
包含转换线、基准线、先行带A、先行带B和滞后线
"""

import pandas as pd
import numpy as np
from datetime import datetime
import time
from strategies.base_strategy import BaseStrategy


class IchimokuShortStrategy(BaseStrategy):
    """
    基于Ichimoku云图的短线交易策略

    策略逻辑：
    1. 看涨信号：
       - 价格突破云图上方
       - 转换线上穿基准线（金叉）
       - 先行带A在先行带B上方（绿云）
       - 滞后线在价格上方

    2. 看跌信号：
       - 价格跌破云图下方
       - 转换线下穿基准线（死叉）
       - 先行带A在先行带B下方（红云）
       - 滞后线在价格下方

    3. 风险管理：
       - 止损：基于ATR动态止损
       - 止盈：基于风险收益比（1:2或1:3）
       - 追踪止损：盈利达到一定比例后激活
    """

    def __init__(self, trader):
        """
        初始化Ichimoku策略

        Args:
            trader: BinanceFuturesTrader实例
        """
        super().__init__(trader)

        # 时间周期配置
        self.kline_interval = '15m'      # 使用15分钟K线（短线交易）
        self.check_interval = 300        # 每5分钟检查一次
        self.lookback_period = 100       # 需要至少100根K线来计算指标
        self.training_lookback = self.lookback_period  # 与TradingManager兼容

        # Ichimoku参数（标准参数）
        self.tenkan_period = 9           # 转换线周期
        self.kijun_period = 26           # 基准线周期
        self.senkou_span_b_period = 52   # 先行带B周期
        self.displacement = 26           # 位移周期

        # 信号确认参数
        self.min_confidence_score = 3    # 最小置信度评分（满分5分）
        self.use_cloud_filter = True     # 是否使用云图过滤
        self.use_chikou_filter = True    # 是否使用滞后线过滤

        # 风险管理参数
        self.atr_period = 14             # ATR周期
        self.atr_multiplier = 2.0        # ATR倍数（用于止损）
        self.risk_reward_ratio = 2.5     # 风险收益比
        self.max_position_size = 0.5     # 最大仓位比例（50%）

        # 追踪止损参数
        self.trailing_stop_enabled = True
        self.trailing_stop_activation = 0.015  # 1.5%激活追踪止损
        self.trailing_stop_distance = 0.008    # 0.8%追踪距离

        # 状态追踪
        self.last_signal = 0             # 上次信号
        self.last_signal_time = None     # 上次信号时间
        self.entry_price = None          # 入场价格
        self.stop_loss_price = None      # 止损价格
        self.take_profit_price = None    # 止盈价格
        self.highest_price = None        # 持仓期间最高价（做多）
        self.lowest_price = None         # 持仓期间最低价（做空）

        self.logger.info(f"Ichimoku短线策略初始化完成")
        self.logger.info(f"参数: Tenkan={self.tenkan_period}, Kijun={self.kijun_period}, "
                        f"Senkou B={self.senkou_span_b_period}, Displacement={self.displacement}")

    def calculate_ichimoku_indicators(self, df):
        """
        计算Ichimoku云图指标

        Args:
            df: DataFrame，包含OHLC数据

        Returns:
            dict: 包含所有Ichimoku指标的字典
        """
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values

        # 1. 转换线 (Tenkan-sen): (9日最高 + 9日最低) / 2
        tenkan_high = pd.Series(high).rolling(window=self.tenkan_period).max()
        tenkan_low = pd.Series(low).rolling(window=self.tenkan_period).min()
        tenkan_sen = (tenkan_high + tenkan_low) / 2

        # 2. 基准线 (Kijun-sen): (26日最高 + 26日最低) / 2
        kijun_high = pd.Series(high).rolling(window=self.kijun_period).max()
        kijun_low = pd.Series(low).rolling(window=self.kijun_period).min()
        kijun_sen = (kijun_high + kijun_low) / 2

        # 3. 先行带A (Senkou Span A): (转换线 + 基准线) / 2，向前移动26期
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(self.displacement)

        # 4. 先行带B (Senkou Span B): (52日最高 + 52日最低) / 2，向前移动26期
        senkou_high = pd.Series(high).rolling(window=self.senkou_span_b_period).max()
        senkou_low = pd.Series(low).rolling(window=self.senkou_span_b_period).min()
        senkou_span_b = ((senkou_high + senkou_low) / 2).shift(self.displacement)

        # 5. 滞后线 (Chikou Span): 当前收盘价，向后移动26期
        chikou_span = pd.Series(close).shift(-self.displacement)

        # 计算ATR用于动态止损
        try:
            import talib
            atr = talib.ATR(high, low, close, timeperiod=self.atr_period)
        except:
            # 如果talib不可用，使用简单的ATR计算
            tr1 = high - low
            tr2 = abs(high - np.roll(close, 1))
            tr3 = abs(low - np.roll(close, 1))
            tr = np.maximum(tr1, np.maximum(tr2, tr3))
            atr = pd.Series(tr).rolling(window=self.atr_period).mean().values

        return {
            'tenkan_sen': tenkan_sen.values,
            'kijun_sen': kijun_sen.values,
            'senkou_span_a': senkou_span_a.values,
            'senkou_span_b': senkou_span_b.values,
            'chikou_span': chikou_span.values,
            'atr': atr,
            'close': close
        }

    def analyze_ichimoku_signal(self, indicators):
        """
        分析Ichimoku指标并生成交易信号

        Args:
            indicators: 包含所有指标的字典

        Returns:
            tuple: (信号, 置信度评分, 分析详情)
                  信号: 1=看涨, -1=看跌, 0=观望
                  置信度: 0-5的评分
        """
        # 获取最新的指标值（倒数第2个，因为最后一个可能未完成）
        idx = -2

        close = indicators['close'][idx]
        tenkan = indicators['tenkan_sen'][idx]
        kijun = indicators['kijun_sen'][idx]
        senkou_a = indicators['senkou_span_a'][idx]
        senkou_b = indicators['senkou_span_b'][idx]
        chikou = indicators['chikou_span'][idx - self.displacement] if idx - self.displacement >= 0 else None

        # 获取前一根K线的数据用于判断交叉
        tenkan_prev = indicators['tenkan_sen'][idx - 1]
        kijun_prev = indicators['kijun_sen'][idx - 1]
        close_prev = indicators['close'][idx - 1]

        # 检查数据有效性
        if np.isnan([tenkan, kijun, senkou_a, senkou_b]).any():
            return 0, 0, "数据不足，等待更多K线"

        # 云图上下边界
        cloud_top = max(senkou_a, senkou_b)
        cloud_bottom = min(senkou_a, senkou_b)

        # 初始化信号和置信度
        signal = 0
        confidence = 0
        details = []

        # ==================== 看涨信号分析 ====================
        bullish_score = 0

        # 1. 转换线和基准线的位置关系（权重：2分）
        if tenkan > kijun:
            bullish_score += 1
            details.append("✓ 转换线在基准线上方")

            # 检查是否刚发生金叉
            if tenkan_prev <= kijun_prev:
                bullish_score += 1
                details.append("✓✓ 转换线金叉基准线（强烈看涨）")

        # 2. 价格与云图的位置关系（权重：2分）
        if close > cloud_top:
            bullish_score += 1
            details.append("✓ 价格在云图上方")

            # 检查是否刚突破云图
            if close_prev <= cloud_top:
                bullish_score += 1
                details.append("✓✓ 价格突破云图上方（强烈看涨）")

        # 3. 云图颜色（先行带A vs 先行带B）（权重：1分）
        if senkou_a > senkou_b:
            bullish_score += 0.5
            details.append("✓ 云图为绿色（看涨）")

        # 4. 滞后线位置（权重：0.5分）
        if chikou is not None and not np.isnan(chikou):
            chikou_ref_close = indicators['close'][idx - self.displacement]
            if chikou > chikou_ref_close:
                bullish_score += 0.5
                details.append("✓ 滞后线在价格上方")

        # ==================== 看跌信号分析 ====================
        bearish_score = 0

        # 1. 转换线和基准线的位置关系（权重：2分）
        if tenkan < kijun:
            bearish_score += 1
            details.append("✗ 转换线在基准线下方")

            # 检查是否刚发生死叉
            if tenkan_prev >= kijun_prev:
                bearish_score += 1
                details.append("✗✗ 转换线死叉基准线（强烈看跌）")

        # 2. 价格与云图的位置关系（权重：2分）
        if close < cloud_bottom:
            bearish_score += 1
            details.append("✗ 价格在云图下方")

            # 检查是否刚跌破云图
            if close_prev >= cloud_bottom:
                bearish_score += 1
                details.append("✗✗ 价格跌破云图下方（强烈看跌）")

        # 3. 云图颜色（权重：1分）
        if senkou_a < senkou_b:
            bearish_score += 0.5
            details.append("✗ 云图为红色（看跌）")

        # 4. 滞后线位置（权重：0.5分）
        if chikou is not None and not np.isnan(chikou):
            chikou_ref_close = indicators['close'][idx - self.displacement]
            if chikou < chikou_ref_close:
                bearish_score += 0.5
                details.append("✗ 滞后线在价格下方")

        self.logger.info(f"上涨score: {bullish_score}, 下跌score: {bearish_score}")
        # ==================== 决定最终信号 ====================
        if bullish_score > bearish_score and bullish_score >= self.min_confidence_score:
            signal = 1
            confidence = min(bullish_score, 5)
        elif bearish_score > bullish_score and bearish_score >= self.min_confidence_score:
            signal = -1
            confidence = min(bearish_score, 5)
        else:
            signal = 0
            confidence = max(bullish_score, bearish_score)
            details.append(f"信号不明确或置信度不足 (看涨:{bullish_score:.1f}, 看跌:{bearish_score:.1f})")

        details_str = " | ".join(details)

        return signal, confidence, details_str

    def generate_signal(self, klines=None):
        """
        生成交易信号

        Args:
            klines: K线数据（可选）

        Returns:
            int: 1=买入, -1=卖出, 2=平仓, 0=观望
        """
        try:
            # 获取K线数据
            if klines is None:
                symbol = self.trader.symbol
                klines = self.trader.get_klines(
                    symbol=symbol,
                    interval=self.kline_interval,
                    limit=self.lookback_period
                )

            if not klines or len(klines) < self.senkou_span_b_period + self.displacement:
                self.logger.warning(f"K线数据不足: {len(klines) if klines else 0} < {self.senkou_span_b_period + self.displacement}")
                return 0

            # 准备DataFrame（统一为6列）
            df = self._klines_to_df(klines)

            # 计算Ichimoku指标
            indicators = self.calculate_ichimoku_indicators(df)

            # 分析信号
            signal, confidence, details = self.analyze_ichimoku_signal(indicators)

            # 记录信号
            current_price = indicators['close'][-2]

            if signal != 0:
                signal_type = "看涨" if signal == 1 else "看跌"
                self.logger.info(f"检测到{signal_type}信号 | 置信度: {confidence:.1f}/5 | 价格: {current_price:.4f}")
                self.logger.info(f"信号详情: {details}")

                # 更新信号状态
                self.last_signal = signal
                self.last_signal_time = datetime.now()
            else:
                self.logger.debug(f"观望 | 价格: {current_price:.4f} | {details}")

            return signal

        except Exception as e:
            self.logger.error(f"生成信号时出错: {str(e)}", exc_info=True)
            return 0

    def calculate_stop_loss_take_profit(self, entry_price, signal, atr):
        """
        计算止损和止盈价格

        Args:
            entry_price: 入场价格
            signal: 交易信号（1=做多, -1=做空）
            atr: 平均真实波幅

        Returns:
            tuple: (止损价格, 止盈价格)
        """
        # 基于ATR的动态止损
        stop_distance = atr * self.atr_multiplier

        if signal == 1:  # 做多
            stop_loss = entry_price - stop_distance
            take_profit = entry_price + stop_distance * self.risk_reward_ratio
        else:  # 做空
            stop_loss = entry_price + stop_distance
            take_profit = entry_price - stop_distance * self.risk_reward_ratio

        return stop_loss, take_profit

    def monitor_position(self):
        """
        监控持仓并执行交易逻辑
        """
        try:
            symbol = self.trader.symbol

            # 获取当前持仓
            position = self.trader.get_position(symbol)

            if position and float(position['positionAmt']) != 0:
                # 有持仓，进行风险管理
                self._manage_position(position)
            else:
                # 无持仓，检查是否有新的交易信号
                signal = self.generate_signal()

                if signal == 1:  # 看涨信号，开多单
                    self._open_long_position()
                elif signal == -1:  # 看跌信号，开空单
                    self._open_short_position()

        except Exception as e:
            self.logger.error(f"监控持仓时出错: {str(e)}", exc_info=True)

    def _open_long_position(self):
        """开多单"""
        try:
            symbol = self.trader.symbol

            # 获取当前价格和ATR
            klines = self.trader.get_klines(symbol, self.kline_interval, self.lookback_period)
            df = self._klines_to_df(klines)

            indicators = self.calculate_ichimoku_indicators(df)
            current_price = indicators['close'][-1]
            current_atr = indicators['atr'][-1]

            # 计算止损止盈
            stop_loss, take_profit = self.calculate_stop_loss_take_profit(
                current_price, 1, current_atr
            )

            # 计算仓位大小
            balance = self.trader.get_balance()
            position_size = balance * self.max_position_size

            # 执行开仓
            self.logger.info(f"开多单 | 价格: {current_price:.4f} | 止损: {stop_loss:.4f} | 止盈: {take_profit:.4f}")

            # 这里调用trader的开仓方法（需要根据实际的trader实现调整）
            # result = self.trader.open_long(symbol, position_size)

            # 记录入场信息
            self.entry_price = current_price
            self.stop_loss_price = stop_loss
            self.take_profit_price = take_profit
            self.highest_price = current_price

        except Exception as e:
            self.logger.error(f"开多单失败: {str(e)}", exc_info=True)

    def _open_short_position(self):
        """开空单"""
        try:
            symbol = self.trader.symbol

            # 获取当前价格和ATR
            klines = self.trader.get_klines(symbol, self.kline_interval, self.lookback_period)
            df = self._klines_to_df(klines)

            indicators = self.calculate_ichimoku_indicators(df)
            current_price = indicators['close'][-1]
            current_atr = indicators['atr'][-1]

            # 计算止损止盈
            stop_loss, take_profit = self.calculate_stop_loss_take_profit(
                current_price, -1, current_atr
            )

            # 计算仓位大小
            balance = self.trader.get_balance()
            position_size = balance * self.max_position_size

            # 执行开仓
            self.logger.info(f"开空单 | 价格: {current_price:.4f} | 止损: {stop_loss:.4f} | 止盈: {take_profit:.4f}")

            # 这里调用trader的开仓方法（需要根据实际的trader实现调整）
            # result = self.trader.open_short(symbol, position_size)

            # 记录入场信息
            self.entry_price = current_price
            self.stop_loss_price = stop_loss
            self.take_profit_price = take_profit
            self.lowest_price = current_price

        except Exception as e:
            self.logger.error(f"开空单失败: {str(e)}", exc_info=True)

    def _manage_position(self, position):
        """
        持仓风险管理

        Args:
            position: 持仓信息
        """
        try:
            symbol = self.trader.symbol
            position_amt = float(position['positionAmt'])
            entry_price = float(position['entryPrice'])

            # 获取当前价格
            klines = self.trader.get_klines(symbol, self.kline_interval, 2)
            current_price = float(klines[-1][4])  # 收盘价

            # 计算盈亏
            if position_amt > 0:  # 多单
                pnl_pct = (current_price - entry_price) / entry_price

                # 更新最高价
                if self.highest_price is None or current_price > self.highest_price:
                    self.highest_price = current_price

                # 检查止损
                if self.stop_loss_price and current_price <= self.stop_loss_price:
                    self.logger.warning(f"触发止损 | 当前价格: {current_price:.4f} <= 止损价: {self.stop_loss_price:.4f}")
                    self.trader.close_position(symbol)
                    self._reset_state()
                    return

                # 检查止盈
                if self.take_profit_price and current_price >= self.take_profit_price:
                    self.logger.info(f"触发止盈 | 当前价格: {current_price:.4f} >= 止盈价: {self.take_profit_price:.4f}")
                    self.trader.close_position(symbol)
                    self._reset_state()
                    return

                # 追踪止损
                if self.trailing_stop_enabled and pnl_pct >= self.trailing_stop_activation:
                    trailing_stop = self.highest_price * (1 - self.trailing_stop_distance)
                    if current_price <= trailing_stop:
                        self.logger.info(f"触发追踪止损 | 当前价格: {current_price:.4f} <= 追踪止损: {trailing_stop:.4f}")
                        self.trader.close_position(symbol)
                        self._reset_state()
                        return

            else:  # 空单
                pnl_pct = (entry_price - current_price) / entry_price

                # 更新最低价
                if self.lowest_price is None or current_price < self.lowest_price:
                    self.lowest_price = current_price

                # 检查止损
                if self.stop_loss_price and current_price >= self.stop_loss_price:
                    self.logger.warning(f"触发止损 | 当前价格: {current_price:.4f} >= 止损价: {self.stop_loss_price:.4f}")
                    self.trader.close_position(symbol)
                    self._reset_state()
                    return

                # 检查止盈
                if self.take_profit_price and current_price <= self.take_profit_price:
                    self.logger.info(f"触发止盈 | 当前价格: {current_price:.4f} <= 止盈价: {self.take_profit_price:.4f}")
                    self.trader.close_position(symbol)
                    self._reset_state()
                    return

                # 追踪止损
                if self.trailing_stop_enabled and pnl_pct >= self.trailing_stop_activation:
                    trailing_stop = self.lowest_price * (1 + self.trailing_stop_distance)
                    if current_price >= trailing_stop:
                        self.logger.info(f"触发追踪止损 | 当前价格: {current_price:.4f} >= 追踪止损: {trailing_stop:.4f}")
                        self.trader.close_position(symbol)
                        self._reset_state()
                        return

            # 记录持仓状态
            self.logger.debug(f"持仓监控 | 数量: {position_amt} | 入场价: {entry_price:.4f} | "
                            f"当前价: {current_price:.4f} | 盈亏: {pnl_pct*100:.2f}%")

        except Exception as e:
            self.logger.error(f"管理持仓时出错: {str(e)}", exc_info=True)

    def _reset_state(self):
        """重置状态"""
        self.entry_price = None
        self.stop_loss_price = None
        self.take_profit_price = None
        self.highest_price = None
        self.lowest_price = None
        self.last_signal = 0

    # 统一K线到DataFrame（兼容6/12列：截取前6列）
    def _klines_to_df(self, klines):
        if not klines or len(klines[0]) < 6:
            raise ValueError(f"K线数据格式不正确，列数={0 if not klines else len(klines[0])}")
        base_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        df = pd.DataFrame([k[:6] for k in klines], columns=base_cols)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
        return df
