"""
K线形态概率交易策略
基于K线形态识别，结合技术指标，输出做多/做空/观望的概率

File: strategies/pattern_probability_strategy.py
"""
import numpy as np
import pandas as pd
import pandas_ta_classic as ta
from datetime import datetime
import time
import logging
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from strategies.base_strategy import BaseStrategy
from utils.enhanced_candlestick_patterns import EnhancedCandlestickPattern, PatternDirection


@dataclass
class TradingSignal:
    """交易信号数据类"""
    signal: int  # 1=做多, -1=做空, 0=观望
    long_prob: float
    short_prob: float
    hold_prob: float
    confidence: float
    patterns_detected: List[str]
    trend_direction: str
    momentum_state: str
    reasoning: str


class PatternProbabilityStrategy(BaseStrategy):
    """
    K线形态概率交易策略
    
    策略特点：
    1. 识别20+种K线形态
    2. 结合趋势、动量、波动率、成交量四维分析
    3. 使用概率模型计算交易方向
    4. 输出做多/做空/观望的概率及置信度
    
    概率计算公式：
    P(做多) = w1*P(形态看涨) + w2*P(趋势向上) + w3*P(动量向上) + w4*P(成交量确认)
    P(做空) = w1*P(形态看跌) + w2*P(趋势向下) + w3*P(动量向下) + w4*P(成交量确认)
    P(观望) = 1 - P(做多) - P(做空)
    """
    
    def __init__(self, trader):
        """初始化策略"""
        super().__init__(trader)
        self.logger = self.get_logger()
        
        # ==================== 时间配置 ====================
        self.kline_interval = '15m'
        self.check_interval = 300  # 5分钟检查一次
        self.lookback_period = 100
        self.training_lookback = 100
        
        # ==================== K线形态识别器 ====================
        self.pattern_detector = EnhancedCandlestickPattern()
        
        # ==================== 技术指标参数 ====================
        # 趋势指标
        self.ema_fast = 8
        self.ema_mid = 21
        self.ema_slow = 55
        self.adx_period = 14
        
        # 动量指标
        self.rsi_period = 14
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        self.stoch_k = 14
        self.stoch_d = 3
        
        # 波动率指标
        self.atr_period = 14
        self.bb_period = 20
        self.bb_std = 2.0
        
        # 成交量指标
        self.volume_ma_period = 20
        
        # ==================== 概率计算权重 ====================
        self.weights = {
            'pattern': 0.35,    # K线形态权重
            'trend': 0.30,      # 趋势权重
            'momentum': 0.25,   # 动量权重
            'volume': 0.10      # 成交量权重
        }
        
        # ==================== 交易阈值 ====================
        self.min_trade_prob = 0.45      # 最小交易概率
        self.min_confidence = 0.55      # 最小置信度
        self.strong_signal_prob = 0.65  # 强信号概率阈值
        
        # ==================== 仓位管理 ====================
        self.stop_loss_pct = 0.02       # 2% 止损
        self.take_profit_pct = 0.06     # 6% 止盈
        self.max_hold_time = 720        # 最大持仓12小时
        
        # 追踪止损
        self.trailing_stop_enabled = True
        self.trailing_activation = 0.025  # 2.5%激活
        self.trailing_distance = 0.012    # 1.2%距离
        
        # ==================== 状态追踪 ====================
        self.position_entry_time = None
        self.position_entry_price = None
        self.max_profit_reached = 0
        self.trailing_stop_price = None
        self.last_signal: Optional[TradingSignal] = None
        
        self.logger.info("=" * 60)
        self.logger.info("K线形态概率交易策略初始化完成")
        self.logger.info(f"时间周期: {self.kline_interval}")
        self.logger.info(f"权重配置: 形态={self.weights['pattern']:.0%}, "
                        f"趋势={self.weights['trend']:.0%}, "
                        f"动量={self.weights['momentum']:.0%}, "
                        f"成交量={self.weights['volume']:.0%}")
        self.logger.info(f"交易阈值: 最小概率={self.min_trade_prob:.0%}, "
                        f"最小置信度={self.min_confidence:.0%}")
        self.logger.info("=" * 60)
    
    # ==================== 技术指标计算 ====================
    
    def calculate_indicators(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        计算所有技术指标
        
        Returns:
            包含所有指标的字典
        """
        try:
            high = df['high']
            low = df['low']
            close = df['close']
            volume = df['volume']
            
            indicators = {}
            
            # ---------- 趋势指标 ----------
            indicators['ema_fast'] = ta.ema(close, length=self.ema_fast)
            indicators['ema_mid'] = ta.ema(close, length=self.ema_mid)
            indicators['ema_slow'] = ta.ema(close, length=self.ema_slow)
            
            # ADX + DI
            adx_df = ta.adx(high, low, close, length=self.adx_period)
            indicators['adx'] = adx_df[f'ADX_{self.adx_period}']
            indicators['plus_di'] = adx_df[f'DMP_{self.adx_period}']
            indicators['minus_di'] = adx_df[f'DMN_{self.adx_period}']
            
            # ---------- 动量指标 ----------
            indicators['rsi'] = ta.rsi(close, length=self.rsi_period)
            
            macd_df = ta.macd(close, fast=self.macd_fast, slow=self.macd_slow, signal=self.macd_signal)
            indicators['macd'] = macd_df[f'MACD_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}']
            indicators['macd_signal'] = macd_df[f'MACDs_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}']
            indicators['macd_hist'] = macd_df[f'MACDh_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}']
            
            stoch_df = ta.stoch(high, low, close, k=self.stoch_k, d=self.stoch_d)
            indicators['stoch_k'] = stoch_df[f'STOCHk_{self.stoch_k}_{self.stoch_d}_3']
            indicators['stoch_d'] = stoch_df[f'STOCHd_{self.stoch_k}_{self.stoch_d}_3']
            
            # ---------- 波动率指标 ----------
            indicators['atr'] = ta.atr(high, low, close, length=self.atr_period)
            
            bb_df = ta.bbands(close, length=self.bb_period, std=self.bb_std)
            indicators['bb_upper'] = bb_df[f'BBU_{self.bb_period}_{self.bb_std}']
            indicators['bb_middle'] = bb_df[f'BBM_{self.bb_period}_{self.bb_std}']
            indicators['bb_lower'] = bb_df[f'BBL_{self.bb_period}_{self.bb_std}']
            
            # ---------- 成交量指标 ----------
            indicators['volume'] = volume
            indicators['volume_ma'] = ta.sma(volume, length=self.volume_ma_period)
            
            # ---------- 价格数据 ----------
            indicators['close'] = close
            indicators['high'] = high
            indicators['low'] = low
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"计算指标出错: {str(e)}")
            return None
    
    # ==================== 各维度概率计算 ====================
    
    def calc_pattern_probability(self, df: pd.DataFrame) -> Tuple[float, float, List[str], str]:
        """
        计算K线形态概率
        
        Returns:
            (看涨概率, 看跌概率, 检测到的形态列表, 最强形态描述)
        """
        patterns = self.pattern_detector.detect_all_patterns(df)
        summary = self.pattern_detector.get_pattern_summary(patterns)
        
        bullish_score = summary['total_bullish_score']
        bearish_score = summary['total_bearish_score']
        
        # 归一化
        total = bullish_score + bearish_score
        if total > 0:
            bull_prob = bullish_score / (total + 0.5)  # 加0.5防止过度自信
            bear_prob = bearish_score / (total + 0.5)
        else:
            bull_prob = 0
            bear_prob = 0
        
        # 检测到的形态名称
        detected = []
        for name, result in patterns.items():
            if result.detected:
                detected.append(f"{result.name}({result.strength:.2f})")
        
        # 最强形态描述
        strongest_desc = ""
        if summary['strongest_pattern']:
            strongest_desc = summary['strongest_pattern'].description
        
        return bull_prob, bear_prob, detected, strongest_desc
    
    def calc_trend_probability(self, indicators: Dict) -> Tuple[float, float, str]:
        """
        计算趋势概率
        
        Returns:
            (看涨概率, 看跌概率, 趋势描述)
        """
        try:
            close = indicators['close'].iloc[-1]
            ema_fast = indicators['ema_fast'].iloc[-1]
            ema_mid = indicators['ema_mid'].iloc[-1]
            ema_slow = indicators['ema_slow'].iloc[-1]
            adx = indicators['adx'].iloc[-1]
            plus_di = indicators['plus_di'].iloc[-1]
            minus_di = indicators['minus_di'].iloc[-1]
            
            score = 0.0
            reasons = []
            
            # EMA排列分析
            if ema_fast > ema_mid > ema_slow:
                score += 0.35
                reasons.append("EMA多头排列")
            elif ema_fast < ema_mid < ema_slow:
                score -= 0.35
                reasons.append("EMA空头排列")
            
            # 价格与EMA关系
            if close > ema_fast:
                score += 0.15
            elif close < ema_fast:
                score -= 0.15
            
            # ADX趋势强度
            if adx > 25:  # 强趋势
                if plus_di > minus_di:
                    score += 0.3
                    reasons.append(f"ADX强趋势上行({adx:.1f})")
                else:
                    score -= 0.3
                    reasons.append(f"ADX强趋势下行({adx:.1f})")
            elif adx > 20:  # 中等趋势
                if plus_di > minus_di:
                    score += 0.15
                else:
                    score -= 0.15
            
            # DI交叉
            di_diff = plus_di - minus_di
            if di_diff > 10:
                score += 0.2
            elif di_diff < -10:
                score -= 0.2
            
            # 转换为概率
            bull_prob = max(0, min(1, 0.5 + score))
            bear_prob = max(0, min(1, 0.5 - score))
            
            # 趋势描述
            if score > 0.3:
                trend_desc = "强上升趋势"
            elif score > 0.1:
                trend_desc = "温和上升趋势"
            elif score < -0.3:
                trend_desc = "强下降趋势"
            elif score < -0.1:
                trend_desc = "温和下降趋势"
            else:
                trend_desc = "横盘震荡"
            
            return bull_prob, bear_prob, f"{trend_desc} ({', '.join(reasons)})"
            
        except Exception as e:
            self.logger.error(f"计算趋势概率出错: {str(e)}")
            return 0.5, 0.5, "计算出错"
    
    def calc_momentum_probability(self, indicators: Dict) -> Tuple[float, float, str]:
        """
        计算动量概率
        
        Returns:
            (看涨概率, 看跌概率, 动量状态描述)
        """
        try:
            rsi = indicators['rsi'].iloc[-1]
            macd = indicators['macd'].iloc[-1]
            macd_signal = indicators['macd_signal'].iloc[-1]
            macd_hist = indicators['macd_hist'].iloc[-1]
            macd_hist_prev = indicators['macd_hist'].iloc[-2]
            stoch_k = indicators['stoch_k'].iloc[-1]
            stoch_d = indicators['stoch_d'].iloc[-1]
            
            score = 0.0
            states = []
            
            # RSI分析
            if rsi > 70:
                score -= 0.2  # 超买
                states.append("RSI超买")
            elif rsi > 55:
                score += 0.1
            elif rsi < 30:
                score += 0.2  # 超卖反弹预期
                states.append("RSI超卖")
            elif rsi < 45:
                score -= 0.1
            
            # MACD分析
            if macd > macd_signal:
                score += 0.15
                if macd_hist > macd_hist_prev:
                    score += 0.1
                    states.append("MACD金叉加速")
            else:
                score -= 0.15
                if macd_hist < macd_hist_prev:
                    score -= 0.1
                    states.append("MACD死叉加速")
            
            # MACD柱状图趋势
            if macd_hist > 0 and macd_hist > macd_hist_prev:
                score += 0.1
            elif macd_hist < 0 and macd_hist < macd_hist_prev:
                score -= 0.1
            
            # 随机指标
            if stoch_k > stoch_d and stoch_k < 80:
                score += 0.15
            elif stoch_k < stoch_d and stoch_k > 20:
                score -= 0.15
            
            if stoch_k > 80:
                states.append("KDJ超买")
            elif stoch_k < 20:
                states.append("KDJ超卖")
            
            # 转换为概率
            bull_prob = max(0, min(1, 0.5 + score))
            bear_prob = max(0, min(1, 0.5 - score))
            
            # 动量状态
            if score > 0.2:
                momentum_state = "强势上涨动能"
            elif score > 0:
                momentum_state = "温和上涨动能"
            elif score < -0.2:
                momentum_state = "强势下跌动能"
            elif score < 0:
                momentum_state = "温和下跌动能"
            else:
                momentum_state = "动能中性"
            
            if states:
                momentum_state += f" ({', '.join(states)})"
            
            return bull_prob, bear_prob, momentum_state
            
        except Exception as e:
            self.logger.error(f"计算动量概率出错: {str(e)}")
            return 0.5, 0.5, "计算出错"
    
    def calc_volume_probability(self, indicators: Dict) -> Tuple[float, float, str]:
        """
        计算成交量确认概率
        
        Returns:
            (看涨概率, 看跌概率, 成交量状态描述)
        """
        try:
            close = indicators['close'].iloc[-1]
            close_prev = indicators['close'].iloc[-2]
            volume = indicators['volume'].iloc[-1]
            volume_ma = indicators['volume_ma'].iloc[-1]
            
            # 成交量比率
            vol_ratio = volume / volume_ma if volume_ma > 0 else 1
            
            # 价格变动方向
            price_up = close > close_prev
            
            score = 0.0
            
            if vol_ratio > 1.5:  # 明显放量
                if price_up:
                    score += 0.3
                    vol_state = "放量上涨，买盘强劲"
                else:
                    score -= 0.3
                    vol_state = "放量下跌，卖盘强劲"
            elif vol_ratio > 1.2:  # 温和放量
                if price_up:
                    score += 0.15
                    vol_state = "温和放量上涨"
                else:
                    score -= 0.15
                    vol_state = "温和放量下跌"
            elif vol_ratio < 0.7:  # 缩量
                vol_state = "缩量整理，等待方向"
            else:
                vol_state = "成交量正常"
            
            bull_prob = max(0, min(1, 0.5 + score))
            bear_prob = max(0, min(1, 0.5 - score))
            
            return bull_prob, bear_prob, f"{vol_state} (量比:{vol_ratio:.2f})"
            
        except Exception as e:
            self.logger.error(f"计算成交量概率出错: {str(e)}")
            return 0.5, 0.5, "计算出错"
    
    # ==================== 综合概率计算 ====================
    
    def calculate_trading_probability(self, df: pd.DataFrame) -> TradingSignal:
        """
        计算综合交易概率
        
        Returns:
            TradingSignal对象
        """
        try:
            # 计算指标
            indicators = self.calculate_indicators(df)
            if indicators is None:
                return self._default_signal()
            
            # 计算各维度概率
            pattern_bull, pattern_bear, patterns_detected, pattern_desc = self.calc_pattern_probability(df)
            trend_bull, trend_bear, trend_desc = self.calc_trend_probability(indicators)
            momentum_bull, momentum_bear, momentum_desc = self.calc_momentum_probability(indicators)
            volume_bull, volume_bear, volume_desc = self.calc_volume_probability(indicators)
            
            # 加权计算综合概率
            total_bull = (
                pattern_bull * self.weights['pattern'] +
                trend_bull * self.weights['trend'] +
                momentum_bull * self.weights['momentum'] +
                volume_bull * self.weights['volume']
            )
            
            total_bear = (
                pattern_bear * self.weights['pattern'] +
                trend_bear * self.weights['trend'] +
                momentum_bear * self.weights['momentum'] +
                volume_bear * self.weights['volume']
            )
            
            # 计算观望概率
            hold_prob = max(0, 1 - total_bull - total_bear)
            
            # 归一化
            total = total_bull + total_bear + hold_prob
            if total > 0:
                long_prob = total_bull / total
                short_prob = total_bear / total
                hold_prob = hold_prob / total
            else:
                long_prob = 0
                short_prob = 0
                hold_prob = 1
            
            # 计算置信度（基于信号的一致性）
            confidence = self._calculate_confidence(
                pattern_bull, pattern_bear,
                trend_bull, trend_bear,
                momentum_bull, momentum_bear,
                indicators
            )
            
            # 生成交易信号
            signal = 0
            if long_prob >= self.min_trade_prob and confidence >= self.min_confidence:
                if long_prob > short_prob:
                    signal = 1
            elif short_prob >= self.min_trade_prob and confidence >= self.min_confidence:
                if short_prob > long_prob:
                    signal = -1
            
            # 构建推理说明
            reasoning = self._build_reasoning(
                signal, long_prob, short_prob, confidence,
                pattern_desc, trend_desc, momentum_desc, volume_desc
            )
            
            # 提取趋势方向
            if "上升" in trend_desc or "上行" in trend_desc:
                trend_direction = "上升"
            elif "下降" in trend_desc or "下行" in trend_desc:
                trend_direction = "下降"
            else:
                trend_direction = "横盘"
            
            result = TradingSignal(
                signal=signal,
                long_prob=float(long_prob),
                short_prob=float(short_prob),
                hold_prob=float(hold_prob),
                confidence=float(confidence),
                patterns_detected=patterns_detected,
                trend_direction=trend_direction,
                momentum_state=momentum_desc,
                reasoning=reasoning
            )
            
            self.last_signal = result
            return result
            
        except Exception as e:
            self.logger.error(f"计算交易概率出错: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return self._default_signal()
    
    def _calculate_confidence(
        self,
        pattern_bull: float, pattern_bear: float,
        trend_bull: float, trend_bear: float,
        momentum_bull: float, momentum_bear: float,
        indicators: Dict
    ) -> float:
        """计算整体置信度"""
        
        # 方向一致性
        bull_signals = sum([
            1 if pattern_bull > 0.5 else 0,
            1 if trend_bull > 0.5 else 0,
            1 if momentum_bull > 0.5 else 0
        ])
        
        bear_signals = sum([
            1 if pattern_bear > 0.5 else 0,
            1 if trend_bear > 0.5 else 0,
            1 if momentum_bear > 0.5 else 0
        ])
        
        # 一致性得分
        consistency = max(bull_signals, bear_signals) / 3
        
        # ADX趋势强度
        adx = indicators['adx'].iloc[-1]
        adx_factor = min(adx / 40, 1) if adx > 0 else 0.5
        
        # 综合置信度
        confidence = consistency * 0.6 + adx_factor * 0.4
        
        return confidence
    
    def _build_reasoning(
        self,
        signal: int,
        long_prob: float,
        short_prob: float,
        confidence: float,
        pattern_desc: str,
        trend_desc: str,
        momentum_desc: str,
        volume_desc: str
    ) -> str:
        """构建推理说明"""
        
        if signal == 1:
            action = "做多"
            prob = long_prob
        elif signal == -1:
            action = "做空"
            prob = short_prob
        else:
            action = "观望"
            prob = 1 - long_prob - short_prob
        
        reasoning = f"""
【交易建议】{action} (概率: {prob:.1%}, 置信度: {confidence:.1%})

【K线形态】{pattern_desc if pattern_desc else '无明显形态'}

【趋势分析】{trend_desc}

【动量状态】{momentum_desc}

【成交量】{volume_desc}

【概率分布】做多={long_prob:.1%} | 做空={short_prob:.1%} | 观望={1-long_prob-short_prob:.1%}
"""
        return reasoning.strip()
    
    def _default_signal(self) -> TradingSignal:
        """返回默认信号（观望）"""
        return TradingSignal(
            signal=0,
            long_prob=0.0,
            short_prob=0.0,
            hold_prob=1.0,
            confidence=0.0,
            patterns_detected=[],
            trend_direction="未知",
            momentum_state="未知",
            reasoning="数据不足或计算出错，建议观望"
        )
    
    # ==================== 策略接口实现 ====================
    
    def generate_signal(self, klines=None) -> int:
        """
        生成交易信号
        
        Returns:
            1: 做多, -1: 做空, 0: 观望
        """
        try:
            # 获取K线数据
            if klines is None:
                klines = self.trader.get_klines(
                    symbol=self.trader.symbol,
                    interval=self.kline_interval,
                    limit=self.lookback_period
                )
            
            if not klines or len(klines) < 50:
                self.logger.warning(f"K线数据不足: {len(klines) if klines else 0}")
                return 0
            
            # 转换为DataFrame
            df = pd.DataFrame(
                klines,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 计算交易概率
            signal = self.calculate_trading_probability(df)
            
            # 打印分析报告
            self._print_analysis_report(signal)
            
            return signal.signal
            
        except Exception as e:
            self.logger.error(f"生成信号出错: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return 0
    
    def _print_analysis_report(self, signal: TradingSignal):
        """打印分析报告"""
        self.logger.info("=" * 70)
        self.logger.info("【K线形态概率分析报告】")
        self.logger.info("=" * 70)
        
        # 概率分布
        self.logger.info(f"📊 概率分布:")
        self.logger.info(f"   做多: {signal.long_prob:.1%}")
        self.logger.info(f"   做空: {signal.short_prob:.1%}")
        self.logger.info(f"   观望: {signal.hold_prob:.1%}")
        self.logger.info(f"   置信度: {signal.confidence:.1%}")
        
        # 检测到的形态
        if signal.patterns_detected:
            self.logger.info(f"🔍 检测到的K线形态: {', '.join(signal.patterns_detected)}")
        else:
            self.logger.info(f"🔍 检测到的K线形态: 无")
        
        # 趋势和动量
        self.logger.info(f"📈 趋势方向: {signal.trend_direction}")
        self.logger.info(f"⚡ 动量状态: {signal.momentum_state}")
        
        # 交易信号
        signal_text = {1: "🟢 做多", -1: "🔴 做空", 0: "⚪ 观望"}
        self.logger.info(f"🎯 交易信号: {signal_text.get(signal.signal, '未知')}")
        
        self.logger.info("-" * 70)
        self.logger.info(signal.reasoning)
        self.logger.info("=" * 70)
    
    def monitor_position(self):
        """监控仓位"""
        try:
            position = self.trader.get_position()
            
            # 无仓位 - 检查入场
            if position is None or float(position['info'].get('positionAmt', 0)) == 0:
                signal = self.generate_signal()
                
                if signal != 0:
                    balance = self.trader.get_balance()
                    available = float(balance['free'])
                    current_price = self.trader.get_market_price()
                    
                    symbol_config = self.trader.symbol_config
                    trade_pct = symbol_config.get('trade_amount_percent', 100)
                    trade_amount = (available * trade_pct / 100) / current_price
                    
                    if signal == 1:
                        self.trader.open_long(amount=trade_amount)
                        self.logger.info(
                            f"✅ 开多仓 | 数量: {trade_amount:.6f} | 价格: {current_price} | "
                            f"概率: {self.last_signal.long_prob:.1%}"
                        )
                    elif signal == -1:
                        self.trader.open_short(amount=trade_amount)
                        self.logger.info(
                            f"✅ 开空仓 | 数量: {trade_amount:.6f} | 价格: {current_price} | "
                            f"概率: {self.last_signal.short_prob:.1%}"
                        )
                    
                    self.position_entry_time = time.time()
                    self.position_entry_price = current_price
                    self.max_profit_reached = 0
                    self.trailing_stop_price = None
            else:
                self._manage_position(position)
                
        except Exception as e:
            self.logger.error(f"监控仓位出错: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
    
    def _manage_position(self, position):
        """管理现有仓位"""
        try:
            pos_amt = float(position['info'].get('positionAmt', 0))
            entry_price = float(position['info'].get('entryPrice', 0))
            current_price = self.trader.get_market_price()
            side = "long" if pos_amt > 0 else "short"
            
            # 计算盈亏
            if side == "long":
                pnl_pct = (current_price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - current_price) / entry_price
            
            # 更新最大利润
            if pnl_pct > self.max_profit_reached:
                self.max_profit_reached = pnl_pct
                
                # 更新追踪止损
                if self.trailing_stop_enabled and pnl_pct >= self.trailing_activation:
                    if side == "long":
                        self.trailing_stop_price = current_price * (1 - self.trailing_distance)
                    else:
                        self.trailing_stop_price = current_price * (1 + self.trailing_distance)
            
            # 检查追踪止损
            if self.trailing_stop_price:
                if side == "long" and current_price <= self.trailing_stop_price:
                    self.logger.info(f"📉 追踪止损触发 | 盈亏: {pnl_pct:.2%}")
                    self.trader.close_position()
                    return
                elif side == "short" and current_price >= self.trailing_stop_price:
                    self.logger.info(f"📉 追踪止损触发 | 盈亏: {pnl_pct:.2%}")
                    self.trader.close_position()
                    return
            
            # 检查止损
            if pnl_pct <= -self.stop_loss_pct:
                self.logger.info(f"🛑 止损触发 | 盈亏: {pnl_pct:.2%}")
                self.trader.close_position()
                return
            
            # 检查止盈
            if pnl_pct >= self.take_profit_pct:
                self.logger.info(f"🎯 止盈触发 | 盈亏: {pnl_pct:.2%}")
                self.trader.close_position()
                return
            
            # 检查持仓时间
            if self.position_entry_time:
                hold_mins = (time.time() - self.position_entry_time) / 60
                if hold_mins >= self.max_hold_time:
                    self.logger.info(f"⏰ 最大持仓时间 | 盈亏: {pnl_pct:.2%}")
                    self.trader.close_position()
                    return
            
            # 检查趋势反转
            signal = self.generate_signal()
            if (side == "long" and signal == -1) or (side == "short" and signal == 1):
                self.logger.info(f"🔄 趋势反转 | 盈亏: {pnl_pct:.2%}")
                self.trader.close_position()
                return
            
            self.logger.debug(f"📊 持仓状态 | 方向: {side} | 盈亏: {pnl_pct:.2%} | 最大: {self.max_profit_reached:.2%}")
            
        except Exception as e:
            self.logger.error(f"管理仓位出错: {str(e)}")
