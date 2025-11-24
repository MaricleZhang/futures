"""
K线形态概率交易策略
基于K线形态识别，结合多个技术指标，输出做多/做空/观望的概率判断

File: strategies/candlestick_pattern_strategy.py
"""
import numpy as np
import pandas as pd
import pandas_ta_classic as ta
from datetime import datetime
import time
import logging
from typing import Dict, Optional, Tuple
from strategies.base_strategy import BaseStrategy
from utils.candlestick_patterns import CandlestickPattern


class CandlestickPatternStrategy(BaseStrategy):
    """
    K线形态概率交易策略
    
    策略逻辑：
    1. 识别多种K线形态（锤子、吞没、十字星等）
    2. 结合技术指标（趋势、动量、波动率）
    3. 计算综合评分
    4. 输出做多/做空/观望的概率
    """
    
    def __init__(self, trader):
        """初始化K线形态策略"""
        super().__init__(trader)
        self.logger = self.get_logger()
        
        # 时间框架配置
        self.kline_interval = '15m'
        self.check_interval = 300  # 5分钟检查一次
        self.lookback_period = 100  # 分析周期
        self.training_lookback = 100  # 兼容TradingManager
        
        # K线形态识别器
        self.pattern_detector = CandlestickPattern()
        
        # 技术指标参数
        self.ema_fast = 10
        self.ema_mid = 20
        self.ema_slow = 50
        self.rsi_period = 14
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        self.atr_period = 14
        self.adx_period = 14
        
        # 概率计算权重
        self.weights = {
            'pattern': 0.40,  # 形态权重
            'trend': 0.30,    # 趋势权重
            'momentum': 0.20, # 动量权重
            'volume': 0.10    # 成交量权重
        }
        
        # 交易配置
        self.min_probability = 0.40  # 最小交易概率阈值
        self.min_confidence = 0.60   # 最小置信度
        self.enable_auto_trade = True  # 是否自动交易
        
        # 仓位管理
        self.stop_loss_pct = 0.02
        self.take_profit_pct = 0.06
        self.max_position_hold_time = 720  # 分钟
        
        # 跟踪变量
        self.position_entry_time = None
        self.position_entry_price = None
        self.max_profit_reached = 0
        self.last_analysis = None
        
        self.logger.info("K线形态概率策略初始化完成")
    
    def calculate_indicators(self, df: pd.DataFrame) -> Dict:
        """
        计算所有技术指标
        
        Returns:
            包含所有指标的字典
        """
        try:
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            volume = df['volume'].values
            
            # 趋势指标
            ema_fast = ta.ema(pd.Series(close), timeperiod=self.ema_fast).values
            ema_mid = ta.ema(pd.Series(close), timeperiod=self.ema_mid).values
            ema_slow = ta.ema(pd.Series(close), timeperiod=self.ema_slow).values
            
            macd_df = ta.macd(
                pd.Series(close), 
                fast=self.macd_fast,
                slow=self.macd_slow,
                signal=self.macd_signal
            )
            macd_col = f'MACD_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}'
            signal_col = f'MACDs_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}'
            hist_col = f'MACDh_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}'
            
            macd = macd_df[macd_col].values
            macd_signal = macd_df[signal_col].values
            macd_hist = macd_df[hist_col].values
            
            adx_df = ta.adx(pd.Series(high), pd.Series(low), pd.Series(close), length=self.adx_period)
            adx = adx_df[f'ADX_{self.adx_period}'].values
            plus_di = adx_df[f'DMP_{self.adx_period}'].values
            minus_di = adx_df[f'DMN_{self.adx_period}'].values
            
            # 动量指标
            rsi = ta.rsi(pd.Series(close), length=self.rsi_period).values
            
            # talib defaults for STOCH are 5, 3, 3. pandas-ta defaults are 14, 3, 3.
            # Assuming we want to keep similar behavior to default talib if not specified otherwise, 
            # but here no params were passed to talib.STOCH, so it used 5, 3, 0, 3, 0.
            # Let's use k=5, d=3, smooth_k=3 for pandas-ta to be close.
            stoch_df = ta.stoch(pd.Series(high), pd.Series(low), pd.Series(close), k=5, d=3, smooth_k=3)
            stoch_k = stoch_df['STOCHk_5_3_3'].values
            stoch_d = stoch_df['STOCHd_5_3_3'].values
            
            # 波动率指标
            atr = ta.atr(pd.Series(high), pd.Series(low), pd.Series(close), length=self.atr_period).values
            
            bb_df = ta.bbands(pd.Series(close), length=20, std=2.0)
            upper_bb = bb_df['BBU_20_2.0'].values
            middle_bb = bb_df['BBM_20_2.0'].values
            lower_bb = bb_df['BBL_20_2.0'].values
            
            # 成交量指标
            avg_volume = ta.sma(pd.Series(volume), length=20).values
            
            return {
                'ema_fast': ema_fast,
                'ema_mid': ema_mid,
                'ema_slow': ema_slow,
                'macd': macd,
                'macd_signal': macd_signal,
                'macd_hist': macd_hist,
                'adx': adx,
                'plus_di': plus_di,
                'minus_di': minus_di,
                'rsi': rsi,
                'stoch_k': stoch_k,
                'stoch_d': stoch_d,
                'atr': atr,
                'upper_bb': upper_bb,
                'middle_bb': middle_bb,
                'lower_bb': lower_bb,
                'volume': volume,
                'avg_volume': avg_volume
            }
            
        except Exception as e:
            self.logger.error(f"计算指标出错: {str(e)}")
            return None
    
    def calculate_pattern_score(self, df: pd.DataFrame) -> Tuple[float, float, Dict]:
        """
        计算K线形态得分
        
        Returns:
            (看涨得分, 看跌得分, 形态详情)
        """
        patterns = self.pattern_detector.detect_all_patterns(df)
        summary = self.pattern_detector.get_pattern_summary(patterns)
        
        bullish_score = summary['total_bullish_strength']
        bearish_score = summary['total_bearish_strength']
        
        # 归一化到0-1范围
        total = bullish_score + bearish_score
        if total > 0:
            bullish_score = bullish_score / total
            bearish_score = bearish_score / total
        
        return bullish_score, bearish_score, summary
    
    def calculate_trend_score(self, indicators: Dict, close: pd.Series) -> Tuple[float, float]:
        """
        计算趋势得分
        
        Returns:
            (看涨得分, 看跌得分)
        """
        score = 0.0
        
        # EMA趋势
        ema_fast = indicators['ema_fast'][-1]
        ema_mid = indicators['ema_mid'][-1]
        ema_slow = indicators['ema_slow'][-1]
        current_price = close.iloc[-1]
        
        # EMA排列得分
        if ema_fast > ema_mid > ema_slow:
            score += 0.3  # 多头排列
        elif ema_fast < ema_mid < ema_slow:
            score -= 0.3  # 空头排列
        
        # 价格与EMA关系
        if current_price > ema_fast:
            score += 0.1
        else:
            score -= 0.1
        
        # MACD
        macd = indicators['macd'][-1]
        macd_signal = indicators['macd_signal'][-1]
        macd_hist = indicators['macd_hist'][-1]
        
        if macd > macd_signal and macd_hist > 0:
            score += 0.2
        elif macd < macd_signal and macd_hist < 0:
            score -= 0.2
        
        # ADX和DI
        adx = indicators['adx'][-1]
        plus_di = indicators['plus_di'][-1]
        minus_di = indicators['minus_di'][-1]
        
        if adx > 25:  # 强趋势
            if plus_di > minus_di:
                score += 0.2
            else:
                score -= 0.2
        
        # 归一化到0-1
        bullish_score = max(0, score)
        bearish_score = max(0, -score)
        
        total = bullish_score + bearish_score
        if total > 0:
            bullish_score = bullish_score / total
            bearish_score = bearish_score / total
        
        return bullish_score, bearish_score
    
    def calculate_momentum_score(self, indicators: Dict) -> Tuple[float, float]:
        """
        计算动量得分
        
        Returns:
            (看涨得分, 看跌得分)
        """
        score = 0.0
        
        # RSI
        rsi = indicators['rsi'][-1]
        if rsi > 70:
            score -= 0.3  # 超买
        elif rsi > 50:
            score += 0.2  # 偏多
        elif rsi < 30:
            score += 0.3  # 超卖（反转信号）
        else:
            score -= 0.2  # 偏空
        
        # Stochastic
        stoch_k = indicators['stoch_k'][-1]
        stoch_d = indicators['stoch_d'][-1]
        
        if stoch_k > stoch_d and stoch_k < 80:
            score += 0.2
        elif stoch_k < stoch_d and stoch_k > 20:
            score -= 0.2
        
        # 布林带位置
        close = indicators['ema_fast'][-1]  # 使用当前价格代理
        upper_bb = indicators['upper_bb'][-1]
        lower_bb = indicators['lower_bb'][-1]
        middle_bb = indicators['middle_bb'][-1]
        
        bb_position = (close - lower_bb) / (upper_bb - lower_bb) if upper_bb != lower_bb else 0.5
        
        if bb_position > 0.8:
            score -= 0.2  # 接近上轨
        elif bb_position < 0.2:
            score += 0.2  # 接近下轨
        
        # 归一化
        bullish_score = max(0, score)
        bearish_score = max(0, -score)
        
        total = bullish_score + bearish_score
        if total > 0:
            bullish_score = bullish_score / total
            bearish_score = bearish_score / total
        
        return bullish_score, bearish_score
    
    def calculate_volume_score(self, indicators: Dict) -> Tuple[float, float]:
        """
        计算成交量得分
        
        Returns:
            (看涨得分, 看跌得分)
        """
        current_volume = indicators['volume'][-1]
        avg_volume = indicators['avg_volume'][-1]
        
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        # 成交量放大表示信号强度增加
        if volume_ratio > 1.5:
            # 高成交量增强当前趋势
            # 这里返回中性，让其他因素决定方向
            return 0.5, 0.5
        else:
            # 低成交量降低信心
            return 0.4, 0.4
    
    def calculate_probabilities(self, df: pd.DataFrame) -> Dict:
        """
        计算做多/做空/观望概率
        
        Returns:
            {
                'long': 0.45,
                'short': 0.25,
                'hold': 0.30,
                'confidence': 0.75,
                'signal': 1/-1/0,
                'analysis': {...}
            }
        """
        try:
            # 计算指标
            indicators = self.calculate_indicators(df)
            if indicators is None:
                return self._default_probabilities()
            
            # 计算各维度得分
            pattern_bull, pattern_bear, pattern_details = self.calculate_pattern_score(df)
            trend_bull, trend_bear = self.calculate_trend_score(indicators, df['close'])
            momentum_bull, momentum_bear = self.calculate_momentum_score(indicators)
            volume_bull, volume_bear = self.calculate_volume_score(indicators)
            
            # 综合评分
            total_bullish = (
                pattern_bull * self.weights['pattern'] +
                trend_bull * self.weights['trend'] +
                momentum_bull * self.weights['momentum'] +
                volume_bull * self.weights['volume']
            )
            
            total_bearish = (
                pattern_bear * self.weights['pattern'] +
                trend_bear * self.weights['trend'] +
                momentum_bear * self.weights['momentum'] +
                volume_bear * self.weights['volume']
            )
            
            # 使用ADX调整置信度
            adx = indicators['adx'][-1]
            confidence_multiplier = min(adx / 40, 1.0) if adx > 0 else 0.5
            
            # Softmax归一化计算概率
            # 使用temperature参数控制概率分布的尖锐度
            temperature = 2.0 - confidence_multiplier  # ADX越高，分布越集中
            
            exp_bull = np.exp(total_bullish / temperature)
            exp_bear = np.exp(total_bearish / temperature)
            exp_hold = np.exp(0.3 / temperature)  # 观望基准值
            
            total_exp = exp_bull + exp_bear + exp_hold
            
            prob_long = exp_bull / total_exp
            prob_short = exp_bear / total_exp
            prob_hold = exp_hold / total_exp
            
            # 整体置信度
            confidence = max(prob_long, prob_short) * confidence_multiplier
            
            # 生成信号
            max_prob = max(prob_long, prob_short, prob_hold)
            if max_prob == prob_long and prob_long >= self.min_probability:
                signal = 1
            elif max_prob == prob_short and prob_short >= self.min_probability:
                signal = -1
            else:
                signal = 0
            
            # 详细分析数据
            analysis = {
                'pattern_score': {
                    'bullish': pattern_bull,
                    'bearish': pattern_bear,
                    'details': pattern_details
                },
                'trend_score': {
                    'bullish': trend_bull,
                    'bearish': trend_bear
                },
                'momentum_score': {
                    'bullish': momentum_bull,
                    'bearish': momentum_bear
                },
                'volume_score': {
                    'bullish': volume_bull,
                    'bearish': volume_bear
                },
                'indicators': {
                    'adx': adx,
                    'rsi': indicators['rsi'][-1],
                    'macd_hist': indicators['macd_hist'][-1]
                }
            }
            
            result = {
                'probabilities': {
                    'long': float(prob_long),
                    'short': float(prob_short),
                    'hold': float(prob_hold)
                },
                'confidence': float(confidence),
                'signal': signal,
                'analysis': analysis
            }
            
            # 记录日志
            self.logger.info(
                f"概率分析 - 做多: {prob_long:.1%}, 做空: {prob_short:.1%}, "
                f"观望: {prob_hold:.1%}, 置信度: {confidence:.1%}, 信号: {signal}"
            )
            
            if pattern_details['bullish_patterns']:
                self.logger.info(f"看涨形态: {pattern_details['bullish_patterns']}")
            if pattern_details['bearish_patterns']:
                self.logger.info(f"看跌形态: {pattern_details['bearish_patterns']}")
            
            self.last_analysis = result
            return result
            
        except Exception as e:
            self.logger.error(f"计算概率出错: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return self._default_probabilities()
    
    def _default_probabilities(self) -> Dict:
        """返回默认概率（全部观望）"""
        return {
            'probabilities': {
                'long': 0.0,
                'short': 0.0,
                'hold': 1.0
            },
            'confidence': 0.0,
            'signal': 0,
            'analysis': {}
        }
    
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
            
            if not klines or len(klines) < self.lookback_period:
                self.logger.warning("K线数据不足")
                return 0
            
            # 转换为DataFrame
            df = self._prepare_dataframe(klines)
            if df is None:
                return 0
            
            # 计算概率
            result = self.calculate_probabilities(df)
            
            return result['signal']
            
        except Exception as e:
            self.logger.error(f"生成信号出错: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return 0
    
    def _prepare_dataframe(self, klines) -> Optional[pd.DataFrame]:
        """转换K线数据为DataFrame"""
        try:
            if not klines or len(klines) < 30:
                self.logger.error("K线数据不足")
                return None
            
            df = pd.DataFrame(
                klines,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            return df
            
        except Exception as e:
            self.logger.error(f"准备DataFrame出错: {str(e)}")
            return None
    
    def monitor_position(self):
        """监控仓位并执行交易逻辑"""
        try:
            position = self.trader.get_position()
            
            # 无仓位 - 检查入场信号
            if position is None or float(position['info'].get('positionAmt', 0)) == 0:
                signal = self.generate_signal()
                
                if signal != 0 and self.enable_auto_trade:
                    # 获取账户余额
                    balance = self.trader.get_balance()
                    available_balance = float(balance['free'])
                    
                    current_price = self.trader.get_market_price()
                    
                    # 计算交易量
                    symbol_config = self.trader.symbol_config
                    trade_percent = symbol_config.get('trade_amount_percent', 100)
                    trade_amount = (available_balance * trade_percent / 100) / current_price
                    
                    # 执行交易
                    if signal == 1:
                        self.trader.open_long(amount=trade_amount)
                        self.logger.info(
                            f"开多仓 - 数量: {trade_amount:.6f}, 价格: {current_price}, "
                            f"概率: {self.last_analysis['probabilities']['long']:.1%}"
                        )
                    elif signal == -1:
                        self.trader.open_short(amount=trade_amount)
                        self.logger.info(
                            f"开空仓 - 数量: {trade_amount:.6f}, 价格: {current_price}, "
                            f"概率: {self.last_analysis['probabilities']['short']:.1%}"
                        )
                    
                    self.position_entry_time = time.time()
                    self.position_entry_price = current_price
                    self.max_profit_reached = 0
            
            # 有仓位 - 管理仓位
            else:
                self._manage_position(position)
                
        except Exception as e:
            self.logger.error(f"监控仓位出错: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
    
    def _manage_position(self, position):
        """管理现有仓位"""
        try:
            position_amount = float(position['info'].get('positionAmt', 0))
            entry_price = float(position['info'].get('entryPrice', 0))
            current_price = self.trader.get_market_price()
            position_side = "long" if position_amount > 0 else "short"
            
            # 计算盈亏
            if position_side == "long":
                profit_rate = (current_price - entry_price) / entry_price
            else:
                profit_rate = (entry_price - current_price) / entry_price
            
            # 更新最大盈利
            if profit_rate > self.max_profit_reached:
                self.max_profit_reached = profit_rate
            
            # 检查持仓时间
            if self.position_entry_time:
                holding_time = (time.time() - self.position_entry_time) / 60
                if holding_time >= self.max_position_hold_time:
                    self.logger.info(f"达到最大持仓时间 ({holding_time:.1f}分钟)，平仓")
                    self.trader.close_position()
                    return
            
            # 止损
            if profit_rate <= -self.stop_loss_pct:
                self.logger.info(f"触发止损 {profit_rate:.2%}")
                self.trader.close_position()
                return
            
            # 止盈
            if profit_rate >= self.take_profit_pct:
                self.logger.info(f"触发止盈 {profit_rate:.2%}")
                self.trader.close_position()
                return
            
            # 趋势反转
            signal = self.generate_signal()
            if (position_side == "long" and signal == -1) or \
               (position_side == "short" and signal == 1):
                self.logger.info(f"检测到趋势反转，平仓 {position_side}")
                self.trader.close_position()
                return
            
            self.logger.debug(
                f"仓位状态 - 方向: {position_side}, 盈亏: {profit_rate:.2%}, "
                f"当前价: {current_price}, 入场价: {entry_price}"
            )
                
        except Exception as e:
            self.logger.error(f"管理仓位出错: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
