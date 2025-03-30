"""
稳定趋势预测策略 - 15分钟时间框架
专注于提供高稳定性、低波动的趋势预测
"""
import numpy as np
import pandas as pd
import talib
from datetime import datetime, timedelta
import time
import logging
from strategies.base_strategy import BaseStrategy


class StableTrendStrategy15m(BaseStrategy):
    """StableTrendStrategy15m - 稳定趋势预测策略
    
    基于15分钟K线的稳定趋势预测策略，通过多重过滤和确认机制，
    减少错误信号和市场噪音的影响，提供更加稳定可靠的趋势预测。
    
    特点:
    1. 多重趋势确认系统：结合多种指标共同确认趋势方向
    2. 先行指标与滞后指标结合：平衡灵敏度和可靠性
    3. 动态噪音过滤：根据市场波动性自适应调整过滤参数
    4. 共识机制：需要多数指标一致认同才生成信号
    5. 趋势变化重要性加权：基于持续性和强度区分主要趋势和次要波动
    6. 市场状态分类：区分趋势市、震荡市和转折市，采用不同的预测策略
    """
    
    def __init__(self, trader):
        """初始化稳定趋势预测策略"""
        super().__init__(trader)
        self.logger = self.get_logger()
        
        # K线设置
        self.kline_interval = '30m'       # 15分钟K线
        self.check_interval = 300         # 检查信号间隔(秒)
        self.lookback_period = 200        # 计算指标所需的K线数量
        self.training_lookback = 200      # 提供该属性以兼容TradingManager
        
        # 趋势平滑参数
        self.ema_short_period = 9         # 短期EMA周期
        self.ema_medium_period = 21       # 中期EMA周期
        self.ema_long_period = 50         # 长期EMA周期
        self.ema_trend_period = 100       # 趋势EMA周期
        
        # 趋势强度参数
        self.adx_period = 14              # ADX周期
        self.adx_trend_threshold = 20     # ADX趋势阈值，高于此值认为是明确趋势
        self.adx_strong_threshold = 40    # ADX强趋势阈值
        
        # 动量指标参数
        self.rsi_period = 14              # RSI周期
        self.rsi_smooth_period = 5        # RSI平滑周期
        self.rsi_overbought = 70          # RSI超买阈值
        self.rsi_oversold = 30            # RSI超卖阈值
        self.macd_fast = 12               # MACD快线周期
        self.macd_slow = 26               # MACD慢线周期
        self.macd_signal = 9              # MACD信号线周期
        
        # 波动率参数
        self.atr_period = 14              # ATR周期
        self.atr_smooth_period = 5        # ATR平滑周期
        self.bb_period = 20               # 布林带周期
        self.bb_std = 2                   # 布林带标准差倍数
        
        # 共识参数
        self.min_consensus_threshold = 0.60  # 最小共识阈值
        self.trend_consensus_threshold = 0.2  # 趋势市场共识阈值
        self.range_consensus_threshold = 0.65  # 震荡市场共识阈值
        
        # 噪声过滤参数
        self.noise_filter_sensitivity = 0.5  # 噪声过滤灵敏度 (0-1)
        self.min_trend_duration = 3       # 最小趋势持续K线数
        self.min_signal_score = 0.3       # 最小信号得分
        
        # 动态参数调整
        self.volatility_adjust = True     # 是否开启波动率自适应
        self.adapt_to_market_state = True # 是否根据市场状态调整策略
        
        # 趋势状态参数
        self.market_state = "unknown"     # 市场状态: "trend", "range", "transition", "unknown"
        self.current_trend = 0            # 当前趋势: 1(上升), -1(下降), 0(中性)
        self.trend_strength = 0           # 趋势强度: 0-100
        self.trend_duration = 0           # 当前趋势持续的K线数量
        self.trend_start_price = None     # 当前趋势的起始价格
        
        # 历史信号记录
        self.signal_history = []          # 最近的信号历史
        self.max_signal_history = 20      # 保存的最大信号数量
        self.last_signal_time = None      # 上次产生信号的时间
        self.last_signal = 0              # 上次信号: 1(买入), -1(卖出), 0(观望)
        
        # 持仓控制参数
        self.max_position_hold_time = 480  # 最大持仓时间(分钟)
        self.stop_loss_pct = 0.02         # 止损比例
        self.take_profit_pct = 0.05       # 止盈比例
        self.trailing_stop = True         # 是否启用追踪止损
        self.trailing_stop_activation = 0.02  # 激活追踪止损的利润百分比
        self.trailing_stop_distance = 0.01    # 追踪止损距离百分比
        
        # 内部状态
        self._last_analyzed_klines = None
        self._last_indicators = None
        self._last_analysis_time = 0
        self.position_entry_time = None
        self.position_entry_price = None
        self.max_profit_reached = 0
    
    def analyze_market_state(self, klines, force_recalculate=False):
        """
        分析市场状态，判断是趋势市、震荡市还是转折市
        
        Args:
            klines (list): K线数据
            force_recalculate (bool): 是否强制重新计算
            
        Returns:
            dict: 市场状态信息，包含状态、趋势方向、趋势强度等
        """
        current_time = time.time()
        
        # 如果距离上次分析时间不足60秒，且不要求强制计算，则返回缓存的结果
        if (not force_recalculate and 
            self._last_analyzed_klines is not None and 
            current_time - self._last_analysis_time < 60):
            return self._last_analyzed_klines
        
        try:
            # 转换K线数据到DataFrame
            df = self._prepare_dataframe(klines)
            if df is None or len(df) < 50:
                return {"state": "unknown", "trend": 0, "strength": 0}
            
            # 计算基础指标
            df = self._calculate_indicators(df)
            
            # 计算趋势强度 (ADX)
            adx_value = df['adx'].iloc[-1]
            plus_di = df['plus_di'].iloc[-1]
            minus_di = df['minus_di'].iloc[-1]
            
            # 判断趋势方向
            trend_direction = 0
            if plus_di > minus_di and plus_di > 20:
                trend_direction = 1  # 上升趋势
            elif minus_di > plus_di and minus_di > 20:
                trend_direction = -1  # 下降趋势
                
            # 判断市场是否处于震荡状态
            is_ranging = False
            bb_width = df['bb_width'].iloc[-1]
            bb_width_mean = df['bb_width'].iloc[-20:].mean()
            price_volatility = df['close'].pct_change().iloc[-20:].std() * 100
            
            # 震荡条件: 
            # 1. ADX较低
            # 2. 布林带宽度较窄
            # 3. 价格波动率低
            if adx_value < self.adx_trend_threshold and bb_width < bb_width_mean * 0.8 and price_volatility < 1.0:
                is_ranging = True
                
            # 判断是否处于转折点
            is_transition = False
            rsi = df['rsi'].iloc[-1]
            rsi_slope = df['rsi'].diff().iloc[-3:].mean()
            macd_hist = df['macd_hist'].iloc[-1]
            macd_hist_prev = df['macd_hist'].iloc[-2]
            
            # 转折条件:
            # 1. RSI从超买/超卖区域反转
            # 2. MACD柱状图穿越零线
            # 3. 短期趋势与中期趋势方向相反
            if ((rsi > 65 and rsi_slope < 0) or (rsi < 35 and rsi_slope > 0) or
                (macd_hist * macd_hist_prev < 0) or
                (df['ema_short'].iloc[-1] > df['ema_medium'].iloc[-1] and 
                 df['ema_medium'].iloc[-1] < df['ema_long'].iloc[-1]) or
                (df['ema_short'].iloc[-1] < df['ema_medium'].iloc[-1] and 
                 df['ema_medium'].iloc[-1] > df['ema_long'].iloc[-1])):
                is_transition = True
                
            # 确定市场状态
            state = "unknown"
            if adx_value >= self.adx_strong_threshold:
                state = "trend"  # 强趋势市场
            elif adx_value >= self.adx_trend_threshold:
                state = "trend"  # 趋势市场
            elif is_ranging:
                state = "range"  # 震荡市场
            elif is_transition:
                state = "transition"  # 转折市场
            else:
                # 使用趋势持续性判断
                recent_trend = self._calculate_trend_consistency(df, 10)
                if abs(recent_trend) > 0.7:
                    state = "trend"
                else:
                    state = "range"
            
            # 计算趋势持续时间
            if self.current_trend == trend_direction:
                self.trend_duration += 1
            else:
                self.trend_duration = 1
                self.trend_start_price = df['close'].iloc[-1]
                
            # 更新市场状态
            self.market_state = state
            self.current_trend = trend_direction
            self.trend_strength = min(100, float(adx_value))
            
            # 更新缓存
            self._last_analyzed_klines = {
                "state": state,
                "trend": trend_direction,
                "strength": min(100, float(adx_value)),
                "adx": float(adx_value),
                "ranging": is_ranging,
                "transition": is_transition,
                "volatility": float(price_volatility),
                "bb_width": float(bb_width),
                "rsi": float(rsi)
            }
            self._last_analysis_time = current_time
            
            self.logger.info(f"市场状态: {state}, 趋势方向: {trend_direction}, plus_di: {plus_di}, minus_di: {minus_di}, 趋势强度: {adx_value:.2f}, " +
                          f"趋势持续: {self.trend_duration}根K线")
            
            return self._last_analyzed_klines
            
        except Exception as e:
            self.logger.error(f"分析市场状态失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {"state": "unknown", "trend": 0, "strength": 0}
    
    def _prepare_dataframe(self, klines):
        """
        将K线数据转换为DataFrame格式
        
        Args:
            klines (list): K线数据列表 [[timestamp, open, high, low, close, volume], ...]
            
        Returns:
            pandas.DataFrame: 转换后的DataFrame
        """
        try:
            if not isinstance(klines, list) or len(klines) < 50:
                self.logger.error("K线数据为空或长度不足")
                return None
            
            # 创建DataFrame
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # 转换数据类型
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
            # 添加时间列
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            return df
            
        except Exception as e:
            self.logger.error(f"准备DataFrame失败: {str(e)}")
            return None
    
    def _calculate_indicators(self, df):
        """
        计算各种技术指标
        
        Args:
            df (pandas.DataFrame): K线数据DataFrame
            
        Returns:
            pandas.DataFrame: 添加了指标的DataFrame
        """
        try:
            # 1. 移动平均线指标
            df['ema_short'] = talib.EMA(df['close'].values, timeperiod=self.ema_short_period)
            df['ema_medium'] = talib.EMA(df['close'].values, timeperiod=self.ema_medium_period)
            df['ema_long'] = talib.EMA(df['close'].values, timeperiod=self.ema_long_period)
            df['ema_trend'] = talib.EMA(df['close'].values, timeperiod=self.ema_trend_period)
            
            # EMA斜率 (变化率)
            df['ema_short_slope'] = (df['ema_short'] - df['ema_short'].shift(3)) / df['ema_short'].shift(3) * 100
            df['ema_medium_slope'] = (df['ema_medium'] - df['ema_medium'].shift(3)) / df['ema_medium'].shift(3) * 100
            
            # 均线交叉和距离
            df['ema_short_medium_cross'] = df['ema_short'] - df['ema_medium']
            df['ema_medium_long_cross'] = df['ema_medium'] - df['ema_long']
            
            # 价格与均线的关系
            df['price_to_ema_short'] = df['close'] / df['ema_short'] - 1
            df['price_to_ema_medium'] = df['close'] / df['ema_medium'] - 1
            df['price_to_ema_long'] = df['close'] / df['ema_long'] - 1
            
            # 2. 趋势指标 - ADX
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            
            df['adx'] = pd.Series(talib.ADX(high, low, close, timeperiod=self.adx_period), index=df.index)
            df['plus_di'] = pd.Series(talib.PLUS_DI(high, low, close, timeperiod=self.adx_period), index=df.index)
            df['minus_di'] = pd.Series(talib.MINUS_DI(high, low, close, timeperiod=self.adx_period), index=df.index)
            
            # ADX斜率 - 用于判断趋势加速/减速
            df['adx_slope'] = df['adx'].diff(3) / 3
            
            # 3. MACD指标
            macd, macd_signal, macd_hist = talib.MACD(
                close, 
                fastperiod=self.macd_fast, 
                slowperiod=self.macd_slow, 
                signalperiod=self.macd_signal
            )
            df['macd'] = pd.Series(macd, index=df.index)
            df['macd_signal'] = pd.Series(macd_signal, index=df.index)
            df['macd_hist'] = pd.Series(macd_hist, index=df.index)
            df['macd_hist_slope'] = df['macd_hist'].diff(3)
            
            # 4. RSI指标
            df['rsi'] = pd.Series(talib.RSI(close, timeperiod=self.rsi_period), index=df.index)
            df['rsi_smooth'] = df['rsi'].rolling(window=self.rsi_smooth_period).mean()
            df['rsi_slope'] = df['rsi'].diff(3)
            
            # 5. 波动率指标 - ATR
            df['atr'] = pd.Series(talib.ATR(high, low, close, timeperiod=self.atr_period), index=df.index)
            df['atr_smooth'] = df['atr'].rolling(window=self.atr_smooth_period).mean()
            df['atr_pct'] = df['atr'] / df['close'] * 100
            
            # 6. 布林带指标
            upper_band, middle_band, lower_band = talib.BBANDS(
                close, timeperiod=self.bb_period, nbdevup=self.bb_std, nbdevdn=self.bb_std
            )
            df['bb_upper'] = pd.Series(upper_band, index=df.index)
            df['bb_middle'] = pd.Series(middle_band, index=df.index)
            df['bb_lower'] = pd.Series(lower_band, index=df.index)
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # 7. 其他指标
            # CCI - 顺势指标
            df['cci'] = pd.Series(talib.CCI(high, low, close, timeperiod=20), index=df.index)
            
            # 随机指标
            df['slowk'], df['slowd'] = talib.STOCH(
                high, low, close, fastk_period=14, slowk_period=3, slowd_period=3
            )
            
            # Chaikin A/D Oscillator
            df['chaikin_ad'] = pd.Series(talib.ADOSC(high, low, close, df['volume'].values, fastperiod=3, slowperiod=10), index=df.index)
            
            # OBV - 能量潮
            df['obv'] = pd.Series(talib.OBV(close, df['volume'].values), index=df.index)
            df['obv_slope'] = (df['obv'] - df['obv'].shift(3)) / abs(df['obv'].shift(3)) * 100
            
            # 8. 价格动量
            df['roc_5'] = pd.Series(talib.ROC(close, timeperiod=5), index=df.index)
            df['roc_10'] = pd.Series(talib.ROC(close, timeperiod=10), index=df.index)
            df['roc_20'] = pd.Series(talib.ROC(close, timeperiod=20), index=df.index)
            
            # 9. 交易量指标
            df['volume_sma'] = df['volume'].rolling(window=10).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # 10. 计算价格与趋势方向的一致性
            df['close_diff'] = df['close'].diff()
            df['consistent_trend'] = ((df['close_diff'] > 0) & (df['ema_short_slope'] > 0)).astype(int) - \
                                     ((df['close_diff'] < 0) & (df['ema_short_slope'] < 0)).astype(int)
            
            return df
            
        except Exception as e:
            self.logger.error(f"计算指标失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return df
    
    def _calculate_trend_consistency(self, df, period=10):
        """
        计算价格趋势的一致性得分
        
        Args:
            df (pandas.DataFrame): K线数据DataFrame
            period (int): 计算周期
            
        Returns:
            float: 一致性得分，范围 [-1, 1]，正值表示上升一致，负值表示下降一致
        """
        if len(df) < period:
            return 0
            
        # 获取最近N个周期的收盘价
        recent_closes = df['close'].iloc[-period:].values
        
        # 计算连续上涨/下跌的比例
        ups = 0
        downs = 0
        
        for i in range(1, len(recent_closes)):
            if recent_closes[i] > recent_closes[i-1]:
                ups += 1
            elif recent_closes[i] < recent_closes[i-1]:
                downs += 1
                
        # 计算一致性得分
        if ups + downs == 0:
            return 0
            
        return (ups - downs) / period
    
    def calculate_trend_score(self, klines):
        """
        计算趋势评分，用于生成稳定的趋势预测
        
        Args:
            klines (list): K线数据
            
        Returns:
            dict: 趋势评分信息，包含总评分、各指标得分等
        """
        try:
            # 准备数据
            df = self._prepare_dataframe(klines)
            if df is None or len(df) < 50:
                return {'total_score': 0, 'trend': 0, 'indicators': {}}
                
            # 计算指标
            df = self._calculate_indicators(df)
            
            # 获取市场状态
            market_state = self.analyze_market_state(klines)
            
            # 根据市场状态调整指标权重
            weights = self._get_adaptive_weights(market_state)
            
            # 计算各指标得分
            indicator_scores = {}
            
            # 1. EMA趋势得分 (-1 到 1)
            ema_score = 0
            # 短期与中期EMA关系
            if df['ema_short'].iloc[-1] > df['ema_medium'].iloc[-1]:
                ema_score += 0.4
            else:
                ema_score -= 0.4
                
            # 中期与长期EMA关系
            if df['ema_medium'].iloc[-1] > df['ema_long'].iloc[-1]:
                ema_score += 0.4
            else:
                ema_score -= 0.4
                
            # EMA斜率
            ema_slope_score = 0
            if df['ema_short_slope'].iloc[-1] > 0.1:
                ema_slope_score += 0.1
            elif df['ema_short_slope'].iloc[-1] < -0.1:
                ema_slope_score -= 0.1
                
            if df['ema_medium_slope'].iloc[-1] > 0.05:
                ema_slope_score += 0.1
            elif df['ema_medium_slope'].iloc[-1] < -0.05:
                ema_slope_score -= 0.1
                
            # 组合EMA得分
            ema_score += ema_slope_score
            indicator_scores['ema'] = ema_score
            
            # 2. ADX趋势得分 (-1 到 1)
            adx_value = df['adx'].iloc[-1]
            plus_di = df['plus_di'].iloc[-1]
            minus_di = df['minus_di'].iloc[-1]
            
            # 标准化ADX为0-1
            adx_norm = min(1.0, adx_value / 50)
            
            # 计算DI差异
            di_diff = plus_di - minus_di
            di_diff_norm = np.clip(di_diff / 30, -1, 1)
            
            # ADX得分 = DI差异 * ADX强度
            adx_score = di_diff_norm * adx_norm
            indicator_scores['adx'] = adx_score
            
            # 3. MACD得分 (-1 到 1)
            macd_value = df['macd'].iloc[-1]
            macd_signal = df['macd_signal'].iloc[-1]
            macd_hist = df['macd_hist'].iloc[-1]
            
            # MACD线相对于信号线
            macd_diff = macd_value - macd_signal
            macd_cross_score = np.clip(macd_diff * 5, -0.5, 0.5)  # 缩放因子5是经验值
            
            # MACD柱状图趋势
            hist_avg = df['macd_hist'].iloc[-5:].mean()
            hist_score = np.clip(hist_avg * 10, -0.5, 0.5)  # 缩放因子10是经验值
            
            # 组合MACD得分
            macd_score = macd_cross_score + hist_score
            indicator_scores['macd'] = macd_score
            
            # 4. RSI得分 (-1 到 1)
            rsi = df['rsi_smooth'].iloc[-1]
            rsi_prev = df['rsi_smooth'].iloc[-2]
            
            # RSI绝对值得分
            if rsi > 70:
                rsi_abs_score = -0.5  # 超买
            elif rsi < 30:
                rsi_abs_score = 0.5   # 超卖
            elif rsi > 60:
                rsi_abs_score = (70 - rsi) / 20  # 60-70区间，线性降低
            elif rsi < 40:
                rsi_abs_score = (30 - rsi) / 20  # 30-40区间，线性增加
            else:
                rsi_abs_score = 0
                
            # RSI动量得分
            rsi_mom_score = (rsi - rsi_prev) / 5  # 变化率/5
            rsi_mom_score = np.clip(rsi_mom_score, -0.5, 0.5)
            
            # 组合RSI得分
            rsi_score = rsi_abs_score + rsi_mom_score
            indicator_scores['rsi'] = rsi_score
            
            # 5. 布林带得分 (-1 到 1)
            bb_pos = df['bb_position'].iloc[-1]
            bb_width = df['bb_width'].iloc[-1]
            bb_width_prev = df['bb_width'].iloc[-4] if len(df) > 4 else bb_width  # 相当于shift(3)
            bb_avg_width = df['bb_width'].iloc[-20:].mean()
            
            # 布林带位置得分
            if bb_pos > 0.8:
                bb_pos_score = -0.6  # 接近上轨，看跌
            elif bb_pos < 0.2:
                bb_pos_score = 0.6   # 接近下轨，看涨
            else:
                bb_pos_score = 0
                
            # 布林带宽度得分（窄带突破信号）
            if bb_width < bb_avg_width * 0.7 and bb_width > bb_width_prev * 1.2:
                if bb_pos > 0.6:
                    bb_width_score = 0.4  # 窄带向上突破
                elif bb_pos < 0.4:
                    bb_width_score = -0.4  # 窄带向下突破
                else:
                    bb_width_score = 0
            else:
                bb_width_score = 0
                
            # 组合布林带得分
            bb_score = bb_pos_score + bb_width_score
            indicator_scores['bollinger'] = bb_score
            
            # 6. 价格动量得分 (-1 到 1)
            roc5 = df['roc_5'].iloc[-1]
            roc20 = df['roc_20'].iloc[-1]
            
            # 短期动量
            short_mom_score = np.clip(roc5 / 3, -0.5, 0.5)
            
            # 长期动量
            long_mom_score = np.clip(roc20 / 5, -0.5, 0.5)
            
            # 组合动量得分
            momentum_score = short_mom_score * 0.6 + long_mom_score * 0.4
            indicator_scores['momentum'] = momentum_score
            
            # 7. 交易量指标得分 (-1 到 1)
            vol_ratio = df['volume_ratio'].iloc[-1]
            obv_slope = df['obv_slope'].iloc[-1]
            
            # 交易量比率得分
            if vol_ratio > 2 and df['close'].iloc[-1] > df['close'].iloc[-2]:
                vol_ratio_score = 0.5  # 放量上涨
            elif vol_ratio > 2 and df['close'].iloc[-1] < df['close'].iloc[-2]:
                vol_ratio_score = -0.5  # 放量下跌
            else:
                vol_ratio_score = 0
                
            # OBV斜率得分
            obv_score = np.clip(obv_slope / 10, -0.5, 0.5)
            
            # 组合交易量得分
            volume_score = vol_ratio_score + obv_score
            indicator_scores['volume'] = volume_score
            
            # 8. 一致性得分 (-1 到 1)
            consistency = self._calculate_trend_consistency(df, 10)
            indicator_scores['consistency'] = consistency
            
            # 9. CCI得分 (-1 到 1)
            cci = df['cci'].iloc[-1]
            cci_score = 0
            
            if cci > 100:
                cci_score = -0.3 + (200 - min(cci, 200)) / 200 * 0.3  # 100-200区间，线性调整
            elif cci < -100:
                cci_score = 0.3 + (min(cci, -200) + 200) / 200 * 0.3  # -100到-200区间，线性调整
            else:
                cci_score = cci / 200  # -100到100区间，线性比例
                
            indicator_scores['cci'] = cci_score
            
            # 10. 随机指标得分 (-1 到 1)
            k = df['slowk'].iloc[-1]
            d = df['slowd'].iloc[-1]
            
            stoch_score = 0
            
            # K线与D线交叉
            if k > d and k < 30:
                stoch_score = 0.5  # 超卖区金叉
            elif k < d and k > 70:
                stoch_score = -0.5  # 超买区死叉
            elif k > d:
                stoch_score = 0.2  # 普通金叉
            elif k < d:
                stoch_score = -0.2  # 普通死叉
                
            indicator_scores['stochastic'] = stoch_score
            
            # 使用市场状态适应性权重计算总分
            total_score = 0
            for indicator, score in indicator_scores.items():
                if indicator in weights:
                    total_score += score * weights[indicator]
            
            # 应用噪音过滤
            filtered_score = self._apply_noise_filter(total_score, df)
            
            # 标准化到[-1, 1]范围
            final_score = np.clip(filtered_score, -1, 1)
            
            # 确定趋势信号
            if final_score > self.min_signal_score:
                trend_signal = 1
            elif final_score < -self.min_signal_score:
                trend_signal = -1
            else:
                trend_signal = 0
                
            # 输出详细结果
            self.logger.info(f"趋势评分: {final_score:.4f}, 趋势信号: {trend_signal}")
            for ind, score in indicator_scores.items():
                self.logger.info(f"  - {ind}: {score:.4f} (权重: {weights.get(ind, 0):.2f})")
                
            # 返回结果
            result = {
                'total_score': final_score,
                'trend': trend_signal,
                'market_state': market_state['state'],
                'trend_strength': market_state['strength'],
                'indicators': indicator_scores
            }
            
            self._last_indicators = result
            return result
            
        except Exception as e:
            self.logger.error(f"计算趋势得分失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {'total_score': 0, 'trend': 0, 'indicators': {}}
    
    def _get_adaptive_weights(self, market_state):
        """
        根据市场状态获取自适应的指标权重
        
        Args:
            market_state (dict): 市场状态信息
            
        Returns:
            dict: 各指标的权重
        """
        state = market_state['state']
        adx = market_state.get('adx', 0)
        
        # 默认权重
        weights = {
            'ema': 0.15,
            'adx': 0.15,
            'macd': 0.15,
            'rsi': 0.1,
            'bollinger': 0.1,
            'momentum': 0.1,
            'volume': 0.05,
            'consistency': 0.1,
            'cci': 0.05,
            'stochastic': 0.05
        }
        
        # 根据市场状态调整权重
        if state == "trend":
            # 趋势市场更重视趋势指标
            weights['ema'] = 0.2
            weights['adx'] = 0.2
            weights['macd'] = 0.15
            weights['momentum'] = 0.15
            weights['consistency'] = 0.1
            weights['rsi'] = 0.05
            weights['bollinger'] = 0.05
            weights['volume'] = 0.05
            weights['cci'] = 0.025
            weights['stochastic'] = 0.025
            
        elif state == "range":
            # 震荡市场更重视震荡指标
            weights['rsi'] = 0.2
            weights['bollinger'] = 0.2
            weights['stochastic'] = 0.15
            weights['cci'] = 0.1
            weights['macd'] = 0.1
            weights['ema'] = 0.05
            weights['adx'] = 0.05
            weights['momentum'] = 0.05
            weights['volume'] = 0.05
            weights['consistency'] = 0.05
            
        elif state == "transition":
            # 转折市场重视动量和反转指标
            weights['rsi'] = 0.15
            weights['macd'] = 0.15
            weights['bollinger'] = 0.15
            weights['momentum'] = 0.15
            weights['volume'] = 0.1
            weights['cci'] = 0.1
            weights['stochastic'] = 0.1
            weights['ema'] = 0.05
            weights['adx'] = 0.025
            weights['consistency'] = 0.025
            
        # 强趋势下进一步调整
        if adx > 40:
            weights['adx'] += 0.05
            weights['ema'] += 0.05
            weights['rsi'] -= 0.05
            weights['bollinger'] -= 0.05
            
        # 确保权重总和为1
        total = sum(weights.values())
        if total != 1.0:
            for key in weights:
                weights[key] /= total
                
        return weights
    
    def _apply_noise_filter(self, score, df):
        """
        应用噪声过滤，减少假信号
        
        Args:
            score (float): 原始趋势得分
            df (pandas.DataFrame): K线数据DataFrame
            
        Returns:
            float: 过滤后的趋势得分
        """
        # 如果得分接近零，可能是噪声，减小其影响
        if abs(score) < 0.1:
            return score * 0.5
            
        # 获取近期波动信息
        recent_volatility = df['atr_pct'].iloc[-5:].mean()
        avg_volatility = df['atr_pct'].iloc[-20:].mean()
        
        # 波动率异常低时，降低信号强度
        if recent_volatility < avg_volatility * 0.5:
            return score * 0.7
            
        # 波动率异常高时，也降低信号强度
        if recent_volatility > avg_volatility * 2:
            return score * 0.8
            
        # 根据趋势持续性调整
        consistency = self._calculate_trend_consistency(df, 5)
        
        # 如果得分方向与近期趋势一致性不符，降低信号强度
        if score > 0 and consistency < 0:
            return score * (1 - self.noise_filter_sensitivity * 0.5)
        elif score < 0 and consistency > 0:
            return score * (1 - self.noise_filter_sensitivity * 0.5)
            
        # 信号与趋势一致，可以略微增强
        if (score > 0 and consistency > 0.6) or (score < 0 and consistency < -0.6):
            return score * 1.1
            
        return score
    
    def generate_signal(self, klines):
        """
        生成交易信号
        
        Args:
            klines (list): K线数据
            
        Returns:
            int: 交易信号，1(买入)，-1(卖出)，0(观望)
        """
        try:
            # 分析市场状态
            market_state = self.analyze_market_state(klines)
            trend_direction = market_state['trend']

            # return trend_direction
            
            # 计算趋势得分
            trend_result = self.calculate_trend_score(klines)
            trend_score = trend_result['total_score']
            trend_signal = trend_result['trend']
            
            # 获取当前持仓
            position = self.trader.get_position()
            position_side = None
            if position and 'info' in position:
                position_amount = float(position['info'].get('positionAmt', 0))
                if position_amount > 0:
                    position_side = "多"
                elif position_amount < 0:
                    position_side = "空"
            
            # 检查信号稳定性
            is_stable_signal = self._check_signal_stability(trend_signal, market_state)
            
            # 信号确认
            if is_stable_signal:
                # 记录信号历史
                current_time = datetime.now()
                signal_record = {
                    'time': current_time,
                    'signal': trend_signal,
                    'score': trend_score,
                    'market_state': market_state['state']
                }
                
                self.signal_history.append(signal_record)
                if len(self.signal_history) > self.max_signal_history:
                    self.signal_history.pop(0)
                    
                self.last_signal_time = current_time
                self.last_signal = trend_signal
                
                # 如果当前没有持仓，或持仓方向与信号相反，则返回信号
                if position_side is None or (position_side == "多" and trend_signal == -1) or (position_side == "空" and trend_signal == 1):
                    return trend_signal
                    
                # 如果持仓方向与信号一致，则继续持有
                if (position_side == "多" and trend_signal == 1) or (position_side == "空" and trend_signal == -1):
                    self.logger.info("信号与当前持仓方向一致，继续持有")
                    return trend_signal
            else:
                # 信号不稳定，但如果当前持仓方向与趋势相反，仍然考虑平仓
                if position_side == "多" and trend_score < -0.2:
                    self.logger.info("虽然信号不够稳定，但趋势明显向下，建议平多仓")
                    return -1
                elif position_side == "空" and trend_score > 0.2:
                    self.logger.info("虽然信号不够稳定，但趋势明显向上，建议平空仓")
                    return 1
                
                self.logger.info("信号不够稳定，保持观望")
            
            return 0
            
        except Exception as e:
            self.logger.error(f"生成信号失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return 0
    
    def _check_signal_stability(self, signal, market_state):
        """
        检查信号的稳定性
        
        Args:
            signal (int): 当前信号
            market_state (dict): 市场状态信息
            
        Returns:
            bool: 信号是否稳定
        """
        # 无信号不需要检查稳定性
        if signal == 0:
            return False
        
        # 根据市场状态选择不同的共识阈值
        if market_state['state'] == "trend":
            consensus_threshold = self.trend_consensus_threshold
        elif market_state['state'] == "range":
            consensus_threshold = self.range_consensus_threshold
        else:
            consensus_threshold = self.min_consensus_threshold
        
        # 在强趋势中可以降低稳定性要求
        if market_state['strength'] > 50:
            consensus_threshold = max(0.5, consensus_threshold - 0.1)
        
        # 检查最近的信号历史是否一致
        if len(self.signal_history) >= 3:
            recent_signals = [record['signal'] for record in self.signal_history[-3:]]
            
            # 计算指定信号的比例
            signal_count = recent_signals.count(signal)
            agreement_ratio = signal_count / len(recent_signals)
            
            # 如果最近的信号一致性高于阈值，认为信号稳定
            if agreement_ratio >= consensus_threshold:
                self.logger.info(f"信号稳定性检查通过: 最近{len(recent_signals)}个信号中，{signal_count}个与当前信号一致")
                return True
            else:
                self.logger.info(f"信号稳定性不足: 最近{len(recent_signals)}个信号中，只有{signal_count}个与当前信号一致")
                return False
        
        # 信号历史不足，检查得分的绝对值
        if abs(self._last_indicators['total_score']) > 0.7:
            self.logger.info(f"信号历史不足，但趋势得分较高({self._last_indicators['total_score']:.4f})，认为信号稳定")
            return True
            
        # 默认情况下，要求趋势持续一定时间
        if (signal == 1 and self.current_trend == 1 and self.trend_duration >= self.min_trend_duration) or \
           (signal == -1 and self.current_trend == -1 and self.trend_duration >= self.min_trend_duration):
            self.logger.info(f"趋势持续{self.trend_duration}根K线，符合最小持续时间要求")
            return True
            
        return False
    
    def monitor_position(self):
        """监控当前持仓，并根据策略决定是否平仓"""
        try:
            # 获取当前持仓
            position = self.trader.get_position()
            
            # 如果没有持仓，检查是否有新的交易信号
            if position is None or float(position['info'].get('positionAmt', 0)) == 0:
                # 获取最新K线数据
                klines = self.trader.get_klines(interval=self.kline_interval, limit=self.lookback_period)
                
                # 生成交易信号
                signal = self.generate_signal(klines)
                
                # 获取当前市场价格
                current_price = self.trader.get_market_price()
                
                # 根据信号执行交易
                if signal == 1:  # 买入信号
                    # 计算交易数量
                    balance = self.trader.get_balance()
                    available_balance = float(balance['free'])
                    
                    # 从config获取交易金额百分比
                    symbol_config = self.trader.symbol_config
                    trade_percent = symbol_config.get('trade_amount_percent', 50)
                    
                    # 计算交易金额
                    trade_amount = (available_balance * trade_percent / 100) / current_price
                    
                    # 开多仓
                    self.trader.open_long(amount=trade_amount)
                    self.logger.info(f"开多仓 - 数量: {trade_amount:.6f}, 价格: {current_price}")
                    
                    # 记录开仓信息
                    self.position_entry_time = time.time()
                    self.position_entry_price = current_price
                    self.max_profit_reached = 0
                    
                elif signal == -1:  # 卖出信号
                    # 计算交易数量
                    balance = self.trader.get_balance()
                    available_balance = float(balance['free'])
                    
                    # 从config获取交易金额百分比
                    symbol_config = self.trader.symbol_config
                    trade_percent = symbol_config.get('trade_amount_percent', 50)
                    
                    # 计算交易金额
                    trade_amount = (available_balance * trade_percent / 100) / current_price
                    
                    # 开空仓
                    self.trader.open_short(amount=trade_amount)
                    self.logger.info(f"开空仓 - 数量: {trade_amount:.6f}, 价格: {current_price}")
                    
                    # 记录开仓信息
                    self.position_entry_time = time.time()
                    self.position_entry_price = current_price
                    self.max_profit_reached = 0
            
            # 如果有持仓，检查是否需要平仓
            else:
                position_amount = float(position['info'].get('positionAmt', 0))
                entry_price = float(position['info'].get('entryPrice', 0))
                current_price = self.trader.get_market_price()
                position_side = "多" if position_amount > 0 else "空"
                
                # 计算持仓时间
                current_time = time.time()
                if self.position_entry_time is not None:
                    holding_time_minutes = (current_time - self.position_entry_time) / 60
                    
                    # 检查最大持仓时间
                    if holding_time_minutes >= self.max_position_hold_time:
                        self.logger.info(f"持仓时间超过{self.max_position_hold_time}分钟，平仓")
                        self.trader.close_position()
                        return
                
                # 计算利润率
                if position_side == "多":
                    profit_rate = (current_price - entry_price) / entry_price
                else:
                    profit_rate = (entry_price - current_price) / entry_price
                
                # 更新最大利润记录
                if profit_rate > self.max_profit_reached:
                    self.max_profit_reached = profit_rate
                
                # 检查止盈
                if profit_rate >= self.take_profit_pct:
                    self.logger.info(f"达到止盈条件，利润率: {profit_rate:.4%}，平仓")
                    self.trader.close_position()
                    return
                
                # 检查止损
                if profit_rate <= -self.stop_loss_pct:
                    self.logger.info(f"达到止损条件，亏损率: {profit_rate:.4%}，平仓")
                    self.trader.close_position()
                    return
                
                # 检查追踪止损
                if self.trailing_stop and profit_rate >= self.trailing_stop_activation:
                    # 计算回撤比例
                    drawdown = self.max_profit_reached - profit_rate
                    
                    # 如果回撤超过追踪止损距离，平仓止盈
                    if drawdown >= self.trailing_stop_distance:
                        self.logger.info(f"触发追踪止损，最大利润: {self.max_profit_reached:.4%}, " +
                                      f"当前利润: {profit_rate:.4%}, 回撤: {drawdown:.4%}")
                        self.trader.close_position()
                        return
                
                # 获取最新趋势信号，检查趋势是否反转
                klines = self.trader.get_klines(interval=self.kline_interval, limit=self.lookback_period)
                trend_result = self.calculate_trend_score(klines)
                trend_score = trend_result['total_score']
                
                # 如果趋势明显反转，考虑平仓
                if position_side == "多" and trend_score < -0.5:
                    self.logger.info(f"趋势明显反转 (得分: {trend_score:.4f})，平多仓")
                    self.trader.close_position()
                    return
                elif position_side == "空" and trend_score > 0.5:
                    self.logger.info(f"趋势明显反转 (得分: {trend_score:.4f})，平空仓")
                    self.trader.close_position()
                    return
                    
                # 如果趋势减弱且利润已经较高，考虑平仓保护利润
                if position_side == "多" and trend_score < 0.2 and profit_rate > self.take_profit_pct * 0.7:
                    self.logger.info(f"趋势减弱 (得分: {trend_score:.4f}) 且已有可观利润 ({profit_rate:.4%})，平多仓保护收益")
                    self.trader.close_position()
                    return
                elif position_side == "空" and trend_score > -0.2 and profit_rate > self.take_profit_pct * 0.7:
                    self.logger.info(f"趋势减弱 (得分: {trend_score:.4f}) 且已有可观利润 ({profit_rate:.4%})，平空仓保护收益")
                    self.trader.close_position()
                    return
            
        except Exception as e:
            self.logger.error(f"监控持仓失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())

    def predict_trend(self, klines, horizon=3):
        """
        预测未来趋势
        
        Args:
            klines (list): K线数据
            horizon (int): 预测未来多少个K线周期的趋势
            
        Returns:
            dict: 趋势预测结果，包含方向、强度、信心度等
        """
        try:
            # 计算当前趋势得分
            trend_result = self.calculate_trend_score(klines)
            
            # 分析市场状态
            market_state = self.analyze_market_state(klines)
            
            # 获取趋势得分和方向
            current_score = trend_result['total_score']
            current_trend = trend_result['trend']
            
            # 计算动量和加速度
            df = self._prepare_dataframe(klines)
            df = self._calculate_indicators(df)
            
            # 计算趋势加速度（趋势得分变化率）
            if len(self.signal_history) >= 3:
                recent_scores = [record['score'] for record in self.signal_history[-3:]]
                score_acceleration = (recent_scores[-1] - recent_scores[0]) / 3
            else:
                # 使用EMA斜率作为替代
                score_acceleration = df['ema_short_slope'].iloc[-1] / 100
            
            # 趋势惯性：当前趋势的持续性预期
            trend_inertia = min(1.0, self.trend_duration / 10) * 0.5
            
            # 趋势强度：市场状态中的趋势强度
            trend_strength = market_state['strength'] / 100
            
            # 趋势稳定性：最近几个K线的一致性
            trend_stability = abs(self._calculate_trend_consistency(df, 5))
            
            # 结合当前得分和各因素预测未来趋势
            # 基本预测公式：future_score = current_score + acceleration * horizon + inertia - decay
            
            # 趋势衰减：较长时间同向趋势会逐渐衰减
            trend_decay = max(0, (self.trend_duration - 10) / 10) * 0.1 * np.sign(current_score)
            
            # 预测未来趋势得分
            future_score = current_score + score_acceleration * horizon + trend_inertia * np.sign(current_score) - trend_decay
            
            # 根据市场状态调整预测
            if market_state['state'] == "range":
                # 震荡市场中趋势更容易反转
                future_score *= (1 - horizon * 0.1)
            elif market_state['state'] == "trend" and trend_strength > 0.6:
                # 强趋势市场中趋势更倾向于延续
                future_score *= (1 + trend_strength * 0.2)
            
            # 限制在[-1, 1]范围内
            future_score = np.clip(future_score, -1, 1)
            
            # 确定预测的趋势方向
            if future_score > self.min_signal_score:
                predicted_trend = 1
            elif future_score < -self.min_signal_score:
                predicted_trend = -1
            else:
                predicted_trend = 0
            
            # 计算预测的信心度
            if market_state['state'] == "trend":
                confidence_base = 0.7
            elif market_state['state'] == "range":
                confidence_base = 0.5
            else:
                confidence_base = 0.6
                
            # 根据趋势强度和稳定性调整信心度
            confidence = confidence_base * (0.5 + 0.5 * trend_strength) * (0.5 + 0.5 * trend_stability)
            
            # 长期趋势方向：基于更长周期的均线
            long_trend = 0
            if df['ema_trend'].iloc[-1] > df['ema_trend'].iloc[-10]:
                long_trend = 1
            elif df['ema_trend'].iloc[-1] < df['ema_trend'].iloc[-10]:
                long_trend = -1
                
            # 如果短期预测与长期趋势相反，降低信心度
            if predicted_trend * long_trend < 0:
                confidence *= 0.8
            
            self.logger.info(f"趋势预测结果 (未来{horizon}个周期):")
            self.logger.info(f"  当前趋势得分: {current_score:.4f}, 加速度: {score_acceleration:.4f}")
            self.logger.info(f"  预测趋势得分: {future_score:.4f}, 方向: {predicted_trend}, 信心度: {confidence:.4f}")
            
            # 返回预测结果
            return {
                'current_score': float(current_score),
                'predicted_score': float(future_score),
                'predicted_trend': predicted_trend,
                'confidence': float(confidence),
                'market_state': market_state['state'],
                'horizon': horizon,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            self.logger.error(f"趋势预测失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {
                'predicted_trend': 0,
                'confidence': 0,
                'error': str(e)
            }
