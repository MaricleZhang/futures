import numpy as np
import pandas as pd
import talib
import time
import logging
from datetime import datetime
import config
from strategies.base_strategy import BaseStrategy

class MediumTrendStrategy(BaseStrategy):
    """MediumTrendStrategy - 中期趋势策略
    
    基于15分钟K线数据的趋势跟踪策略，结合多种技术指标和市场结构分析，
    专注于捕捉中期趋势机会，同时添加反转信号检测功能以提前识别趋势转变点。
    
    特点:
    1. 中期趋势跟踪: 使用15分钟K线识别和跟踪市场趋势
    2. 多重趋势确认: 结合EMA、MACD、ADX等多个指标交叉验证趋势方向
    3. 波动率适应: 通过ATR动态调整进出场条件
    4. 反转信号检测: 识别潜在的趋势反转点
    5. 动态止盈止损: 根据市场波动性和趋势强度调整止盈止损条件
    """
    
    def __init__(self, trader):
        """初始化中期趋势策略"""
        super().__init__(trader)
        self.logger = self.get_logger()
        
        # K线设置
        self.kline_interval = '5m'     # 15分钟K线
        self.check_interval = 300       # 检查信号间隔(秒)
        self.lookback_period = 100      # 计算指标所需的K线数量
        self.training_lookback = 100    # 提供该属性以兼容TradingManager
        
        # 趋势参数
        self.ema_short_period = 8        # 短期EMA周期
        self.ema_medium_period = 21      # 中期EMA周期
        self.ema_long_period = 55        # 长期EMA周期
        self.adx_period = 14             # ADX周期
        self.adx_threshold = 25          # ADX阈值，高于此值认为有趋势
        self.macd_fast = 12              # MACD快线周期
        self.macd_slow = 26              # MACD慢线周期
        self.macd_signal = 9             # MACD信号线周期
        
        # 波动率参数
        self.atr_period = 14             # ATR周期
        self.atr_multiplier = 2.0        # ATR乘数，用于止损计算
        
        # 成交量参数
        self.volume_ma_period = 10       # 成交量均线周期
        self.volume_threshold = 1.2      # 成交量阈值，高于此值认为量能充足
        
        # 反转信号参数
        self.reversal_sensitivity = 0.65  # 反转信号灵敏度（0-1）
        self.reversal_confirmation = 2    # 反转信号确认所需的指标数量
        
        # 交易控制参数
        self.max_position_hold_time = 720   # 最大持仓时间(分钟)
        self.profit_target_pct = 0.015      # 目标利润率 1.5%
        self.stop_loss_pct = 0.008          # 止损率 0.8%
        self.max_trades_per_day = 4         # 每日最大交易次数
        
        # 交易状态
        self.trade_count_day = 0         # 当前日交易次数
        self.last_trade_day = None       # 上次交易的日期
        self.position_entry_time = None  # 开仓时间
        self.position_entry_price = None # 开仓价格
        
        # 兼容TradingManager的方法
        self.last_training_time = time.time()
        
        # 添加上一次信号记录
        self.last_signal = 0  # 初始化为观望信号
        self.reversal_detected = False  # 反转信号检测状态
        self.reversal_direction = 0     # 反转方向（1=向上反转，-1=向下反转）
        
    def calculate_indicators(self, klines):
        """计算技术指标"""
        try:
            if not isinstance(klines, list) or len(klines) < 50:
                self.logger.error("K线数据为空或长度不足")
                return None
            
            # 创建DataFrame
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 创建指标DataFrame
            indicators = {}
            
            # === 1. 移动平均线指标 ===
            indicators['ema_short'] = df['close'].ewm(span=self.ema_short_period, adjust=False).mean()
            indicators['ema_medium'] = df['close'].ewm(span=self.ema_medium_period, adjust=False).mean()
            indicators['ema_long'] = df['close'].ewm(span=self.ema_long_period, adjust=False).mean()
            
            # EMA交叉和距离
            indicators['ema_short_medium_cross'] = indicators['ema_short'] - indicators['ema_medium']
            indicators['ema_medium_long_cross'] = indicators['ema_medium'] - indicators['ema_long']
            
            # 价格相对于EMA
            indicators['price_to_ema_short'] = df['close'] / indicators['ema_short'] - 1
            indicators['price_to_ema_medium'] = df['close'] / indicators['ema_medium'] - 1
            indicators['price_to_ema_long'] = df['close'] / indicators['ema_long'] - 1
            
            # === 2. 趋势强度指标 - ADX ===
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            
            indicators['adx'] = pd.Series(talib.ADX(high, low, close, timeperiod=self.adx_period), index=df.index)
            indicators['plus_di'] = pd.Series(talib.PLUS_DI(high, low, close, timeperiod=self.adx_period), index=df.index)
            indicators['minus_di'] = pd.Series(talib.MINUS_DI(high, low, close, timeperiod=self.adx_period), index=df.index)
            
            # ADX斜率 - 用于判断趋势加速/减速
            indicators['adx_slope'] = indicators['adx'].diff(3) / 3
            
            # === 3. MACD指标 ===
            macd, macd_signal, macd_hist = talib.MACD(
                close, 
                fastperiod=self.macd_fast, 
                slowperiod=self.macd_slow, 
                signalperiod=self.macd_signal
            )
            indicators['macd'] = pd.Series(macd, index=df.index)
            indicators['macd_signal'] = pd.Series(macd_signal, index=df.index)
            indicators['macd_hist'] = pd.Series(macd_hist, index=df.index)
            indicators['macd_hist_slope'] = pd.Series(macd_hist, index=df.index).diff(3)
            
            # === 4. 波动率指标 - ATR ===
            indicators['atr'] = pd.Series(talib.ATR(high, low, close, timeperiod=self.atr_period), index=df.index)
            indicators['atr_pct'] = indicators['atr'] / df['close'] * 100  # 百分比表示
            
            # === 5. RSI指标 ===
            indicators['rsi'] = pd.Series(talib.RSI(close, timeperiod=14), index=df.index)
            indicators['rsi_slope'] = indicators['rsi'].diff(3)
            
            # === 6. 布林带指标 ===
            bbands_period = 20
            upper_band, middle_band, lower_band = talib.BBANDS(
                close, timeperiod=bbands_period, nbdevup=2, nbdevdn=2
            )
            indicators['bb_upper'] = pd.Series(upper_band, index=df.index)
            indicators['bb_middle'] = pd.Series(middle_band, index=df.index)
            indicators['bb_lower'] = pd.Series(lower_band, index=df.index)
            indicators['bb_width'] = (indicators['bb_upper'] - indicators['bb_lower']) / indicators['bb_middle']
            indicators['bb_position'] = (df['close'] - indicators['bb_lower']) / (indicators['bb_upper'] - indicators['bb_lower'])
            
            # === 7. 成交量指标 ===
            indicators['volume'] = df['volume']
            indicators['volume_ma'] = df['volume'].rolling(window=self.volume_ma_period).mean()
            indicators['volume_ratio'] = df['volume'] / indicators['volume_ma']
            
            # 价量关系
            indicators['price_volume_corr'] = df['close'].pct_change().rolling(10).corr(df['volume'].pct_change())
            
            # === 8. 动量指标 ===
            indicators['roc_12'] = pd.Series(talib.ROC(close, timeperiod=12), index=df.index)
            indicators['cci'] = pd.Series(talib.CCI(high, low, close, timeperiod=14), index=df.index)
            
            # 随机指标
            indicators['slowk'], indicators['slowd'] = talib.STOCH(
                high, low, close, fastk_period=14, slowk_period=3, slowd_period=3
            )
            indicators['slowk'] = pd.Series(indicators['slowk'], index=df.index)
            indicators['slowd'] = pd.Series(indicators['slowd'], index=df.index)
            
            # === 9. 综合趋势评分 ===
            # 初始化趋势评分为0
            trend_score = pd.Series(0.0, index=df.index)
            
            # 添加EMA趋势分量 (-1 到 1)
            ema_trend = pd.Series(0.0, index=df.index)
            ema_trend[indicators['ema_short'] > indicators['ema_medium']] += 0.4
            ema_trend[indicators['ema_medium'] > indicators['ema_long']] += 0.4
            ema_trend[indicators['ema_short'] < indicators['ema_medium']] -= 0.4
            ema_trend[indicators['ema_medium'] < indicators['ema_long']] -= 0.4
            
            # 添加ADX分量 (0 到 0.5, 根据ADX强度)
            adx_factor = indicators['adx'] / 100  # 归一化ADX (0-1)
            adx_trend = pd.Series(0.0, index=df.index)
            adx_trend[indicators['plus_di'] > indicators['minus_di']] = adx_factor * 0.5
            adx_trend[indicators['plus_di'] < indicators['minus_di']] = -adx_factor * 0.5
            
            # 添加MACD分量 (-0.4 到 0.4)
            macd_factor = indicators['macd_hist'] / (df['close'] * 0.01)  # 相对于价格1%归一化
            macd_factor = macd_factor.clip(-0.4, 0.4)  # 限制在-0.4到0.4之间
            
            # 添加RSI分量 (-0.4 到 0.4)
            rsi_factor = (indicators['rsi'] - 50) / 40  # 归一化为-1.25到1.25
            rsi_factor = rsi_factor.clip(-0.4, 0.4)  # 限制在-0.4到0.4之间
            
            # 添加成交量确认分量 (0 到 0.3)
            volume_factor = (indicators['volume_ratio'] - 1).clip(-1, 1) * 0.3
            
            # 组合所有分量
            self.logger.info(f'EMA Trend: {ema_trend.iloc[-1]:.4f}')
            self.logger.info(f'ADX Trend: {adx_trend.iloc[-1]:.4f}')
            self.logger.info(f'MACD Factor: {macd_factor.iloc[-1]:.4f}')
            self.logger.info(f'RSI Factor: {rsi_factor.iloc[-1]:.4f}')
            self.logger.info(f'Volume Factor: {volume_factor.iloc[-1]:.4f}')
            trend_score = ema_trend + adx_trend + macd_factor + rsi_factor + volume_factor
            self.logger.info(f'Final Trend Score: {trend_score.iloc[-1]:.4f}')
            
            # 总分在-2到2之间，归一化到-1到1
            trend_score = trend_score.clip(-1, 1)
            
            indicators['trend_score'] = trend_score
            
            # 趋势方向 (1: 上升, -1: 下降, 0: 盘整)
            trend_direction = pd.Series(0, index=df.index)
            trend_direction[trend_score > 0.3] = 1
            trend_direction[trend_score < -0.3] = -1
            
            indicators['trend_direction'] = trend_direction
            
            # === 10. 趋势持续性评估 ===
            # 计算价格波动的一致性
            price_changes = df['close'].pct_change(3)
            indicators['price_consistency'] = price_changes.rolling(5).apply(
                lambda x: np.sign(x).sum() / 5  # 结果在-1到1之间
            )
            
            # === 11. 趋势衰竭指标 ===
            # 价格与趋势方向一致性检查
            price_adv = df['close'] > df['close'].shift(1)
            indicators['trend_exhaustion'] = pd.Series(0, index=df.index)
            
            # 使用直接赋值而不是.loc操作，防止dict对象报错
            # 计算趋势衰竭指标
            indicators['trend_exhaustion'] = pd.Series(0, index=df.index, dtype='float64')
            for i in range(len(df)):
                # 如果是上升趋势
                if trend_direction.iloc[i] == 1 and indicators['rsi'].iloc[i] > 50:
                    indicators['trend_exhaustion'].iloc[i] = min((indicators['rsi'].iloc[i] - 50) / 20, 1)
                # 如果是下降趋势
                elif trend_direction.iloc[i] == -1 and indicators['rsi'].iloc[i] < 50:
                    indicators['trend_exhaustion'].iloc[i] = min((50 - indicators['rsi'].iloc[i]) / 20, 1)
            
            return indicators, df
            
        except Exception as e:
            self.logger.error(f"计算指标失败: {str(e)}")
            return None, None
    
    def detect_reversal(self, klines):
        """
        检测市场是否处于反转状态
        
        Args:
            klines (list): K线数据，格式为 [[timestamp, open, high, low, close, volume], ...]
        
        Returns:
            tuple: (is_reversed, direction, strength)
                is_reversed (bool): 是否检测到反转
                direction (int): 反转方向，1 为向上反转（由空转多），-1 为向下反转（由多转空）
                strength (float): 反转信号强度，范围 0-1
        """
        try:
            # 计算技术指标
            indicators, df = self.calculate_indicators(klines)
            if indicators is None or df is None:
                return False, 0, 0
            
            # 获取最近几根K线的指标
            latest_idx = -1  # 最新的K线
            
            # 1. 基于价格行为的反转信号
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            open_price = df['open'].values
            
            # 检查是否形成锤子线或吊颈线形态
            hammer_detected = False
            if (close[-1] > open_price[-1] and  # 阳线
                (close[-1] - open_price[-1]) < (high[-1] - low[-1]) * 0.3 and  # 实体较小
                (open_price[-1] - low[-1]) > (high[-1] - open_price[-1]) * 2):  # 下影线明显
                hammer_detected = True
                
            shooting_star_detected = False
            if (close[-1] < open_price[-1] and  # 阴线
                (open_price[-1] - close[-1]) < (high[-1] - low[-1]) * 0.3 and  # 实体较小
                (high[-1] - open_price[-1]) > (open_price[-1] - low[-1]) * 2):  # 上影线明显
                shooting_star_detected = True
            
            # 2. 基于RSI的反转信号
            rsi = indicators['rsi'].iloc[latest_idx]
            rsi_prev = indicators['rsi'].iloc[latest_idx-1]
            rsi_prev2 = indicators['rsi'].iloc[latest_idx-2]
            rsi_slope = indicators['rsi_slope'].iloc[latest_idx]
            
            # 超卖反转信号 (RSI < 30 且开始回升)
            oversold_reversal = (rsi < 35 and rsi_slope > 0 and rsi > rsi_prev)
            
            # 超买反转信号 (RSI > 70 且开始下降)
            overbought_reversal = (rsi > 65 and rsi_slope < 0 and rsi < rsi_prev)
            
            # RSI背离检查
            price_high_idx = df['high'].iloc[-5:].idxmax()
            price_low_idx = df['low'].iloc[-5:].idxmin()
            
            rsi_high_idx = indicators['rsi'].iloc[-5:].idxmax()
            rsi_low_idx = indicators['rsi'].iloc[-5:].idxmin()
            
            # 看跌背离 - 价格创新高但RSI未能创新高
            bearish_divergence = (price_high_idx > rsi_high_idx and 
                                df['close'].iloc[-1] > df['close'].iloc[-2] and
                                indicators['rsi'].iloc[-1] < indicators['rsi'].iloc[rsi_high_idx])
            
            # 看涨背离 - 价格创新低但RSI未能创新低
            bullish_divergence = (price_low_idx > rsi_low_idx and 
                                df['close'].iloc[-1] < df['close'].iloc[-2] and
                                indicators['rsi'].iloc[-1] > indicators['rsi'].iloc[rsi_low_idx])
            
            # 3. 基于MACD的反转信号
            macd_hist = indicators['macd_hist'].iloc[latest_idx]
            macd_hist_prev = indicators['macd_hist'].iloc[latest_idx-1]
            macd_hist_prev2 = indicators['macd_hist'].iloc[latest_idx-2]
            macd_hist_slope = indicators['macd_hist_slope'].iloc[latest_idx]
            
            # MACD柱状图由负转正
            macd_bullish_cross = (macd_hist_prev < 0 and macd_hist > 0)
            
            # MACD柱状图由正转负
            macd_bearish_cross = (macd_hist_prev > 0 and macd_hist < 0)
            
            # MACD背离
            macd_bearish_div = (df['close'].iloc[-1] > df['close'].iloc[-3] and
                              indicators['macd_hist'].iloc[-1] < indicators['macd_hist'].iloc[-3])
            
            macd_bullish_div = (df['close'].iloc[-1] < df['close'].iloc[-3] and
                              indicators['macd_hist'].iloc[-1] > indicators['macd_hist'].iloc[-3])
            
            # 4. 基于布林带的反转信号
            bb_position = indicators['bb_position'].iloc[latest_idx]
            bb_width = indicators['bb_width'].iloc[latest_idx]
            bb_width_prev = indicators['bb_width'].iloc[latest_idx-5:latest_idx].mean()
            
            # 价格触及布林带边界
            price_at_upper_band = bb_position > 0.95
            price_at_lower_band = bb_position < 0.05
            
            # 布林带宽度变化 - 挤压后扩展表示可能出现反转
            bb_squeeze_expansion = (bb_width > bb_width_prev * 1.2 and 
                                  bb_width_prev < indicators['bb_width'].iloc[-10:-5].mean() * 0.8)
            
            # 5. 基于随机指标的反转信号
            stoch_k = indicators['slowk'].iloc[latest_idx]
            stoch_d = indicators['slowd'].iloc[latest_idx]
            stoch_k_prev = indicators['slowk'].iloc[latest_idx-1]
            stoch_d_prev = indicators['slowd'].iloc[latest_idx-1]
            
            # 随机指标超买超卖区反转
            stoch_oversold_reversal = (stoch_k < 20 and stoch_k > stoch_k_prev and stoch_k > stoch_d)
            stoch_overbought_reversal = (stoch_k > 80 and stoch_k < stoch_k_prev and stoch_k < stoch_d)
            
            # 6. 趋势指标反转
            trend_score = indicators['trend_score'].iloc[latest_idx]
            trend_score_prev = indicators['trend_score'].iloc[latest_idx-1]
            trend_score_change = trend_score - trend_score_prev
            
            trend_direction = indicators['trend_direction'].iloc[latest_idx]
            trend_direction_prev = indicators['trend_direction'].iloc[latest_idx-1]
            
            # 趋势方向变化
            trend_direction_change = (trend_direction != trend_direction_prev and trend_direction_prev != 0)
            
            # 趋势衰竭指标
            trend_exhaustion = indicators['trend_exhaustion'].iloc[latest_idx]
            
            # 7. 交易量突变信号
            volume = df['volume'].iloc[latest_idx]
            volume_ratio = indicators['volume_ratio'].iloc[latest_idx]
            volume_spike = volume_ratio > 1.5  # 交易量是平均的1.5倍以上
            
            # 8. 波动率变化信号
            atr_pct = indicators['atr_pct'].iloc[latest_idx]
            atr_pct_avg = indicators['atr_pct'].iloc[latest_idx-5:latest_idx].mean()
            volatility_spike = atr_pct > atr_pct_avg * 1.3  # 波动率是平均的1.3倍以上
            
            # 9. ADX趋势强度变化
            adx = indicators['adx'].iloc[latest_idx]
            adx_prev = indicators['adx'].iloc[latest_idx-1]
            adx_slope = indicators['adx_slope'].iloc[latest_idx]
            
            # 趋势强度见顶或见底
            adx_peaking = (adx > 25 and adx < adx_prev and adx_prev < indicators['adx'].iloc[latest_idx-2])
            
            # 计算看涨信号数量
            bullish_signals = [
                1 if oversold_reversal else 0,              # RSI超卖反转
                1 if bullish_divergence else 0,             # RSI看涨背离
                1 if macd_bullish_cross else 0,             # MACD金叉
                1 if macd_bullish_div else 0,               # MACD看涨背离
                1 if price_at_lower_band else 0,            # 价格触及布林带下轨
                1 if stoch_oversold_reversal else 0,        # 随机指标超卖反转
                1 if (trend_score < 0 and trend_score_change > 0.1) else 0,  # 下降趋势减弱
                1 if (trend_direction == -1 and trend_exhaustion > 0.7) else 0,  # 下降趋势衰竭
                1 if (hammer_detected and trend_direction <= 0) else 0,  # 在下降趋势中出现锤子线
                1 if (volume_spike and df['close'].iloc[-1] > df['open'].iloc[-1]) else 0  # 放量阳线
            ]
            
            # 计算看跌信号数量
            bearish_signals = [
                1 if overbought_reversal else 0,            # RSI超买反转
                1 if bearish_divergence else 0,             # RSI看跌背离
                1 if macd_bearish_cross else 0,             # MACD死叉
                1 if macd_bearish_div else 0,               # MACD看跌背离
                1 if price_at_upper_band else 0,            # 价格触及布林带上轨
                1 if stoch_overbought_reversal else 0,      # 随机指标超买反转
                1 if (trend_score > 0 and trend_score_change < -0.1) else 0,  # 上升趋势减弱
                1 if (trend_direction == 1 and trend_exhaustion > 0.7) else 0,  # 上升趋势衰竭
                1 if (shooting_star_detected and trend_direction >= 0) else 0,  # 在上升趋势中出现吊颈线
                1 if (volume_spike and df['close'].iloc[-1] < df['open'].iloc[-1]) else 0  # 放量阴线
            ]
            
            # 计算权重
            weights = [0.15, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05]
            
            # 计算加权得分
            bullish_score = sum(s * w for s, w in zip(bullish_signals, weights))
            bearish_score = sum(s * w for s, w in zip(bearish_signals, weights))
            
            # 调整反转确认阈值
            reversal_threshold = self.reversal_sensitivity
            
            # 确定是否有反转信号
            is_bullish_reversal = bullish_score > reversal_threshold and bullish_score > bearish_score * 1.5
            is_bearish_reversal = bearish_score > reversal_threshold and bearish_score > bullish_score * 1.5
            
            # 计算有多少个信号被触发
            bullish_signal_count = sum(bullish_signals)
            bearish_signal_count = sum(bearish_signals)
            
            # 只有当足够多的信号同时被触发时才确认反转
            if is_bullish_reversal and bullish_signal_count >= self.reversal_confirmation:
                self.logger.info(f"检测到看涨反转信号，强度: {bullish_score:.2f}, 信号数: {bullish_signal_count}")
                for i, signal in enumerate(bullish_signals):
                    if signal:
                        self.logger.info(f"看涨信号 #{i+1} 被触发")
                return True, 1, bullish_score
            elif is_bearish_reversal and bearish_signal_count >= self.reversal_confirmation:
                self.logger.info(f"检测到看跌反转信号，强度: {bearish_score:.2f}, 信号数: {bearish_signal_count}")
                for i, signal in enumerate(bearish_signals):
                    if signal:
                        self.logger.info(f"看跌信号 #{i+1} 被触发")
                return True, -1, bearish_score
            else:
                # 如果信号强度不够，但接近阈值，记录警告
                if bullish_score > reversal_threshold * 0.7:
                    self.logger.info(f"潜在看涨反转信号，强度: {bullish_score:.2f}，未达确认阈值")
                if bearish_score > reversal_threshold * 0.7:
                    self.logger.info(f"潜在看跌反转信号，强度: {bearish_score:.2f}，未达确认阈值")
                return False, 0, max(bullish_score, bearish_score)
            
        except Exception as e:
            self.logger.error(f"检测反转失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False, 0, 0
    
    def generate_signal(self, klines):
        """生成交易信号
        返回值：
        -1: 卖出信号
        0: 观望信号
        1: 买入信号
        """
        try:
            # 首先检测是否有反转信号
            is_reversed, reversal_direction, reversal_strength = self.detect_reversal(klines)
            
            # 保存反转信号状态
            self.reversal_detected = is_reversed
            self.reversal_direction = reversal_direction
            
            # 如果检测到强烈的反转信号
            if is_reversed and reversal_strength > 0.75:
                self.logger.info(f"检测到强烈的{'向上' if reversal_direction > 0 else '向下'}反转信号，强度: {reversal_strength:.4f}")
                return reversal_direction  # 返回反转信号
            
            # 计算指标
            indicators, df = self.calculate_indicators(klines)
            if indicators is None or df is None:
                return 0
            
            # 获取最新指标值
            latest_idx = -1
            
            trend_score = indicators['trend_score'].iloc[latest_idx]
            trend_direction = indicators['trend_direction'].iloc[latest_idx]
            adx = indicators['adx'].iloc[latest_idx]
            
            self.logger.info(f"趋势评分: {trend_score:.4f}, 趋势方向: {trend_direction}, 趋势强度(ADX): {adx:.2f}")
            
            # 轻微反转处理
            if is_reversed and reversal_strength > 0.6:
                self.logger.info(f"检测到{'向上' if reversal_direction > 0 else '向下'}反转信号，强度: {reversal_strength:.4f}")
                if (self.last_signal == 1 and reversal_direction < 0) or (self.last_signal == -1 and reversal_direction > 0):
                    self.logger.info("检测到趋势反转信号，与当前持仓方向相反")
                    return reversal_direction
            
            # 获取当前持仓
            position = self.trader.get_position()
            position_side = None
            if position and 'info' in position:
                position_amount = float(position['info'].get('positionAmt', 0))
                if position_amount > 0:
                    position_side = "多"
                elif position_amount < 0:
                    position_side = "空"
                
                # 检查趋势与持仓方向是否一致
                if (position_side == "多" and trend_score < -0.3) :
                    self.logger.info("趋势方向与持仓方向不一致，生成反转信号-1")
                    return -1
                if (position_side == "空" and trend_score > 0.3) :
                    self.logger.info("趋势方向与持仓方向不一致，生成反转信号1")
                    return 1
            
            # 检查交易频率限制
            current_day = datetime.now().day
            if current_day != self.last_trade_day:
                self.last_trade_day = current_day
                self.trade_count_day = 0
            
            # if self.trade_count_day >= self.max_trades_per_day:
            #     self.logger.info(f"已达到每日最大交易次数({self.max_trades_per_day})，今日不再开新仓")
            #     # 仍然可以生成平仓信号
            #     if position_side and ((position_side == "多" and trend_score < -0.3) or (position_side == "空" and trend_score > 0.3)):
            #         return 2
                return 0
            
            # 判断是否为强趋势市场
            strong_trend = adx >= self.adx_threshold
            
            # 获取其他重要指标
            rsi = indicators['rsi'].iloc[latest_idx]
            macd_hist = indicators['macd_hist'].iloc[latest_idx]
            volume_ratio = indicators['volume_ratio'].iloc[latest_idx]
            price_to_ema_medium = indicators['price_to_ema_medium'].iloc[latest_idx]
            self.logger.info(f'RSI: {float(rsi):.2f}')
            self.logger.info(f'MACD Histogram: {float(macd_hist):.6f}')
            self.logger.info(f'Volume Ratio: {float(volume_ratio):.2f}')
            self.logger.info(f'Price to EMA Medium: {float(price_to_ema_medium):.4f}')
            
            # 根据趋势强度和方向生成信号
            if trend_direction > 0 and trend_score > 0.4:
                # 买入前的额外过滤条件
                # if (rsi < 70 and  # RSI不超买
                #     volume_ratio >= 0.8 and  # 成交量充足
                #     macd_hist > 0):  # MACD柱状图为正
                    self.logger.info(f"生成买入信号 (RSI={rsi:.2f}, 成交量比={volume_ratio:.2f})")
                    signal = 1
                    self.last_signal = signal
                    self.trade_count_day += 1
                # else:
                #     self.logger.info("虽有上升趋势，但未满足买入条件")
                #     signal = 0
            elif trend_direction < 0 and trend_score < -0.4:
                # 卖出前的额外过滤条件
                # if (rsi > 30 and  # RSI不超卖
                #     volume_ratio >= 0.8 and  # 成交量充足
                #     macd_hist < 0):  # MACD柱状图为负
                    self.logger.info(f"生成卖出信号 (RSI={rsi:.2f}, 成交量比={volume_ratio:.2f})")
                    signal = -1
                    self.last_signal = signal
                    self.trade_count_day += 1
                # else:
                #     self.logger.info("虽有下降趋势，但未满足卖出条件")
                #     signal = 0
            else:
                # 无明确趋势或趋势不够强
                signal = 0
                
                # 检查价格是否远离均线 - 可能是回调买入/卖出机会
                if trend_score > 0.2 and price_to_ema_medium < -0.01 and rsi > 40 and rsi < 60:
                    self.logger.info(f"价格回调到均线下方，可能是买入机会 (价格/均线偏差={price_to_ema_medium:.4f})")
                    signal = 1
                    self.last_signal = signal
                    self.trade_count_day += 1
                elif trend_score < -0.2 and price_to_ema_medium > 0.01 and rsi > 40 and rsi < 60:
                    self.logger.info(f"价格反弹到均线上方，可能是卖出机会 (价格/均线偏差={price_to_ema_medium:.4f})")
                    signal = -1
                    self.last_signal = signal
                    self.trade_count_day += 1
            
            return signal
            
        except Exception as e:
            self.logger.error(f"生成信号失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return 0  # 发生错误时返回观望信号
            
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
                
                # 检查止盈
                if profit_rate >= self.profit_target_pct:
                    self.logger.info(f"达到止盈条件，利润率: {profit_rate:.4%}，平仓")
                    self.trader.close_position()
                    return
                
                # 检查止损
                if profit_rate <= -self.stop_loss_pct:
                    self.logger.info(f"达到止损条件，亏损率: {profit_rate:.4%}，平仓")
                    self.trader.close_position()
                    return
                
                # 检查反转信号
                klines = self.trader.get_klines(interval=self.kline_interval, limit=self.lookback_period)
                indicators, _ = self.calculate_indicators(klines)
                
                if indicators is not None:
                    # 获取最新指标
                    latest_idx = len(indicators['trend_direction']) - 1
                    trend_direction = indicators['trend_direction'].iloc[latest_idx]
                    trend_score = indicators['trend_score'].iloc[latest_idx]
                    rsi = indicators['rsi'].iloc[latest_idx]
                    
                    # 检查反转信号
                    is_reversed, reversal_direction, _ = self.detect_reversal(klines)
                    
                    # 多仓反转条件
                    if position_side == "多" and (
                        (is_reversed and reversal_direction < 0) or  # 检测到向下反转
                        trend_direction < 0 or                       # 趋势方向向下
                        rsi > 75                                     # RSI超买
                    ):
                        self.logger.info("多仓趋势反转信号，平仓")
                        self.trader.close_position()
                        return
                    
                    # 空仓反转条件
                    if position_side == "空" and (
                        (is_reversed and reversal_direction > 0) or  # 检测到向上反转
                        trend_direction > 0 or                       # 趋势方向向上
                        rsi < 25                                     # RSI超卖
                    ):
                        self.logger.info("空仓趋势反转信号，平仓")
                        self.trader.close_position()
                        return
                    
                    # 增强的反转信号检测
                    rapid_reversal = False
                    
                    # 检查快速反转条件
                    if position_side == "多":
                        if (trend_score < -0.3 and self.last_signal > 0) or \
                           (indicators['macd_hist'].iloc[latest_idx] < 0 and indicators['macd_hist'].iloc[latest_idx-1] > 0):
                            rapid_reversal = True
                            
                    elif position_side == "空":
                        if (trend_score > 0.3 and self.last_signal < 0) or \
                           (indicators['macd_hist'].iloc[latest_idx] > 0 and indicators['macd_hist'].iloc[latest_idx-1] < 0):
                            rapid_reversal = True
                    
                    # 如果检测到快速反转，立即平仓
                    if rapid_reversal:
                        self.logger.info("检测到快速趋势反转，立即平仓")
                        self.trader.close_position()
                        return
                    
                    # 动态调整止损
                    atr_value = indicators['atr'].iloc[latest_idx]
                    current_volatility = atr_value / current_price
                    
                    # 根据波动率和趋势强度动态调整止损
                    if abs(trend_score) < 0.2:  # 趋势减弱时收紧止损
                        dynamic_stop_loss = min(self.stop_loss_pct * 0.7, current_volatility * 1.5)
                    else:
                        dynamic_stop_loss = min(self.stop_loss_pct, current_volatility * self.atr_multiplier)
                    
                    # 检查动态止损
                    if profit_rate <= -dynamic_stop_loss:
                        self.logger.info(f"触发动态止损，亏损率: {profit_rate:.4%}，平仓")
                        self.trader.close_position()
                        return
            
        except Exception as e:
            self.logger.error(f"监控持仓失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())