import numpy as np
import pandas as pd
import talib
import time
import config
from strategies.base_strategy import BaseStrategy

class SimpleTrendStrategy(BaseStrategy):
    """SimpleTrendStrategy - 简化趋势跟踪策略
    
    一个不使用机器学习模型的纯规则交易策略，专注于识别和跟踪市场趋势。
    使用多种趋势指标和动量分析，生成交易信号。
    
    特点：
    1. 多重趋势确认: 使用EMA、ADX、MACD等多个指标交叉验证趋势方向
    2. 波动率过滤: 通过ATR动态调整进出场条件
    3. 量价结合: 通过成交量确认价格趋势真实性
    4. 直观规则: 简单明了的规则生成交易信号，无机器学习黑盒
    5. 增强反转检测: 改进的信号反转检测，能快速识别市场反转点
    """
    
    def __init__(self, trader):
        """初始化简化趋势跟踪策略"""
        super().__init__(trader)
        self.logger = self.get_logger()
        
        # K线设置
        self.kline_interval = '3m'      # 3分钟K线
        self.check_interval = 60       # 检查信号间隔(秒)
        self.lookback_period = 100      # 计算指标所需的K线数量
        self.training_lookback = 100    # 提供该属性以兼容TradingManager
        
        # 趋势参数
        self.ema_short_period = 5       # 短期EMA周期
        self.ema_medium_period = 20     # 中期EMA周期
        self.ema_long_period = 50       # 长期EMA周期
        self.adx_period = 14            # ADX周期
        self.adx_threshold = 25         # ADX阈值，高于此值认为有趋势
        self.macd_fast = 12             # MACD快线周期
        self.macd_slow = 26             # MACD慢线周期
        self.macd_signal = 9            # MACD信号线周期
        
        # 波动率参数
        self.atr_period = 14            # ATR周期
        self.atr_multiplier = 2.0       # ATR乘数，用于止损计算
        
        # 成交量参数
        self.volume_ma_period = 10      # 成交量均线周期
        self.volume_threshold = 1.2     # 成交量阈值，高于此值认为量能充足
        
        # 交易控制参数
        self.max_position_hold_time = 150     # 最大持仓时间(分钟)
        self.profit_target_pct = 0.004       # 目标利润率 0.4%
        self.stop_loss_pct = 0.002           # 止损率 0.2%
        self.max_trades_per_hour = 4         # 每小时最大交易次数
        
        # 交易状态
        self.trade_count_hour = 0       # 当前小时交易次数
        self.last_trade_hour = None     # 上次交易的小时
        self.position_entry_time = None # 开仓时间
        self.position_entry_price = None # 开仓价格
        
        # 兼容TradingManager的方法
        self.last_training_time = time.time()
        
        # 添加信号历史记录 - 改进点1
        self.last_signal = 0  # 初始化为观望信号
        self.signal_history = [0] * 5  # 保存最近5个信号
        self.trend_score_history = []  # 保存趋势评分历史
        
        # 反转检测参数 - 改进点2
        self.reversal_sensitivity = 0.4  # 反转敏感度 (0-1)，越大越敏感
        self.consec_signals_for_reversal = 2  # 确认反转需要的连续信号数
        self.reversal_confirmation_threshold = 0.25  # 确认反转的阈值
        
    def calculate_indicators(self, klines):
        """计算技术指标"""
        try:
            if not isinstance(klines, list) or len(klines) < 50:
                self.logger.error("K线数据为空或长度不足")
                return None, None
            
            # 创建DataFrame
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 创建指标字典 - 使用numpy数组替代Series以避免iloc问题
            indicators = {}
            
            # === 1. 移动平均线指标 ===
            indicators['ema_short'] = df['close'].ewm(span=self.ema_short_period, adjust=False).mean().values
            indicators['ema_medium'] = df['close'].ewm(span=self.ema_medium_period, adjust=False).mean().values
            indicators['ema_long'] = df['close'].ewm(span=self.ema_long_period, adjust=False).mean().values
            
            # EMA交叉和距离
            indicators['ema_short_medium_cross'] = indicators['ema_short'] - indicators['ema_medium']
            indicators['ema_medium_long_cross'] = indicators['ema_medium'] - indicators['ema_long']
            
            # 价格相对于EMA
            indicators['price_to_ema_short'] = df['close'].values / indicators['ema_short'] - 1
            indicators['price_to_ema_medium'] = df['close'].values / indicators['ema_medium'] - 1
            
            # === 2. 趋势强度指标 - ADX ===
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            
            # 直接保存为numpy数组
            indicators['high'] = high
            indicators['low'] = low
            indicators['close'] = close
            indicators['open'] = df['open'].values
            
            indicators['adx'] = talib.ADX(high, low, close, timeperiod=self.adx_period)
            indicators['plus_di'] = talib.PLUS_DI(high, low, close, timeperiod=self.adx_period)
            indicators['minus_di'] = talib.MINUS_DI(high, low, close, timeperiod=self.adx_period)
            
            # === 3. MACD指标 ===
            macd, macd_signal, macd_hist = talib.MACD(
                close, 
                fastperiod=self.macd_fast, 
                slowperiod=self.macd_slow, 
                signalperiod=self.macd_signal
            )
            indicators['macd'] = macd
            indicators['macd_signal'] = macd_signal
            indicators['macd_hist'] = macd_hist
            
            # === 4. 波动率指标 - ATR ===
            indicators['atr'] = talib.ATR(high, low, close, timeperiod=self.atr_period)
            indicators['atr_pct'] = indicators['atr'] / close * 100  # 百分比表示
            
            # === 5. RSI指标 ===
            indicators['rsi'] = talib.RSI(close, timeperiod=14)
            
            # === 6. 成交量指标 ===
            volume = df['volume'].values
            indicators['volume'] = volume
            indicators['volume_ma'] = df['volume'].rolling(window=self.volume_ma_period).mean().values
            indicators['volume_ratio'] = volume / indicators['volume_ma']
            
            # === 7. 综合趋势评分 ===
            # 初始化趋势评分为0
            trend_score = np.zeros(len(close))
            
            # 添加EMA趋势分量 (-1 到 1)
            ema_trend = np.zeros(len(close))
            ema_trend[(indicators['ema_short'] > indicators['ema_medium'])] += 0.5
            ema_trend[(indicators['ema_medium'] > indicators['ema_long'])] += 0.5
            ema_trend[(indicators['ema_short'] < indicators['ema_medium'])] -= 0.5
            ema_trend[(indicators['ema_medium'] < indicators['ema_long'])] -= 0.5
            
            # 添加ADX分量 (0 到 0.5, 根据ADX强度)
            adx_factor = indicators['adx'] / 100  # 归一化ADX (0-1)
            adx_trend = np.zeros(len(close))
            adx_trend[indicators['plus_di'] > indicators['minus_di']] = adx_factor[indicators['plus_di'] > indicators['minus_di']] * 0.5
            adx_trend[indicators['plus_di'] < indicators['minus_di']] = -adx_factor[indicators['plus_di'] < indicators['minus_di']] * 0.5
            
            # 添加MACD分量 (-0.5 到 0.5)
            macd_factor = indicators['macd_hist'] / (close * 0.01)  # 相对于价格1%归一化
            macd_factor = np.clip(macd_factor, -0.5, 0.5)  # 限制在-0.5到0.5之间
            
            # 添加RSI分量 (-0.5 到 0.5)
            rsi_factor = (indicators['rsi'] - 50) / 50  # 归一化为-1到1
            rsi_factor = rsi_factor * 0.5  # 缩放为-0.5到0.5
            
            # 添加成交量确认分量 (0 到 0.5)
            volume_factor = np.clip(indicators['volume_ratio'] - 1, -1, 1) * 0.25
            
            # 组合所有分量
            trend_score = ema_trend + adx_trend + macd_factor + rsi_factor + volume_factor
            
            # 总分在-2到2之间，归一化到-1到1
            trend_score = trend_score / 2
            
            indicators['trend_score'] = trend_score
            
            # 趋势方向 (1: 上升, -1: 下降, 0: 盘整)
            trend_direction = np.zeros(len(close))
            trend_direction[trend_score > 0.3] = 1
            trend_direction[trend_score < -0.3] = -1
            
            indicators['trend_direction'] = trend_direction
            
            # === 新增短期趋势反转指标 ===
            # 计算短期动量
            momentum = np.zeros(len(close))
            for i in range(3, len(close)):
                momentum[i] = close[i] - close[i-3]
            indicators['momentum'] = momentum
            
            # 计算动量MA
            momentum_ma = np.zeros(len(momentum))
            for i in range(5, len(momentum)):
                momentum_ma[i] = np.mean(momentum[i-5:i])
            indicators['momentum_ma'] = momentum_ma
            
            # 计算价格变化率
            price_change = np.zeros(len(close))
            for i in range(1, len(close)):
                price_change[i] = (close[i] / close[i-1]) - 1
            indicators['price_change'] = price_change
            
            # 计算波动率 (标准差)
            volatility = np.zeros(len(price_change))
            for i in range(10, len(price_change)):
                volatility[i] = np.std(price_change[i-10:i])
            indicators['volatility'] = volatility
            
            # 计算短期趋势强度
            indicators['short_trend'] = talib.HT_TRENDLINE(close)
            
            # 计算趋势斜率
            trend_slope = np.zeros(len(indicators['short_trend']))
            for i in range(1, len(indicators['short_trend'])):
                trend_slope[i] = indicators['short_trend'][i] - indicators['short_trend'][i-1]
            indicators['trend_slope'] = trend_slope
            
            # 添加短期反转因子
            momentum_factor = np.clip(indicators['momentum_ma'] / (close * 0.001), -0.5, 0.5)
            
            # 安全计算波动率因子
            volatility_mean = np.mean(volatility[volatility > 0]) if np.any(volatility > 0) else 1
            volatility_factor = np.clip((volatility / volatility_mean) - 1, -0.3, 0.3)
            
            slope_factor = np.clip(indicators['trend_slope'] / (close * 0.001), -0.4, 0.4)
            
            # 更新趋势评分
            trend_score = trend_score + momentum_factor + volatility_factor + slope_factor
            
            trend_score = trend_score / 2.2

            indicators['trend_score'] = trend_score  # 更新最终趋势评分
            
            # === 新增反转模式识别 (改进点3) ===
            # 计算K线形态
            doji = talib.CDLDOJI(df['open'].values, high, low, close)
            indicators['doji'] = doji / 100
            
            engulfing = talib.CDLENGULFING(df['open'].values, high, low, close)
            indicators['engulfing'] = engulfing / 100
            
            # 添加背离检测 (简化版，不使用rolling.apply)
            # 价格高点/低点
            price_peaks = np.zeros(len(close))
            for i in range(2, len(close)-2):
                window = close[i-2:i+3]
                if close[i] == np.max(window):
                    price_peaks[i] = 1
                elif close[i] == np.min(window):
                    price_peaks[i] = -1
            indicators['price_peaks'] = price_peaks
            
            # RSI高点/低点
            rsi_peaks = np.zeros(len(indicators['rsi']))
            for i in range(2, len(indicators['rsi'])-2):
                if not np.isnan(indicators['rsi'][i-2:i+3]).any():  # 检查无NaN
                    window = indicators['rsi'][i-2:i+3]
                    if indicators['rsi'][i] == np.max(window):
                        rsi_peaks[i] = 1
                    elif indicators['rsi'][i] == np.min(window):
                        rsi_peaks[i] = -1
            indicators['rsi_peaks'] = rsi_peaks
            
            # 背离计算 (1: 看涨背离, -1: 看跌背离, 0: 无背离)
            divergence = np.zeros(len(close))
            
            # 看涨背离: 价格创新低但RSI未创新低
            bullish_div_mask = (price_peaks == -1) & (rsi_peaks != -1)
            divergence[bullish_div_mask] = 1
            
            # 看跌背离: 价格创新高但RSI未创新高
            bearish_div_mask = (price_peaks == 1) & (rsi_peaks != 1)
            divergence[bearish_div_mask] = -1
            
            indicators['divergence'] = divergence
            
            # 反转概率计算 (改进点4)
            reversal_up_prob = np.zeros(len(close))
            reversal_down_prob = np.zeros(len(close))
            
            # 计算上涨反转概率因子
            reversal_up_prob += (indicators['rsi'] < 30) * 0.4  # RSI超卖
            reversal_up_prob += (indicators['divergence'] == 1) * 0.3  # 看涨背离
            reversal_up_prob += (indicators['engulfing'] > 0) * 0.2  # 看涨吞没
            reversal_up_prob += (indicators['doji'] != 0) * 0.1  # 十字星
            
            # 计算下跌反转概率因子
            reversal_down_prob += (indicators['rsi'] > 70) * 0.4  # RSI超买
            reversal_down_prob += (indicators['divergence'] == -1) * 0.3  # 看跌背离
            reversal_down_prob += (indicators['engulfing'] < 0) * 0.2  # 看跌吞没
            reversal_down_prob += (indicators['doji'] != 0) * 0.1  # 十字星
            
            indicators['reversal_up_prob'] = reversal_up_prob
            indicators['reversal_down_prob'] = reversal_down_prob
            
            return indicators, df
            
        except Exception as e:
            self.logger.error(f"计算指标失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None, None
    
    def detect_reversal(self, indicators, trend_direction):
        """
        增强的反转检测函数 (改进点5)
        
        Args:
            indicators: 技术指标字典
            trend_direction: 当前趋势方向
            
        Returns:
            bool: 是否检测到反转
            int: 反转方向 (1 表示向上反转, -1 表示向下反转)
        """
        try:
            if indicators is None:
                return False, 0
                
            # 安全访问函数
            def safe_get(key, idx=-1, default=0):
                """超级安全的指标访问函数"""
                try:
                    if key not in indicators:
                        return default
                        
                    value = indicators[key]
                    
                    if value is None:
                        return default
                        
                    if isinstance(value, pd.Series):
                        if abs(idx) >= len(value):
                            return default
                        return value.iloc[idx]
                    elif isinstance(value, np.ndarray):
                        if abs(idx) >= len(value):
                            return default
                        return value[idx]
                    else:
                        return value if idx == -1 else default
                except Exception as e:
                    self.logger.error(f"安全访问指标 {key} 出错: {str(e)}")
                    return default
            
            # 当前趋势分数 - 使用安全访问函数
            current_trend_score = safe_get('trend_score')
            
            # 添加到历史记录
            self.trend_score_history.append(current_trend_score)
            if len(self.trend_score_history) > 10:
                self.trend_score_history.pop(0)
            
            # 1. 趋势评分反转检测
            if len(self.trend_score_history) >= 3:
                # 检查趋势评分是否开始逆转
                if trend_direction > 0 and self.trend_score_history[-1] < self.trend_score_history[-2] < self.trend_score_history[-3]:
                    # 向下反转
                    trend_change_intensity = abs(self.trend_score_history[-1] - self.trend_score_history[-3])
                    if trend_change_intensity > self.reversal_confirmation_threshold:
                        self.logger.info(f"趋势评分开始下降 {trend_change_intensity:.4f}，检测到向下反转")
                        return True, -1
                        
                elif trend_direction < 0 and self.trend_score_history[-1] > self.trend_score_history[-2] > self.trend_score_history[-3]:
                    # 向上反转
                    trend_change_intensity = abs(self.trend_score_history[-1] - self.trend_score_history[-3])
                    if trend_change_intensity > self.reversal_confirmation_threshold:
                        self.logger.info(f"趋势评分开始上升 {trend_change_intensity:.4f}，检测到向上反转")
                        return True, 1
            
            # 2. 技术形态反转检测
            rsi = safe_get('rsi')
            
            # 计算随机指标
            stoch_cross_up = False
            stoch_cross_down = False
            try:
                # 确保我们使用numpy数组
                high = indicators.get('high', [])
                low = indicators.get('low', [])
                close = indicators.get('close', [])
                
                if len(high) > 0 and len(low) > 0 and len(close) > 0:
                    stoch_k, stoch_d = talib.STOCH(
                        high, low, close,
                        fastk_period=5,
                        slowk_period=3,
                        slowd_period=3
                    )
                    if len(stoch_k) > 1 and len(stoch_d) > 1:
                        stoch_cross_up = (stoch_k[-1] > stoch_d[-1] and stoch_k[-2] <= stoch_d[-2])
                        stoch_cross_down = (stoch_k[-1] < stoch_d[-1] and stoch_k[-2] >= stoch_d[-2])
            except Exception as e:
                self.logger.warning(f"计算随机指标时出错: {str(e)}")
            
            # 金叉/死叉检测
            macd_cross_up = False
            macd_cross_down = False
            try:
                macd = safe_get('macd')
                macd_signal = safe_get('macd_signal')
                macd_prev = safe_get('macd', -2)
                macd_signal_prev = safe_get('macd_signal', -2)
                
                if not (macd == 0 and macd_signal == 0 and macd_prev == 0 and macd_signal_prev == 0):
                    macd_cross_up = (macd > macd_signal and macd_prev <= macd_signal_prev)
                    macd_cross_down = (macd < macd_signal and macd_prev >= macd_signal_prev)
            except Exception as e:
                self.logger.warning(f"检测MACD交叉时出错: {str(e)}")
            
            # 超买/超卖区域反转
            rsi_overbought_reversal = False
            rsi_oversold_reversal = False
            try:
                rsi_prev = safe_get('rsi', -2)
                if rsi > 0 and rsi_prev > 0:  # 确保RSI有效
                    rsi_overbought_reversal = (rsi > 70 and rsi_prev > rsi)
                    rsi_oversold_reversal = (rsi < 30 and rsi_prev < rsi)
            except Exception as e:
                self.logger.warning(f"检测RSI超买超卖时出错: {str(e)}")
            
            # 3. 价格行为反转
            price_candle_pattern = 0
            try:
                # 获取必要的数据
                close_curr = safe_get('close')
                close_prev = safe_get('close', -2)
                open_curr = safe_get('open')
                open_prev = safe_get('open', -2)
                atr_prev = safe_get('atr', -2)
                
                if atr_prev > 0:  # 确保ATR有效
                    # 大阴线后的阳线，可能是反转
                    if (close_curr > open_curr and 
                        close_prev < open_prev and
                        abs(close_prev - open_prev) > atr_prev * 0.8):
                        price_candle_pattern = 1
                        
                    # 大阳线后的阴线，可能是反转
                    elif (close_curr < open_curr and 
                        close_prev > open_prev and
                        abs(close_prev - open_prev) > atr_prev * 0.8):
                        price_candle_pattern = -1
            except Exception as e:
                self.logger.warning(f"检测价格形态时出错: {str(e)}")
            
            # 反转信号汇总
            reversal_up_prob = safe_get('reversal_up_prob')
            divergence = safe_get('divergence')
            
            reversal_up_signals = sum([
                macd_cross_up,
                stoch_cross_up,
                rsi_oversold_reversal,
                price_candle_pattern == 1,
                reversal_up_prob > 0.5,
                divergence == 1
            ])
            
            reversal_down_prob = safe_get('reversal_down_prob')
            
            reversal_down_signals = sum([
                macd_cross_down,
                stoch_cross_down,
                rsi_overbought_reversal,
                price_candle_pattern == -1,
                reversal_down_prob > 0.5,
                divergence == -1
            ])
            
            # 检查反转信号是否足够强
            min_signals_required = max(2, int(3 * self.reversal_sensitivity))
            
            if trend_direction > 0 and reversal_down_signals >= min_signals_required:
                self.logger.info(f"多个指标显示向下反转 [{reversal_down_signals}]，检测到向下反转")
                return True, -1
                
            elif trend_direction < 0 and reversal_up_signals >= min_signals_required:
                self.logger.info(f"多个指标显示向上反转 [{reversal_up_signals}]，检测到向上反转")
                return True, 1
                
            # 没有检测到明确的反转
            return False, 0
            
        except Exception as e:
            self.logger.error(f"检测反转失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False, 0
    
    def generate_signal(self, klines):
        """生成交易信号
        返回值：
        -1: 卖出信号
        0: 观望信号
        1: 买入信号
        2: 趋势反转信号 (改进点6)
        """
        try:
            # 计算指标
            indicators, df = self.calculate_indicators(klines)
            if indicators is None or df is None:
                return 0
            
            # 获取最新指标值
            latest_idx = -1
            
            trend_score = indicators['trend_score'][latest_idx]
            trend_direction = indicators['trend_direction'][latest_idx]
            
            self.logger.info(f"趋势评分: {trend_score:.4f}, 趋势方向: {trend_direction}, 上一次信号: {self.last_signal}")
            
            # 检查反转信号 (改进点7)
            reversal_detected, reversal_direction = self.detect_reversal(indicators, trend_direction)
            
            if reversal_detected:
                if (self.last_signal == 1 and reversal_direction < 0) or (self.last_signal == -1 and reversal_direction > 0):
                    self.logger.info(f"检测到趋势反转: 方向={reversal_direction}, 当前趋势={trend_direction}")
                    
                    # 更新信号历史
                    self.signal_history.pop(0)
                    self.signal_history.append(2)  # 使用2表示反转信号
                    
                    return 2  # 返回反转信号
            
            # 生成常规信号
            signal = 0  # 默认为观望
            
            if trend_direction > 0 and trend_score > 0.4:
                signal = 1  # 买入
            elif trend_direction < 0 and trend_score < -0.4:
                signal = -1  # 卖出
                
            # 更新信号历史
            self.signal_history.pop(0)
            self.signal_history.append(signal)
            
            # 更新上一次信号
            if signal != 0:  # 不是观望信号时更新
                self.last_signal = signal
            
            return signal
            
        except Exception as e:
            self.logger.error(f"生成信号失败: {str(e)}")
            return 0  # 发生错误时返回观望信号
    
    