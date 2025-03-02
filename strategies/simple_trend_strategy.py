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
    """
    
    def __init__(self, trader):
        """初始化简化趋势跟踪策略"""
        super().__init__(trader)
        self.logger = self.get_logger()
        
        # K线设置
        self.kline_interval = '3m'      # 5分钟K线
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
        
        # 添加上一次信号记录
        self.last_signal = 0  # 初始化为观望信号
        
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
            
            # === 2. 趋势强度指标 - ADX ===
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            
            indicators['adx'] = pd.Series(talib.ADX(high, low, close, timeperiod=self.adx_period), index=df.index)
            indicators['plus_di'] = pd.Series(talib.PLUS_DI(high, low, close, timeperiod=self.adx_period), index=df.index)
            indicators['minus_di'] = pd.Series(talib.MINUS_DI(high, low, close, timeperiod=self.adx_period), index=df.index)
            
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
            
            # === 4. 波动率指标 - ATR ===
            indicators['atr'] = pd.Series(talib.ATR(high, low, close, timeperiod=self.atr_period), index=df.index)
            indicators['atr_pct'] = indicators['atr'] / df['close'] * 100  # 百分比表示
            
            # === 5. RSI指标 ===
            indicators['rsi'] = pd.Series(talib.RSI(close, timeperiod=14), index=df.index)
            
            # === 6. 成交量指标 ===
            indicators['volume'] = df['volume']
            indicators['volume_ma'] = df['volume'].rolling(window=self.volume_ma_period).mean()
            indicators['volume_ratio'] = df['volume'] / indicators['volume_ma']
            
            # === 7. 综合趋势评分 ===
            # 初始化趋势评分为0
            trend_score = pd.Series(0.0, index=df.index)
            
            # 添加EMA趋势分量 (-1 到 1)
            ema_trend = pd.Series(0.0, index=df.index)
            ema_trend[indicators['ema_short'] > indicators['ema_medium']] += 0.5
            ema_trend[indicators['ema_medium'] > indicators['ema_long']] += 0.5
            ema_trend[indicators['ema_short'] < indicators['ema_medium']] -= 0.5
            ema_trend[indicators['ema_medium'] < indicators['ema_long']] -= 0.5
            
            # 添加ADX分量 (0 到 0.5, 根据ADX强度)
            adx_factor = indicators['adx'] / 100  # 归一化ADX (0-1)
            adx_trend = pd.Series(0.0, index=df.index)
            adx_trend[indicators['plus_di'] > indicators['minus_di']] = adx_factor * 0.5
            adx_trend[indicators['plus_di'] < indicators['minus_di']] = -adx_factor * 0.5
            
            # 添加MACD分量 (-0.5 到 0.5)
            macd_factor = indicators['macd_hist'] / (df['close'] * 0.01)  # 相对于价格1%归一化
            macd_factor = macd_factor.clip(-0.5, 0.5)  # 限制在-0.5到0.5之间
            
            # 添加RSI分量 (-0.5 到 0.5)
            rsi_factor = (indicators['rsi'] - 50) / 50  # 归一化为-1到1
            rsi_factor = rsi_factor * 0.5  # 缩放为-0.5到0.5
            
            # 添加成交量确认分量 (0 到 0.5)
            volume_factor = (indicators['volume_ratio'] - 1).clip(-1, 1) * 0.25
            
            # 打印各个趋势评分组成部分
            self.logger.info(f"""趋势评分组成部分:
                EMA趋势: {ema_trend.iloc[-1]:.4f}
                ADX趋势: {adx_trend.iloc[-1]:.4f}
                MACD因子: {macd_factor.iloc[-1]:.4f}
                RSI因子: {rsi_factor.iloc[-1]:.4f}
                成交量因子: {volume_factor.iloc[-1]:.4f}
            """)
            
            # 组合所有分量
            trend_score = ema_trend + adx_trend + macd_factor + rsi_factor + volume_factor
            
            # 总分在-2到2之间，归一化到-1到1
            trend_score = trend_score / 2
            
            indicators['trend_score'] = trend_score
            
            # 趋势方向 (1: 上升, -1: 下降, 0: 盘整)
            trend_direction = pd.Series(0, index=df.index)
            trend_direction[trend_score > 0.3] = 1
            trend_direction[trend_score < -0.3] = -1
            
            indicators['trend_direction'] = trend_direction
            
            # === 新增短期趋势反转指标 ===
            # 计算短期动量
            indicators['momentum'] = df['close'].diff(3)
            indicators['momentum_ma'] = indicators['momentum'].rolling(window=5).mean()
            
            # 计算价格波动率
            indicators['price_change'] = df['close'].pct_change()
            indicators['volatility'] = indicators['price_change'].rolling(window=10).std()
            
            # 计算短期趋势强度
            indicators['short_trend'] = pd.Series(talib.HT_TRENDLINE(close), index=df.index)
            indicators['trend_slope'] = indicators['short_trend'].diff()
            
            # 修改趋势评分计算，增加短期反转因子
            momentum_factor = (indicators['momentum_ma'] / (df['close'] * 0.001)).clip(-0.5, 0.5)
            volatility_factor = (indicators['volatility'] / indicators['volatility'].mean() - 1).clip(-0.3, 0.3)
            slope_factor = (indicators['trend_slope'] / (df['close'] * 0.001)).clip(-0.4, 0.4)
            
            # 将短期因子添加到趋势评分中
            trend_score = trend_score + momentum_factor + volatility_factor + slope_factor
            
            return indicators, df
            
        except Exception as e:
            self.logger.error(f"计算指标失败: {str(e)}")
            return None, None
    
    def generate_signal(self, klines):
        """生成交易信号
        返回值：
        -1: 卖出信号
        0: 观望信号
        1: 买入信号
        2: 趋势反转信号
        """
        try:
            # 检测反转
            is_reversed, reversal_direction, reversal_strength = self.detect_reversal(klines)
            
            # 如果检测到反转且强度足够高
            if is_reversed and reversal_strength > 0.6:
                self.logger.info(f"检测到强烈的{'向上' if reversal_direction > 0 else '向下'}反转信号，强度: {reversal_strength:.4f}")
                return 2  # 返回反转信号
            
            # 计算指标
            indicators, df = self.calculate_indicators(klines)
            if indicators is None or df is None:
                return 0
            
            # 获取最新指标值
            latest_idx = -1
            
            trend_score = indicators['trend_score'].iloc[latest_idx]
            trend_direction = indicators['trend_direction'].iloc[latest_idx]
            
            self.logger.info(f"趋势评分: {trend_score:.4f}, 趋势方向: {trend_direction}, 上一次信号: {self.last_signal}")
            
            # 轻微反转处理
            if is_reversed and reversal_strength > 0.4:
                self.logger.info(f"检测到{'向上' if reversal_direction > 0 else '向下'}反转信号，强度: {reversal_strength:.4f}")
                if (self.last_signal == 1 and reversal_direction < 0) or (self.last_signal == -1 and reversal_direction > 0):
                    self.logger.info("检测到趋势反转信号")
                    signal = 2
                    return signal
            
            # 检查趋势反转 (原有逻辑保留作为备选)
            if (self.last_signal == 1 and trend_score < 0) or (self.last_signal == -1 and trend_score > 0):
                self.logger.info("检测到趋势评分反转信号")
                signal = 2
                return signal
            
            # 生成常规信号
            if trend_direction > 0 and trend_score > 0.4:
                signal = 1
            elif trend_direction < 0 and trend_score < -0.4:
                signal = -1
            else:
                signal = 0
            
            # 更新上一次信号（仅当不是反转信号时）
            if signal != 2 and signal != 0:
                self.last_signal = signal
            
            return signal
            
        except Exception as e:
            self.logger.error(f"生成信号失败: {str(e)}")
            return 0  # 发生错误时返回观望信号
    
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
            
            # 1. 基于RSI的反转信号
            rsi = indicators['rsi'].iloc[latest_idx]
            rsi_prev = indicators['rsi'].iloc[latest_idx-1]
            rsi_change = rsi - rsi_prev
            
            # 超卖反转信号 (RSI < 30 且开始回升)
            oversold_reversal = (rsi < 30 and rsi_change > 0)
            
            # 超买反转信号 (RSI > 70 且开始下降)
            overbought_reversal = (rsi > 70 and rsi_change < 0)
            
            # 2. 基于MACD的反转信号
            macd_hist = indicators['macd_hist'].iloc[latest_idx]
            macd_hist_prev = indicators['macd_hist'].iloc[latest_idx-1]
            macd_hist_change = macd_hist - macd_hist_prev
            
            # MACD柱状图由负转正
            macd_bullish_cross = (macd_hist_prev < 0 and macd_hist > 0)
            
            # MACD柱状图由正转负
            macd_bearish_cross = (macd_hist_prev > 0 and macd_hist < 0)
            
            # 3. 基于EMA的反转信号
            ema_short = indicators['ema_short'].iloc[latest_idx]
            ema_medium = indicators['ema_medium'].iloc[latest_idx]
            
            ema_short_prev = indicators['ema_short'].iloc[latest_idx-1]
            ema_medium_prev = indicators['ema_medium'].iloc[latest_idx-1]
            
            # EMA短线上穿中线
            ema_bullish_cross = (ema_short_prev < ema_medium_prev and ema_short > ema_medium)
            
            # EMA短线下穿中线
            ema_bearish_cross = (ema_short_prev > ema_medium_prev and ema_short < ema_medium)
            
            # 4. 基于K线形态的反转信号
            price_change = (df['close'].iloc[latest_idx] - df['close'].iloc[latest_idx-1]) / df['close'].iloc[latest_idx-1]
            
            # 当前K线实体较大
            large_body = abs(df['close'].iloc[latest_idx] - df['open'].iloc[latest_idx]) > indicators['atr'].iloc[latest_idx] * 1.5
            
            # 成交量放大
            volume_surge = df['volume'].iloc[latest_idx] > df['volume'].iloc[latest_idx-3:latest_idx].mean() * 1.5
            
            # 5. 趋势强度变化
            trend_strength = indicators['trend_score'].iloc[latest_idx]
            trend_strength_prev = indicators['trend_score'].iloc[latest_idx-1]
            trend_change = trend_strength - trend_strength_prev
            
            # 综合计算向上反转信号强度 (0-1)
            bullish_signals = [
                1 if oversold_reversal else 0,  # RSI超卖回升
                1 if macd_bullish_cross else 0,  # MACD金叉
                1 if ema_bullish_cross else 0,   # EMA金叉
                1 if (large_body and price_change > 0) else 0,  # 大阳线
                1 if (volume_surge and price_change > 0) else 0,  # 放量上涨
                min(max(trend_change * 2, 0), 1) if trend_change > 0 else 0  # 趋势由负转正的强度
            ]
            
            # 综合计算向下反转信号强度 (0-1)
            bearish_signals = [
                1 if overbought_reversal else 0,  # RSI超买回落
                1 if macd_bearish_cross else 0,   # MACD死叉
                1 if ema_bearish_cross else 0,    # EMA死叉
                1 if (large_body and price_change < 0) else 0,  # 大阴线
                1 if (volume_surge and price_change < 0) else 0,  # 放量下跌
                min(max(-trend_change * 2, 0), 1) if trend_change < 0 else 0  # 趋势由正转负的强度
            ]
            
            # 计算向上和向下反转的信号强度 (加权平均)
            weights = [0.2, 0.2, 0.2, 0.15, 0.15, 0.1]  # 各信号权重
            bullish_strength = sum(signal * weight for signal, weight in zip(bullish_signals, weights))
            bearish_strength = sum(signal * weight for signal, weight in zip(bearish_signals, weights))
            
            # 6. 市场波动率和趋势疲劳评估
            volatility = indicators['atr_pct'].iloc[latest_idx]
            avg_volatility = indicators['atr_pct'].iloc[latest_idx-5:latest_idx].mean()
            volatility_change = volatility / avg_volatility - 1
            
            # 波动率突变会增强反转信号
            if volatility_change > 0.5:  # 波动率显著增加
                bullish_strength *= (1 + 0.2 * min(volatility_change, 1))
                bearish_strength *= (1 + 0.2 * min(volatility_change, 1))
            
            # 7. 判断最终反转方向和强度
            is_bullish_reversal = bullish_strength > 0.4 and bullish_strength > bearish_strength * 1.5
            is_bearish_reversal = bearish_strength > 0.4 and bearish_strength > bullish_strength * 1.5
            
            if is_bullish_reversal:
                return True, 1, bullish_strength
            elif is_bearish_reversal:
                return True, -1, bearish_strength
            else:
                return False, 0, max(bullish_strength, bearish_strength)
            
        except Exception as e:
            self.logger.error(f"检测反转失败: {str(e)}")
            return False, 0, 0
    
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
                indicators, df = self.calculate_indicators(klines)
                
                if indicators is not None:
                    # 获取最新指标
                    trend_direction = indicators['trend_direction'].iloc[-1]
                    rsi = indicators['rsi'].iloc[-1]
                    
                    # 多仓反转条件
                    if position_side == "多" and (trend_direction < 0 or rsi > 75):
                        self.logger.info(f"多仓趋势反转信号，平仓")
                        self.trader.close_position()
                        return
                    
                    # 空仓反转条件
                    if position_side == "空" and (trend_direction > 0 or rsi < 25):
                        self.logger.info(f"空仓趋势反转信号，平仓")
                        self.trader.close_position()
                        return
                    
                    # 增强的反转信号检测
                    rapid_reversal = False
                    
                    # 检查快速反转条件
                    if position_side == "多":
                        if (indicators['momentum_ma'].iloc[-1] < 0 and abs(indicators['momentum_ma'].iloc[-1]) > indicators['volatility'].iloc[-1] * 2) or \
                           (indicators['price_change'].iloc[-1] < -indicators['volatility'].iloc[-1] * 1.5) or \
                           (indicators['trend_score'].iloc[-1] < -0.3 and self.last_signal > 0):
                            rapid_reversal = True
                            
                    elif position_side == "空":
                        if (indicators['momentum_ma'].iloc[-1] > 0 and abs(indicators['momentum_ma'].iloc[-1]) > indicators['volatility'].iloc[-1] * 2) or \
                           (indicators['price_change'].iloc[-1] > indicators['volatility'].iloc[-1] * 1.5) or \
                           (indicators['trend_score'].iloc[-1] > 0.3 and self.last_signal < 0):
                            rapid_reversal = True
                    
                    # 如果检测到快速反转，立即平仓
                    if rapid_reversal:
                        self.logger.info(f"检测到快速趋势反转，立即平仓")
                        self.trader.close_position()
                        return
                    
                    # 动态调整止损
                    if abs(indicators['trend_score'].iloc[-1]) < 0.2:  # 趋势减弱时收紧止损
                        dynamic_stop_loss = self.stop_loss_pct * 0.7
                    else:
                        dynamic_stop_loss = self.stop_loss_pct
                    
                    # 检查动态止损
                    if profit_rate <= -dynamic_stop_loss:
                        self.logger.info(f"触发动态止损，亏损率: {profit_rate:.4%}，平仓")
                        self.trader.close_position()
                        return
                
        except Exception as e:
            self.logger.error(f"监控持仓失败: {str(e)}")
    
    # 添加与TradingManager兼容的方法
    def should_retrain(self):
        """兼容TradingManager的方法，该策略不需要重新训练"""
        return False
    
    def train_model(self, klines):
        """兼容TradingManager的方法，该策略不需要训练模型"""
        self.logger.info("SimpleTrendStrategy不需要训练模型")
        return True
    
    def run(self):
        """运行策略"""
        try:
            self.logger.info("启动简化趋势跟踪策略")
            while True:
                try:
                    self.monitor_position()
                except Exception as e:
                    self.logger.error(f"策略执行出错: {str(e)}")
                time.sleep(self.check_interval)
        except Exception as e:
            self.logger.error(f"策略运行失败: {str(e)}")
            raise