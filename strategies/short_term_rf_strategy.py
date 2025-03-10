import numpy as np
import pandas as pd
import talib
import time
from datetime import datetime
from strategies.base_rf_strategy import BaseRFStrategy

class ShortTermRFStrategy(BaseRFStrategy):
    """ShortTermRFStrategy - 随机森林短期趋势交易策略
    
    基于随机森林算法的短期趋势交易策略，使用5分钟K线数据识别市场趋势和波段交易机会。
    通过提取价格、成交量、趋势、动量等多维特征，预测未来价格走势并生成交易信号。
    
    特点:
    1. 多维特征提取: 综合分析价格形态、技术指标、波动率等数据
    2. 自适应预测: 随机森林的特性使得模型能够适应不同市场环境
    3. 概率输出: 不仅预测方向，还提供概率用于交易决策
    4. 定期再训练: 随着新数据的产生，定期更新模型以适应市场变化
    5. 风险管理: 内置止盈止损和交易频率控制机制
    """
    
    MODEL_NAME = "ShortTermRF"
    VERSION = "1.0.0"
    
    def __init__(self, trader):
        """初始化随机森林短期趋势交易策略"""
        super().__init__(trader)
        
        # K线设置
        self.kline_interval = '5m'           # 5分钟K线
        self.check_interval = 60             # 检查信号间隔(秒)
        self.training_lookback = 1000        # 训练所需K线数量
        self.retraining_interval = 3600 * 4  # 4小时重新训练一次
        
        # 随机森林参数
        self.n_estimators = 200    # 决策树数量
        self.max_depth = 10        # 最大深度
        self.min_samples_split = 15  # 分裂所需最小样本数
        self.min_samples_leaf = 5   # 叶节点最小样本数
        self.confidence_threshold = 0.45  # 信号置信度阈值
        self.prob_diff_threshold = 0.20   # 概率差异阈值
        
        # 风险控制参数
        self.max_position_hold_time = 60  # 最大持仓时间(分钟)
        self.profit_target_pct = 0.5    # 目标利润率 50%
        self.stop_loss_pct = 0.01        # 止损率 1%
        self.max_trades_per_hour = 5      # 每小时最大交易次数
        
        # 交易状态跟踪
        self.last_trade_hour = None       # 上次交易的小时
        
        # 进行初始化模型
        self.initialize_model()
        
        # 进行初始训练
        self._initial_training()
    
    def prepare_features(self, klines):
        """准备特征数据，从K线中提取用于预测的特征"""
        try:
            if not isinstance(klines, list) or len(klines) < 50:
                self.logger.error("K线数据不足，无法提取特征")
                return None
            
            # 创建DataFrame
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 检查是否有缺失值
            if df.isnull().any().any():
                self.logger.warning("K线数据存在缺失值，进行插值处理")
                df = df.interpolate(method='linear')
            
            # 提取基础数据
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            volume = df['volume'].values
            
            # 创建特征DataFrame
            features = pd.DataFrame(index=df.index)
            
            # 1. 价格特征 - 价格变化率
            for period in [1, 2, 3, 5, 8, 13]:
                features[f'price_change_{period}'] = df['close'].pct_change(period)
            
            # 2. 移动平均线特征
            # 先计算所有SMA和EMA
            for period in [5, 10, 20, 30, 50]:
                # SMA
                sma = talib.SMA(close, timeperiod=period)
                features[f'sma_{period}'] = pd.Series(sma, index=df.index)
                features[f'price_to_sma_{period}'] = df['close'] / features[f'sma_{period}'] - 1
                
                # EMA
                ema = talib.EMA(close, timeperiod=period)
                features[f'ema_{period}'] = pd.Series(ema, index=df.index)
                features[f'price_to_ema_{period}'] = df['close'] / features[f'ema_{period}'] - 1
            
            # 再计算均线交叉信号
            for period in [5, 10, 20, 30]:
                features[f'ema_cross_{period}_50'] = features[f'ema_{period}'] - features[f'ema_50']
            
            # 3. 动量指标
            # RSI - 相对强弱指数
            for period in [7, 14, 21]:
                features[f'rsi_{period}'] = pd.Series(talib.RSI(close, timeperiod=period), index=df.index)
                
            # MACD - 移动平均收敛/发散
            macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
            features['macd'] = pd.Series(macd, index=df.index)
            features['macd_signal'] = pd.Series(macd_signal, index=df.index)
            features['macd_hist'] = pd.Series(macd_hist, index=df.index)
            features['macd_hist_diff'] = features['macd_hist'].diff()
            
            # CCI - 商品通道指数
            features['cci'] = pd.Series(talib.CCI(high, low, close, timeperiod=14), index=df.index)
            
            # 4. 波动率指标
            # ATR - 真实波动幅度均值
            features['atr'] = pd.Series(talib.ATR(high, low, close, timeperiod=14), index=df.index)
            features['atr_percent'] = features['atr'] / df['close'] * 100
            
            # 布林带指标
            upperband, middleband, lowerband = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
            features['bb_upper'] = pd.Series(upperband, index=df.index)
            features['bb_middle'] = pd.Series(middleband, index=df.index)
            features['bb_lower'] = pd.Series(lowerband, index=df.index)
            features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / features['bb_middle']
            features['bb_position'] = (df['close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
            
            # 5. 量价关系
            features['volume_change'] = df['volume'].pct_change()
            features['volume_ma'] = df['volume'].rolling(10).mean()
            features['volume_ratio'] = df['volume'] / features['volume_ma']
            
            # OBV - 能量潮
            features['obv'] = pd.Series(talib.OBV(close, volume), index=df.index)
            features['obv_change'] = features['obv'].pct_change(3)
            
            # 成交量加权均价
            df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
            features['price_to_vwap'] = df['close'] / df['vwap'] - 1
            
            # 6. 趋势指标
            # ADX - 平均趋向指数
            features['adx'] = pd.Series(talib.ADX(high, low, close, timeperiod=14), index=df.index)
            features['plus_di'] = pd.Series(talib.PLUS_DI(high, low, close, timeperiod=14), index=df.index)
            features['minus_di'] = pd.Series(talib.MINUS_DI(high, low, close, timeperiod=14), index=df.index)
            features['di_diff'] = features['plus_di'] - features['minus_di']
            
            # Aroon - 阿隆指标
            aroon_down, aroon_up = talib.AROON(high, low, timeperiod=14)
            features['aroon_down'] = pd.Series(aroon_down, index=df.index)
            features['aroon_up'] = pd.Series(aroon_up, index=df.index)
            features['aroon_diff'] = features['aroon_up'] - features['aroon_down']
            
            # 7. 价格模式
            # K线形态特征
            features['body_ratio'] = abs(df['close'] - df['open']) / (df['high'] - df['low'])
            features['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (df['high'] - df['low'])
            features['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / (df['high'] - df['low'])
            
            # 移除包含NaN的行
            features = features.replace([np.inf, -np.inf], np.nan).dropna()
            
            # 确保特征索引与原始数据对齐
            features.index = range(len(features))
            
            return features
            
        except Exception as e:
            self.logger.error(f"准备特征数据失败: {str(e)}")
            return None
    
    def generate_labels(self, klines):
        """生成训练标签，预测未来价格变化"""
        try:
            if not isinstance(klines, list) or len(klines) < 50:
                self.logger.error("K线数据不足，无法生成标签")
                return None
                
            # 创建DataFrame
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
            # 未来价格变化比例（6个周期后）
            future_periods = 6  # 预测未来6个周期（30分钟）
            df['future_change'] = df['close'].shift(-future_periods) / df['close'] - 1
            
            # 计算未来最高和最低价与当前价格的比例
            df['future_high'] = df['high'].rolling(future_periods).max().shift(-future_periods)
            df['future_low'] = df['low'].rolling(future_periods).min().shift(-future_periods)
            df['future_high_change'] = df['future_high'] / df['close'] - 1
            df['future_low_change'] = df['future_low'] / df['close'] - 1
            
            # 创建标签：-1=看跌，0=盘整，1=看涨
            # 根据未来价格变化幅度判断趋势方向
            threshold = 0.003  # 0.3% 的阈值
            labels = pd.Series(0, index=df.index)  # 默认为盘整
            
            # 如果未来价格上涨超过阈值，标记为看涨
            labels.loc[df['future_change'] > threshold] = 1
            
            # 如果未来价格下跌超过阈值，标记为看跌
            labels.loc[df['future_change'] < -threshold] = -1
            
            # 移除包含NaN的行
            valid_indices = ~df['future_change'].isnull()
            labels = labels[valid_indices]
            
            # 重置索引
            labels.index = range(len(labels))
            
            # 输出标签分布
            label_counts = labels.value_counts()
            self.logger.info(f"标签分布: 看跌={label_counts.get(-1, 0)}, 盘整={label_counts.get(0, 0)}, 看涨={label_counts.get(1, 0)}")
            
            return labels
            
        except Exception as e:
            self.logger.error(f"生成标签失败: {str(e)}")
            return None
    
    def monitor_position(self):
        """监控当前持仓，并根据策略决定是否平仓或开新仓"""
        try:
            # 获取当前持仓
            position = self.trader.get_position()
            
            # 获取最新K线数据
            klines = self.trader.get_klines(interval=self.kline_interval, limit=self.training_lookback)
            if len(klines) < 100:  # 确保有足够的数据进行分析
                self.logger.error(f"K线数据不足，需要至少100根")
                return
                
            # 获取当前市场价格
            current_price = self.trader.get_market_price()
            
            # 如果需要重新训练模型
            if self.should_retrain():
                if self.train_model(klines):
                    self.logger.info("模型重新训练完成")
                else:
                    self.logger.error("模型重新训练失败")
                self.last_training_time = time.time()
            
            # 如果没有持仓，检查是否有新的交易信号
            if position is None or float(position['info'].get('positionAmt', 0)) == 0:
                # 生成交易信号
                signal = self.generate_signal(klines)
                
                # 根据信号执行交易
                if signal == 1:  # 买入信号
                    # 检查每小时交易次数限制
                    current_hour = datetime.now().hour
                    if self.last_trade_hour != current_hour:
                        self.last_trade_hour = current_hour
                        self.trade_count_hour = 0
                        
                    if self.trade_count_hour >= self.max_trades_per_hour:
                        self.logger.info(f"已达到每小时最大交易次数({self.max_trades_per_hour})，跳过买入信号")
                        return
                        
                    # 计算交易数量
                    balance = self.trader.get_balance()
                    available_balance = float(balance['free'])
                    
                    # 从config获取交易金额百分比
                    symbol_config = self.trader.symbol_config
                    trade_percent = symbol_config.get('trade_amount_percent', 50)
                    
                    # 计算交易金额
                    trade_amount = (available_balance * trade_percent / 100) / current_price
                    
                    # 执行开多仓操作
                    order = self.trader.open_long(amount=trade_amount)
                    if order:
                        self.logger.info(f"开多仓 - 数量: {trade_amount:.6f}, 价格: {current_price}")
                        
                        # 记录开仓信息
                        self.position_entry_time = time.time()
                        self.position_entry_price = current_price
                        self.trade_count_hour += 1
                
                elif signal == -1:  # 卖出信号
                    # 检查每小时交易次数限制
                    current_hour = datetime.now().hour
                    if self.last_trade_hour != current_hour:
                        self.last_trade_hour = current_hour
                        self.trade_count_hour = 0
                        
                    if self.trade_count_hour >= self.max_trades_per_hour:
                        self.logger.info(f"已达到每小时最大交易次数({self.max_trades_per_hour})，跳过卖出信号")
                        return
                        
                    # 计算交易数量
                    balance = self.trader.get_balance()
                    available_balance = float(balance['free'])
                    
                    # 从config获取交易金额百分比
                    symbol_config = self.trader.symbol_config
                    trade_percent = symbol_config.get('trade_amount_percent', 50)
                    
                    # 计算交易金额
                    trade_amount = (available_balance * trade_percent / 100) / current_price
                    
                    # 执行开空仓操作
                    order = self.trader.open_short(amount=trade_amount)
                    if order:
                        self.logger.info(f"开空仓 - 数量: {trade_amount:.6f}, 价格: {current_price}")
                        
                        # 记录开仓信息
                        self.position_entry_time = time.time()
                        self.position_entry_price = current_price
                        self.trade_count_hour += 1
            
            # 如果有持仓，检查是否需要平仓
            else:
                position_amount = float(position['info'].get('positionAmt', 0))
                entry_price = float(position['info'].get('entryPrice', 0))
                position_side = "多" if position_amount > 0 else "空"
                
                # 计算持仓时间
                current_time = time.time()
                holding_time_minutes = 0
                
                if self.position_entry_time:
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
                
                # 检查反向信号平仓条件
                signal = self.generate_signal(klines)
                if (position_side == "多" and signal == -1) or (position_side == "空" and signal == 1):
                    self.logger.info(f"出现反向信号，当前{position_side}仓，信号为{signal}，平仓")
                    self.trader.close_position()
                    return
                
        except Exception as e:
            self.logger.error(f"监控持仓失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
