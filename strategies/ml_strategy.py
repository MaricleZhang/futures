import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from .base_strategy import BaseStrategy
from utils.logger import Logger
import talib
import time
import config

class MLStrategy(BaseStrategy):
    def __init__(self, trader):
        """初始化机器学习策略"""
        super().__init__(trader)
        self.logger = Logger.get_logger()
        
        # 模型参数
        self.lookback = config.AI_KLINES_LIMIT
        self.base_threshold = 0.60  # 基础阈值
        self.dynamic_threshold = self.base_threshold  # 动态阈值
        self.threshold_adjustment_factor = 0.05  # 阈值调整因子
        
        # 技术指标参数
        self.rsi_period = config.RSI_PERIOD
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        self.atr_period = 14
        self.stoch_k = 14
        self.stoch_d = 3
        self.stoch_slow = 3
        
        # 市场情绪指标
        self.sentiment_window = 20  # 市场情绪窗口
        self.volatility_window = 20  # 波动率窗口
        self.volume_ma_window = 20  # 成交量均线窗口
        
        # 初始化模型
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        
        # 止损和止盈设置
        self.stop_loss_percent = config.DEFAULT_STOP_LOSS_PERCENT / 100
        self.take_profit_percent = config.DEFAULT_TAKE_PROFIT_PERCENT / 100
        
        # 记录上一次的信号和交易时间
        self.last_signal = 0
        self.last_trade_time = 0
        
        # 交易统计
        self.total_trades = 0
        self.winning_trades = 0
        self.total_profit = 0
        self.realized_profit = 0
        self.last_trade_price = 0
        self.last_trade_type = None
        self.current_position_entry_price = 0
        
    def prepare_features(self, klines):
        """准备特征数据"""
        try:
            if not isinstance(klines, list) or len(klines) == 0:
                self.logger.error("K线数据为空或格式不正确")
                return pd.DataFrame()
                
            # 创建DataFrame
            try:
                df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                if df.empty:
                    self.logger.error("创建的DataFrame为空")
                    return pd.DataFrame()
            except Exception as e:
                self.logger.error(f"创建DataFrame失败: {str(e)}")
                return pd.DataFrame()
            
            # 转换为数值类型
            try:
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                # 检查是否有足够的有效数据
                if df[['open', 'high', 'low', 'close', 'volume']].isnull().values.any():
                    self.logger.error("数值转换后存在空值")
                    return pd.DataFrame()
            except Exception as e:
                self.logger.error(f"数值转换失败: {str(e)}")
                return pd.DataFrame()
                
            # 基础技术指标
            try:
                df['RSI'] = talib.RSI(df['close'], timeperiod=self.rsi_period)
                macd, signal, hist = talib.MACD(df['close'], 
                                              fastperiod=self.macd_fast, 
                                              slowperiod=self.macd_slow, 
                                              signalperiod=self.macd_signal)
                df['MACD'] = macd
                df['MACD_SIGNAL'] = signal
                df['MACD_HIST'] = hist
                
                # 添加随机指标
                slowk, slowd = talib.STOCH(df['high'], df['low'], df['close'],
                                         fastk_period=self.stoch_k,
                                         slowk_period=self.stoch_d,
                                         slowk_matype=0,
                                         slowd_period=self.stoch_slow,
                                         slowd_matype=0)
                df['STOCH_K'] = slowk
                df['STOCH_D'] = slowd
            except Exception as e:
                self.logger.error(f"计算技术指标失败: {str(e)}")
                return pd.DataFrame()
            
            # 价格动量指标
            try:
                df['returns'] = df['close'].pct_change()
                df['log_returns'] = np.log(df['close']/df['close'].shift(1))
                df['volatility'] = df['returns'].rolling(window=self.volatility_window).std()
                
                # 趋势指标
                df['EMA_10'] = talib.EMA(df['close'], timeperiod=10)
                df['EMA_20'] = talib.EMA(df['close'], timeperiod=20)
                df['EMA_50'] = talib.EMA(df['close'], timeperiod=50)
                df['trend_strength'] = ((df['EMA_10'] - df['EMA_20']) / df['EMA_20'] * 100)
            except Exception as e:
                self.logger.error(f"计算动量指标失败: {str(e)}")
                return pd.DataFrame()
            
            # 市场情绪指标
            try:
                df['price_momentum'] = df['close'].pct_change(self.sentiment_window)
                df['volume_momentum'] = df['volume'].pct_change(self.sentiment_window)
                df['market_sentiment'] = df['price_momentum']  # 由于没有 taker_buy_base 数据，暂时只用价格动量
            except Exception as e:
                self.logger.error(f"计算市场情绪指标失败: {str(e)}")
                return pd.DataFrame()
            
            # 成交量指标
            try:
                # 现有的成交量指标
                df['volume_ma'] = df['volume'].rolling(window=self.volume_ma_window).mean()
                df['volume_ratio'] = df['volume'] / df['volume_ma']
                df['volume_trend'] = df['volume'].pct_change(5)
                
                # 添加OBV (On Balance Volume)
                df['OBV'] = talib.OBV(df['close'], df['volume'])
                df['OBV_ma'] = df['OBV'].rolling(window=20).mean()
                df['OBV_ratio'] = df['OBV'] / df['OBV_ma']
                
                # 添加MFI (Money Flow Index)
                df['MFI'] = talib.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=14)
                
                # 添加VWAP (Volume Weighted Average Price)
                df['VWAP'] = (df['close'] * df['volume']).rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
                df['VWAP_ratio'] = df['close'] / df['VWAP']
                
                # 添加成交量波动率
                df['volume_volatility'] = df['volume'].rolling(window=20).std() / df['volume_ma']
                
                # 添加相对成交量强度
                df['volume_rsi'] = talib.RSI(df['volume'], timeperiod=14)
                
            except Exception as e:
                self.logger.error(f"计算成交量指标失败: {str(e)}")
                return pd.DataFrame()
            
            # ATR和布林带
            try:
                df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=self.atr_period)
                upper, middle, lower = talib.BBANDS(df['close'], timeperiod=20)
                df['BB_width'] = (upper - lower) / middle
                df['BB_position'] = (df['close'] - lower) / (upper - lower)
            except Exception as e:
                self.logger.error(f"计算ATR和布林带失败: {str(e)}")
                return pd.DataFrame()
            
            # 准备特征
            feature_columns = [
                'RSI', 'MACD', 'MACD_SIGNAL', 'MACD_HIST',
                'STOCH_K', 'STOCH_D',
                'returns', 'log_returns', 'volatility',
                'trend_strength', 'volume_ratio', 'ATR',
                'BB_width', 'BB_position',
                'market_sentiment', 'volume_momentum',
                'volume_trend', 'price_momentum',
                # 新增的成交量特征
                'OBV_ratio', 'MFI', 'VWAP_ratio',
                'volume_volatility', 'volume_rsi'
            ]
            
            try:
                features = df[feature_columns].copy()
                features = features.ffill().fillna(0)
                
                if features.empty:
                    self.logger.error("生成的特征数据为空")
                    return pd.DataFrame()
                    
                # 检查特征数据是否包含无效值
                if features.isnull().values.any():
                    self.logger.error("特征数据包含无效值")
                    return pd.DataFrame()
                    
                return features
                
            except Exception as e:
                self.logger.error(f"准备特征数据失败: {str(e)}")
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"特征准备过程出错: {str(e)}")
            return pd.DataFrame()
        
    def prepare_labels(self, df):
        """准备训练标签
        
        使用多个时间窗口的加权收益率和趋势强度来生成买入(2)、观望(1)和卖出(0)信号。
        目标是实现1:3:1的标签分布。
        
        Args:
            df: 包含价格和技术指标的DataFrame
            
        Returns:
            numpy.ndarray: 标签数组，值为0(卖出)、1(观望)或2(买入)
        """
        try:
            if len(df) < 20:  # 确保有足够的数据
                self.logger.error("数据长度不足以生成标签")
                return None
                
            # 计算多个时间窗口的未来收益率
            future_returns = []
            lookback_windows = [1, 3, 5, 10, 15]  # 增加更长的时间窗口
            weights = [0.35, 0.25, 0.2, 0.15, 0.05]    # 调整权重分布
            
            for window in lookback_windows:
                future_return = df['returns'].shift(-window)
                future_returns.append(future_return)
                
            # 计算加权收益率
            weighted_returns = pd.Series(0, index=df.index)
            for w, r in zip(weights, future_returns):
                weighted_returns += w * r
                
            # 计算趋势强度
            fast_ma = df['close'].rolling(window=5).mean()
            slow_ma = df['close'].rolling(window=20).mean()
            trend_strength = (fast_ma - slow_ma) / slow_ma
            
            # 计算波动率
            returns = df['close'].pct_change()
            volatility = returns.rolling(window=20).std()
            
            # 增加趋势持续性指标
            trend_consistency = ((df['close'] > fast_ma) & (fast_ma > slow_ma)).astype(int) * 2 - 1
            trend_duration = trend_consistency.rolling(window=10).mean()
            
            # 综合考虑收益率、趋势强度和趋势持续性
            adjusted_returns = weighted_returns + 0.5 * trend_strength + 0.2 * trend_duration
            
            # 动态阈值：基于波动率调整分位数范围
            volatility_percentile = volatility.rank(pct=True)
            base_threshold = 0.2  # 基础阈值
            threshold_adjustment = 0.1 * (1 - volatility_percentile)  # 波动率越大，阈值范围越小
            
            buy_threshold = adjusted_returns.quantile(0.8 - threshold_adjustment)
            sell_threshold = adjusted_returns.quantile(0.2 + threshold_adjustment)
            
            # 生成标签
            labels = pd.Series(1, index=df.index)  # 默认为观望(1)
            labels[adjusted_returns > buy_threshold] = 2  # 买入
            labels[adjusted_returns < sell_threshold] = 0  # 卖出
            
            # 统计标签分布
            label_counts = labels.value_counts()
            total = len(labels)
            self.logger.info(f"标签分布: 卖出={label_counts.get(0, 0)} ({label_counts.get(0, 0)/total:.1%}), "
                           f"观望={label_counts.get(1, 0)} ({label_counts.get(1, 0)/total:.1%}), "
                           f"买入={label_counts.get(2, 0)} ({label_counts.get(2, 0)/total:.1%})")
            
            return labels.values
            
        except Exception as e:
            self.logger.error(f"生成标签时出错: {str(e)}")
            return None

    def train_model(self, klines):
        """训练模型"""
        try:
            if len(klines) < 100:  
                self.logger.warning("训练数据不足，需要至少100根K线")
                return False
                
            # 准备特征数据
            features = self.prepare_features(klines[:-1])
            if features.empty:
                self.logger.error("特征准备失败，无法训练模型")
                return False
                
            # 准备标签数据
            try:
                labels = self.prepare_labels(pd.DataFrame({'returns': features['returns']}))
                if labels is None:
                    self.logger.error("标签准备失败，无法训练模型")
                    return False
            except Exception as e:
                self.logger.error(f"准备标签数据失败: {str(e)}")
                return False
            
            # 去除NaN值
            try:
                valid_idx = ~np.isnan(labels)
                features = features[valid_idx]
                labels = labels[valid_idx]
                
                if len(features) < 50:
                    self.logger.warning("有效训练数据不足，需要至少50个有效样本")
                    return False
            except Exception as e:
                self.logger.error(f"数据清洗失败: {str(e)}")
                return False
            
            # 标准化特征
            try:
                features_scaled = self.scaler.fit_transform(features)
            except Exception as e:
                self.logger.error(f"特征标准化失败: {str(e)}")
                return False
            
            # 训练模型
            try:
                self.model.fit(features_scaled, labels)
                
                # 计算训练集准确率和损失
                train_pred = self.model.predict(features_scaled)
                train_accuracy = np.mean(train_pred == labels)
                
                # 计算交叉熵损失
                try:
                    train_proba = self.model.predict_proba(features_scaled)
                    train_loss = -np.mean(labels * np.log(train_proba[:, 1] + 1e-10) + 
                                        (1 - labels) * np.log(1 - train_proba[:, 1] + 1e-10))
                except Exception as e:
                    self.logger.error(f"计算训练损失失败: {str(e)}")
                    train_loss = 0
                
                self.logger.info(f"模型训练指标 - Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}")
                return True
                
            except Exception as e:
                self.logger.error(f"模型训练失败: {str(e)}")
                return False
                
        except Exception as e:
            self.logger.error(f"训练过程出错: {str(e)}")
            return False
            
    def log_trade_stats(self, current_price=None):
        """记录交易统计信息"""
        # 记录总体交易统计
        if self.total_trades > 0:
            win_rate = (self.winning_trades / self.total_trades) * 100
            avg_profit = self.total_profit / self.total_trades
            self.logger.info(f"交易统计 - 总交易次数: {self.total_trades}, 胜率: {win_rate:.2f}%, "
                           f"总盈利: {self.total_profit:.2f} USDT (已实现: {self.realized_profit:.2f} USDT), "
                           f"平均盈利: {avg_profit:.2f} USDT")
        
        # 记录当前持仓盈亏
        if current_price and self.last_trade_price > 0:
            position = self.trader.get_position()
            if position:
                position_size = float(position['info'].get('positionAmt', 0))
                entry_price = float(position['info'].get('entryPrice', 0))
                if position_size != 0:
                    if position_size > 0:  
                        unrealized_profit = (current_price - entry_price) * abs(position_size)
                        profit_rate = ((current_price / entry_price) - 1) * 100
                    else:  
                        unrealized_profit = (entry_price - current_price) * abs(position_size)
                        profit_rate = ((entry_price / current_price) - 1) * 100
                    
                    self.logger.info(f"当前持仓 - 方向: {'多' if position_size > 0 else '空'}, "
                                   f"数量: {abs(position_size):.4f}, "
                                   f"开仓价: {entry_price:.2f}, "
                                   f"当前价: {current_price:.2f}, "
                                   f"未实现盈亏: {unrealized_profit:.2f} USDT, "
                                   f"盈亏率: {profit_rate:.2f}%")
                    
                    # 记录当前持仓的入场价格
                    if self.current_position_entry_price != entry_price:
                        self.current_position_entry_price = entry_price
                        self.last_trade_price = entry_price
                        self.last_trade_type = 'long' if position_size > 0 else 'short'
                
    def update_trade_stats(self, current_price):
        """更新交易统计"""
        position = self.trader.get_position()
        position_size = float(position['info'].get('positionAmt', 0)) if position else 0
        
        # 如果之前有持仓，现在没有持仓，说明已经平仓
        if self.last_trade_price > 0 and self.last_trade_type and position_size == 0:
            profit = 0
            if self.last_trade_type == 'long':
                profit = (current_price - self.last_trade_price) * self.trader.position_size
            elif self.last_trade_type == 'short':
                profit = (self.last_trade_price - current_price) * self.trader.position_size
                
            if profit != 0:
                self.total_trades += 1
                if profit > 0:
                    self.winning_trades += 1
                self.total_profit += profit
                self.realized_profit += profit
                self.logger.info(f"平仓统计 - 方向: {self.last_trade_type}, "
                               f"开仓价: {self.last_trade_price:.2f}, "
                               f"平仓价: {current_price:.2f}, "
                               f"实现盈亏: {profit:.2f} USDT")
                self.log_trade_stats()
                
                # 重置交易记录
                self.last_trade_price = 0
                self.last_trade_type = None
                self.current_position_entry_price = 0
                
    def adjust_threshold(self, market_volatility, prediction_confidence):
        """动态调整预测阈值"""
        # 基于市场波动性调整阈值
        volatility_adjustment = market_volatility * self.threshold_adjustment_factor
        
        # 基于预测置信度调整阈值
        confidence_adjustment = (1 - prediction_confidence) * self.threshold_adjustment_factor
        
        # 计算新的动态阈值
        self.dynamic_threshold = min(0.85, max(0.55,
            self.base_threshold + volatility_adjustment + confidence_adjustment
        ))
        
        self.logger.info(f"动态阈值调整 - 新阈值: {self.dynamic_threshold:.4f} "
                        f"(波动性调整: {volatility_adjustment:.4f}, "
                        f"置信度调整: {confidence_adjustment:.4f})")
        
    def generate_signals(self, klines):
        """生成交易信号"""
        try:
            current_time = time.time()
            
            # 检查最小交易间隔
            if current_time - self.last_trade_time < config.AI_MIN_TRADE_INTERVAL * 60:
                self.logger.info("未达到最小交易间隔")
                return 0
                
            if len(klines) < self.lookback:
                self.logger.warning("K线数据不足，无法生成信号")
                return 0
                
            # 基于时间间隔重新训练模型
            if not hasattr(self, 'last_train_time'):
                self.last_train_time = 0
                
            if current_time - self.last_train_time > config.AI_TRAIN_INTERVAL * 60:
                self.logger.info("开始定期重新训练模型")
                if self.train_model(klines):
                    self.last_train_time = current_time
                    self.logger.info("模型重新训练完成")
                else:
                    self.logger.warning("模型重新训练失败")
                    
            # 准备最新的特征数据
            try:
                features = self.prepare_features(klines)
                if features.empty:
                    self.logger.warning("特征数据为空，无法生成信号")
                    return 0
                    
                latest_features = features.iloc[-1:]
                current_price = float(klines[-1][4])  
            except Exception as e:
                self.logger.error(f"特征准备失败: {str(e)}")
                return 0
            
            # 更新交易统计
            try:
                self.update_trade_stats(current_price)
                self.log_trade_stats(current_price)
            except Exception as e:
                self.logger.error(f"更新交易统计失败: {str(e)}")
                # 继续执行，因为这不是关键错误
            
            # 标准化特征
            try:
                latest_features_scaled = self.scaler.transform(latest_features)
            except Exception as e:
                self.logger.error(f"特征标准化失败: {str(e)}")
                return 0
            
            # 预测
            try:
                prob = self.model.predict_proba(latest_features_scaled)[0]
                self.logger.info(f"上涨概率: {prob[1]:.2f}, 下跌概率: {prob[0]:.2f}")
                
                # 生成交易信号
                try:
                    signal = self.generate_signal(features, prob[1])
                    
                    # 更新最后交易时间
                    if signal != 0:
                        self.last_trade_time = current_time
                        
                    return signal
                    
                except Exception as e:
                    self.logger.error(f"信号生成失败: {str(e)}")
                    return 0
                    
            except Exception as e:
                self.logger.error(f"模型预测失败: {str(e)}")
                return 0
                
        except Exception as e:
            self.logger.error(f"生成信号时发生错误: {str(e)}")
            return 0
            
    def generate_signal(self, features, pred_proba):
        """生成交易信号"""
        try:
            # 获取最新的市场特征
            market_volatility = features['volatility'].iloc[-1]
            trend_strength = features['trend_strength'].iloc[-1]
            rsi = features['RSI'].iloc[-1]
            
            # 获取当前持仓信息
            position = self.trader.get_position()
            position_size = float(position['info'].get('positionAmt', 0)) if position else 0
            
            # 计算预测置信度
            prediction_confidence = abs(pred_proba - 0.5) * 2
            
            # 调整预测阈值
            self.adjust_threshold(market_volatility, prediction_confidence)
            
            # 趋势确认
            trend_confirmed = (trend_strength > 0.01) if pred_proba > 0.5 else (trend_strength < -0.01)
            
            # RSI过滤
            rsi_confirm_long = rsi < 70  # 非超买
            rsi_confirm_short = rsi > 30  # 非超卖
            
            # 根据当前持仓调整信号阈值
            if position_size > 0:  # 持有多仓
                sell_threshold = self.dynamic_threshold - 0.05  # 降低平仓阈值
                buy_threshold = self.dynamic_threshold + 0.1   # 提高加仓阈值
            elif position_size < 0:  # 持有空仓
                sell_threshold = self.dynamic_threshold + 0.1  # 提高加仓阈值
                buy_threshold = self.dynamic_threshold - 0.05  # 降低平仓阈值
            else:  # 无持仓
                sell_threshold = self.dynamic_threshold
                buy_threshold = self.dynamic_threshold
            
            # 生成信号
            if pred_proba > buy_threshold and trend_confirmed and rsi_confirm_long:
                return 1  # 买入信号
            elif pred_proba < (1 - sell_threshold) and trend_confirmed and rsi_confirm_short:
                return -1  # 卖出信号
            return 0  # 持仓不变
            
        except Exception as e:
            self.logger.error(f"信号生成过程出错: {str(e)}")
            return 0

    def should_close_position(self, position_info, current_price):
        """判断是否应该平仓"""
        if not position_info:
            return False
            
        entry_price = float(position_info['entryPrice'])
        position_amount = float(position_info['positionAmt'])
        
        # 计算盈亏百分比
        if position_amount > 0:  
            profit_percent = (current_price - entry_price) / entry_price
        else:  
            profit_percent = (entry_price - current_price) / entry_price
            
        # 止损或止盈
        if profit_percent <= -self.stop_loss_percent:
            self.logger.info(f"触发止损: {profit_percent:.2%}")
            return True
        elif profit_percent >= self.take_profit_percent:
            self.logger.info(f"触发止盈: {profit_percent:.2%}")
            return True
            
        return False
