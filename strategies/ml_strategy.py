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
        self.lookback = config.AI_KLINES_LIMIT  # 使用配置文件中的K线数量
        self.prediction_threshold = 0.65  # 提高预测概率阈值以减少误判
        
        # 技术指标参数
        self.rsi_period = config.RSI_PERIOD
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        
        # 初始化模型
        self.model = RandomForestClassifier(
            n_estimators=200,  # 增加树的数量
            max_depth=8,       # 增加树的深度
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1  # 使用所有CPU核心
        )
        self.scaler = StandardScaler()
        
        # 止损和止盈设置
        self.stop_loss_percent = config.DEFAULT_STOP_LOSS_PERCENT / 100
        self.take_profit_percent = config.DEFAULT_TAKE_PROFIT_PERCENT / 100
        
        # 记录上一次的信号和交易时间
        self.last_signal = 0
        self.last_trade_time = 0
        self.trades_today = 0
        self.last_day = None
        
    def prepare_features(self, klines):
        """准备特征数据"""
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                                         'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignored'])
        
        # 转换为数值类型
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])
            
        # 基础技术指标
        df['RSI'] = talib.RSI(df['close'], timeperiod=self.rsi_period)
        macd, signal, hist = talib.MACD(df['close'], 
                                      fastperiod=self.macd_fast, 
                                      slowperiod=self.macd_slow, 
                                      signalperiod=self.macd_signal)
        df['MACD'] = macd
        df['MACD_SIGNAL'] = signal
        df['MACD_HIST'] = hist
        
        # 价格动量指标
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close']/df['close'].shift(1))
        df['volatility'] = df['returns'].rolling(window=20).std()
        
        # 趋势指标
        df['EMA_10'] = talib.EMA(df['close'], timeperiod=10)
        df['EMA_20'] = talib.EMA(df['close'], timeperiod=20)
        df['EMA_50'] = talib.EMA(df['close'], timeperiod=50)
        df['trend_strength'] = ((df['EMA_10'] - df['EMA_20']) / df['EMA_20'] * 100)
        
        # 成交量指标
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # ATR和布林带
        df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        upper, middle, lower = talib.BBANDS(df['close'], timeperiod=20)
        df['BB_width'] = (upper - lower) / middle
        
        # 准备特征
        feature_columns = [
            'RSI', 'MACD', 'MACD_SIGNAL', 'MACD_HIST',
            'returns', 'log_returns', 'volatility',
            'trend_strength', 'volume_ratio', 'ATR',
            'BB_width'
        ]
        features = df[feature_columns].ffill().fillna(0)
        
        return features
        
    def prepare_labels(self, df):
        """准备训练标签"""
        # 使用未来的收益率作为标签
        future_returns = df['returns'].shift(-1)
        labels = (future_returns > 0).astype(int)
        return labels
        
    def train_model(self, klines):
        """训练模型"""
        if len(klines) < 100:  # 确保有足够的数据
            return False
            
        features = self.prepare_features(klines[:-1])  # 不使用最新的数据点
        labels = self.prepare_labels(pd.DataFrame({'returns': features['returns']}))
        
        # 去除NaN值
        valid_idx = ~labels.isna()
        features = features[valid_idx]
        labels = labels[valid_idx]
        
        if len(features) < 50:  # 确保有足够的训练数据
            return False
            
        # 标准化特征
        features_scaled = self.scaler.fit_transform(features)
        
        # 训练模型
        try:
            self.model.fit(features_scaled, labels)
            
            # 计算训练集准确率和损失
            train_pred = self.model.predict(features_scaled)
            train_accuracy = np.mean(train_pred == labels)
            train_loss = -np.mean(labels * np.log(self.model.predict_proba(features_scaled)[:, 1] + 1e-10) + 
                                (1 - labels) * np.log(1 - self.model.predict_proba(features_scaled)[:, 1] + 1e-10))
            
            self.logger.info(f"模型训练指标 - Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}")
            return True
        except Exception as e:
            self.logger.error(f"模型训练失败: {str(e)}")
            return False
            
    def generate_signals(self, klines):
        """生成交易信号"""
        current_time = time.time()
        
        # 检查每日交易限制
        current_day = time.strftime("%Y-%m-%d")
        if current_day != self.last_day:
            self.trades_today = 0
            self.last_day = current_day
        
        if self.trades_today >= config.MAX_TRADES_PER_DAY:
            self.logger.info("已达到每日最大交易次数限制")
            return 0
        
        # 检查最小交易间隔
        if current_time - self.last_trade_time < config.MIN_TRADE_INTERVAL:
            self.logger.info("未达到最小交易间隔")
            return 0
            
        if len(klines) < self.lookback:
            self.logger.warning("K线数据不足，无法生成信号")
            return 0
            
        # 每100根K线重新训练一次模型
        if len(klines) % config.RETRAIN_INTERVAL == 0:
            if not self.train_model(klines):
                return 0
                
        # 准备最新的特征数据
        features = self.prepare_features(klines)
        latest_features = features.iloc[-1:]
        
        # 标准化特征
        latest_features_scaled = self.scaler.transform(latest_features)
        
        # 预测
        try:
            prob = self.model.predict_proba(latest_features_scaled)[0]
            self.logger.info(f"上涨概率: {prob[1]:.2f}, 下跌概率: {prob[0]:.2f}")
            
            # 根据预测概率生成信号
            if prob[1] > self.prediction_threshold:  # 看多信号
                    return 1  # 开多
            elif prob[0] > self.prediction_threshold:  # 看空信号
                    return -1  # 开空
            else:
                self.logger.info("预测概率未达到阈值，观望信号")
                return 0
                
        except Exception as e:
            self.logger.error(f"生成信号时发生错误: {str(e)}")
            return 0
            
    def should_close_position(self, position_info, current_price):
        """判断是否应该平仓"""
        if not position_info:
            return False
            
        entry_price = float(position_info['entryPrice'])
        position_amount = float(position_info['positionAmt'])
        
        # 计算盈亏百分比
        if position_amount > 0:  # 多头
            profit_percent = (current_price - entry_price) / entry_price
        else:  # 空头
            profit_percent = (entry_price - current_price) / entry_price
            
        # 止损或止盈
        if profit_percent <= -self.stop_loss_percent:
            self.logger.info(f"触发止损: {profit_percent:.2%}")
            return True
        elif profit_percent >= self.take_profit_percent:
            self.logger.info(f"触发止盈: {profit_percent:.2%}")
            return True
            
        return False
