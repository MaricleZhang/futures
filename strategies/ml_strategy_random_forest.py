import numpy as np
import pandas as pd
from strategies.ml_strategy import MLStrategy
import talib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import time
import config

class RandomForestStrategy(MLStrategy):
    """RandomForestMaster - 随机森林交易策略
    
    一个基于随机森林的期货交易策略模型，使用集成学习方法，
    结合技术指标、市场情绪和波动特征来预测市场走势。
    
    特点：
    1. 多特征融合：整合技术指标、市场情绪和波动性等多维度特征
    2. 三分类预测：买入、卖出、持仓三种行为的概率预测
    3. 动态阈值：使用置信度阈值过滤低置信度的交易信号
    4. 特征重要性：可解释性强，能够理解每个特征的重要程度
    """
    
    MODEL_NAME = "RandomForestMaster"
    VERSION = "1.0.0"
    
    def __init__(self, trader):
        """初始化随机森林策略"""
        super().__init__(trader)
        self.logger = self.get_logger()
        
        # 随机森林特定参数
        self.n_estimators = 50   # 保持树的数量
        self.max_depth = 4       # 保持树的深度
        self.min_samples_split = 20
        self.min_samples_leaf = 10
        self.confidence_threshold = 0.45  # 降低置信度阈值，使模型更容易产生交易信号
        
        # K线设置
        self.kline_interval = '1m'
        self.training_lookback = 1000
        self.retraining_interval = 300
        self.last_training_time = 0
        
        # 初始化StandardScaler
        self.scaler = StandardScaler()
        
        # 初始化模型
        self.initialize_model()
        
        # 模型评估指标
        self.feature_importance = None
        self.val_accuracies = []
        
        # 获取历史数据进行训练
        try:
            self.logger.info("获取历史K线数据...")
            historical_data = self.trader.get_klines(
                interval=self.kline_interval,
                limit=self.training_lookback
            )
            
            if historical_data and len(historical_data) > 0:
                self.logger.info(f"成功获取{len(historical_data)}根{self.kline_interval}K线数据")
                if self.train_model(historical_data):
                    self.logger.info("模型训练完成")
                else:
                    self.logger.error("模型训练失败")
            else:
                self.logger.error("获取历史数据失败")
        except Exception as e:
            self.logger.error(f"初始化训练失败: {str(e)}")

    def initialize_model(self):
        """初始化随机森林模型"""
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced_subsample',  
            bootstrap=True,
            max_features='sqrt',
            max_samples=0.7  
        )

    def prepare_features(self, klines):
        """准备特征数据"""
        if not isinstance(klines, list) or len(klines) == 0:
            self.logger.error("K线数据为空或格式不正确")
            return None
            
        try:
            # 创建DataFrame
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            if df.empty:
                self.logger.error("创建的DataFrame为空")
                return None
                
            # 转换为数值类型
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
            # 添加更多特征
            try:
                # === 1. 波动性指标 ===
                # ATR - 短期和长期
                df['ATR_short'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=5)  
                df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)  
                
                # === 2. 动量指标 ===
                # RSI - 短期和中期
                df['RSI_short'] = talib.RSI(df['close'], timeperiod=6)  
                df['RSI'] = talib.RSI(df['close'], timeperiod=14)  
                
                # MACD - 快速设置
                macd, signal, hist = talib.MACD(df['close'], fastperiod=6, slowperiod=13, signalperiod=4)
                df['MACD'] = macd
                df['MACD_signal'] = signal
                df['MACD_hist'] = hist
                
                # === 3. 趋势指标 ===
                # ADX和DMI
                df['ADX'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
                df['DI_plus'] = talib.PLUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
                df['DI_minus'] = talib.MINUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
                
                # === 4. 价格通道 ===
                # 布林带 - 短期和标准期
                upper_short, middle_short, lower_short = talib.BBANDS(df['close'], timeperiod=10, nbdevup=2, nbdevdn=2)
                df['BB_upper_short'] = upper_short
                df['BB_middle_short'] = middle_short
                df['BB_lower_short'] = lower_short
                
                upper, middle, lower = talib.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2)
                df['BB_upper'] = upper
                df['BB_middle'] = middle
                df['BB_lower'] = lower
                
                # === 5. 价格动态特征 ===
                # 价格变化率 - 多个时间窗口
                df['price_change_1m'] = df['close'].pct_change(1)
                df['price_change_3m'] = df['close'].pct_change(3)
                df['price_change_5m'] = df['close'].pct_change(5)
                
                # 价格加速度（二阶导数）
                df['price_acceleration'] = df['price_change_1m'].diff()
                
                # === 6. 成交量特征 ===
                # 成交量变化率 - 多个时间窗口
                df['volume_change_1m'] = df['volume'].pct_change(1)
                df['volume_change_3m'] = df['volume'].pct_change(3)
                df['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(10).mean()
                
                # === 7. 波动率特征 ===
                # 价格波动率
                df['volatility_1m'] = df['close'].rolling(2).std()
                df['volatility_3m'] = df['close'].rolling(3).std()
                df['volatility_5m'] = df['close'].rolling(5).std()
                
                # === 8. 趋势强度 ===
                # 移动平均线斜率
                df['ma_slope_short'] = talib.LINEARREG_SLOPE(df['close'], timeperiod=5)
                df['ma_slope_mid'] = talib.LINEARREG_SLOPE(df['close'], timeperiod=10)
                
                # === 9. 价格形态 ===
                # 蜡烛图特征
                df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
                df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
                df['body_size'] = abs(df['open'] - df['close'])
                
                # 移除NaN值
                df = df.dropna()
                
                return df
                
            except Exception as e:
                self.logger.error(f"特征准备失败: {str(e)}")
                return None
                
        except Exception as e:
            self.logger.error(f"数据预处理失败: {str(e)}")
            return None

    def generate_signals(self, klines):
        """生成交易信号"""
        try:
            # 检查是否需要重新训练
            current_time = time.time()
            if current_time - self.last_training_time > self.retraining_interval:
                self.train_model(klines)
                self.last_training_time = current_time

            # 准备特征
            features = self.prepare_features(klines)
            if features is None or features.empty:
                return 0

            # 选择最新的数据点
            latest_features = features.iloc[-1:]
            
            # 选择特征列
            feature_columns = [col for col in features.columns 
                             if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            X = latest_features[feature_columns].values

            # 确保scaler已经被训练
            if not hasattr(self, 'scaler') or self.scaler is None:
                self.logger.error("Scaler未初始化，重新训练模型")
                self.train_model(klines)
                return 0

            try:
                X_scaled = self.scaler.transform(X)
            except Exception as e:
                self.logger.error(f"特征标准化失败: {str(e)}")
                self.train_model(klines)
                return 0

            # 预测概率
            probabilities = self.model.predict_proba(X_scaled)[0]
            
            # 记录预测概率
            self.logger.info(f"预测概率: 卖出={probabilities[0]:.4f}, 观望={probabilities[1]:.4f}, 买入={probabilities[2]:.4f}")
            
            # 获取最高概率及其对应的类别
            max_prob = max(probabilities)
            predicted_class = np.argmax(probabilities)
            
            # 记录预测结果和置信度
            self.logger.info(f"预测结果: {['卖出', '观望', '买入'][predicted_class]} (置信度: {max_prob:.4f})")
            
            # 根据置信度阈值和预测类别生成信号
            if max_prob >= self.confidence_threshold:
                if predicted_class == 0:  # 卖出信号
                    return -1
                elif predicted_class == 2:  # 买入信号
                    return 1
            elif max_prob < 0.4:  # 如果最高概率太低，考虑次高概率
                sorted_probs = sorted(enumerate(probabilities), key=lambda x: x[1], reverse=True)
                second_class, second_prob = sorted_probs[1]
                if second_prob >= self.confidence_threshold * 0.9:  # 稍微降低次优选择的阈值
                    if second_class == 0:  # 卖出信号
                        return -1
                    elif second_class == 2:  # 买入信号
                        return 1
            
            return 0  # 持仓不变
            
        except Exception as e:
            self.logger.error(f"生成信号失败: {str(e)}")
            return 0

    def train_model(self, klines):
        """训练随机森林模型"""
        try:
            # 准备特征和标签
            features_df = self.prepare_features(klines)
            if features_df is None or features_df.empty:
                return False

            # 生成标签（-1: 卖出, 0: 持有, 1: 买入）
            labels = self.generate_labels(features_df)
            if labels is None:
                return False

            # 选择特征列
            feature_columns = [col for col in features_df.columns 
                             if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            # 确保特征和标签的长度一致
            X = features_df[feature_columns].values
            X = X[:-5]  # 去掉最后5行，因为标签生成时去掉了最后5个点
            y = labels

            if len(X) != len(y):
                self.logger.error(f"特征和标签长度不匹配: X={len(X)}, y={len(y)}")
                return False

            # 标准化特征
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)

            # 训练模型
            self.model.fit(X_scaled, y)

            # 记录特征重要性
            self.feature_importance = dict(zip(feature_columns, self.model.feature_importances_))
            
            # 计算训练集准确率
            train_accuracy = self.model.score(X_scaled, y)
            self.val_accuracies.append(train_accuracy)
            
            self.logger.info(f"模型训练完成，训练集准确率: {train_accuracy:.4f}")
            self.logger.info("Top 5 重要特征:")
            sorted_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
            for feature, importance in sorted_features:
                self.logger.info(f"{feature}: {importance:.4f}")

            return True

        except Exception as e:
            self.logger.error(f"模型训练失败: {str(e)}")
            return False

    def generate_labels(self, df):
        """生成训练标签"""
        try:
            if len(df) < 6:
                self.logger.error("数据长度不足以生成标签")
                return None
                
            # 计算多个时间窗口的未来收益
            future_returns_1m = df['close'].pct_change(-1)
            future_returns_3m = df['close'].pct_change(-3)
            future_returns_5m = df['close'].pct_change(-5)
            
            # 计算波动率
            volatility = df['volatility_1m'].ffill().rolling(5, min_periods=1).mean()
            avg_volatility = volatility.mean()
            
            # 动态阈值：基于平均波动率调整
            base_threshold = 0.0004  
            vol_multiplier = 0.8     
            
            # 计算动态阈值，使用平均波动率
            dynamic_threshold = base_threshold * (1 + vol_multiplier * (volatility / avg_volatility).ffill())
            
            # 初始化标签
            labels = np.zeros(len(df)-5)
            
            # 确保所有数据都有效
            future_returns_1m = future_returns_1m.ffill()
            future_returns_3m = future_returns_3m.ffill()
            future_returns_5m = future_returns_5m.ffill()
            
            # 截断数据以匹配长度
            future_returns_1m = future_returns_1m[:-5].values
            future_returns_3m = future_returns_3m[:-5].values
            future_returns_5m = future_returns_5m[:-5].values
            dynamic_threshold = dynamic_threshold[:-5].values
            
            # 生成标签（考虑多个时间窗口的走势）
            for i in range(len(labels)):
                # 计算综合收益率（进一步增加短期权重）
                weighted_return = (
                    0.8 * future_returns_1m[i] +  
                    0.15 * future_returns_3m[i] + 
                    0.05 * future_returns_5m[i]   
                )
                
                # 根据动态阈值判断买卖信号
                if weighted_return > dynamic_threshold[i]:
                    labels[i] = 2  # 买入
                elif weighted_return < -dynamic_threshold[i]:
                    labels[i] = 0  # 卖出
                else:
                    labels[i] = 1  # 持有
            
            if len(labels) == 0:
                self.logger.error("生成的标签为空")
                return None
                
            # 打印标签分布情况
            unique, counts = np.unique(labels, return_counts=True)
            distribution = dict(zip(unique, counts))
            self.logger.info(f"标签分布: 卖出={distribution.get(0, 0)}, "
                           f"观望={distribution.get(1, 0)}, "
                           f"买入={distribution.get(2, 0)}")
            
            # 计算买卖信号比例
            total_samples = len(labels)
            action_ratio = (distribution.get(0, 0) + distribution.get(2, 0)) / total_samples
            self.logger.info(f"买卖信号比例: {action_ratio:.2%}")
            
            return labels
            
        except Exception as e:
            self.logger.error(f"生成标签失败: {str(e)}")
            return None
