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
        self.n_estimators = 100  # 增加树的数量
        self.max_depth = 5      # 略微增加深度
        self.min_samples_split = 30  # 增加分裂所需样本数
        self.min_samples_leaf = 15   # 增加叶节点最小样本数
        self.confidence_threshold = 0.42  # 置信度阈值
        self.prob_diff_threshold = 0.11   # 降低概率差阈值
        
        # K线设置
        self.kline_interval = '1m'
        self.training_lookback = 1000
        self.retraining_interval = 300  # 5分钟重新训练一次
        self.last_training_time = 0
        
        # 初始化StandardScaler
        self.scaler = StandardScaler()
        
        # 初始化模型
        self.initialize_model()
        
        # 模型评估指标
        self.feature_importance = None
        self.val_accuracies = []
        
        # 趋势状态
        self.trend_state = 0  # -1: 下跌, 0: 盘整, 1: 上涨
        self.trend_confidence = 0.0  # 趋势置信度
        
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
            class_weight='balanced',  # 使用balanced而不是balanced_subsample
            bootstrap=True,
            max_features='sqrt',
            max_samples=0.8  # 增加采样比例
        )

    def prepare_features(self, klines):
        """准备特征数据"""
        try:
            if not isinstance(klines, list) or len(klines) < 20:  # 确保有足够的数据来计算指标
                self.logger.error("K线数据为空或长度不足")
                return None

            # 创建基础DataFrame
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # 转换数据类型
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 创建特征DataFrame
            features = pd.DataFrame(index=df.index)
            
            # 基础价格特征
            features['close'] = df['close']
            features['volume'] = df['volume']
            
            # 价格变化
            features['price_change_1m'] = df['close'].pct_change()
            features['price_change_3m'] = df['close'].pct_change(3)
            features['price_change_5m'] = df['close'].pct_change(5)
            
            # 成交量变化
            features['volume_change_1m'] = df['volume'].pct_change()
            features['volume_change_3m'] = df['volume'].pct_change(3)
            features['volume_change_5m'] = df['volume'].pct_change(5)
            
            # 移动平均
            features['SMA_5'] = df['close'].rolling(window=5).mean()
            features['SMA_10'] = df['close'].rolling(window=10).mean()
            features['SMA_20'] = df['close'].rolling(window=20).mean()
            
            # 成交量移动平均
            volume_ma = df['volume'].rolling(window=20).mean()
            features['volume_ma_ratio'] = df['volume'] / volume_ma
            
            # 布林带
            features['BB_middle'] = df['close'].rolling(window=20).mean()
            features['BB_upper'] = features['BB_middle'] + df['close'].rolling(window=20).std() * 2
            features['BB_lower'] = features['BB_middle'] - df['close'].rolling(window=20).std() * 2
            
            # 短期布林带
            features['BB_middle_short'] = df['close'].rolling(window=10).mean()
            features['BB_upper_short'] = features['BB_middle_short'] + df['close'].rolling(window=10).std() * 2
            features['BB_lower_short'] = features['BB_middle_short'] - df['close'].rolling(window=10).std() * 2
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            features['RSI'] = 100 - (100 / (1 + gain / loss))
            
            # MACD
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            features['MACD'] = exp1 - exp2
            features['Signal_Line'] = features['MACD'].ewm(span=9, adjust=False).mean()
            
            # ATR
            high_low = df['high'] - df['low']
            high_close = (df['high'] - df['close'].shift()).abs()
            low_close = (df['low'] - df['close'].shift()).abs()
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            features['ATR'] = true_range.rolling(window=14).mean()
            
            # ADX
            plus_dm = df['high'].diff()
            minus_dm = df['low'].diff()
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm > 0] = 0
            
            tr = true_range
            plus_di = 100 * (plus_dm.rolling(window=14).mean() / tr.rolling(window=14).mean())
            minus_di = 100 * (minus_dm.rolling(window=14).mean() / tr.rolling(window=14).mean())
            dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).abs())
            features['ADX'] = dx.rolling(window=14).mean()
            
            # 删除任何包含NaN的行
            valid_mask = ~(features.isna().any(axis=1))
            features = features[valid_mask]
            
            # 选择特征列
            feature_columns = [col for col in features.columns 
                             if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            features = features[feature_columns]
            
            return features
            
        except Exception as e:
            self.logger.error(f"特征准备失败: {str(e)}")
            return None

    def generate_labels(self, klines):
        """生成训练标签"""
        try:
            if not isinstance(klines, list) or len(klines) < 20:
                self.logger.error("K线数据为空或长度不足")
                return None

            # 创建基础DataFrame
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # 转换数据类型
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # 计算未来收益率
            future_returns = []
            lookback_windows = [1, 3, 5, 10]  # 不同的时间窗口
            weights = [0.4, 0.3, 0.2, 0.1]    # 对应的权重
            
            for window in lookback_windows:
                future_close = df['close'].shift(-window)
                future_return = (future_close - df['close']) / df['close']
                future_returns.append(future_return)
            
            # 计算加权收益率
            weighted_returns = pd.Series(0.0, index=df.index)
            for w, r in zip(weights, future_returns):
                weighted_returns += w * r
            
            # 计算趋势特征
            df['SMA_fast'] = df['close'].rolling(window=5).mean()
            df['SMA_slow'] = df['close'].rolling(window=20).mean()
            
            # 计算趋势强度
            trend_strength = (df['SMA_fast'] - df['SMA_slow']) / df['SMA_slow']
            
            # 动态阈值：根据波动率调整
            volatility = df['close'].pct_change().rolling(window=20).std()
            threshold_factor = 1.5
            buy_threshold = threshold_factor * volatility
            sell_threshold = -threshold_factor * volatility
            
            # 生成标签
            labels = pd.Series(1, index=df.index)  # 默认全部为观望(1)
            
            valid_mask = ~(weighted_returns.isna() | trend_strength.isna() | buy_threshold.isna())
            
            # 计算调整后的收益率
            trend_factor = 0.3  # 趋势影响权重
            adjusted_returns = weighted_returns[valid_mask] + trend_factor * trend_strength[valid_mask]
            
            # 生成标签
            labels[valid_mask & (adjusted_returns > buy_threshold[valid_mask])] = 2  # 买入
            labels[valid_mask & (adjusted_returns < sell_threshold[valid_mask])] = 0  # 卖出
            
            # 删除NaN值
            labels = labels.dropna()
            
            # 统计标签分布
            label_counts = labels.value_counts()
            self.logger.info(f"标签分布: 卖出={label_counts.get(0, 0)}, "
                           f"观望={label_counts.get(1, 0)}, "
                           f"买入={label_counts.get(2, 0)}")
            
            # 计算买卖信号比例
            total_signals = len(labels)
            action_signals = sum(1 for l in labels if l != 1)
            signal_ratio = (action_signals / total_signals) * 100
            self.logger.info(f"买卖信号比例: {signal_ratio:.2f}%")
            
            return labels.values
            
        except Exception as e:
            self.logger.error(f"生成标签失败: {str(e)}")
            return None

    def update_trend_state(self, klines):
        """更新趋势状态"""
        try:
            if not isinstance(klines, list) or len(klines) < 20:
                self.logger.error("K线数据为空或长度不足")
                return False
                
            # 创建基础DataFrame
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # 转换数据类型
            df['close'] = pd.to_numeric(df['close'], errors='coerce')
            
            # 计算快速和慢速移动平均
            fast_ma = df['close'].rolling(window=5).mean()
            slow_ma = df['close'].rolling(window=20).mean()
            
            # 计算趋势强度
            trend_strength = (fast_ma - slow_ma) / slow_ma
            
            # 获取最新的趋势强度
            latest_trend = trend_strength.iloc[-1]
            
            # 更新趋势状态和置信度
            if latest_trend > 0.001:  # 上涨趋势
                self.trend_state = 1
                self.trend_confidence = min(abs(latest_trend) * 10, 1.0)
            elif latest_trend < -0.001:  # 下跌趋势
                self.trend_state = -1
                self.trend_confidence = min(abs(latest_trend) * 10, 1.0)
            else:  # 无明显趋势
                self.trend_state = 0
                self.trend_confidence = 0.0
                
            self.logger.info(f"趋势状态: {['下跌', '无趋势', '上涨'][self.trend_state + 1]}, "
                           f"置信度: {self.trend_confidence:.4f}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"趋势状态更新失败: {str(e)}")
            return False

    def generate_signals(self, klines):
        """生成交易信号"""
        try:
            # 检查是否需要重新训练
            current_time = time.time()
            if current_time - self.last_training_time > self.retraining_interval:
                self.train_model(klines)
                self.last_training_time = current_time

            # 准备特征
            if not isinstance(klines, list) or len(klines) < 20:
                self.logger.error("K线数据为空或长度不足")
                return 0
                
            # 创建基础DataFrame
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # 转换数据类型
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 准备特征
            features = pd.DataFrame(index=df.index)
            
            # 基础价格特征
            features['close'] = df['close']
            features['volume'] = df['volume']
            
            # 价格变化
            features['price_change_1m'] = df['close'].pct_change()
            features['price_change_3m'] = df['close'].pct_change(3)
            features['price_change_5m'] = df['close'].pct_change(5)
            
            # 成交量变化
            features['volume_change_1m'] = df['volume'].pct_change()
            features['volume_change_3m'] = df['volume'].pct_change(3)
            features['volume_change_5m'] = df['volume'].pct_change(5)
            
            # 移动平均
            features['SMA_5'] = df['close'].rolling(window=5).mean()
            features['SMA_10'] = df['close'].rolling(window=10).mean()
            features['SMA_20'] = df['close'].rolling(window=20).mean()
            
            # 成交量移动平均
            volume_ma = df['volume'].rolling(window=20).mean()
            features['volume_ma_ratio'] = df['volume'] / volume_ma
            
            # 布林带
            features['BB_middle'] = df['close'].rolling(window=20).mean()
            features['BB_upper'] = features['BB_middle'] + df['close'].rolling(window=20).std() * 2
            features['BB_lower'] = features['BB_middle'] - df['close'].rolling(window=20).std() * 2
            
            # 短期布林带
            features['BB_middle_short'] = df['close'].rolling(window=10).mean()
            features['BB_upper_short'] = features['BB_middle_short'] + df['close'].rolling(window=10).std() * 2
            features['BB_lower_short'] = features['BB_middle_short'] - df['close'].rolling(window=10).std() * 2
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            features['RSI'] = 100 - (100 / (1 + gain / loss))
            
            # MACD
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            features['MACD'] = exp1 - exp2
            features['Signal_Line'] = features['MACD'].ewm(span=9, adjust=False).mean()
            
            # ATR
            high_low = df['high'] - df['low']
            high_close = (df['high'] - df['close'].shift()).abs()
            low_close = (df['low'] - df['close'].shift()).abs()
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            features['ATR'] = true_range.rolling(window=14).mean()
            
            # ADX
            plus_dm = df['high'].diff()
            minus_dm = df['low'].diff()
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm > 0] = 0
            
            tr = true_range
            plus_di = 100 * (plus_dm.rolling(window=14).mean() / tr.rolling(window=14).mean())
            minus_di = 100 * (minus_dm.rolling(window=14).mean() / tr.rolling(window=14).mean())
            dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).abs())
            features['ADX'] = dx.rolling(window=14).mean()
            
            # 删除包含NaN的行
            features = features.dropna()
            
            # 选择特征列
            feature_columns = [col for col in features.columns 
                             if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            # 更新趋势状态
            self.update_trend_state(klines)
            
            # 选择最新的数据点
            latest_features = features[feature_columns].iloc[-1:]
            
            # 确保scaler已经被训练
            if not hasattr(self, 'scaler') or self.scaler is None:
                self.logger.error("Scaler未初始化，重新训练模型")
                self.train_model(klines)
                return 0

            try:
                X_scaled = self.scaler.transform(latest_features)
            except Exception as e:
                self.logger.error(f"特征标准化失败: {str(e)}")
                self.train_model(klines)
                return 0

            # 预测概率
            probabilities = self.model.predict_proba(X_scaled)[0]
            
            # 记录预测概率
            self.logger.info(f"预测概率: 卖出={probabilities[0]:.4f}, "
                           f"观望={probabilities[1]:.4f}, "
                           f"买入={probabilities[2]:.4f}")
            
            # 获取最高概率及其对应的类别
            max_prob = max(probabilities)
            predicted_class = np.argmax(probabilities)
            
            # 记录预测结果和置信度
            self.logger.info(f"最终预测: {['卖出', '观望', '买入'][predicted_class]} "
                           f"(置信度: {max_prob:.4f})")
            
            # 生成交易信号
            if max_prob >= self.confidence_threshold:
                if predicted_class == 0:  # 卖出信号
                    # if probabilities[0] - probabilities[2] > self.prob_diff_threshold:
                        return -1
                elif predicted_class == 2:  # 买入信号
                    if probabilities[2] - probabilities[0] > self.prob_diff_threshold:
                        return 1
            
            return 0  # 持仓不变
            
        except Exception as e:
            self.logger.error(f"生成信号失败: {str(e)}")
            return 0

    def train_model(self, klines):
        """训练随机森林模型"""
        try:
            # 创建基础DataFrame
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # 转换数据类型
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 准备特征
            features = pd.DataFrame(index=df.index)
            
            # 基础价格特征
            features['close'] = df['close']
            features['volume'] = df['volume']
            
            # 价格变化
            features['price_change_1m'] = df['close'].pct_change()
            features['price_change_3m'] = df['close'].pct_change(3)
            features['price_change_5m'] = df['close'].pct_change(5)
            
            # 成交量变化
            features['volume_change_1m'] = df['volume'].pct_change()
            features['volume_change_3m'] = df['volume'].pct_change(3)
            features['volume_change_5m'] = df['volume'].pct_change(5)
            
            # 移动平均
            features['SMA_5'] = df['close'].rolling(window=5).mean()
            features['SMA_10'] = df['close'].rolling(window=10).mean()
            features['SMA_20'] = df['close'].rolling(window=20).mean()
            
            # 成交量移动平均
            volume_ma = df['volume'].rolling(window=20).mean()
            features['volume_ma_ratio'] = df['volume'] / volume_ma
            
            # 布林带
            features['BB_middle'] = df['close'].rolling(window=20).mean()
            features['BB_upper'] = features['BB_middle'] + df['close'].rolling(window=20).std() * 2
            features['BB_lower'] = features['BB_middle'] - df['close'].rolling(window=20).std() * 2
            
            # 短期布林带
            features['BB_middle_short'] = df['close'].rolling(window=10).mean()
            features['BB_upper_short'] = features['BB_middle_short'] + df['close'].rolling(window=10).std() * 2
            features['BB_lower_short'] = features['BB_middle_short'] - df['close'].rolling(window=10).std() * 2
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            features['RSI'] = 100 - (100 / (1 + gain / loss))
            
            # MACD
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            features['MACD'] = exp1 - exp2
            features['Signal_Line'] = features['MACD'].ewm(span=9, adjust=False).mean()
            
            # ATR
            high_low = df['high'] - df['low']
            high_close = (df['high'] - df['close'].shift()).abs()
            low_close = (df['low'] - df['close'].shift()).abs()
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            features['ATR'] = true_range.rolling(window=14).mean()
            
            # ADX
            plus_dm = df['high'].diff()
            minus_dm = df['low'].diff()
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm > 0] = 0
            
            tr = true_range
            plus_di = 100 * (plus_dm.rolling(window=14).mean() / tr.rolling(window=14).mean())
            minus_di = 100 * (minus_dm.rolling(window=14).mean() / tr.rolling(window=14).mean())
            dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).abs())
            features['ADX'] = dx.rolling(window=14).mean()
            
            # 计算未来收益率
            future_returns = []
            lookback_windows = [1, 3, 5, 10]  # 不同的时间窗口
            weights = [0.4, 0.3, 0.2, 0.1]    # 对应的权重
            
            for window in lookback_windows:
                future_close = df['close'].shift(-window)
                future_return = (future_close - df['close']) / df['close']
                future_returns.append(future_return)
            
            # 计算加权收益率
            weighted_returns = pd.Series(0.0, index=df.index)
            for w, r in zip(weights, future_returns):
                weighted_returns += w * r
            
            # 计算趋势特征
            trend_strength = (features['SMA_5'] - features['SMA_20']) / features['SMA_20']
            
            # 动态阈值：根据波动率调整
            volatility = df['close'].pct_change().rolling(window=20).std()
            threshold_factor = 1.5
            buy_threshold = threshold_factor * volatility
            sell_threshold = -threshold_factor * volatility
            
            # 生成标签
            labels = pd.Series(1, index=df.index)  # 默认全部为观望(1)
            
            # 计算调整后的收益率
            trend_factor = 0.3  # 趋势影响权重
            adjusted_returns = weighted_returns + trend_factor * trend_strength
            
            # 生成标签
            labels[adjusted_returns > buy_threshold] = 2  # 买入
            labels[adjusted_returns < sell_threshold] = 0  # 卖出
            
            # 删除任何包含NaN的行
            valid_mask = ~(features.isna().any(axis=1) | labels.isna())
            features = features[valid_mask]
            labels = labels[valid_mask]
            
            # 选择特征列
            feature_columns = [col for col in features.columns 
                             if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            features = features[feature_columns]
            
            # 确保特征和标签长度匹配
            if len(features) != len(labels):
                self.logger.error(f"特征和标签长度不匹配: X={len(features)}, y={len(labels)}")
                return False
                
            # 记录标签分布
            label_counts = labels.value_counts()
            self.logger.info(f"标签分布: 卖出={label_counts.get(0, 0)}, "
                           f"观望={label_counts.get(1, 0)}, "
                           f"买入={label_counts.get(2, 0)}")
            
            # 计算买卖信号比例
            total_signals = len(labels)
            action_signals = sum(1 for l in labels if l != 1)
            signal_ratio = (action_signals / total_signals) * 100
            self.logger.info(f"买卖信号比例: {signal_ratio:.2f}%")
            
            # 标准化特征
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(features)
            y = labels.values
            
            # 初始化并训练模型
            self.model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                class_weight='balanced',
                bootstrap=True,
                n_jobs=-1
            )
            
            # 训练模型
            self.model.fit(X, y)
            
            # 计算训练集准确率
            train_accuracy = self.model.score(X, y)
            self.logger.info(f"模型训练完成，训练集准确率: {train_accuracy:.4f}")
            
            # 获取特征重要性
            feature_importance = pd.Series(
                self.model.feature_importances_,
                index=features.columns
            ).sort_values(ascending=False)
            
            # 记录Top 5重要特征
            self.logger.info("Top 5 重要特征:")
            for feature, importance in feature_importance[:5].items():
                self.logger.info(f"{feature}: {importance:.4f}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"模型训练失败: {str(e)}")
            return False
