import numpy as np
import pandas as pd
import talib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import time
import config
from strategies.base_rf_strategy import BaseRFStrategy

class LongTermRFStrategy(BaseRFStrategy):
    """LongTermRFStrategy - 长线随机森林交易策略
    
    针对长线交易优化的随机森林策略模型，使用15分钟K线数据，
    结合多个长期技术指标和成交量特征来捕捉大趋势机会。
    
    特点：
    1. 长期特征：使用15分钟到4小时的技术指标
    2. 低频交易：每15分钟更新一次模型，降低交易频率
    3. 严格风控：包含持仓时间、交易频率、止盈止损等多重风控措施
    4. 趋势跟踪：通过多周期趋势分析提高持仓收益
    """
    
    MODEL_NAME = "LongTermRF"
    VERSION = "1.0.0"
    
    def __init__(self, trader):
        """初始化长线随机森林策略"""
        super().__init__(trader)
        
        # 随机森林参数
        self.max_depth = 12         # 树的最大深度
        self.min_samples_split = 40 # 分裂所需最小样本数
        self.min_samples_leaf = 20  # 叶节点最小样本数
        self.confidence_threshold = 0.5   # 信号置信度阈值
        self.prob_diff_threshold = 0.12   # 概率差异阈值
        
        # K线设置
        self.kline_interval = '15m'  # 15分钟K线
        self.training_lookback = 2000  # 训练数据回看周期
        self.retraining_interval = 900  # 15分钟重新训练
        
        # 风险控制参数
        self.max_position_hold_time = 480  # 最大持仓时间(分钟)
        self.profit_target_pct = 0.015     # 目标利润率
        self.stop_loss_pct = 0.01          # 止损率
        self.max_trades_per_hour = 2       # 每小时最大交易次数
        self.min_vol_percentile = 50       # 最小成交量百分位
        
        # 初始化模型并开始训练
        self.initialize_model()
        self._initial_training()
    
    def prepare_features(self, klines):
        """准备特征数据"""
        try:
            if not isinstance(klines, list) or len(klines) < 48:
                self.logger.error("K线数据为空或长度不足")
                return None
            
            # 创建DataFrame
            df = pd.DataFrame(klines)
            df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 创建特征DataFrame
            features = pd.DataFrame(index=df.index)
            
            # 价格动量指标 - 长期
            features['momentum_15m'] = df['close'] - df['close'].shift(1)
            features['momentum_1h'] = df['close'] - df['close'].shift(4)
            features['momentum_4h'] = df['close'] - df['close'].shift(16)
            
            # 价格变化率和加速度
            features['price_change_15m'] = df['close'].pct_change(1)
            features['price_change_1h'] = df['close'].pct_change(4)
            features['price_change_4h'] = df['close'].pct_change(16)
            features['price_acceleration'] = features['price_change_15m'] - features['price_change_15m'].shift(1)
            
            # 波动率指标
            features['volatility_1h'] = df['close'].rolling(window=4).std()
            features['volatility_4h'] = df['close'].rolling(window=16).std()
            features['volatility_ratio'] = features['volatility_1h'] / features['volatility_4h']
            
            # 成交量分析
            features['volume_ma_12'] = df['volume'].rolling(window=12).mean()
            features['volume_ma_24'] = df['volume'].rolling(window=24).mean()
            features['volume_ratio_12'] = df['volume'] / features['volume_ma_12']
            features['volume_ratio_24'] = df['volume'] / features['volume_ma_24']
            features['volume_trend'] = df['volume'].pct_change()
            
            # 价格压力指标
            features['high_low_range'] = (df['high'] - df['low']) / df['close']
            features['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
            
            # 趋势强度指标
            features['trend_strength_1h'] = features['price_change_1h'].abs() * features['volume_ratio_12']
            features['trend_strength_4h'] = features['price_change_4h'].abs() * features['volume_ratio_24']
            
            # 布林带 - 多周期
            for window in [24, 48, 96]:
                bb_middle = df['close'].rolling(window=window).mean()
                bb_std = df['close'].rolling(window=window).std()
                features[f'BB_width_{window}'] = (bb_std * 2) / bb_middle
                features[f'BB_position_{window}'] = (df['close'] - (bb_middle - bb_std * 2)) / (bb_std * 4)
            
            # RSI指标 - 多周期
            for period in [12, 24, 48]:
                features[f'RSI_{period}'] = pd.Series(talib.RSI(df['close'].values, timeperiod=period), index=df.index)
            
            # MACD指标
            exp12 = df['close'].ewm(span=12, adjust=False).mean()
            exp26 = df['close'].ewm(span=26, adjust=False).mean()
            macd = exp12 - exp26
            signal = macd.ewm(span=9, adjust=False).mean()
            features['MACD'] = macd
            features['MACD_signal'] = signal
            features['MACD_hist'] = macd - signal
            
            # 删除NaN值
            valid_mask = ~(features.isna().any(axis=1))
            features = features[valid_mask]
            
            return features
            
        except Exception as e:
            self.logger.error(f"特征准备失败: {str(e)}")
            return None
    
    def generate_labels(self, klines):
        """生成训练标签"""
        try:
            if not isinstance(klines, list) or len(klines) < 48:
                return None
            
            # 创建DataFrame
            df = pd.DataFrame(klines)
            df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 计算未来收益
            future_returns = []
            lookback_windows = [1, 2, 4, 8]  # 长期预测窗口
            weights = [0.4, 0.3, 0.2, 0.1]   # 权重偏向近期
            
            for window in lookback_windows:
                future_close = df['close'].shift(-window)
                future_return = (future_close - df['close']) / df['close']
                future_returns.append(future_return)
            
            # 计算加权收益
            weighted_returns = pd.Series(0.0, index=df.index)
            for w, r in zip(weights, future_returns):
                weighted_returns += w * r
            
            # 生成标签：-1表示下跌，0表示盘整，1表示上涨
            labels = pd.Series(0, index=df.index)  # 默认为0（盘整）
            threshold = 0.001  # 设置阈值，区分上涨/下跌和盘整
            
            # 根据收益率大小分类
            labels[weighted_returns < -threshold] = -1  # 下跌
            labels[weighted_returns > threshold] = 1    # 上涨
            # weighted_returns在[-threshold, threshold]之间的保持为0（盘整）
            
            # 删除NaN值
            labels = labels.dropna()
            
            return labels
            
        except Exception as e:
            self.logger.error(f"标签生成失败: {str(e)}")
            return None
            
    def train_model(self, klines):
        """训练模型"""
        try:
            features = self.prepare_features(klines)
            if features is None:
                return False
            
            labels = self.generate_labels(klines)
            if labels is None:
                return False
            
            # 确保特征和标签使用相同的索引
            common_index = features.index.intersection(labels.index)
            features = features.loc[common_index]
            labels = labels.loc[common_index]
            
            if len(features) != len(labels):
                self.logger.error("特征和标签长度不匹配")
                return False
            
            # 标准化特征
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(features)
            y = labels
            
            # 计算样本权重
            unique_labels, label_counts = np.unique(y, return_counts=True)
            class_weights = dict(zip(unique_labels, len(y) / (len(unique_labels) * label_counts)))
            
            # 训练模型
            self.model.set_params(class_weight=class_weights)
            self.model.fit(X, y)
            
            # 计算训练集上的预测结果
            y_pred = self.model.predict(X)
            train_accuracy = accuracy_score(y, y_pred)
            
            # 计算各类别的精确率、召回率和F1分数
            precision = precision_score(y, y_pred, average='weighted')
            recall = recall_score(y, y_pred, average='weighted')
            f1 = f1_score(y, y_pred, average='weighted')
            
            # 输出评估指标
            self.logger.info(f"模型评估指标:")
            self.logger.info(f"训练集准确率: {train_accuracy:.4f}")
            self.logger.info(f"加权精确率: {precision:.4f}")
            self.logger.info(f"加权召回率: {recall:.4f}")
            self.logger.info(f"加权F1分数: {f1:.4f}")
            
            # 输出混淆矩阵
            cm = confusion_matrix(y, y_pred)
            self.logger.info(f"混淆矩阵:")
            self.logger.info(f"卖出 观望 买入")
            for i, row in enumerate(cm):
                label = "实际卖出" if i == 0 else "实际观望" if i == 1 else "实际买入"
                self.logger.info(f"{row} # {label}")
            
            # 记录特征重要性
            self.feature_importance = pd.Series(
                self.model.feature_importances_,
                index=features.columns
            ).sort_values(ascending=False)
            
            # 输出Top5重要特征
            self.logger.info("Top 5 重要特征:")
            for feature, importance in self.feature_importance[:5].items():
                self.logger.info(f"{feature}: {importance:.4f}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"模型训练失败: {str(e)}")
            return False