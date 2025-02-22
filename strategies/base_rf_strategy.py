import numpy as np
import pandas as pd
import talib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import time
import config
from strategies.base_strategy import BaseStrategy

class BaseRFStrategy(BaseStrategy):
    """BaseRFStrategy - 基础随机森林交易策略
    
    为所有基于随机森林的策略提供基础功能和共享代码。
    各个具体策略通过继承此类并重写相关方法来实现自己的逻辑。
    """
    
    MODEL_NAME = "BaseRF"
    VERSION = "1.0.0"
    
    def __init__(self, trader):
        """初始化基础随机森林策略"""
        super().__init__(trader)
        self.logger = self.get_logger()
        
        # 随机森林参数 - 子类可以重写这些参数
        self.n_estimators = 200     # 树的数量
        self.max_depth = 8          # 树的最大深度
        self.min_samples_split = 20 # 分裂所需最小样本数
        self.min_samples_leaf = 10  # 叶节点最小样本数
        self.confidence_threshold = 0.4  # 信号置信度阈值
        self.prob_diff_threshold = 0.08   # 概率差异阈值
        
        # K线设置 - 子类必须重写这些参数
        self.kline_interval = None  # K线周期
        self.training_lookback = None  # 训练数据回看周期
        self.retraining_interval = None  # 重新训练间隔
        self.last_training_time = 0
        
        # 风险控制参数 - 子类可以重写这些参数
        self.max_position_hold_time = 30  # 最大持仓时间(分钟)
        self.profit_target_pct = 0.003    # 目标利润率
        self.stop_loss_pct = 0.002        # 止损率
        self.max_trades_per_hour = 12     # 每小时最大交易次数
        self.min_vol_percentile = 30      # 最小成交量百分位
        
        # 交易状态
        self.trade_count_hour = 0         # 当前小时交易次数
        self.last_trade_hour = None       # 上次交易的小时
        self.position_entry_time = None   # 开仓时间
        self.position_entry_price = None  # 开仓价格
        
        # 模型相关
        self.scaler = StandardScaler()
        self.model = None
        self.feature_importance = None
    
    def initialize_model(self):
        """初始化随机森林模型"""
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced',
            bootstrap=True,
            max_features='sqrt',
            max_samples=0.8
        )
    
    def _initial_training(self):
        """初始化训练"""
        if not self.kline_interval:
            raise ValueError("子类必须设置 kline_interval")
        if not self.training_lookback:
            raise ValueError("子类必须设置 training_lookback")
            
        try:
            self.logger.info("获取历史K线数据进行初始训练...")
            historical_data = self.trader.get_klines(
                symbol=self.trader.symbol,
                interval=self.kline_interval,
                limit=self.training_lookback
            )
            
            if historical_data and len(historical_data) > 0:
                self.logger.info(f"成功获取{len(historical_data)}根{self.kline_interval}K线数据")
                if self.train_model(historical_data):
                    self.logger.info("初始模型训练完成")
                else:
                    self.logger.error("初始模型训练失败")
            else:
                self.logger.error("获取历史数据失败")
        except Exception as e:
            self.logger.error(f"初始化训练失败: {str(e)}")
            
    def prepare_features(self, klines):
        """准备特征数据 - 子类应该重写此方法"""
        raise NotImplementedError("子类必须实现 prepare_features 方法")
        
    def generate_labels(self, klines):
        """生成训练标签 - 子类应该重写此方法"""
        raise NotImplementedError("子类必须实现 generate_labels 方法")
        
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
            
            # 训练模型
            self.model.fit(X, y)
            
            # 计算特征重要性
            self.feature_importance = pd.DataFrame({
                'feature': features.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # 评估模型
            y_pred = self.model.predict(X)
            accuracy = accuracy_score(y, y_pred)
            precision = precision_score(y, y_pred, average='weighted')
            recall = recall_score(y, y_pred, average='weighted')
            f1 = f1_score(y, y_pred, average='weighted')
            
            self.logger.info(f"模型评估结果:")
            self.logger.info(f"Accuracy: {accuracy:.4f}")
            self.logger.info(f"Precision: {precision:.4f}")
            self.logger.info(f"Recall: {recall:.4f}")
            self.logger.info(f"F1 Score: {f1:.4f}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"模型训练失败: {str(e)}")
            return False
            
    def should_retrain(self):
        """检查是否需要重新训练"""
        if not self.retraining_interval:
            raise ValueError("子类必须设置 retraining_interval")
            
        current_time = time.time()
        if current_time - self.last_training_time >= self.retraining_interval:
            return True
        return False
        
    def generate_signal(self, klines):
        """生成交易信号
        返回值：
        -1: 卖出信号
        0: 观望信号
        1: 买入信号
        """
        try:
            # 检查交易限制
            if not self.check_trade_limits():
                return 0  # 返回观望信号
            
            # 准备特征
            features = self.prepare_features(klines)
            if features is None:
                return 0
            
            # 标准化特征
            features_scaled = self.scaler.transform(features)
            
            # 模型预测
            probabilities = self.model.predict_proba(features_scaled)[-1]
            sell_prob, hold_prob, buy_prob = probabilities  # 概率对应 -1, 0, 1
            
            # 输出预测概率
            self.logger.info(f"预测概率: 卖出={sell_prob:.4f}, 观望={hold_prob:.4f}, 买入={buy_prob:.4f}")
            
            # 检查成交量条件
            df = pd.DataFrame(klines[-20:], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            volume_percentile = pd.to_numeric(df['volume']).rank(pct=True).iloc[-1] * 100
            
            # if volume_percentile < self.min_vol_percentile:
            #     self.logger.info(f"成交量百分位({volume_percentile:.2f}%)低于阈值({self.min_vol_percentile}%)")
            #     return 0
            
            # 获取预测
            prediction = np.argmax(probabilities)
            max_prob = max(probabilities)
            
            # 检查置信度
            if prediction != 0 and max_prob < self.confidence_threshold:  # 0是观望
                self.logger.info(f"信号置信度({max_prob:.4f})低于阈值({self.confidence_threshold})")
                return 0
            
            # 更新交易计数
            if prediction != 0:  # 不是观望信号时增加计数
                self.trade_count_hour += 1
            
            # 将预测值从[0,1,2]映射到[-1,0,1]
            signal_mapping = {0: -1, 1: 0, 2: 1}
            return signal_mapping[prediction]
            
        except Exception as e:
            self.logger.error(f"生成信号失败: {str(e)}")
            return 0  # 发生错误时返回观望信号

