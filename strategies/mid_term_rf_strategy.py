import numpy as np
import pandas as pd
import talib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import time
import config
from strategies.base_rf_strategy import BaseRFStrategy

class MidTermRFStrategy(BaseRFStrategy):
    """MidTermRFStrategy - 中线随机森林交易策略
    
    针对中线交易优化的随机森林策略模型，使用5分钟K线数据，
    结合多个中期技术指标和成交量特征来捕捉价格趋势机会。
    
    特点：
    1. 中期特征：使用5分钟到30分钟的技术指标
    2. 中频交易：每5分钟更新一次模型，平衡响应速度和稳定性
    3. 严格风控：包含持仓时间、交易频率、止盈止损等多重风控措施
    4. 趋势跟踪：通过多周期趋势分析提高持仓收益
    """
    
    MODEL_NAME = "MidTermRF"
    VERSION = "1.0.0"
    
    def __init__(self, trader):
        """初始化中线随机森林策略"""
        super().__init__(trader)
        
        # 随机森林参数
        self.max_depth = 10         # 树的最大深度
        self.min_samples_split = 30 # 分裂所需最小样本数
        self.min_samples_leaf = 15  # 叶节点最小样本数
        self.confidence_threshold = 0.45  # 信号置信度阈值
        self.prob_diff_threshold = 0.1    # 概率差异阈值
        
        # K线设置
        self.kline_interval = '5m'  # 5分钟K线
        self.training_lookback = 1000  # 训练数据回看周期
        self.retraining_interval = 300  # 5分钟重新训练
        
        # 风险控制参数
        self.max_position_hold_time = 120  # 最大持仓时间(分钟)
        self.profit_target_pct = 0.008     # 目标利润率
        self.stop_loss_pct = 0.005         # 止损率
        self.max_trades_per_hour = 6       # 每小时最大交易次数
        self.min_vol_percentile = 40       # 最小成交量百分位
        
        # 初始化模型并开始训练
        self.initialize_model()
        self._initial_training()
    
    def initialize_model(self):
        """初始化随机森林模型"""
        self.model = RandomForestClassifier(
            n_estimators=200,
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
        try:
            self.logger.info("获取历史K线数据进行初始训练...")
            historical_data = self.trader.get_klines(
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
        """准备特征数据"""
        try:
            if not isinstance(klines, list) or len(klines) < 30:
                self.logger.error("K线数据为空或长度不足")
                return None
            
            # 创建DataFrame
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 创建特征DataFrame
            features = pd.DataFrame(index=df.index)
            
            # 价格动量指标 - 中期
            features['momentum_5m'] = df['close'] - df['close'].shift(1)
            features['momentum_15m'] = df['close'] - df['close'].shift(3)
            features['momentum_30m'] = df['close'] - df['close'].shift(6)
            
            # 价格变化率和加速度
            features['price_change_5m'] = df['close'].pct_change(1)
            features['price_change_15m'] = df['close'].pct_change(3)
            features['price_change_30m'] = df['close'].pct_change(6)
            features['price_acceleration'] = features['price_change_5m'] - features['price_change_5m'].shift(1)
            
            # 波动率指标
            features['volatility_15m'] = df['close'].rolling(window=3).std()
            features['volatility_30m'] = df['close'].rolling(window=6).std()
            features['volatility_ratio'] = features['volatility_15m'] / features['volatility_30m']
            
            # 成交量分析
            features['volume_ma_6'] = df['volume'].rolling(window=6).mean()
            features['volume_ma_12'] = df['volume'].rolling(window=12).mean()
            features['volume_ratio_6'] = df['volume'] / features['volume_ma_6']
            features['volume_ratio_12'] = df['volume'] / features['volume_ma_12']
            features['volume_trend'] = df['volume'].pct_change()
            
            # 价格压力指标
            features['high_low_range'] = (df['high'] - df['low']) / df['close']
            features['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
            
            # 趋势强度指标
            features['trend_strength_15m'] = features['price_change_15m'].abs() * features['volume_ratio_6']
            features['trend_strength_30m'] = features['price_change_30m'].abs() * features['volume_ratio_12']
            
            # 布林带 - 多周期
            for window in [12, 24, 36]:
                bb_middle = df['close'].rolling(window=window).mean()
                bb_std = df['close'].rolling(window=window).std()
                features[f'BB_width_{window}'] = (bb_std * 2) / bb_middle
                features[f'BB_position_{window}'] = (df['close'] - (bb_middle - bb_std * 2)) / (bb_std * 4)
            
            # 价格突破指标
            features['breakout_15m'] = df['close'] > df['high'].rolling(window=3).max().shift(1)
            features['breakdown_15m'] = df['close'] < df['low'].rolling(window=3).min().shift(1)
            
            # RSI指标 - 多周期
            for period in [6, 12, 24]:
                features[f'RSI_{period}'] = pd.Series(talib.RSI(df['close'].values, timeperiod=period), index=df.index)
            
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
            if not isinstance(klines, list) or len(klines) < 30:
                return None
            
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 计算未来收益
            future_returns = []
            lookback_windows = [1, 2, 4, 6]  # 中期预测窗口
            weights = [0.4, 0.3, 0.2, 0.1]   # 权重偏向近期
            
            for window in lookback_windows:
                future_close = df['close'].shift(-window)
                future_return = (future_close - df['close']) / df['close']
                future_returns.append(future_return)
            
            # 计算加权收益
            weighted_returns = pd.Series(0.0, index=df.index)
            for w, r in zip(weights, future_returns):
                weighted_returns += w * r
            
            # 将加权收益保存到DataFrame中
            df['returns'] = weighted_returns
            
            # 生成标签
            labels = pd.Series(0, index=df.index)
            labels[weighted_returns > self.profit_target_pct] = 1  # 上涨信号
            labels[weighted_returns < -self.stop_loss_pct] = -1    # 下跌信号
            
            # 删除包含 NaN 的行
            valid_mask = ~(labels.isna())
            return labels[valid_mask]
            
        except Exception as e:
            self.logger.error(f"标签生成失败: {str(e)}")
            return None
    
    def check_trade_limits(self):
        """检查交易限制"""
        current_hour = time.localtime().tm_hour
        
        # 重置每小时交易计数
        if self.last_trade_hour != current_hour:
            self.trade_count_hour = 0
            self.last_trade_hour = current_hour
        
        # 检查交易频率限制
        if self.trade_count_hour >= self.max_trades_per_hour:
            self.logger.info("已达到每小时最大交易次数限制")
            return False
        
        # 检查持仓时间限制
        if self.position_entry_time:
            hold_time = (time.time() - self.position_entry_time) / 60
            if hold_time > self.max_position_hold_time:
                self.logger.info("已超过最大持仓时间限制")
                return False
        
        return True
    
    def update_trade_status(self):
        """更新交易状态"""
        self.trade_count_hour += 1
        self.position_entry_time = time.time()
        self.position_entry_price = self.trader.get_latest_price()
    
    def check_exit_signals(self):
        """检查平仓信号"""
        if not self.position_entry_price:
            return False
            
        current_price = self.trader.get_latest_price()
        price_change = (current_price - self.position_entry_price) / self.position_entry_price
        
        # 止盈检查
        if abs(price_change) >= self.profit_target_pct:
            self.logger.info(f"达到目标利润: {price_change:.4%}")
            return True
            
        # 止损检查
        if abs(price_change) >= self.stop_loss_pct:
            self.logger.info(f"触发止损: {price_change:.4%}")
            return True
            
        return False
