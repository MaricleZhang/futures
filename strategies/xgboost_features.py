"""
XGBoost特征适配器模块
将LSTM的序列特征转换为XGBoost的扁平特征矩阵

Features:
- 复用LSTMFeatureExtractor进行特征计算
- 将序列特征转换为扁平特征矩阵
- 支持训练数据准备和标签生成

File: strategies/xgboost_features.py
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional
import logging

from strategies.lstm_features import LSTMFeatureExtractor


class XGBoostFeatureAdapter:
    """XGBoost特征适配器
    
    将LSTM的序列特征转换为XGBoost的扁平特征
    """
    
    def __init__(self, lstm_extractor: Optional[LSTMFeatureExtractor] = None):
        """初始化适配器
        
        Args:
            lstm_extractor: LSTM特征提取器实例，如果为None则创建新实例
        """
        self.logger = logging.getLogger(__name__)
        self.lstm_extractor = lstm_extractor or LSTMFeatureExtractor()
        
        # 归一化参数
        self.global_mean = None
        self.global_std = None
        self.scaler_fitted = False
        
    def extract_features(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """提取特征
        
        从OHLCV数据中提取扁平特征矩阵，适用于XGBoost
        
        Args:
            df: 包含OHLCV列的DataFrame (open, high, low, close, volume)
            
        Returns:
            特征矩阵 (samples, features)，如果失败返回None
        """
        try:
            # 使用LSTM特征提取器计算特征
            df_features = self.lstm_extractor.calculate_features(df)
            
            if df_features is None:
                self.logger.error("特征计算失败")
                return None

            # 获取特征列名
            feature_names = self.lstm_extractor.feature_names
            if feature_names is None:
                self.logger.error("特征名称未初始化")
                return None
            
            # 检查特征列是否存在
            missing_cols = [col for col in feature_names if col not in df_features.columns]
            if missing_cols:
                self.logger.error(f"缺少特征列: {missing_cols}")
                return None
            
            # 提取特征数据为扁平矩阵
            features = df_features[feature_names].values
            
            # 处理NaN和无穷值
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            
            self.logger.debug(f"提取特征完成: shape={features.shape}")
            
            return features.astype(np.float32)
            
        except Exception as e:
            self.logger.error(f"特征提取出错: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def prepare_training_data(
        self,
        df: pd.DataFrame,
        future_periods: int = 5,
        threshold: float = 0.002
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """准备训练数据
        
        提取特征并生成标签，用于XGBoost模型训练
        
        Args:
            df: 包含OHLCV列的DataFrame
            future_periods: 预测未来K线数
            threshold: 涨跌阈值 (如0.002表示0.2%)
            
        Returns:
            X: 特征矩阵 (samples, features)
            y: 标签数组 (0=跌, 1=持, 2=涨)
            如果失败返回 (None, None)
        """
        try:
            # 使用LSTM特征提取器计算特征
            df_features = self.lstm_extractor.calculate_features(df)
            
            if df_features is None:
                self.logger.error("特征计算失败")
                return None, None
            
            # 获取特征列名
            feature_names = self.lstm_extractor.feature_names
            if feature_names is None:
                self.logger.error("特征名称未初始化")
                return None, None
            
            # 计算未来收益标签
            close_prices = df_features['close'].values
            n_samples = len(close_prices)
            
            # 需要足够的数据来计算未来收益
            if n_samples <= future_periods:
                self.logger.error(f"数据不足: {n_samples} <= {future_periods}")
                return None, None
            
            # 计算未来收益
            future_returns = np.zeros(n_samples)
            for i in range(n_samples - future_periods):
                future_returns[i] = (close_prices[i + future_periods] - close_prices[i]) / close_prices[i]
            
            # 生成标签
            labels = np.ones(n_samples, dtype=np.int64)  # 默认为1(持)
            labels[future_returns > threshold] = 2   # 涨
            labels[future_returns < -threshold] = 0  # 跌

            # 提取特征矩阵
            features = df_features[feature_names].values
            
            # 处理NaN和无穷值
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 移除最后future_periods个样本（没有有效标签）
            # 同时移除前面因技术指标计算产生NaN的行
            # 技术指标需要的最大回溯期约为26（MACD slow period）
            warmup_period = max(
                self.lstm_extractor.macd_slow,
                self.lstm_extractor.bb_period,
                self.lstm_extractor.adx_period,
                self.lstm_extractor.volume_ma_period
            )
            
            valid_start = warmup_period
            valid_end = n_samples - future_periods
            
            if valid_start >= valid_end:
                self.logger.error(f"有效数据不足: start={valid_start}, end={valid_end}")
                return None, None
            
            X = features[valid_start:valid_end].astype(np.float32)
            y = labels[valid_start:valid_end]
            
            self.logger.info(f"准备训练数据完成: X.shape={X.shape}, y.shape={y.shape}")
            self.logger.info(f"标签分布: 跌={np.sum(y==0)}, 持={np.sum(y==1)}, 涨={np.sum(y==2)}")
            
            return X, y
            
        except Exception as e:
            self.logger.error(f"准备训练数据出错: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None, None
    
    def normalize_features(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """标准化特征
        
        Args:
            X: 特征矩阵 (samples, features)
            fit: 是否拟合归一化参数（训练时为True，推理时为False）
            
        Returns:
            标准化后的特征矩阵
        """
        try:
            if fit:
                # 训练时：计算全局统计量
                self.global_mean = X.mean(axis=0)
                self.global_std = X.std(axis=0)
                self.global_std = np.where(self.global_std == 0, 1, self.global_std)
                self.scaler_fitted = True
                self.logger.info(f"Scaler fitted: mean range [{self.global_mean.min():.4f}, {self.global_mean.max():.4f}]")
            
            if self.scaler_fitted and self.global_mean is not None:
                X_norm = (X - self.global_mean) / self.global_std
            else:
                self.logger.warning("Scaler not fitted, returning original features")
                return X
            
            return X_norm.astype(np.float32)
            
        except Exception as e:
            self.logger.error(f"标准化特征出错: {str(e)}")
            return X
    
    def save_scaler(self, path: str) -> bool:
        """保存归一化参数
        
        Args:
            path: 保存路径 (如 'strategies/models/symbol/xgboost_scaler.npz')
            
        Returns:
            是否保存成功
        """
        try:
            if not self.scaler_fitted:
                self.logger.warning("Scaler not fitted, nothing to save")
                return False
            
            from pathlib import Path
            save_path = Path(path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            np.savez(
                path,
                global_mean=self.global_mean,
                global_std=self.global_std,
                feature_names=np.array(self.lstm_extractor.feature_names, dtype=object)
            )
            self.logger.info(f"XGBoost scaler saved to {path}")
            return True
        except Exception as e:
            self.logger.error(f"保存scaler失败: {str(e)}")
            return False
    
    def load_scaler(self, path: str) -> bool:
        """加载归一化参数
        
        Args:
            path: 加载路径
            
        Returns:
            是否加载成功
        """
        try:
            from pathlib import Path
            scaler_path = Path(path)
            
            if not scaler_path.exists():
                self.logger.warning(f"Scaler file not found: {path}")
                return False
            
            data = np.load(path, allow_pickle=True)
            self.global_mean = data['global_mean']
            self.global_std = data['global_std']
            self.scaler_fitted = True
            
            # 可选：加载特征名称
            if 'feature_names' in data:
                loaded_names = data['feature_names'].tolist()
                if self.lstm_extractor.feature_names is None:
                    self.lstm_extractor.feature_names = loaded_names
            
            self.logger.info(f"XGBoost scaler loaded from {path}")
            return True
        except Exception as e:
            self.logger.error(f"加载scaler失败: {str(e)}")
            return False
    
    def get_feature_count(self) -> int:
        """获取特征数量
        
        Returns:
            特征数量
        """
        return self.lstm_extractor.get_feature_count()
    
    def get_feature_names(self) -> list:
        """获取特征名称列表
        
        Returns:
            特征名称列表
        """
        return self.lstm_extractor.feature_names or []
