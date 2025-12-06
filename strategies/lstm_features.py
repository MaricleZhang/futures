"""
LSTM特征工程模块
用于深度学习交易策略的特征提取和预处理

Features:
- 价格特征: 收盘价变化率、高低价波动、K线实体比例
- 技术指标: RSI, MACD, Bollinger Bands, ADX, ATR
- 成交量特征: 成交量变化率、成交量MA比率
- 归一化: 滚动窗口标准化

File: strategies/lstm_features.py
"""
import numpy as np
import pandas as pd
import pandas_ta_classic as ta
from typing import Tuple, Optional
import logging


class LSTMFeatureExtractor:
    """LSTM特征提取器"""
    
    def __init__(self, sequence_length: int = 60):
        """初始化特征提取器
        
        Args:
            sequence_length: 输入序列长度(K线数量)
        """
        self.logger = logging.getLogger(__name__)
        self.sequence_length = sequence_length
        
        # 技术指标参数
        self.rsi_period = 14
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        self.bb_period = 20
        self.bb_std = 2.0
        self.adx_period = 14
        self.atr_period = 14
        self.ema_fast = 12
        self.ema_slow = 26
        self.volume_ma_period = 20
        
        # 特征名称
        self.feature_names = None
        
        # 全局归一化参数 (训练时拟合，推理时使用)
        self.global_mean = None
        self.global_std = None
        self.scaler_fitted = False
        
    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算所有特征
        
        Args:
            df: 包含OHLCV的DataFrame
            
        Returns:
            包含所有特征的DataFrame
        """
        try:
            df = df.copy()
            
            # 确保数值类型
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # ==================== 价格特征 ====================
            # 收盘价变化率
            df['price_change'] = df['close'].pct_change()
            
            # 高低价波动率
            df['hl_range'] = (df['high'] - df['low']) / df['close']
            
            # K线实体比例 (实体大小相对于整根K线)
            df['body_ratio'] = abs(df['close'] - df['open']) / (df['high'] - df['low'] + 1e-10)
            
            # K线方向 (阳线1, 阴线-1)
            df['candle_direction'] = np.where(df['close'] >= df['open'], 1, -1)
            
            # 上下影线比例
            df['upper_shadow'] = (df['high'] - df[['close', 'open']].max(axis=1)) / (df['high'] - df['low'] + 1e-10)
            df['lower_shadow'] = (df[['close', 'open']].min(axis=1) - df['low']) / (df['high'] - df['low'] + 1e-10)
            
            # ==================== 技术指标特征 ====================
            close = df['close']
            high = df['high']
            low = df['low']
            volume = df['volume']
            
            # RSI
            df['rsi'] = ta.rsi(close, length=self.rsi_period)
            df['rsi_norm'] = (df['rsi'] - 50) / 50  # 归一化到[-1, 1]
            
            # MACD
            macd_df = ta.macd(close, fast=self.macd_fast, slow=self.macd_slow, signal=self.macd_signal)
            df['macd'] = macd_df[f'MACD_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}']
            df['macd_signal'] = macd_df[f'MACDs_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}']
            df['macd_hist'] = macd_df[f'MACDh_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}']
            # 归一化MACD
            df['macd_norm'] = df['macd'] / (df['close'] + 1e-10)
            df['macd_signal_norm'] = df['macd_signal'] / (df['close'] + 1e-10)
            df['macd_hist_norm'] = df['macd_hist'] / (df['close'] + 1e-10)
            
            # Bollinger Bands
            bb_df = ta.bbands(close, length=self.bb_period, std=self.bb_std)
            df['bb_upper'] = bb_df[f'BBU_{self.bb_period}_{self.bb_std}']
            df['bb_middle'] = bb_df[f'BBM_{self.bb_period}_{self.bb_std}']
            df['bb_lower'] = bb_df[f'BBL_{self.bb_period}_{self.bb_std}']
            # BB位置 (价格在BB中的相对位置, 0-1)
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
            # BB宽度
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            
            # ADX
            adx_df = ta.adx(high, low, close, length=self.adx_period)
            df['adx'] = adx_df[f'ADX_{self.adx_period}']
            df['plus_di'] = adx_df[f'DMP_{self.adx_period}']
            df['minus_di'] = adx_df[f'DMN_{self.adx_period}']
            # DI差值归一化
            df['di_diff'] = (df['plus_di'] - df['minus_di']) / 100
            df['adx_norm'] = df['adx'] / 100
            
            # ATR
            df['atr'] = ta.atr(high, low, close, length=self.atr_period)
            df['atr_norm'] = df['atr'] / df['close']  # ATR相对于价格
            
            # EMA趋势
            df['ema_fast'] = ta.ema(close, length=self.ema_fast)
            df['ema_slow'] = ta.ema(close, length=self.ema_slow)
            df['ema_diff'] = (df['ema_fast'] - df['ema_slow']) / df['close']
            df['price_ema_diff'] = (df['close'] - df['ema_fast']) / df['close']
            
            # ==================== 成交量特征 ====================
            df['volume_change'] = df['volume'].pct_change()
            df['volume_ma'] = ta.sma(volume, length=self.volume_ma_period)
            df['volume_ratio'] = df['volume'] / (df['volume_ma'] + 1e-10)
            # 成交量加权价格变化
            df['vwap_diff'] = df['price_change'] * df['volume_ratio']
            
            # ==================== 动量特征 ====================
            # 多周期价格变化
            df['price_change_5'] = df['close'].pct_change(5)
            df['price_change_10'] = df['close'].pct_change(10)
            df['price_change_20'] = df['close'].pct_change(20)
            
            # 记录特征名称
            self.feature_names = [
                'price_change', 'hl_range', 'body_ratio', 'candle_direction',
                'upper_shadow', 'lower_shadow', 'rsi_norm',
                'macd_norm', 'macd_signal_norm', 'macd_hist_norm',
                'bb_position', 'bb_width', 'di_diff', 'adx_norm', 'atr_norm',
                'ema_diff', 'price_ema_diff', 'volume_ratio',
            ]
            
            return df
            
        except Exception as e:
            self.logger.error(f"特征计算出错: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def prepare_sequences(
        self, 
        df: pd.DataFrame, 
        target_column: str = None,
        future_periods: int = 5,
        threshold: float = 0.002
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """准备LSTM输入序列
        
        Args:
            df: 包含特征的DataFrame
            target_column: 目标列名(用于训练)
            future_periods: 预测未来K线数
            threshold: 涨跌阈值
            
        Returns:
            X: 形状为(samples, sequence_length, features)的特征矩阵
            y: 标签数组(0=跌, 1=持, 2=涨) 或 None(预测时)
        """
        try:
            # 获取特征列
            feature_cols = self.feature_names
            
            # 检查特征是否存在
            missing_cols = [col for col in feature_cols if col not in df.columns]
            if missing_cols:
                self.logger.error(f"缺少特征列: {missing_cols}")
                return None, None
            
            # 提取特征数据
            features = df[feature_cols].values
            
            # 处理NaN
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 创建序列
            X = []
            y = [] if target_column is None else []
            
            for i in range(self.sequence_length, len(features) - future_periods):
                X.append(features[i - self.sequence_length:i])
                
                if target_column is None:
                    # 预测模式: 计算未来收益
                    future_return = (df['close'].iloc[i + future_periods] - df['close'].iloc[i]) / df['close'].iloc[i]
                    if future_return > threshold:
                        y.append(2)  # 涨
                    elif future_return < -threshold:
                        y.append(0)  # 跌
                    else:
                        y.append(1)  # 持
            
            X = np.array(X, dtype=np.float32)
            
            if target_column is None and len(y) > 0:
                y = np.array(y, dtype=np.int64)
            else:
                y = None
            
            self.logger.info(f"准备序列完成: X.shape={X.shape}, y.shape={y.shape if y is not None else 'None'}")
            
            return X, y
            
        except Exception as e:
            self.logger.error(f"准备序列出错: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None, None
    
    def get_feature_count(self) -> int:
        """获取特征数量"""
        return len(self.feature_names) if self.feature_names else 18
    
    def normalize_features(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """标准化特征
        
        Args:
            X: 形状为(samples, sequence_length, features)的特征矩阵
            fit: 是否拟合全局统计量（训练时为True，推理时为False）
            
        Returns:
            标准化后的特征矩阵
        """
        try:
            if fit:
                # 训练时：计算全局统计量
                # 将所有样本展平计算全局均值和标准差
                X_flat = X.reshape(-1, X.shape[-1])  # (samples * seq_len, features)
                self.global_mean = X_flat.mean(axis=0)
                self.global_std = X_flat.std(axis=0)
                self.global_std = np.where(self.global_std == 0, 1, self.global_std)
                self.scaler_fitted = True
                self.logger.info(f"Scaler fitted: mean range [{self.global_mean.min():.4f}, {self.global_mean.max():.4f}]")
            
            if self.scaler_fitted and self.global_mean is not None:
                # 使用全局统计量归一化
                X_norm = (X - self.global_mean) / self.global_std
            else:
                # 回退到逐序列归一化（兼容旧代码）
                self.logger.warning("Scaler not fitted, using per-sequence normalization")
                X_norm = np.zeros_like(X)
                for i in range(X.shape[0]):
                    seq = X[i]
                    mean = seq.mean(axis=0, keepdims=True)
                    std = seq.std(axis=0, keepdims=True)
                    std = np.where(std == 0, 1, std)
                    X_norm[i] = (seq - mean) / std
            
            return X_norm
            
        except Exception as e:
            self.logger.error(f"标准化特征出错: {str(e)}")
            return X
    
    def save_scaler(self, path: str) -> bool:
        """保存归一化参数
        
        Args:
            path: 保存路径 (如 'models/scaler.npz')
            
        Returns:
            是否保存成功
        """
        try:
            if not self.scaler_fitted:
                self.logger.warning("Scaler not fitted, nothing to save")
                return False
            
            # 确保目录存在
            from pathlib import Path
            save_path = Path(path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            np.savez(
                path,
                global_mean=self.global_mean,
                global_std=self.global_std,
                feature_names=np.array(self.feature_names, dtype=object)
            )
            self.logger.info(f"Scaler saved to {path}")
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
                if self.feature_names is None:
                    self.feature_names = loaded_names
            
            self.logger.info(f"Scaler loaded from {path}")
            return True
        except Exception as e:
            self.logger.error(f"加载scaler失败: {str(e)}")
            return False
    
    def transform_single(self, features: np.ndarray) -> np.ndarray:
        """转换单个序列（推理时使用）
        
        Args:
            features: 形状为(sequence_length, features)的特征矩阵
            
        Returns:
            标准化后的特征矩阵
        """
        if self.scaler_fitted and self.global_mean is not None:
            return (features - self.global_mean) / self.global_std
        else:
            # 回退到逐序列归一化
            mean = features.mean(axis=0, keepdims=True)
            std = features.std(axis=0, keepdims=True)
            std = np.where(std == 0, 1, std)
            return (features - mean) / std
