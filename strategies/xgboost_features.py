"""
XGBoost特征提取模块
用于计算技术指标和准备模型输入特征

File: strategies/xgboost_features.py
"""
import numpy as np
import pandas as pd
import pandas_ta_classic as ta
from pathlib import Path
import logging


class XGBoostFeatureExtractor:
    """XGBoost特征提取器"""
    
    def __init__(self, lookback_period: int = 150):
        """初始化特征提取器
        
        Args:
            lookback_period: 回溯周期，需要多少根K线来计算特征
        """
        self.lookback_period = lookback_period
        self.logger = logging.getLogger(__name__)
        
        # 特征归一化参数
        self.feature_mean = None
        self.feature_std = None
        self.scaler_fitted = False
        
        # 特征列表
        self.feature_names = [
            'rsi_14', 'rsi_28',
            'macd', 'macd_signal', 'macd_hist',
            'bb_position', 'bb_width',
            'atr_pct',
            'ema_ratio_fast', 'ema_ratio_slow',
            'volume_ratio',
            'adx', 'plus_di', 'minus_di', 'di_diff',
            'slowk', 'slowd',
            'momentum_10', 'momentum_20',
            'price_change_5', 'price_change_10', 'price_change_20',
            'volatility_10', 'volatility_20',
            'close_to_high', 'close_to_low'
        ]
    
    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算所有技术指标特征
        
        Args:
            df: 包含OHLCV数据的DataFrame
            
        Returns:
            添加了特征列的DataFrame
        """
        try:
            df = df.copy()
            
            # 基础数据
            close = df['close']
            high = df['high']
            low = df['low']
            volume = df['volume']
            
            # === RSI ===
            df['rsi_14'] = ta.rsi(close, length=14)
            df['rsi_28'] = ta.rsi(close, length=28)
            
            # === MACD ===
            macd_df = ta.macd(close, fast=12, slow=26, signal=9)
            df['macd'] = macd_df['MACD_12_26_9']
            df['macd_signal'] = macd_df['MACDs_12_26_9']
            df['macd_hist'] = macd_df['MACDh_12_26_9']
            
            # === Bollinger Bands ===
            bb_df = ta.bbands(close, length=20, std=2.0)
            bb_upper = bb_df['BBU_20_2.0']
            bb_lower = bb_df['BBL_20_2.0']
            bb_middle = bb_df['BBM_20_2.0']
            df['bb_position'] = (close - bb_lower) / (bb_upper - bb_lower + 1e-10)
            df['bb_width'] = (bb_upper - bb_lower) / (bb_middle + 1e-10)
            
            # === ATR ===
            atr = ta.atr(high, low, close, length=14)
            df['atr_pct'] = atr / close
            
            # === EMA ===
            ema_10 = ta.ema(close, length=10)
            ema_30 = ta.ema(close, length=30)
            ema_60 = ta.ema(close, length=60)
            df['ema_ratio_fast'] = ema_10 / ema_30
            df['ema_ratio_slow'] = ema_30 / ema_60
            
            # === Volume ===
            volume_ma = ta.sma(volume, length=20)
            df['volume_ratio'] = volume / (volume_ma + 1e-10)
            
            # === ADX / DI ===
            adx_df = ta.adx(high, low, close, length=14)
            df['adx'] = adx_df['ADX_14']
            df['plus_di'] = adx_df['DMP_14']
            df['minus_di'] = adx_df['DMN_14']
            df['di_diff'] = df['plus_di'] - df['minus_di']
            
            # === Stochastic ===
            stoch_df = ta.stoch(high, low, close, k=14, d=3, smooth_k=3)
            df['slowk'] = stoch_df['STOCHk_14_3_3']
            df['slowd'] = stoch_df['STOCHd_14_3_3']
            
            # === Momentum ===
            df['momentum_10'] = ta.mom(close, length=10)
            df['momentum_20'] = ta.mom(close, length=20)
            
            # === Price Change ===
            df['price_change_5'] = close.pct_change(5)
            df['price_change_10'] = close.pct_change(10)
            df['price_change_20'] = close.pct_change(20)
            
            # === Volatility ===
            df['volatility_10'] = close.pct_change().rolling(10).std()
            df['volatility_20'] = close.pct_change().rolling(20).std()
            
            # === Price position ===
            rolling_high = high.rolling(20).max()
            rolling_low = low.rolling(20).min()
            df['close_to_high'] = (close - rolling_low) / (rolling_high - rolling_low + 1e-10)
            df['close_to_low'] = (rolling_high - close) / (rolling_high - rolling_low + 1e-10)
            
            # 删除NaN行
            df = df.dropna()
            
            self.logger.info(f"Calculated {len(self.feature_names)} features for {len(df)} samples")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating features: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def get_feature_array(self, df: pd.DataFrame) -> np.ndarray:
        """从DataFrame中提取特征数组
        
        Args:
            df: 包含特征列的DataFrame
            
        Returns:
            特征数组 (n_samples, n_features)
        """
        available_features = [f for f in self.feature_names if f in df.columns]
        if len(available_features) != len(self.feature_names):
            missing = set(self.feature_names) - set(available_features)
            self.logger.warning(f"Missing features: {missing}")
        
        return df[available_features].values
    
    def get_latest_features(self, df: pd.DataFrame) -> np.ndarray:
        """获取最新一根K线的特征
        
        Args:
            df: 包含特征列的DataFrame
            
        Returns:
            特征数组 (1, n_features)
        """
        features = self.get_feature_array(df)
        if len(features) == 0:
            return None
        return features[-1:, :]
    
    def normalize_features(self, features: np.ndarray, fit: bool = False) -> np.ndarray:
        """特征归一化
        
        Args:
            features: 特征数组
            fit: 是否拟合归一化参数
            
        Returns:
            归一化后的特征数组
        """
        if fit or not self.scaler_fitted:
            self.feature_mean = np.mean(features, axis=0)
            self.feature_std = np.std(features, axis=0) + 1e-10
            self.scaler_fitted = True
        
        # 标准化
        normalized = (features - self.feature_mean) / self.feature_std
        
        # 裁剪极端值 (3-sigma clipping)
        normalized = np.clip(normalized, -3, 3)
        
        return normalized
    
    def save_scaler(self, path: str):
        """保存归一化参数
        
        Args:
            path: 保存路径
        """
        if not self.scaler_fitted:
            self.logger.warning("Scaler not fitted, nothing to save")
            return
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            mean=self.feature_mean,
            std=self.feature_std
        )
        self.logger.info(f"Scaler saved to {path}")
    
    def load_scaler(self, path: str) -> bool:
        """加载归一化参数
        
        Args:
            path: 加载路径
            
        Returns:
            是否成功加载
        """
        try:
            data = np.load(path)
            self.feature_mean = data['mean']
            self.feature_std = data['std']
            self.scaler_fitted = True
            self.logger.info(f"Scaler loaded from {path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load scaler: {str(e)}")
            return False
    
    def get_feature_count(self) -> int:
        """获取特征数量"""
        return len(self.feature_names)
    
    def prepare_training_data(
        self, 
        df: pd.DataFrame, 
        future_periods: int = 5, 
        threshold: float = 0.003
    ) -> tuple:
        """准备训练数据
        
        Args:
            df: 包含OHLCV数据的DataFrame
            future_periods: 预测未来几根K线
            threshold: 涨跌阈值
            
        Returns:
            (X, y) 特征和标签数组
        """
        # 计算特征
        df = self.calculate_features(df)
        if df is None or len(df) == 0:
            return None, None
        
        # 计算未来收益
        df['future_return'] = df['close'].shift(-future_periods) / df['close'] - 1
        
        # 创建标签: 0=跌, 1=持, 2=涨
        df['label'] = 1  # 默认持
        df.loc[df['future_return'] > threshold, 'label'] = 2  # 涨
        df.loc[df['future_return'] < -threshold, 'label'] = 0  # 跌
        
        # 删除无标签的行
        df = df.dropna(subset=['future_return'])
        
        # 提取特征和标签
        X = self.get_feature_array(df)
        y = df['label'].values.astype(int)
        
        self.logger.info(f"Prepared {len(X)} training samples")
        
        return X, y
