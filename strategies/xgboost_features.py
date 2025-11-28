"""
XGBoost特征工程模块
提供技术指标计算和特征提取功能

File: strategies/xgboost_features.py
"""
import numpy as np
import pandas as pd
import pandas_ta_classic as ta
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler
import logging


class FeatureEngineer:
    """
    XGBoost特征工程类
    
    职责:
    1. 计算技术指标
    2. 从指标中提取ML特征
    3. 特征归一化
    4. 特征名称管理
    """
    
    def __init__(self):
        """初始化特征工程器"""
        self.logger = logging.getLogger(__name__)
        
        # 技术指标参数
        self.ema_fast = 8
        self.ema_mid = 21
        self.ema_slow = 55
        self.adx_period = 14
        self.rsi_period = 14
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        self.stoch_k = 14
        self.stoch_d = 3
        self.atr_period = 14
        self.bb_period = 20
        self.bb_std = 2.0
        self.volume_ma_period = 20
        
        # 归一化器
        self.scaler = None
        
    def calculate_technical_indicators(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        计算所有技术指标
        
        Args:
            df: OHLCV数据框
            
        Returns:
            包含所有指标的字典
        """
        try:
            high = df['high']
            low = df['low']
            close = df['close']
            volume = df['volume']
            
            indicators = {}
            
            # ---------- 趋势指标 ----------
            indicators['ema_fast'] = ta.ema(close, length=self.ema_fast)
            indicators['ema_mid'] = ta.ema(close, length=self.ema_mid)
            indicators['ema_slow'] = ta.ema(close, length=self.ema_slow)
            
            # ADX + DI
            adx_df = ta.adx(high, low, close, length=self.adx_period)
            indicators['adx'] = adx_df[f'ADX_{self.adx_period}']
            indicators['plus_di'] = adx_df[f'DMP_{self.adx_period}']
            indicators['minus_di'] = adx_df[f'DMN_{self.adx_period}']
            
            # ---------- 动量指标 ----------
            indicators['rsi'] = ta.rsi(close, length=self.rsi_period)
            
            macd_df = ta.macd(close, fast=self.macd_fast, slow=self.macd_slow, signal=self.macd_signal)
            indicators['macd'] = macd_df[f'MACD_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}']
            indicators['macd_signal'] = macd_df[f'MACDs_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}']
            indicators['macd_hist'] = macd_df[f'MACDh_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}']
            
            stoch_df = ta.stoch(high, low, close, k=self.stoch_k, d=self.stoch_d)
            indicators['stoch_k'] = stoch_df[f'STOCHk_{self.stoch_k}_{self.stoch_d}_3']
            indicators['stoch_d'] = stoch_df[f'STOCHd_{self.stoch_k}_{self.stoch_d}_3']
            
            # ---------- 波动率指标 ----------
            indicators['atr'] = ta.atr(high, low, close, length=self.atr_period)
            
            bb_df = ta.bbands(close, length=self.bb_period, std=self.bb_std)
            indicators['bb_upper'] = bb_df[f'BBU_{self.bb_period}_{self.bb_std}']
            indicators['bb_middle'] = bb_df[f'BBM_{self.bb_period}_{self.bb_std}']
            indicators['bb_lower'] = bb_df[f'BBL_{self.bb_period}_{self.bb_std}']
            
            # ---------- 成交量指标 ----------
            indicators['volume'] = volume
            indicators['volume_ma'] = ta.sma(volume, length=self.volume_ma_period)
            
            # ---------- 价格数据 ----------
            indicators['close'] = close
            indicators['high'] = high
            indicators['low'] = low
            indicators['open'] = df['open']
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"计算指标出错: {str(e)}")
            return None
    
    def extract_features(self, df: pd.DataFrame, indicators: Dict) -> Tuple[Optional[pd.DataFrame], List[str]]:
        """
        从技术指标提取ML特征
        
        Args:
            df: 原始OHLCV数据
            indicators: 技术指标字典
            
        Returns:
            (特征DataFrame, 特征名称列表)
        """
        try:
            features = pd.DataFrame(index=df.index)
            feature_names = []
            
            # ---------- 价格特征 ----------
            close = indicators['close']
            high = indicators['high']
            low = indicators['low']
            open_price = indicators['open']
            
            # 价格变化率
            features['price_change_1'] = close.pct_change(1)
            features['price_change_5'] = close.pct_change(5)
            features['price_change_10'] = close.pct_change(10)
            feature_names.extend(['price_change_1', 'price_change_5', 'price_change_10'])
            
            # 最高/最低价位置
            features['high_position'] = (close - low) / (high - low + 1e-10)
            feature_names.append('high_position')
            
            # K线实体大小
            features['candle_body'] = abs(close - open_price) / open_price
            feature_names.append('candle_body')
            
            # ---------- 趋势特征 ----------
            ema_fast = indicators['ema_fast']
            ema_mid = indicators['ema_mid']
            ema_slow = indicators['ema_slow']
            
            # 价格与均线距离
            features['price_to_ema_fast'] = (close - ema_fast) / ema_fast
            features['price_to_ema_mid'] = (close - ema_mid) / ema_mid
            features['price_to_ema_slow'] = (close - ema_slow) / ema_slow
            feature_names.extend(['price_to_ema_fast', 'price_to_ema_mid', 'price_to_ema_slow'])
            
            # EMA排列状态
            features['ema_alignment'] = ((ema_fast > ema_mid).astype(int) * 2 + 
                                        (ema_mid > ema_slow).astype(int))  # 0-3的值
            feature_names.append('ema_alignment')
            
            # ADX和DI
            features['adx'] = indicators['adx']
            features['plus_di'] = indicators['plus_di']
            features['minus_di'] = indicators['minus_di']
            features['di_diff'] = indicators['plus_di'] - indicators['minus_di']
            feature_names.extend(['adx', 'plus_di', 'minus_di', 'di_diff'])
            
            # ---------- 动量特征 ----------
            features['rsi'] = indicators['rsi']
            feature_names.append('rsi')
            
            # RSI偏离50的程度
            features['rsi_deviation'] = indicators['rsi'] - 50
            feature_names.append('rsi_deviation')
            
            # MACD
            features['macd'] = indicators['macd']
            features['macd_signal'] = indicators['macd_signal']
            features['macd_hist'] = indicators['macd_hist']
            feature_names.extend(['macd', 'macd_signal', 'macd_hist'])
            
            # MACD柱状图变化
            features['macd_hist_change'] = indicators['macd_hist'].diff()
            feature_names.append('macd_hist_change')
            
            # KDJ
            features['stoch_k'] = indicators['stoch_k']
            features['stoch_d'] = indicators['stoch_d']
            features['stoch_diff'] = indicators['stoch_k'] - indicators['stoch_d']
            feature_names.extend(['stoch_k', 'stoch_d', 'stoch_diff'])
            
            # ---------- 波动率特征 ----------
            atr = indicators['atr']
            features['atr'] = atr
            features['atr_ratio'] = atr / close  # ATR相对于价格的比例
            feature_names.extend(['atr', 'atr_ratio'])
            
            # 布林带
            bb_upper = indicators['bb_upper']
            bb_middle = indicators['bb_middle']
            bb_lower = indicators['bb_lower']
            
            features['bb_position'] = (close - bb_lower) / (bb_upper - bb_lower + 1e-10)
            features['bb_width'] = (bb_upper - bb_lower) / bb_middle
            feature_names.extend(['bb_position', 'bb_width'])
            
            # ---------- 成交量特征 ----------
            volume = indicators['volume']
            volume_ma = indicators['volume_ma']
            
            features['volume_ratio'] = volume / (volume_ma + 1e-10)
            features['volume_price_trend'] = (volume / (volume_ma + 1e-10)) * (close.pct_change(1))
            feature_names.extend(['volume_ratio', 'volume_price_trend'])
            
            # ---------- 交叉特征 ----------
            # 趋势强度 × 动量
            features['trend_momentum'] = features['adx'] * features['rsi_deviation'] / 100
            feature_names.append('trend_momentum')
            
            # 波动率 × 成交量
            features['volatility_volume'] = features['atr_ratio'] * features['volume_ratio']
            feature_names.append('volatility_volume')
            
            # 删除包含NaN的行
            features = features.ffill().fillna(0)
            
            return features, feature_names
            
        except Exception as e:
            self.logger.error(f"提取特征出错: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None, []
    
    def normalize_features(self, features: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        特征归一化
        
        Args:
            features: 特征DataFrame
            fit: 是否拟合归一化器(训练时为True，预测时为False)
            
        Returns:
            归一化后的特征DataFrame
        """
        try:
            if fit:
                self.scaler = StandardScaler()
                normalized = self.scaler.fit_transform(features)
            else:
                if self.scaler is None:
                    self.logger.warning("归一化器未初始化，跳过归一化")
                    return features
                normalized = self.scaler.transform(features)
            
            return pd.DataFrame(normalized, columns=features.columns, index=features.index)
            
        except Exception as e:
            self.logger.error(f"归一化特征出错: {str(e)}")
            return features
    
    def get_feature_names(self) -> List[str]:
        """
        获取特征名称列表
        
        Returns:
            特征名称列表
        """
        # 这个方法返回所有特征的名称
        # 与extract_features方法中的feature_names保持一致
        return [
            # 价格特征
            'price_change_1', 'price_change_5', 'price_change_10',
            'high_position', 'candle_body',
            # 趋势特征
            'price_to_ema_fast', 'price_to_ema_mid', 'price_to_ema_slow',
            'ema_alignment', 'adx', 'plus_di', 'minus_di', 'di_diff',
            # 动量特征
            'rsi', 'rsi_deviation',
            'macd', 'macd_signal', 'macd_hist', 'macd_hist_change',
            'stoch_k', 'stoch_d', 'stoch_diff',
            # 波动率特征
            'atr', 'atr_ratio', 'bb_position', 'bb_width',
            # 成交量特征
            'volume_ratio', 'volume_price_trend',
            # 交叉特征
            'trend_momentum', 'volatility_volume'
        ]
    
    def prepare_data(self, df: pd.DataFrame, normalize: bool = True, fit_scaler: bool = True) -> Tuple[Optional[pd.DataFrame], List[str]]:
        """
        一站式数据准备: 计算指标 -> 提取特征 -> 归一化
        
        Args:
            df: 原始OHLCV数据
            normalize: 是否归一化
            fit_scaler: 是否拟合归一化器
            
        Returns:
            (处理好的特征DataFrame, 特征名称列表)
        """
        # 计算技术指标
        indicators = self.calculate_technical_indicators(df)
        if indicators is None:
            return None, []
        
        # 提取特征
        features, feature_names = self.extract_features(df, indicators)
        if features is None:
            return None, []
        
        # 归一化
        if normalize:
            features = self.normalize_features(features, fit=fit_scaler)
        
        return features, feature_names
