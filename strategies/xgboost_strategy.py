"""
XGBoost交易策略
使用XGBoost模型预测价格趋势进行交易

File: strategies/xgboost_strategy.py
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
import time
import logging

from strategies.base_strategy import BaseStrategy
from strategies.xgboost_features import XGBoostFeatureExtractor
import config


class XGBoostStrategy(BaseStrategy):
    """XGBoost交易策略
    
    策略逻辑:
    1. 使用XGBoost模型预测价格涨跌
    2. 根据预测置信度决定开仓
    3. 使用止损止盈和追踪止损管理风险
    
    信号:
    - 买入: 模型预测上涨且置信度足够高
    - 卖出: 模型预测下跌且置信度足够高
    - 观望: 预测为持或置信度不足
    """
    
    def __init__(self, trader, interval='15m'):
        """初始化XGBoost策略
        
        Args:
            trader: 交易者实例
            interval: K线周期
        """
        super().__init__(trader)
        self.logger = self.get_logger()
        
        # 策略配置
        xgb_config = config.XGBOOST_STRATEGY_CONFIG
        self.kline_interval = interval
        self.lookback_period = xgb_config.get('lookback_period', 150)
        
        # 时间间隔配置
        interval_map = {
            '1m': 60, '3m': 180, '5m': 300, '15m': 300, '30m': 600,
            '1h': 900, '2h': 1800, '4h': 3600, '6h': 7200, '12h': 14400, '1d': 28800
        }
        self.check_interval = interval_map.get(interval, 300)
        
        # 交易参数
        self.confidence_threshold = xgb_config.get('confidence_threshold', 0.50)
        self.stop_loss_pct = 0.03  # 3% 止损
        self.take_profit_pct = 0.06  # 6% 止盈
        
        # 追踪止损参数
        self.trailing_stop_enabled = True
        self.trailing_activation = 0.02  # 2% 利润激活
        self.trailing_distance = 0.01  # 1% 追踪距离
        
        # 持仓跟踪
        self.position_entry_time = None
        self.position_entry_price = None
        self.max_profit_reached = 0
        self.last_signal = 0
        
        # 特征提取器
        self.feature_extractor = XGBoostFeatureExtractor(
            lookback_period=self.lookback_period
        )
        
        # 加载模型和scaler
        self.model = None
        self._load_model()
        
        self.logger.info(f"XGBoost Strategy initialized")
        self.logger.info(f"Interval: {interval}, Lookback: {self.lookback_period}")
        self.logger.info(f"Confidence threshold: {self.confidence_threshold}")
    
    def _load_model(self):
        """加载XGBoost模型和scaler"""
        try:
            # 确定模型路径
            symbol = self.trader.symbol if self.trader else 'ZECUSDT'
            symbol_dir = symbol.lower().replace('/', '')
            
            models_base = Path(config.XGBOOST_STRATEGY_CONFIG.get(
                'models_base_dir', 'strategies/models'
            ))
            
            model_path = models_base / symbol_dir / 'xgboost_model.json'
            scaler_path = models_base / symbol_dir / 'xgboost_scaler.npz'
            
            # 加载模型
            if model_path.exists():
                self.model = xgb.Booster()
                self.model.load_model(str(model_path))
                self.logger.info(f"Model loaded from {model_path}")
            else:
                self.logger.warning(f"Model not found at {model_path}")
                return
            
            # 加载scaler
            if scaler_path.exists():
                self.feature_extractor.load_scaler(str(scaler_path))
            else:
                self.logger.warning(f"Scaler not found at {scaler_path}")
                
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
    
    def _prepare_dataframe(self, klines) -> pd.DataFrame:
        """将K线数据转换为DataFrame
        
        Args:
            klines: K线数据列表
            
        Returns:
            DataFrame
        """
        try:
            if not klines or len(klines) < 30:
                self.logger.error("Insufficient k-line data")
                return None
            
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume'
            ])
            
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error preparing DataFrame: {str(e)}")
            return None
    
    def generate_signal(self, klines=None) -> int:
        """生成交易信号
        
        Args:
            klines: K线数据 (可选，不提供则从trader获取)
            
        Returns:
            int: 交易信号 (1=买入, -1=卖出, 0=观望)
        """
        try:
            # 获取K线数据
            if klines is None:
                klines = self.trader.get_klines(
                    symbol=self.trader.symbol,
                    interval=self.kline_interval,
                    limit=self.lookback_period + 50
                )
            
            if not klines or len(klines) < self.lookback_period:
                self.logger.warning("Insufficient k-line data for analysis")
                return 0
            
            # 检查模型是否加载
            if self.model is None:
                self.logger.warning("Model not loaded")
                return 0
            
            # 转换为DataFrame
            df = self._prepare_dataframe(klines)
            if df is None:
                return 0
            
            # 计算特征
            df = self.feature_extractor.calculate_features(df)
            if df is None or len(df) == 0:
                return 0
            
            # 获取最新特征
            features = self.feature_extractor.get_latest_features(df)
            if features is None:
                return 0
            
            # 归一化
            if self.feature_extractor.scaler_fitted:
                features = self.feature_extractor.normalize_features(features)
            
            # 模型预测
            dmatrix = xgb.DMatrix(features)
            proba = self.model.predict(dmatrix)[0]
            
            # proba是一个包含3个类别概率的数组: [跌, 持, 涨]
            if len(proba) == 3:
                prob_down, prob_hold, prob_up = proba
            else:
                # 如果是单值预测，转换为概率
                prob_up = proba if proba > 0.5 else 0
                prob_down = 1 - proba if proba < 0.5 else 0
                prob_hold = 1 - prob_up - prob_down
            
            # 决策逻辑
            signal = 0
            max_prob = max(prob_down, prob_hold, prob_up)
            
            if prob_up > self.confidence_threshold and prob_up == max_prob:
                signal = 1
                self.logger.info(f"📈 BUY signal: prob_up={prob_up:.2%}, "
                               f"prob_down={prob_down:.2%}, prob_hold={prob_hold:.2%}")
            elif prob_down > self.confidence_threshold and prob_down == max_prob:
                signal = -1
                self.logger.info(f"📉 SELL signal: prob_down={prob_down:.2%}, "
                               f"prob_up={prob_up:.2%}, prob_hold={prob_hold:.2%}")
            else:
                self.logger.debug(f"⏸️ HOLD: prob_up={prob_up:.2%}, "
                                f"prob_down={prob_down:.2%}, prob_hold={prob_hold:.2%}")
            
            self.last_signal = signal
            return signal
            
        except Exception as e:
            self.logger.error(f"Error generating signal: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return 0
    
    def monitor_position(self):
        """监控持仓并执行交易逻辑"""
        try:
            position = self.trader.get_position()
            
            # 无持仓 - 检查入场信号
            if position is None or float(position['info'].get('positionAmt', 0)) == 0:
                signal = self.generate_signal()
                
                if signal != 0:
                    balance = self.trader.get_balance()
                    available_balance = float(balance['free'])
                    current_price = self.trader.get_market_price()
                    
                    # 计算交易量
                    symbol_config = getattr(self.trader, 'symbol_config', {})
                    trade_percent = symbol_config.get('trade_amount_percent', 100)
                    trade_amount = (available_balance * trade_percent / 100) / current_price
                    
                    if signal == 1:
                        self.trader.open_long(amount=trade_amount)
                        self.logger.info(f"🟢 LONG opened - Amount: {trade_amount:.6f}, "
                                       f"Price: {current_price}")
                    elif signal == -1:
                        self.trader.open_short(amount=trade_amount)
                        self.logger.info(f"🔴 SHORT opened - Amount: {trade_amount:.6f}, "
                                       f"Price: {current_price}")
                    
                    self.position_entry_time = time.time()
                    self.position_entry_price = current_price
                    self.max_profit_reached = 0
            else:
                # 有持仓 - 管理持仓
                self._manage_position(position)
                
        except Exception as e:
            self.logger.error(f"Error monitoring position: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
    
    def _manage_position(self, position):
        """管理现有持仓
        
        Args:
            position: 当前持仓信息
        """
        try:
            pos_side = position['info'].get('positionSide', 'LONG')
            entry_price = float(position['info'].get('entryPrice', 0))
            current_price = self.trader.get_market_price()
            
            # 计算盈亏比例
            if pos_side == 'LONG' or float(position['info'].get('positionAmt', 0)) > 0:
                pnl_pct = (current_price - entry_price) / entry_price
                is_long = True
            else:
                pnl_pct = (entry_price - current_price) / entry_price
                is_long = False
            
            # 更新最大利润
            self.max_profit_reached = max(self.max_profit_reached, pnl_pct)
            
            # === 止损检查 ===
            if pnl_pct < -self.stop_loss_pct:
                self.logger.info(f"🛑 Stop loss triggered: PnL={pnl_pct:.2%}")
                self.trader.close_position()
                self._reset_position_tracking()
                return
            
            # === 止盈检查 ===
            if pnl_pct >= self.take_profit_pct:
                self.logger.info(f"🎯 Take profit triggered: PnL={pnl_pct:.2%}")
                self.trader.close_position()
                self._reset_position_tracking()
                return
            
            # === 追踪止损 ===
            if self.trailing_stop_enabled and self.max_profit_reached >= self.trailing_activation:
                trailing_stop_level = self.max_profit_reached - self.trailing_distance
                if pnl_pct < trailing_stop_level:
                    self.logger.info(f"🔄 Trailing stop triggered: Max={self.max_profit_reached:.2%}, "
                                   f"Current={pnl_pct:.2%}")
                    self.trader.close_position()
                    self._reset_position_tracking()
                    return
            
            # === 反向信号检查 ===
            signal = self.generate_signal()
            if (is_long and signal == -1) or (not is_long and signal == 1):
                self.logger.info(f"🔀 Reverse signal detected, closing position")
                self.trader.close_position()
                self._reset_position_tracking()
                
        except Exception as e:
            self.logger.error(f"Error managing position: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
    
    def _reset_position_tracking(self):
        """重置持仓跟踪变量"""
        self.position_entry_time = None
        self.position_entry_price = None
        self.max_profit_reached = 0
