"""
XGBoost价格预测策略
使用XGBoost模型预测市场三种状态的概率: 上涨/下跌/观望

File: strategies/xgboost_price_strategy.py
"""
import numpy as np
import pandas as pd
from datetime import datetime
import time
import logging
import pickle
import os
from pathlib import Path
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

try:
    import xgboost as xgb
except ImportError:
    raise ImportError("请安装xgboost: pip install xgboost>=2.0.0")

from strategies.base_strategy import BaseStrategy
from strategies.xgboost_features import FeatureEngineer


@dataclass
class PredictionResult:
    """预测结果数据类"""
    long_prob: float  # 做多概率
    short_prob: float  # 做空概率
    hold_prob: float  # 观望概率
    confidence: float  # 置信度
    signal: int  # 交易信号 (1=做多, -1=做空, 0=观望)
    predicted_class: int  # 预测类别 (0=下跌, 1=观望, 2=上涨)


class XGBoostPriceStrategy(BaseStrategy):
    """
    XGBoost价格预测策略
    
    策略特点:
    1. 使用XGBoost三分类模型预测市场状态
    2. 基于30+技术指标特征
    3. 输出上涨/下跌/观望的概率
    4. 支持模型训练、保存和加载
    5. 支持定期重训练以适应市场变化
    
    分类标签定义:
    - 类别0 (下跌): 未来价格下跌 > label_threshold
    - 类别1 (观望): 未来价格变化在 [-label_threshold, label_threshold] 之间
    - 类别2 (上涨): 未来价格上涨 > label_threshold
    """
    
    def __init__(self, trader, interval='15m'):
        """初始化策略"""
        super().__init__(trader)
        self.logger = self.get_logger()
        
        # ==================== 时间配置 ====================
        self.kline_interval = interval
        self.check_interval = 300  # 5分钟检查一次
        self.lookback_period = 200  # 策略需要的历史数据
        self.training_lookback = 5000  # 训练数据窗口
        
        # ==================== 模型配置 ====================
        self.model = None
        self.feature_engineer = FeatureEngineer()
        self.feature_names = []
        
        # 模型参数
        self.xgb_params = {
            'objective': 'multi:softprob',  # 多分类概率输出
            'num_class': 3,  # 三分类
            'max_depth': 3,  # 降低树深度防止过拟合
            'learning_rate': 0.03,  # 降低学习率
            'n_estimators': 300,  # 增加树的数量(配合早停)
            'min_child_weight': 3,  # 增加叶子节点最小权重
            'subsample': 0.7,  # 降低样本采样率
            'colsample_bytree': 0.7,  # 降低特征采样率
            'reg_alpha': 0.5,  # 增加L1正则化
            'reg_lambda': 1.5,  # 增加L2正则化
            'random_state': 42,
            'n_jobs': -1,
            'eval_metric': 'mlogloss',
            'early_stopping_rounds': 10  # 早停轮数
        }
        
        # ==================== 标签生成配置 ====================
        self.label_threshold = 0.015  # 1.5% 涨跌阈值
        self.prediction_horizon = 5  # 预测未来5根K线
        
        # ==================== 训练配置 ====================
        self.retrain_enabled = True
        self.retrain_interval = 100  # 每100次检查重训练一次
        self.check_count = 0
        self.min_training_samples = 500
        self.model_dir = Path(__file__).parent.parent / 'models'
        self.model_dir.mkdir(exist_ok=True)
        
        # ==================== 交易阈值 ====================
        self.min_trade_prob = 0.40  # 最小交易概率 (降低以增加交易机会)
        self.min_confidence = 0.10  # 最小置信度 (大幅降低，0.55太高了)
        self.strong_signal_prob = 0.65  # 强信号概率阈值
        
        # ==================== 仓位管理 ====================
        self.stop_loss_pct = 0.02  # 2% 止损
        self.take_profit_pct = 0.06  # 6% 止盈
        self.max_hold_time = 720  # 最大持仓12小时(分钟)
        
        # 追踪止损
        self.trailing_stop_enabled = True
        self.trailing_activation = 0.025  # 2.5%激活
        self.trailing_distance = 0.012  # 1.2%距离
        
        # ==================== 状态追踪 ====================
        self.position_entry_time = None
        self.position_entry_price = None
        self.max_profit_reached = 0
        self.trailing_stop_price = None
        self.last_prediction: Optional[PredictionResult] = None
        
        # ==================== 初始化模型 ====================
        self._load_or_initialize_model()
        
        self.logger.info("=" * 60)
        self.logger.info("XGBoost价格预测策略初始化完成")
        self.logger.info(f"时间周期: {self.kline_interval}")
        self.logger.info(f"标签阈值: {self.label_threshold:.1%}")
        self.logger.info(f"预测时间跨度: {self.prediction_horizon}根K线")
        self.logger.info(f"交易阈值: 最小概率={self.min_trade_prob:.0%}, "
                        f"最小置信度={self.min_confidence:.0%}")
        self.logger.info("=" * 60)
    
    def _load_or_initialize_model(self):
        """加载或初始化模型"""
        model_path = self.model_dir / 'xgboost_price_model.pkl'
        scaler_path = self.model_dir / 'xgboost_scaler.pkl'
        
        if model_path.exists() and scaler_path.exists():
            try:
                self.logger.info("正在加载已有模型...")
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                with open(scaler_path, 'rb') as f:
                    self.feature_engineer.scaler = pickle.load(f)
                self.logger.info(f"✓ 模型加载成功: {model_path}")
            except Exception as e:
                self.logger.warning(f"模型加载失败: {str(e)}, 将重新训练")
                self.model = None
        else:
            self.logger.info("未找到预训练模型，将在首次运行时训练")
    
    def generate_labels(self, df: pd.DataFrame) -> pd.Series:
        """
        生成训练标签
        
        标签定义:
        - 0: 下跌 (未来价格跌幅 > label_threshold)
        - 1: 观望 (未来价格变化在阈值范围内)
        - 2: 上涨 (未来价格涨幅 > label_threshold)
        
        Args:
            df: OHLCV数据
            
        Returns:
            标签序列
        """
        close = df['close']
        
        # 计算未来收益率
        future_returns = close.shift(-self.prediction_horizon) / close - 1
        
        # 生成标签
        labels = pd.Series(1, index=df.index)  # 默认为观望
        labels[future_returns > self.label_threshold] = 2  # 上涨
        labels[future_returns < -self.label_threshold] = 0  # 下跌
        
        return labels
    
    def prepare_training_data(self, klines) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
        """
        准备训练数据
        
        Args:
            klines: K线数据
            
        Returns:
            (特征DataFrame, 标签Series)
        """
        try:
            # 转换为DataFrame
            df = pd.DataFrame(
                klines,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 生成标签
            labels = self.generate_labels(df)
            
            # 提取特征
            features, self.feature_names = self.feature_engineer.prepare_data(
                df, normalize=True, fit_scaler=True
            )
            
            if features is None or len(features) == 0:
                return None, None
            
            # 删除无效标签的样本(最后prediction_horizon行)
            valid_idx = ~labels.isna()
            features = features[valid_idx]
            labels = labels[valid_idx]
            
            # 确保没有NaN
            valid_idx = ~(features.isna().any(axis=1) | labels.isna())
            features = features[valid_idx]
            labels = labels[valid_idx]
            
            return features, labels
            
        except Exception as e:
            self.logger.error(f"准备训练数据出错: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None, None
    
    def train_model(self) -> bool:
        """
        训练XGBoost模型
        
        Returns:
            是否训练成功
        """
        try:
            self.logger.info("开始训练XGBoost模型...")
            
            # 获取训练数据
            klines = self.trader.get_klines(
                symbol=self.trader.symbol,
                interval=self.kline_interval,
                limit=self.training_lookback
            )
            
            if not klines or len(klines) < self.min_training_samples:
                self.logger.warning(f"训练数据不足: {len(klines) if klines else 0}")
                return False
            
            # 准备训练数据
            X, y = self.prepare_training_data(klines)
            
            if X is None or len(X) < self.min_training_samples:
                self.logger.warning(f"有效训练样本不足: {len(X) if X is not None else 0}")
                return False
            
            # 检查类别分布
            class_counts = y.value_counts()
            self.logger.info(f"训练样本分布: {dict(class_counts)}")
            
            # 分割训练集和验证集
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # 训练模型
            self.model = xgb.XGBClassifier(**self.xgb_params)
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            
            # 评估模型
            train_acc = self.model.score(X_train, y_train)
            val_acc = self.model.score(X_val, y_val)
            
            self.logger.info(f"✓ 模型训练完成")
            self.logger.info(f"  训练准确率: {train_acc:.2%}")
            self.logger.info(f"  验证准确率: {val_acc:.2%}")
            self.logger.info(f"  训练样本数: {len(X_train)}")
            self.logger.info(f"  验证样本数: {len(X_val)}")
            
            # 保存模型
            self.save_model()
            
            # 记录特征重要性
            self._log_feature_importance()
            
            return True
            
        except Exception as e:
            self.logger.error(f"训练模型出错: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    def _log_feature_importance(self, top_n: int = 10):
        """记录特征重要性"""
        try:
            if self.model is None:
                return
            
            importance = self.model.feature_importances_
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            self.logger.info(f"Top {top_n} 重要特征:")
            for idx, row in feature_importance.head(top_n).iterrows():
                self.logger.info(f"  {row['feature']}: {row['importance']:.4f}")
                
        except Exception as e:
            self.logger.error(f"记录特征重要性出错: {str(e)}")
    
    def predict_probabilities(self, klines) -> Optional[PredictionResult]:
        """
        预测三种状态的概率
        
        Args:
            klines: K线数据
            
        Returns:
            预测结果
        """
        try:
            # 检查模型
            if self.model is None:
                self.logger.warning("模型未训练，正在训练...")
                if not self.train_model():
                    return None
            
            # 转换为DataFrame
            df = pd.DataFrame(
                klines,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 提取特征(使用已有的scaler，不重新fit)
            features, _ = self.feature_engineer.prepare_data(
                df, normalize=True, fit_scaler=False
            )
            
            if features is None or len(features) == 0:
                return None
            
            # 预测最后一个样本
            X = features.iloc[[-1]]
            
            # 预测概率
            probas = self.model.predict_proba(X)[0]
            predicted_class = self.model.predict(X)[0]
            
            # 解析概率 (类别顺序: 0=下跌, 1=观望, 2=上涨)
            short_prob = float(probas[0])
            hold_prob = float(probas[1])
            long_prob = float(probas[2])
            
            # 计算置信度(最大概率与次大概率的差距)
            sorted_probs = sorted(probas, reverse=True)
            confidence = float(sorted_probs[0] - sorted_probs[1])
            
            # 生成交易信号
            signal = 0
            if long_prob >= self.min_trade_prob and confidence >= self.min_confidence:
                if long_prob > short_prob and long_prob > hold_prob:
                    signal = 1
            elif short_prob >= self.min_trade_prob and confidence >= self.min_confidence:
                if short_prob > long_prob and short_prob > hold_prob:
                    signal = -1
            
            result = PredictionResult(
                long_prob=long_prob,
                short_prob=short_prob,
                hold_prob=hold_prob,
                confidence=confidence,
                signal=signal,
                predicted_class=int(predicted_class)
            )
            
            self.last_prediction = result
            return result
            
        except Exception as e:
            self.logger.error(f"预测出错: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def generate_signal(self, klines=None) -> int:
        """
        生成交易信号
        
        Args:
            klines: K线数据(可选)
            
        Returns:
            交易信号 (1=做多, -1=做空, 0=观望)
        """
        try:
            # 获取K线数据
            if klines is None:
                klines = self.trader.get_klines(
                    symbol=self.trader.symbol,
                    interval=self.kline_interval,
                    limit=self.lookback_period
                )
            
            if not klines or len(klines) < 100:
                self.logger.warning(f"K线数据不足: {len(klines) if klines else 0}")
                return 0
            
            # 检查是否需要重训练
            self.check_count += 1
            if self.retrain_enabled and self.check_count % self.retrain_interval == 0:
                self.logger.info(f"达到重训练间隔({self.retrain_interval}次检查)，开始重训练...")
                self.train_model()
            
            # 预测
            prediction = self.predict_probabilities(klines)
            
            if prediction is None:
                return 0
            
            # 打印预测结果
            self._print_prediction_report(prediction)
            
            return prediction.signal
            
        except Exception as e:
            self.logger.error(f"生成信号出错: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return 0
    
    def _print_prediction_report(self, pred: PredictionResult):
        """打印预测报告"""
        self.logger.info("=" * 70)
        self.logger.info("【XGBoost预测报告】")
        self.logger.info("=" * 70)
        
        # 概率分布
        self.logger.info(f"📊 概率预测:")
        self.logger.info(f"   做多: {pred.long_prob:.1%}")
        self.logger.info(f"   做空: {pred.short_prob:.1%}")
        self.logger.info(f"   观望: {pred.hold_prob:.1%}")
        self.logger.info(f"   置信度: {pred.confidence:.1%}")
        
        # 预测类别
        class_names = {0: "下跌", 1: "观望", 2: "上涨"}
        self.logger.info(f"🎯 预测类别: {class_names.get(pred.predicted_class, '未知')}")
        
        # 交易信号
        signal_text = {1: "🟢 做多", -1: "🔴 做空", 0: "⚪ 观望"}
        self.logger.info(f"📈 交易信号: {signal_text.get(pred.signal, '未知')}")
        
        self.logger.info("=" * 70)
    
    def monitor_position(self):
        """监控仓位"""
        try:
            position = self.trader.get_position()
            
            # 无仓位 - 检查入场
            if position is None or float(position['info'].get('positionAmt', 0)) == 0:
                signal = self.generate_signal()
                
                if signal != 0:
                    balance = self.trader.get_balance()
                    available = float(balance['free'])
                    current_price = self.trader.get_market_price()
                    
                    symbol_config = self.trader.symbol_config
                    trade_pct = symbol_config.get('trade_amount_percent', 100)
                    trade_amount = (available * trade_pct / 100) / current_price
                    
                    if signal == 1:
                        self.trader.open_long(amount=trade_amount)
                        self.logger.info(
                            f"✅ 开多仓 | 数量: {trade_amount:.6f} | 价格: {current_price} | "
                            f"概率: {self.last_prediction.long_prob:.1%}"
                        )
                    elif signal == -1:
                        self.trader.open_short(amount=trade_amount)
                        self.logger.info(
                            f"✅ 开空仓 | 数量: {trade_amount:.6f} | 价格: {current_price} | "
                            f"概率: {self.last_prediction.short_prob:.1%}"
                        )
                    
                    self.position_entry_time = time.time()
                    self.position_entry_price = current_price
                    self.max_profit_reached = 0
                    self.trailing_stop_price = None
            else:
                self._manage_position(position)
                
        except Exception as e:
            self.logger.error(f"监控仓位出错: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
    
    def _manage_position(self, position):
        """管理现有仓位"""
        try:
            pos_amt = float(position['info'].get('positionAmt', 0))
            entry_price = float(position['info'].get('entryPrice', 0))
            current_price = self.trader.get_market_price()
            side = "long" if pos_amt > 0 else "short"
            
            # 计算盈亏
            if side == "long":
                pnl_pct = (current_price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - current_price) / entry_price
            
            # 更新最大利润
            if pnl_pct > self.max_profit_reached:
                self.max_profit_reached = pnl_pct
                
                # 更新追踪止损
                if self.trailing_stop_enabled and pnl_pct >= self.trailing_activation:
                    if side == "long":
                        self.trailing_stop_price = current_price * (1 - self.trailing_distance)
                    else:
                        self.trailing_stop_price = current_price * (1 + self.trailing_distance)
            
            # 检查追踪止损
            if self.trailing_stop_price:
                if side == "long" and current_price <= self.trailing_stop_price:
                    self.logger.info(f"📉 追踪止损触发 | 盈亏: {pnl_pct:.2%}")
                    self.trader.close_position()
                    return
                elif side == "short" and current_price >= self.trailing_stop_price:
                    self.logger.info(f"📉 追踪止损触发 | 盈亏: {pnl_pct:.2%}")
                    self.trader.close_position()
                    return
            
            # 检查止损
            if pnl_pct <= -self.stop_loss_pct:
                self.logger.info(f"🛑 止损触发 | 盈亏: {pnl_pct:.2%}")
                self.trader.close_position()
                return
            
            # 检查止盈
            if pnl_pct >= self.take_profit_pct:
                self.logger.info(f"🎯 止盈触发 | 盈亏: {pnl_pct:.2%}")
                self.trader.close_position()
                return
            
            # 检查持仓时间
            if self.position_entry_time:
                hold_mins = (time.time() - self.position_entry_time) / 60
                if hold_mins >= self.max_hold_time:
                    self.logger.info(f"⏰ 最大持仓时间 | 盈亏: {pnl_pct:.2%}")
                    self.trader.close_position()
                    return
            
            # 检查反向信号
            signal = self.generate_signal()
            if (side == "long" and signal == -1) or (side == "short" and signal == 1):
                self.logger.info(f"🔄 反向信号触发 | 盈亏: {pnl_pct:.2%}")
                self.trader.close_position()
                return
                
        except Exception as e:
            self.logger.error(f"管理仓位出错: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
    
    def save_model(self):
        """保存模型到磁盘"""
        try:
            if self.model is None:
                self.logger.warning("模型未训练，无法保存")
                return
            
            model_path = self.model_dir / 'xgboost_price_model.pkl'
            scaler_path = self.model_dir / 'xgboost_scaler.pkl'
            
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.feature_engineer.scaler, f)
            
            self.logger.info(f"✓ 模型已保存: {model_path}")
            
        except Exception as e:
            self.logger.error(f"保存模型出错: {str(e)}")
    
    def load_model(self):
        """从磁盘加载模型"""
        self._load_or_initialize_model()
