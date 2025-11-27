"""
XGBoost 价格预测策略
使用机器学习预测价格走势，结合技术指标特征

File: strategies/xgboost_price_strategy.py
"""
import numpy as np
import pandas as pd
import pandas_ta_classic as ta
from datetime import datetime
import time
import logging
import pickle
import os
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from strategies.base_strategy import BaseStrategy

try:
    import xgboost as xgb
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


@dataclass
class PredictionSignal:
    """预测信号数据类"""
    signal: int  # 1=做多, -1=做空, 0=观望
    predicted_direction: str  # 'up', 'down', 'neutral'
    confidence: float
    predicted_change: float  # 预测价格变化百分比
    feature_importance: Dict[str, float]
    reasoning: str


class XGBoostPriceStrategy(BaseStrategy):
    """
    XGBoost 价格预测策略
    
    核心逻辑：
    1. 使用技术指标作为特征
    2. XGBoost 模型预测未来价格方向
    3. 结合预测置信度过滤信号
    4. 动态止损止盈
    
    特征工程：
    - 价格变化率 (ROC)
    - RSI, MACD, Bollinger Bands
    - EMA 交叉信号
    - 成交量变化
    - ATR 波动率
    - ADX 趋势强度
    """
    
    def __init__(self, trader):
        """初始化策略"""
        super().__init__(trader)
        
        if not HAS_XGBOOST:
            raise ImportError("请安装 xgboost 和 sklearn: pip install xgboost scikit-learn")
        
        self.logger = self.get_logger()
        
        # ==================== 时间配置 ====================
        self.kline_interval = '15m'
        self.check_interval = 300  # 5分钟检查
        self.lookback_period = 1000  # 增加训练数据，降低过拟合
        self.training_lookback = self.lookback_period  # For compatibility with TradingManager
        self.prediction_horizon = 6  # 预测未来4根K线
        
        # ==================== 模型配置 ====================
        self.model: Optional[xgb.XGBClassifier] = None
        self.scaler = StandardScaler()
        self.model_path = 'models/xgboost_price_model.pkl'
        self.scaler_path = 'models/xgboost_scaler.pkl'
        self.retrain_interval = 24 * 60 * 60  # 24小时重训练
        self.last_train_time = 0
        self.min_train_samples = 200
        
        # XGBoost 参数 - 优化以降低过拟合
        self.xgb_params = {
            'n_estimators': 50,         # 从100降到50，减少树的数量
            'max_depth': 3,             # 从5降到3，降低树的深度
            'learning_rate': 0.05,      # 从0.1降到0.05，降低学习率
            'subsample': 0.7,           # 从0.8降到0.7，增加随机性
            'colsample_bytree': 0.7,    # 从0.8降到0.7，增加随机性
            'min_child_weight': 3,      # 新增：防止过拟合
            'gamma': 0.1,               # 新增：剪枝参数，控制分裂
            'reg_alpha': 0.1,           # 新增：L1正则化
            'reg_lambda': 1.0,          # 新增：L2正则化
            'objective': 'multi:softprob',
            'num_class': 3,             # 上涨/下跌/横盘
            'eval_metric': 'mlogloss',
            'use_label_encoder': False,
            'random_state': 42
        }
        
        # ==================== 信号配置 ====================
        self.min_confidence = 0.50  # 最小置信度（从55%降到50%）
        self.min_predicted_change = 0.003  # 最小预测变化 0.3%
        self.classification_threshold = 0.005  # 分类阈值 0.5%（从0.2%提高）
        
        # ==================== 仓位管理 ====================
        self.stop_loss_pct = 0.02  # 2% 止损
        self.take_profit_pct = 0.05  # 5% 止盈
        self.max_hold_time = 720  # 最大持仓12小时（分钟）
        
        # 追踪止损
        self.trailing_stop_enabled = True
        self.trailing_activation = 0.025  # 2.5%激活
        self.trailing_distance = 0.012  # 1.2%距离
        
        # ==================== 状态追踪 ====================
        self.position_entry_time = None
        self.position_entry_price = None
        self.max_profit_reached = 0
        self.trailing_stop_price = None
        self.last_signal: Optional[PredictionSignal] = None
        self.feature_names: List[str] = []
        
        # 加载已有模型
        self._load_model()
        
        self.logger.info("=" * 60)
        self.logger.info("XGBoost 价格预测策略初始化完成")
        self.logger.info(f"时间周期: {self.kline_interval}")
        self.logger.info(f"预测周期: {self.prediction_horizon} 根K线")
        self.logger.info(f"最小置信度: {self.min_confidence:.0%}")
        self.logger.info("=" * 60)

    # ==================== 特征工程 ====================
    
    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算所有特征（优化后）"""
        try:
            high = df['high']
            low = df['low']
            close = df['close']
            volume = df['volume']
            
            features = pd.DataFrame(index=df.index)
            
            # ---------- 价格变化率（精简）----------
            # 只保留关键周期，移除高度相关的短周期
            features['roc_5'] = ta.roc(close, length=5)
            features['roc_10'] = ta.roc(close, length=10)
            features['roc_20'] = ta.roc(close, length=20)
            
            # ---------- RSI ----------
            features['rsi_14'] = ta.rsi(close, length=14)
            # 移除 rsi_7，与 rsi_14 高度相关
            
            # ---------- MACD（精简）----------
            macd = ta.macd(close, fast=12, slow=26, signal=9)
            # 只保留 MACD 直方图，这是最重要的信号
            features['macd_hist'] = macd['MACDh_12_26_9']
            features['macd_signal'] = macd['MACDs_12_26_9']
            
            # ---------- Bollinger Bands ----------
            bb = ta.bbands(close, length=20, std=2)
            features['bb_width'] = (bb['BBU_20_2.0'] - bb['BBL_20_2.0']) / bb['BBM_20_2.0']
            features['bb_position'] = (close - bb['BBL_20_2.0']) / (bb['BBU_20_2.0'] - bb['BBL_20_2.0'])
            
            # ---------- EMA（精简）----------
            ema_8 = ta.ema(close, length=8)
            ema_21 = ta.ema(close, length=21)
            ema_55 = ta.ema(close, length=55)
            
            # EMA 交叉信号（只保留最重要的两个）
            features['ema_8_21_diff'] = (ema_8 - ema_21) / close
            features['ema_21_55_diff'] = (ema_21 - ema_55) / close
            
            # ---------- ATR 波动率 ----------
            atr = ta.atr(high, low, close, length=14)
            features['atr_pct'] = atr / close
            
            # ---------- ADX 趋势强度 ----------
            adx = ta.adx(high, low, close, length=14)
            features['adx'] = adx['ADX_14']
            features['di_diff'] = adx['DMP_14'] - adx['DMN_14']
            
            # ---------- 成交量特征 ----------
            volume_ma = ta.sma(volume, length=20)
            features['volume_ratio'] = volume / volume_ma
            # 移除 volume_change，只保留 volume_ratio
            
            # ---------- 动量指标 ----------
            features['mom_10'] = ta.mom(close, length=10)
            features['willr_14'] = ta.willr(high, low, close, length=14)
            
            # ---------- 价格位置 ----------
            features['high_low_range'] = (high - low) / close
            
            # ---------- 新增：价格趋势强度 ----------
            # 添加更多有价值的特征
            features['close_20_std'] = close.rolling(20).std() / close  # 20期标准差
            features['volume_20_std'] = volume.rolling(20).std() / volume  # 成交量波动
            
            return features
            
        except Exception as e:
            self.logger.error(f"计算特征出错: {str(e)}")
            return pd.DataFrame()
    
    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """准备训练数据"""
        try:
            features = self.calculate_features(df)
            
            # 计算目标变量：未来N根K线的价格变化方向
            future_return = df['close'].shift(-self.prediction_horizon) / df['close'] - 1
            
            # 分类标签: 0=下跌, 1=横盘, 2=上涨
            # 使用更高的阈值，减少横盘类别的噪声
            threshold = self.classification_threshold
            labels = pd.Series(1, index=df.index)  # 默认横盘
            labels[future_return > threshold] = 2   # 上涨
            labels[future_return < -threshold] = 0  # 下跌
            
            # 合并并删除NaN
            data = pd.concat([features, labels.rename('label')], axis=1)
            data = data.dropna()
            
            X = data.drop(columns=['label']).values
            y = data['label'].values
            
            self.feature_names = list(data.drop(columns=['label']).columns)
            
            return X, y
            
        except Exception as e:
            self.logger.error(f"准备训练数据出错: {str(e)}")
            return np.array([]), np.array([])

    # ==================== 模型训练与预测 ====================
    
    def train_model(self, df: pd.DataFrame) -> bool:
        """训练 XGBoost 模型"""
        try:
            X, y = self.prepare_training_data(df)
            
            if len(X) < self.min_train_samples:
                self.logger.warning(f"训练样本不足: {len(X)} < {self.min_train_samples}")
                return False
            
            # 统计类别分布
            from collections import Counter
            label_dist = Counter(y)
            self.logger.info(f"类别分布: 下跌={label_dist[0]}, 横盘={label_dist[1]}, 上涨={label_dist[2]}")
            
            # 划分训练集和验证集
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, shuffle=False
            )
            
            # 计算样本权重以平衡类别
            from sklearn.utils.class_weight import compute_sample_weight
            sample_weights = compute_sample_weight('balanced', y_train)
            
            # 标准化
            self.scaler.fit(X_train)
            X_train_scaled = self.scaler.transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            
            # 训练模型（添加早停机制）
            # XGBoost 2.0+ 需要在初始化时传入 early_stopping_rounds
            self.model = xgb.XGBClassifier(
                **self.xgb_params,
                early_stopping_rounds=10,  # 早停：验证集10轮无改善则停止
                callbacks=[xgb.callback.EvaluationMonitor(show_stdv=False)]
            )
            self.model.fit(
                X_train_scaled, y_train,
                sample_weight=sample_weights,  # 使用样本权重
                eval_set=[(X_val_scaled, y_val)],
                verbose=False
            )
            
            # 评估
            train_acc = self.model.score(X_train_scaled, y_train)
            val_acc = self.model.score(X_val_scaled, y_val)
            
            # 获取最佳迭代次数（XGBoost 2.0+ 使用 best_iteration 属性）
            best_iteration = getattr(self.model, 'best_iteration', self.xgb_params['n_estimators'])
            if best_iteration is None:
                best_iteration = self.xgb_params['n_estimators']
            
            self.logger.info(f"模型训练完成 | 样本数: {len(X)} | 最佳迭代: {best_iteration}/{self.xgb_params['n_estimators']}")
            self.logger.info(f"训练准确率: {train_acc:.2%} | 验证准确率: {val_acc:.2%}")
            
            # 保存模型
            self._save_model()
            self.last_train_time = time.time()
            
            return True
            
        except Exception as e:
            self.logger.error(f"训练模型出错: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    def predict(self, df: pd.DataFrame) -> Tuple[int, float, float]:
        """
        预测价格方向
        
        Returns:
            (预测类别, 置信度, 预测变化率)
        """
        try:
            if self.model is None:
                return 1, 0.0, 0.0  # 无模型返回横盘
            
            features = self.calculate_features(df)
            if features.empty:
                return 1, 0.0, 0.0
            
            # 取最后一行
            X = features.iloc[[-1]].values
            
            # 检查NaN
            if np.isnan(X).any():
                self.logger.warning("特征包含NaN值")
                return 1, 0.0, 0.0
            
            # 标准化
            X_scaled = self.scaler.transform(X)
            
            # 预测
            pred_proba = self.model.predict_proba(X_scaled)[0]
            pred_class = np.argmax(pred_proba)
            confidence = pred_proba[pred_class]
            
            # 计算预测变化率（基于概率加权）
            # 假设: 下跌=-1%, 横盘=0%, 上涨=+1%
            predicted_change = pred_proba[0] * (-0.01) + pred_proba[1] * 0 + pred_proba[2] * 0.01
            
            return int(pred_class), float(confidence), float(predicted_change)
            
        except Exception as e:
            self.logger.error(f"预测出错: {str(e)}")
            return 1, 0.0, 0.0
    
    def _save_model(self):
        """保存模型"""
        try:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            
            self.logger.info(f"模型已保存: {self.model_path}")
        except Exception as e:
            self.logger.error(f"保存模型出错: {str(e)}")
    
    def _load_model(self):
        """加载模型"""
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                with open(self.scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                self.logger.info(f"模型已加载: {self.model_path}")
        except Exception as e:
            self.logger.warning(f"加载模型失败: {str(e)}")
            self.model = None

    # ==================== 信号生成 ====================
    
    def generate_prediction_signal(self, df: pd.DataFrame) -> PredictionSignal:
        """生成预测信号"""
        try:
            # 检查是否需要重训练
            if self.model is None or (time.time() - self.last_train_time > self.retrain_interval):
                self.logger.info("开始训练/重训练模型...")
                self.train_model(df)
            
            # 预测
            pred_class, confidence, predicted_change = self.predict(df)
            
            # 转换预测类别
            direction_map = {0: 'down', 1: 'neutral', 2: 'up'}
            predicted_direction = direction_map.get(pred_class, 'neutral')
            
            # 生成交易信号
            signal = 0
            reasons = []
            
            if confidence >= self.min_confidence:
                if pred_class == 2 and predicted_change >= self.min_predicted_change:
                    signal = 1
                    reasons.append(f"预测上涨 (置信度: {confidence:.1%})")
                elif pred_class == 0 and predicted_change <= -self.min_predicted_change:
                    signal = -1
                    reasons.append(f"预测下跌 (置信度: {confidence:.1%})")
                else:
                    reasons.append(f"预测横盘或变化不足 (预测变化: {predicted_change:.2%})")
            else:
                reasons.append(f"置信度不足 ({confidence:.1%} < {self.min_confidence:.1%})")
            
            # 获取特征重要性
            feature_importance = self._get_feature_importance()
            
            # 构建推理
            reasoning = self._build_reasoning(
                signal, confidence, predicted_direction, 
                predicted_change, reasons, feature_importance
            )
            
            result = PredictionSignal(
                signal=signal,
                predicted_direction=predicted_direction,
                confidence=confidence,
                predicted_change=predicted_change,
                feature_importance=feature_importance,
                reasoning=reasoning
            )
            
            self.last_signal = result
            return result
            
        except Exception as e:
            self.logger.error(f"生成预测信号出错: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return self._default_signal()
    
    def _get_feature_importance(self) -> Dict[str, float]:
        """获取特征重要性"""
        try:
            if self.model is None or not self.feature_names:
                return {}
            
            importance = self.model.feature_importances_
            feature_imp = dict(zip(self.feature_names, importance))
            
            # 排序取前10
            sorted_imp = dict(sorted(feature_imp.items(), key=lambda x: x[1], reverse=True)[:10])
            return sorted_imp
            
        except Exception as e:
            return {}
    
    def _build_reasoning(
        self,
        signal: int,
        confidence: float,
        predicted_direction: str,
        predicted_change: float,
        reasons: List[str],
        feature_importance: Dict[str, float]
    ) -> str:
        """构建推理说明"""
        
        action = {1: "🟢 做多", -1: "🔴 做空", 0: "⚪ 观望"}.get(signal, "观望")
        direction_cn = {'up': '上涨', 'down': '下跌', 'neutral': '横盘'}.get(predicted_direction, '未知')
        
        # 特征重要性字符串
        top_features = "\n".join([f"  - {k}: {v:.3f}" for k, v in list(feature_importance.items())[:5]])
        
        reasoning = f"""
【交易建议】{action} | 置信度: {confidence:.1%}

【预测方向】{direction_cn} | 预测变化: {predicted_change:+.2%}

【决策原因】
{chr(10).join(['  - ' + r for r in reasons])}

【重要特征】
{top_features if top_features else '  - 模型未训练'}
"""
        return reasoning.strip()
    
    def _default_signal(self) -> PredictionSignal:
        """返回默认信号"""
        return PredictionSignal(
            signal=0,
            predicted_direction='neutral',
            confidence=0.0,
            predicted_change=0.0,
            feature_importance={},
            reasoning="数据不足或模型未就绪，建议观望"
        )

    # ==================== 策略接口 ====================
    
    def generate_signal(self, klines=None) -> int:
        """生成交易信号"""
        try:
            if klines is None:
                klines = self.trader.get_klines(
                    symbol=self.trader.symbol,
                    interval=self.kline_interval,
                    limit=self.lookback_period
                )
            
            if not klines or len(klines) < 100:
                self.logger.warning(f"K线数据不足: {len(klines) if klines else 0}")
                return 0
            
            df = pd.DataFrame(
                klines,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            signal = self.generate_prediction_signal(df)
            self._print_analysis_report(signal)
            
            return signal.signal
            
        except Exception as e:
            self.logger.error(f"生成信号出错: {str(e)}")
            return 0
    
    def _print_analysis_report(self, signal: PredictionSignal):
        """打印分析报告"""
        self.logger.info("=" * 70)
        self.logger.info("【XGBoost 价格预测策略分析报告】")
        self.logger.info("=" * 70)
        
        # 预测状态
        direction_icon = {'up': '📈', 'down': '📉', 'neutral': '➡️'}
        self.logger.info(f"{direction_icon.get(signal.predicted_direction, '❓')} 预测方向: {signal.predicted_direction}")
        self.logger.info(f"📊 置信度: {signal.confidence:.1%}")
        self.logger.info(f"📈 预测变化: {signal.predicted_change:+.2%}")
        
        # 交易信号
        signal_text = {1: "🟢 做多", -1: "🔴 做空", 0: "⚪ 观望"}
        self.logger.info(f"🎯 交易信号: {signal_text.get(signal.signal, '未知')}")
        
        # 重要特征
        if signal.feature_importance:
            self.logger.info("📋 重要特征:")
            for feat, imp in list(signal.feature_importance.items())[:5]:
                self.logger.info(f"   {feat}: {imp:.3f}")
        
        self.logger.info("-" * 70)
        self.logger.info(signal.reasoning)
        self.logger.info("=" * 70)
    
    def monitor_position(self):
        """监控仓位"""
        try:
            position = self.trader.get_position()
            
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
                        self.logger.info(f"✅ 开多仓 | 数量: {trade_amount:.6f} | 价格: {current_price}")
                    elif signal == -1:
                        self.trader.open_short(amount=trade_amount)
                        self.logger.info(f"✅ 开空仓 | 数量: {trade_amount:.6f} | 价格: {current_price}")
                    
                    self.position_entry_time = time.time()
                    self.position_entry_price = current_price
                    self.max_profit_reached = 0
                    self.trailing_stop_price = None
            else:
                self._manage_position(position)
                
        except Exception as e:
            self.logger.error(f"监控仓位出错: {str(e)}")
    
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
            
            # 检查预测反转
            signal = self.generate_signal()
            if (side == "long" and signal == -1) or (side == "short" and signal == 1):
                self.logger.info(f"🔄 预测反转 | 盈亏: {pnl_pct:.2%}")
                self.trader.close_position()
                return
            
            self.logger.debug(f"📊 持仓状态 | 方向: {side} | 盈亏: {pnl_pct:.2%}")
            
        except Exception as e:
            self.logger.error(f"管理仓位出错: {str(e)}")
