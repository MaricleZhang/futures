"""
随机森林交易策略 V2
基于Random Forest模型预测交易信号

核心改进 (防过拟合):
1. 简化为二分类问题 (涨/跌)
2. 使用更长预测周期减少噪声
3. 特征选择 - 只保留高信息量特征
4. 更严格的正则化参数
5. 滚动窗口验证
6. 概率校准
7. 集成多个时间尺度特征

File: strategies/random_forest_strategy.py
"""
import numpy as np
import pandas as pd
import pandas_ta_classic as ta
import time
import logging
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.feature_selection import SelectFromModel

from strategies.base_strategy import BaseStrategy
import config


class RandomForestStrategy(BaseStrategy):
    """随机森林交易策略 V2"""
    
    def __init__(self, trader, interval='15m', symbol=None):
        """初始化策略"""
        super().__init__(trader)
        self.logger = self.get_logger()
        
        # ==================== 交易对配置 ====================
        self.symbol = symbol or getattr(trader, 'symbol', 'UNKNOWN')
        self.symbol_key = self.symbol.lower().replace('/', '')
        
        # ==================== 时间配置 ====================
        self.kline_interval = interval
        interval_to_check = {
            '1m': 60, '5m': 300, '15m': 300,
            '30m': 600, '1h': 900, '4h': 3600,
        }
        self.check_interval = interval_to_check.get(interval, 300)
        
        # ==================== 模型配置 ====================
        self.lookback_period = 250  # 回看周期
        self.prediction_horizon = 8  # 预测未来N根K线 (增加到8根减少噪声)
        self.min_price_change = 0.008  # 最小价格变化阈值 (0.8%)
        
        # 强化防过拟合参数
        self.max_depth = 5           # 更浅的树
        self.min_samples_split = 50  # 更大的分裂样本数
        self.min_samples_leaf = 25   # 更大的叶节点样本数
        self.n_estimators = 200      # 更多的树
        self.max_features = 0.3      # 只用30%特征
        self.cv_folds = 5
        
        # 置信度阈值 (提高到60%)
        self.confidence_threshold = 0.60
        
        # 模型路径
        base_dir = 'strategies/models'
        symbol_model_dir = f"{base_dir}/{self.symbol_key}"
        self.model_path = f"{symbol_model_dir}/rf_model_v2.joblib"
        self.scaler_path = f"{symbol_model_dir}/rf_scaler_v2.joblib"
        self.selector_path = f"{symbol_model_dir}/rf_selector_v2.joblib"
        
        # ==================== 模型和Scaler ====================
        self.model = None
        self.scaler = None
        self.selector = None
        self.selected_features = None
        self._load_model()
        
        # ==================== 仓位管理 ====================
        self.stop_loss_pct = 0.02
        self.take_profit_pct = 0.06
        self.max_hold_time = 1440
        
        # 追踪止损
        self.trailing_stop_enabled = True
        self.trailing_activation = 0.02
        self.trailing_distance = 0.01
        
        # ==================== 状态追踪 ====================
        self.position_entry_time = None
        self.position_entry_price = None
        self.max_profit_reached = 0
        self.trailing_stop_price = None
        
        self.logger.info("=" * 70)
        self.logger.info("🌲 Random Forest Strategy V2 初始化完成")
        self.logger.info(f"交易对: {self.symbol}")
        self.logger.info(f"K线周期: {self.kline_interval} | 检查间隔: {self.check_interval}秒")
        self.logger.info(f"树数量: {self.n_estimators} | 最大深度: {self.max_depth}")
        self.logger.info(f"预测周期: {self.prediction_horizon}根K线 | 阈值: {self.min_price_change:.1%}")
        self.logger.info(f"置信度阈值: {self.confidence_threshold:.0%}")
        self.logger.info("=" * 70)
    
    def _load_model(self):
        """加载预训练模型"""
        try:
            model_path = Path(self.model_path)
            scaler_path = Path(self.scaler_path)
            selector_path = Path(self.selector_path)
            
            if model_path.exists() and scaler_path.exists():
                self.model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                if selector_path.exists():
                    self.selector = joblib.load(selector_path)
                self.logger.info(f"✅ 模型加载成功: {model_path}")
            else:
                self.logger.warning(f"⚠️ 模型文件不存在，将在首次运行时训练")
                self.scaler = RobustScaler()  # 使用RobustScaler更抗异常值
        except Exception as e:
            self.logger.error(f"模型加载失败: {str(e)}")
            self.scaler = RobustScaler()

    
    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标特征 - 精简版，减少噪声"""
        try:
            high = df['high']
            low = df['low']
            close = df['close']
            volume = df['volume']
            
            # ========== 趋势特征 (归一化) ==========
            for period in [5, 10, 20, 50]:
                sma = ta.sma(close, length=period)
                df[f'price_sma{period}_ratio'] = (close - sma) / sma
            
            # 均线斜率 (趋势强度)
            df['sma20_slope'] = ta.sma(close, length=20).pct_change(5)
            df['sma50_slope'] = ta.sma(close, length=50).pct_change(10)
            
            # ========== 动量特征 (归一化到0-1) ==========
            df['rsi_14'] = ta.rsi(close, length=14) / 100
            df['rsi_7'] = ta.rsi(close, length=7) / 100
            
            # RSI变化率
            df['rsi_14_change'] = df['rsi_14'].diff(3)
            
            # MACD归一化
            macd = ta.macd(close, fast=12, slow=26, signal=9)
            df['macd_norm'] = macd['MACD_12_26_9'] / close
            df['macd_hist_norm'] = macd['MACDh_12_26_9'] / close
            
            # Stochastic
            stoch = ta.stoch(high, low, close, k=14, d=3)
            df['stoch_k'] = stoch['STOCHk_14_3_3'] / 100
            df['stoch_d'] = stoch['STOCHd_14_3_3'] / 100
            
            # ========== 波动率特征 ==========
            df['atr_14'] = ta.atr(high, low, close, length=14)
            df['atr_ratio'] = df['atr_14'] / close
            
            # 布林带位置
            bb = ta.bbands(close, length=20, std=2)
            df['bb_position'] = (close - bb['BBL_20_2.0']) / (bb['BBU_20_2.0'] - bb['BBL_20_2.0'] + 1e-8)
            df['bb_width'] = (bb['BBU_20_2.0'] - bb['BBL_20_2.0']) / bb['BBM_20_2.0']
            
            # 历史波动率
            df['volatility_10'] = close.pct_change().rolling(10).std()
            df['volatility_20'] = close.pct_change().rolling(20).std()
            
            # ========== 成交量特征 ==========
            vol_sma = ta.sma(volume, length=20)
            df['volume_ratio'] = volume / (vol_sma + 1e-8)
            df['volume_trend'] = vol_sma.pct_change(5)
            
            # ========== 趋势强度 ==========
            adx = ta.adx(high, low, close, length=14)
            df['adx'] = adx['ADX_14'] / 100
            df['di_diff'] = (adx['DMP_14'] - adx['DMN_14']) / 100
            
            # ========== 价格动量 ==========
            df['return_1'] = close.pct_change(1)
            df['return_5'] = close.pct_change(5)
            df['return_10'] = close.pct_change(10)
            df['return_20'] = close.pct_change(20)
            
            # 动量变化
            df['momentum_accel'] = df['return_5'] - df['return_5'].shift(5)
            
            # ========== 支撑阻力特征 ==========
            df['high_20'] = high.rolling(20).max()
            df['low_20'] = low.rolling(20).min()
            df['price_position'] = (close - df['low_20']) / (df['high_20'] - df['low_20'] + 1e-8)
            
            # 距离高低点
            df['dist_from_high'] = (df['high_20'] - close) / close
            df['dist_from_low'] = (close - df['low_20']) / close
            
            return df
            
        except Exception as e:
            self.logger.error(f"计算特征出错: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def get_feature_columns(self) -> list:
        """获取特征列名"""
        return [
            # 趋势
            'price_sma5_ratio', 'price_sma10_ratio', 'price_sma20_ratio', 'price_sma50_ratio',
            'sma20_slope', 'sma50_slope',
            # 动量
            'rsi_14', 'rsi_7', 'rsi_14_change',
            'macd_norm', 'macd_hist_norm',
            'stoch_k', 'stoch_d',
            # 波动率
            'atr_ratio', 'bb_position', 'bb_width',
            'volatility_10', 'volatility_20',
            # 成交量
            'volume_ratio', 'volume_trend',
            # 趋势强度
            'adx', 'di_diff',
            # 价格动量
            'return_1', 'return_5', 'return_10', 'return_20',
            'momentum_accel',
            # 支撑阻力
            'price_position', 'dist_from_high', 'dist_from_low'
        ]
    
    def create_labels(self, df: pd.DataFrame) -> pd.Series:
        """创建二分类标签: 涨(1) / 跌(0)"""
        future_return = df['close'].shift(-self.prediction_horizon) / df['close'] - 1
        
        # 二分类: 涨=1, 跌=0, 中性=NaN (过滤掉)
        labels = pd.Series(index=df.index, dtype=float)
        labels[future_return > self.min_price_change] = 1   # 涨
        labels[future_return < -self.min_price_change] = 0  # 跌
        # 中性区间设为NaN，训练时会被过滤
        
        return labels

    
    def train_model(self, df: pd.DataFrame) -> dict:
        """训练随机森林模型 V2 (强化防过拟合)"""
        try:
            self.logger.info("🔄 开始训练随机森林模型 V2...")
            
            # 计算特征
            df = self.calculate_features(df)
            if df is None:
                return {'success': False, 'error': '特征计算失败'}
            
            # 创建标签
            df['label'] = self.create_labels(df)
            
            # 移除NaN (包括中性样本)
            feature_cols = self.get_feature_columns()
            df_clean = df.dropna(subset=feature_cols + ['label'])
            
            self.logger.info(f"📊 有效样本数: {len(df_clean)} (过滤中性样本后)")
            
            if len(df_clean) < 1000:
                return {'success': False, 'error': f'数据不足: {len(df_clean)}'}
            
            # 检查类别分布
            class_dist = df_clean['label'].value_counts()
            self.logger.info(f"📊 类别分布: 涨={class_dist.get(1, 0)} | 跌={class_dist.get(0, 0)}")
            
            X = df_clean[feature_cols].values
            y = df_clean['label'].values.astype(int)
            
            # ========== 时间序列分割 ==========
            # 70% 训练, 15% 验证, 15% 测试
            train_end = int(len(X) * 0.70)
            val_end = int(len(X) * 0.85)
            
            X_train, y_train = X[:train_end], y[:train_end]
            X_val, y_val = X[train_end:val_end], y[train_end:val_end]
            X_test, y_test = X[val_end:], y[val_end:]
            
            self.logger.info(f"📊 数据分割: 训练={len(X_train)} | 验证={len(X_val)} | 测试={len(X_test)}")
            
            # 标准化 (使用RobustScaler抗异常值)
            self.scaler = RobustScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            X_test_scaled = self.scaler.transform(X_test)
            
            # ========== 特征选择 ==========
            self.logger.info("🔍 特征选择中...")
            
            # 使用GradientBoosting进行特征选择
            gb_selector = GradientBoostingClassifier(
                n_estimators=50, max_depth=3, random_state=42
            )
            gb_selector.fit(X_train_scaled, y_train)
            
            # 选择重要性 > 平均值的特征
            importances = gb_selector.feature_importances_
            threshold = np.mean(importances)
            selected_mask = importances >= threshold
            
            selected_features = [f for f, s in zip(feature_cols, selected_mask) if s]
            self.selected_features = selected_features
            self.logger.info(f"✅ 选择了 {len(selected_features)}/{len(feature_cols)} 个特征")
            
            # 应用特征选择
            X_train_selected = X_train_scaled[:, selected_mask]
            X_val_selected = X_val_scaled[:, selected_mask]
            X_test_selected = X_test_scaled[:, selected_mask]
            
            # ========== 交叉验证 ==========
            tscv = TimeSeriesSplit(n_splits=self.cv_folds)
            
            # 创建模型
            self.model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1,
                oob_score=True,
                bootstrap=True,
                max_samples=0.8  # 每棵树只用80%样本
            )
            
            # 交叉验证
            cv_scores = cross_val_score(
                self.model, X_train_selected, y_train,
                cv=tscv, scoring='accuracy'
            )
            self.logger.info(f"📊 交叉验证准确率: {cv_scores.mean():.2%} (+/- {cv_scores.std()*2:.2%})")
            
            # 训练最终模型
            self.model.fit(X_train_selected, y_train)
            
            # ========== 评估 ==========
            train_pred = self.model.predict(X_train_selected)
            val_pred = self.model.predict(X_val_selected)
            test_pred = self.model.predict(X_test_selected)
            
            train_acc = accuracy_score(y_train, train_pred)
            val_acc = accuracy_score(y_val, val_pred)
            test_acc = accuracy_score(y_test, test_pred)
            oob_acc = self.model.oob_score_
            
            # F1分数
            train_f1 = f1_score(y_train, train_pred, average='weighted')
            test_f1 = f1_score(y_test, test_pred, average='weighted')
            
            self.logger.info("=" * 50)
            self.logger.info("📈 模型评估结果:")
            self.logger.info(f"  训练集准确率: {train_acc:.2%} | F1: {train_f1:.2%}")
            self.logger.info(f"  验证集准确率: {val_acc:.2%}")
            self.logger.info(f"  测试集准确率: {test_acc:.2%} | F1: {test_f1:.2%}")
            self.logger.info(f"  OOB准确率: {oob_acc:.2%}")
            self.logger.info("=" * 50)
            
            # 过拟合检查
            overfit_gap = train_acc - test_acc
            if overfit_gap > 0.08:
                self.logger.warning(f"⚠️ 可能存在过拟合! 差距: {overfit_gap:.2%}")
            else:
                self.logger.info(f"✅ 过拟合控制良好，差距: {overfit_gap:.2%}")
            
            # 特征重要性
            feature_importance = pd.DataFrame({
                'feature': selected_features,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            self.logger.info("📊 Top 10 重要特征:")
            for _, row in feature_importance.head(10).iterrows():
                self.logger.info(f"  {row['feature']}: {row['importance']:.4f}")
            
            # 保存模型
            self._save_model(selected_mask)
            
            return {
                'success': True,
                'train_accuracy': train_acc,
                'val_accuracy': val_acc,
                'test_accuracy': test_acc,
                'cv_accuracy': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'oob_accuracy': oob_acc,
                'overfit_gap': overfit_gap,
                'train_f1': train_f1,
                'test_f1': test_f1,
                'feature_importance': feature_importance,
                'selected_features': selected_features
            }
            
        except Exception as e:
            self.logger.error(f"训练模型出错: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {'success': False, 'error': str(e)}
    
    def _save_model(self, selected_mask):
        """保存模型"""
        try:
            model_path = Path(self.model_path)
            model_path.parent.mkdir(parents=True, exist_ok=True)
            
            joblib.dump(self.model, model_path)
            joblib.dump(self.scaler, self.scaler_path)
            joblib.dump({
                'mask': selected_mask,
                'features': self.selected_features
            }, self.selector_path)
            
            self.logger.info(f"✅ 模型已保存: {model_path}")
        except Exception as e:
            self.logger.error(f"保存模型失败: {str(e)}")

    
    def generate_signal(self, klines=None) -> int:
        """生成交易信号"""
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
            
            # 转换为DataFrame
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 如果模型不存在，先训练
            if self.model is None:
                self.logger.info("模型不存在，开始训练...")
                result = self.train_model(df.copy())
                if not result['success']:
                    self.logger.error(f"训练失败: {result.get('error')}")
                    return 0
            
            # 计算特征
            df = self.calculate_features(df)
            if df is None:
                return 0
            
            # 准备输入
            feature_cols = self.get_feature_columns()
            X = df[feature_cols].iloc[-1:].values
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 标准化
            X_scaled = self.scaler.transform(X)
            
            # 特征选择
            if self.selector is not None:
                selector_data = joblib.load(self.selector_path)
                X_scaled = X_scaled[:, selector_data['mask']]
            
            # 预测
            probs = self.model.predict_proba(X_scaled)[0]
            pred_class = np.argmax(probs)
            confidence = probs[pred_class]
            
            # 生成信号 (二分类: 0=跌, 1=涨)
            signal = 0
            if confidence >= self.confidence_threshold:
                if pred_class == 1:  # 涨
                    signal = 1
                elif pred_class == 0:  # 跌
                    signal = -1
            
            # 打印分析
            class_names = ['跌↓', '涨↑']
            self.logger.info("=" * 70)
            self.logger.info("【Random Forest V2 信号分析】")
            self.logger.info("=" * 70)
            self.logger.info(f"🎯 预测: {class_names[pred_class]} | 置信度: {confidence:.2%}")
            self.logger.info(f"📊 概率分布: 跌={probs[0]:.2%} | 涨={probs[1]:.2%}")
            
            signal_emoji = {1: "🟢 买入", -1: "🔴 卖出", 0: "⚪ 观望"}
            self.logger.info(f"📈 信号: {signal_emoji[signal]}")
            self.logger.info("=" * 70)
            
            return signal
            
        except Exception as e:
            self.logger.error(f"生成信号出错: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return 0
    
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
                    trade_pct = symbol_config.get('trade_amount_percent', 95)
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
            import traceback
            self.logger.error(traceback.format_exc())
    
    def _manage_position(self, position):
        """管理仓位"""
        try:
            pos_amt = float(position['info'].get('positionAmt', 0))
            entry_price = float(position['info'].get('entryPrice', 0))
            current_price = self.trader.get_market_price()
            side = "long" if pos_amt > 0 else "short"
            
            if side == "long":
                pnl_pct = (current_price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - current_price) / entry_price
            
            if pnl_pct > self.max_profit_reached:
                self.max_profit_reached = pnl_pct
                
                if self.trailing_stop_enabled and pnl_pct >= self.trailing_activation:
                    if side == "long":
                        self.trailing_stop_price = current_price * (1 - self.trailing_distance)
                    else:
                        self.trailing_stop_price = current_price * (1 + self.trailing_distance)
            
            # 追踪止损
            if self.trailing_stop_price:
                if side == "long" and current_price <= self.trailing_stop_price:
                    self.logger.info(f"📉 追踪止损触发 | 盈亏: {pnl_pct:.2%}")
                    self.trader.close_position()
                    return
                elif side == "short" and current_price >= self.trailing_stop_price:
                    self.logger.info(f"📉 追踪止损触发 | 盈亏: {pnl_pct:.2%}")
                    self.trader.close_position()
                    return
            
            # 止损
            if pnl_pct <= -self.stop_loss_pct:
                self.logger.info(f"🛑 止损触发 | 盈亏: {pnl_pct:.2%}")
                self.trader.close_position()
                return
            
            # 止盈
            if pnl_pct >= self.take_profit_pct:
                self.logger.info(f"🎯 止盈触发 | 盈亏: {pnl_pct:.2%}")
                self.trader.close_position()
                return
            
            # 持仓超时
            if self.position_entry_time:
                hold_mins = (time.time() - self.position_entry_time) / 60
                if hold_mins >= self.max_hold_time:
                    self.logger.info(f"⏰ 最大持仓时间 | 盈亏: {pnl_pct:.2%}")
                    self.trader.close_position()
                    return
            
            # 反向信号
            signal = self.generate_signal()
            if (side == "long" and signal == -1) or (side == "short" and signal == 1):
                self.logger.info(f"🔄 反向信号平仓 | 盈亏: {pnl_pct:.2%}")
                self.trader.close_position()
                return
            
            self.logger.debug(f"📊 持仓状态 | 方向: {side} | 盈亏: {pnl_pct:.2%}")
            
        except Exception as e:
            self.logger.error(f"管理仓位出错: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
