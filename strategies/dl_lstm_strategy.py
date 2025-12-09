"""
深度学习LSTM交易策略
基于LSTM模型预测交易信号

策略逻辑:
1. 使用LSTM模型分析K线特征
2. 预测未来价格走势(涨/跌/持)
3. 结合置信度过滤低确信号
4. 动态止损止盈

File: strategies/dl_lstm_strategy.py
"""
import numpy as np
import pandas as pd
import torch
import time
import logging
from pathlib import Path

from strategies.base_strategy import BaseStrategy
from strategies.lstm_features import LSTMFeatureExtractor
from strategies.lstm_model import LSTMClassifier
import config


class DLLSTMStrategy(BaseStrategy):
    """深度学习LSTM交易策略"""
    
    def __init__(self, trader, interval='15m', symbol=None):
        """初始化策略
        
        Args:
            trader: 交易器实例
            interval: K线周期
            symbol: 交易对(可选，默认从trader获取)
        """
        super().__init__(trader)
        self.logger = self.get_logger()
        
        # ==================== 交易对配置 ====================
        self.symbol = symbol or getattr(trader, 'symbol', 'UNKNOWN')
        self.symbol_key = self.symbol.lower().replace('/', '')
        
        # ==================== 时间配置 ====================
        self.kline_interval = interval
        
        # 根据interval设置检查频率
        interval_to_check = {
            '1m': 60, '5m': 300, '15m': 300,
            '30m': 600, '1h': 900, '4h': 3600,
        }
        self.check_interval = interval_to_check.get(interval, 300)
        
        # ==================== 模型配置 ====================
        dl_config = config.DL_STRATEGY_CONFIG
        self.sequence_length = dl_config.get('sequence_length', 60)
        self.hidden_size = dl_config.get('hidden_size', 128)
        self.num_layers = dl_config.get('num_layers', 2)
        self.input_features = dl_config.get('input_features', 18)
        self.output_classes = dl_config.get('output_classes', 3)
        self.confidence_threshold = dl_config.get('confidence_threshold', 0.50)
        
        # 根据交易对自动选择模型路径
        base_dir = dl_config.get('models_base_dir', 'strategies/models')
        symbol_model_dir = f"{base_dir}/{self.symbol_key}"
        
        self.model_path = f"{symbol_model_dir}/lstm_model.pth"
        self.scaler_path = f"{symbol_model_dir}/lstm_scaler.npz"
        
        # 默认模型路径(回退用)
        self.default_model_path = dl_config.get('default_model_path', 'strategies/models/lstm_model.pth')
        self.default_scaler_path = dl_config.get('default_scaler_path', 'strategies/models/lstm_scaler.npz')
        
        # 温度缩放参数 (用于校准置信度)
        self.temperature = dl_config.get('temperature', 2.0)  # 温度越高，概率越平滑
        
        # 回测需要的属性
        self.lookback_period = self.sequence_length + 50  # 确保有足够数据计算特征
        self.training_lookback = self.lookback_period
        
        # ==================== 特征提取器 ====================
        self.feature_extractor = LSTMFeatureExtractor(sequence_length=self.sequence_length)
        
        # ==================== 模型和Scaler加载 ====================
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self._load_model()
        
        # ==================== 仓位管理 ====================
        self.stop_loss_pct = 0.02        # 2%止损
        self.take_profit_pct = 0.06      # 6%止盈
        self.max_hold_time = 1440         # 1天
        
        # 动态止损止盈
        self.use_dynamic_stops = True
        self.atr_multiplier_sl = 1.5
        self.atr_multiplier_tp = 3.0
        
        # 追踪止损
        self.trailing_stop_enabled = True
        self.trailing_activation = 0.02   # 2%激活
        self.trailing_distance = 0.01     # 1%距离
        
        # ==================== 状态追踪 ====================
        self.position_entry_time = None
        self.position_entry_price = None
        self.max_profit_reached = 0
        self.trailing_stop_price = None
        self.current_stop_loss_pct = self.stop_loss_pct
        self.current_take_profit_pct = self.take_profit_pct
        self.last_signal = 0
        self.last_signal_time = None
        
        self.logger.info("=" * 70)
        self.logger.info("🧠 Deep Learning LSTM Strategy 初始化完成")
        self.logger.info(f"交易对: {self.symbol} | 模型目录: {self.symbol_key}")
        self.logger.info(f"K线周期: {self.kline_interval} | 检查间隔: {self.check_interval}秒")
        self.logger.info(f"序列长度: {self.sequence_length} | 隐藏层: {self.hidden_size}")
        self.logger.info(f"置信度阈值: {self.confidence_threshold:.0%}")
        self.logger.info(f"设备: {self.device}")
        self.logger.info("=" * 70)
    
    def _load_model(self):
        """加载预训练模型和归一化参数"""
        try:
            # 尝试加载交易对专属模型
            model_path = Path(self.model_path)
            if not model_path.is_absolute():
                model_path = Path(__file__).parent.parent / self.model_path
            
            # 如果交易对专属模型不存在，尝试回退到默认模型
            if not model_path.exists():
                self.logger.warning(f"⚠️ 交易对专属模型不存在: {model_path}")
                model_path = Path(self.default_model_path)
                if not model_path.is_absolute():
                    model_path = Path(__file__).parent.parent / self.default_model_path
                if model_path.exists():
                    self.logger.info(f"📍 回退到默认模型: {model_path}")
            
            if model_path.exists():
                self.model = LSTMClassifier(
                    input_size=self.input_features,
                    hidden_size=self.hidden_size,
                    num_layers=self.num_layers,
                    num_classes=self.output_classes,
                    dropout=0.0  # 推理时不用dropout
                ).to(self.device)
                
                checkpoint = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.eval()
                self.logger.info(f"✅ 模型加载成功: {model_path}")
            else:
                self.logger.warning(f"⚠️ 模型文件不存在: {model_path}")
                self.logger.warning("将使用随机权重，仅用于回测测试")
                # 创建未训练的模型(仅用于回测框架测试)
                self.model = LSTMClassifier(
                    input_size=self.input_features,
                    hidden_size=self.hidden_size,
                    num_layers=self.num_layers,
                    num_classes=self.output_classes,
                    dropout=0.0
                ).to(self.device)
                self.model.eval()
            
            # 尝试加载交易对专属scaler
            scaler_path = Path(self.scaler_path)
            if not scaler_path.is_absolute():
                scaler_path = Path(__file__).parent.parent / self.scaler_path
            
            # 如果交易对专属scaler不存在，尝试回退到默认scaler
            if not scaler_path.exists():
                self.logger.warning(f"⚠️ 交易对专属Scaler不存在: {scaler_path}")
                scaler_path = Path(self.default_scaler_path)
                if not scaler_path.is_absolute():
                    scaler_path = Path(__file__).parent.parent / self.default_scaler_path
                if scaler_path.exists():
                    self.logger.info(f"📍 回退到默认Scaler: {scaler_path}")
            
            if self.feature_extractor.load_scaler(str(scaler_path)):
                self.logger.info(f"✅ Scaler加载成功: {scaler_path}")
            else:
                self.logger.warning(f"⚠️ Scaler文件不存在: {scaler_path}")
                self.logger.warning("将使用逐序列归一化（可能影响预测准确性）")
                
        except Exception as e:
            self.logger.error(f"模型加载失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
    
    def generate_signal(self, klines=None) -> int:
        """生成交易信号
        
        Args:
            klines: K线数据(可选)
            
        Returns:
            信号: 1=买入, -1=卖出, 0=观望
        """
        try:
            # 获取K线数据
            if klines is None:
                klines = self.trader.get_klines(
                    symbol=self.trader.symbol,
                    interval=self.kline_interval,
                    limit=self.lookback_period
                )
            
            if not klines or len(klines) < self.lookback_period - 10:
                self.logger.warning(f"K线数据不足: {len(klines) if klines else 0}")
                return 0
            
            # 转换为DataFrame
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 计算特征
            df = self.feature_extractor.calculate_features(df)
            if df is None:
                return 0
            
            # 准备输入序列
            feature_cols = self.feature_extractor.feature_names
            features = df[feature_cols].iloc[-self.sequence_length:].values
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 使用全局归一化参数进行归一化
            features = self.feature_extractor.transform_single(features)
            
            # 模型预测
            X = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                logits = self.model(X)
                # 使用温度缩放校准置信度
                logits_scaled = logits / self.temperature
                probs = torch.softmax(logits_scaled, dim=1)[0]
            
            probs = probs.cpu().numpy()
            pred_class = np.argmax(probs)
            confidence = probs[pred_class]
            
            # 生成信号
            # 0=跌(卖), 1=持(观望), 2=涨(买)
            signal = 0
            if confidence >= self.confidence_threshold:
                if pred_class == 2:  # 涨
                    signal = 1  # 买入
                elif pred_class == 0:  # 跌
                    signal = -1  # 卖出
            
            # 打印分析结果
            class_names = ['跌↓', '持→', '涨↑']
            self.logger.info("=" * 70)
            self.logger.info("【Deep Learning LSTM 信号分析】")
            self.logger.info("=" * 70)
            self.logger.info(f"🎯 预测: {class_names[pred_class]} | 置信度: {confidence:.2%}")
            self.logger.info(f"📊 概率分布: 跌={probs[0]:.2%} | 持={probs[1]:.2%} | 涨={probs[2]:.2%}")
            
            signal_emoji = {1: "🟢 买入", -1: "🔴 卖出", 0: "⚪ 观望"}
            self.logger.info(f"📈 信号: {signal_emoji[signal]}")
            self.logger.info("=" * 70)
            
            # 更新状态
            if signal != 0:
                self.last_signal = signal
                self.last_signal_time = time.time()
            
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
            
            # 无仓位
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
                        self.logger.info(
                            f"✅ 开多仓 | 数量: {trade_amount:.6f} | 价格: {current_price}"
                        )
                    elif signal == -1:
                        self.trader.open_short(amount=trade_amount)
                        self.logger.info(
                            f"✅ 开空仓 | 数量: {trade_amount:.6f} | 价格: {current_price}"
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
        """管理仓位"""
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
                
                # 追踪止损
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
                self.logger.info(f"🔄 反向信号平仓 | 盈亏: {pnl_pct:.2%}")
                self.trader.close_position()
                return
            
            self.logger.debug(
                f"📊 持仓状态 | 方向: {side} | 盈亏: {pnl_pct:.2%} | "
                f"最大: {self.max_profit_reached:.2%}"
            )
            
        except Exception as e:
            self.logger.error(f"管理仓位出错: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
