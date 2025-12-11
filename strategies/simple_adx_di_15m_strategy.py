"""
Enhanced ADX-DI Strategy for 15-minute timeframe
优化版ADX-DI策略，提高胜率

主要优化：
1. 添加EMA趋势过滤
2. 添加RSI动量确认
3. 添加MACD辅助判断
4. 添加ATR波动率过滤
5. 添加成交量确认
6. 优化ADX阈值和DI差异要求
7. 添加回调入场机制
8. 调整止损止盈比例

File: strategies/enhanced_adx_di_strategy.py
"""
import numpy as np
import pandas as pd
import pandas_ta_classic as ta
from datetime import datetime
import time
import logging
from strategies.base_strategy import BaseStrategy


class SimpleADXDIStrategy15m(BaseStrategy):
    """
    简单版ADX-DI趋势策略
    
    核心改进：
    1. 多重确认机制：ADX + DI + EMA + RSI + MACD + 成交量
    2. 趋势质量评估：只在高质量趋势中交易
    3. 回调入场：等待价格回调到支撑/阻力
    4. 动态止损止盈：根据ATR调整
    5. ADX动量：只在ADX上升时入场
    """
    
    def __init__(self, trader, interval='15m'):
        """初始化策略"""
        super().__init__(trader)
        self.logger = self.get_logger()
        
        # ==================== 时间配置 ====================
        self.kline_interval = interval  # 动态设置，不再硬编码
        
        # 根据 interval 设置合理的检查频率
        interval_to_check = {
            '1m': 60,      # 1分钟检查一次
            '5m': 300,     # 5分钟检查一次  
            '15m': 300,    # 5分钟检查一次
            '30m': 600,    # 10分钟检查一次
            '1h': 900,     # 15分钟检查一次
            '4h': 3600,    # 1小时检查一次
        }
        self.check_interval = interval_to_check.get(interval, 300)  # 默认5分钟
        self.lookback_period = 100        
        # ==================== ADX和DI参数 ====================
        self.adx_period = 14
        self.adx_min_threshold = 18      # 降低到18（允许中等趋势）
        self.adx_strong_threshold = 25   # 强趋势25（降低门槛）
        self.di_diff_threshold = 10      # 降低到10（增加信号）
        
        # ==================== 趋势过滤参数 ====================
        self.ema_fast = 21
        self.ema_slow = 55
        self.ema_trend = 100  # 主趋势EMA
        
        # ==================== 动量指标参数 ====================
        self.rsi_period = 14
        self.rsi_overbought = 70  # RSI超买（扩大健康区域）
        self.rsi_oversold = 30    # RSI超卖（扩大健康区域）
        
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        
        # ==================== 波动率和成交量 ====================
        self.atr_period = 14
        self.atr_multiplier_sl = 1.2   # 止损ATR倍数（收紧）
        self.atr_multiplier_tp = 4.0   # 止盈ATR倍数（扩大）
        
        self.volume_ma_period = 20
        self.volume_threshold = 1.2    # 成交量倍数
        
        # ==================== 仓位管理 ====================
        self.stop_loss_pct = 0.02        # 2%止损（收紧）
        self.take_profit_pct = 0.08      # 8%止盈（扩大盈亏比）
        self.max_hold_time = 720         # 12小时
        
        # 动态止损止盈
        self.use_dynamic_stops = True
        
        # 追踪止损
        self.trailing_stop_enabled = True
        self.trailing_activation = 0.03   # 3%激活
        self.trailing_distance = 0.015    # 1.5%距离
        
        # ==================== 信号质量要求 ====================
        self.min_signal_score = 5  # 最低5分（增加交易频率）
        
        # ==================== 状态追踪 ====================
        self.position_entry_time = None
        self.position_entry_price = None
        self.max_profit_reached = 0
        self.trailing_stop_price = None
        self.last_signal = 0
        self.last_signal_time = None
        
        
        self.logger.info("=" * 70)
        self.logger.info("🚀 Enhanced ADX-DI Strategy 初始化完成")
        self.logger.info(f"K线周期: {self.kline_interval} | 检查间隔: {self.check_interval}秒")
        self.logger.info(f"ADX阈值: {self.adx_min_threshold} (强趋势: {self.adx_strong_threshold})")
        self.logger.info(f"DI差异阈值: {self.di_diff_threshold}")
        self.logger.info(f"EMA周期: {self.ema_fast}/{self.ema_slow}/{self.ema_trend}")
        self.logger.info(f"最低信号分数: {self.min_signal_score}/10")
        self.logger.info("=" * 70)
    
    def calculate_indicators(self, df: pd.DataFrame) -> dict:
        """计算所有技术指标"""
        try:
            high = df['high']
            low = df['low']
            close = df['close']
            volume = df['volume']
            
            indicators = {}
            
            # ---------- ADX和DI ----------
            adx_df = ta.adx(high, low, close, length=self.adx_period)
            indicators['adx'] = adx_df[f'ADX_{self.adx_period}']
            indicators['plus_di'] = adx_df[f'DMP_{self.adx_period}']
            indicators['minus_di'] = adx_df[f'DMN_{self.adx_period}']
            
            # ---------- EMA趋势 ----------
            indicators['ema_fast'] = ta.ema(close, length=self.ema_fast)
            indicators['ema_slow'] = ta.ema(close, length=self.ema_slow)
            indicators['ema_trend'] = ta.ema(close, length=self.ema_trend)
            
            # ---------- RSI ----------
            indicators['rsi'] = ta.rsi(close, length=self.rsi_period)
            
            # ---------- MACD ----------
            macd_df = ta.macd(close, fast=self.macd_fast, slow=self.macd_slow, signal=self.macd_signal)
            indicators['macd'] = macd_df[f'MACD_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}']
            indicators['macd_signal'] = macd_df[f'MACDs_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}']
            indicators['macd_hist'] = macd_df[f'MACDh_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}']
            
            # ---------- ATR ----------
            indicators['atr'] = ta.atr(high, low, close, length=self.atr_period)
            
            # ---------- 成交量 ----------
            indicators['volume'] = volume
            indicators['volume_ma'] = ta.sma(volume, length=self.volume_ma_period)
            
            # ---------- 价格 ----------
            indicators['close'] = close
            indicators['high'] = high
            indicators['low'] = low
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"计算指标出错: {str(e)}")
            return None
    
    def analyze_signal_quality(self, indicators: dict, idx: int = -1) -> dict:
        """
        分析信号质量并打分
        
        评分标准（满分10分）：
        1. ADX强度 (2分)
        2. DI差异 (2分)
        3. EMA趋势 (2分)
        4. RSI位置 (1分)
        5. MACD确认 (1分)
        6. ADX动量 (1分)
        7. 成交量确认 (1分)
        
        Returns:
            {
                'signal': 1/-1/0,
                'score': float,
                'details': str,
                'stop_loss': float,
                'take_profit': float
            }
        """
        try:
            # 获取当前值
            adx = indicators['adx'].iloc[idx]
            adx_prev = indicators['adx'].iloc[idx-1]
            plus_di = indicators['plus_di'].iloc[idx]
            minus_di = indicators['minus_di'].iloc[idx]
            
            close = indicators['close'].iloc[idx]
            ema_fast = indicators['ema_fast'].iloc[idx]
            ema_slow = indicators['ema_slow'].iloc[idx]
            ema_trend = indicators['ema_trend'].iloc[idx]
            
            rsi = indicators['rsi'].iloc[idx]
            macd = indicators['macd'].iloc[idx]
            macd_signal = indicators['macd_signal'].iloc[idx]
            macd_hist = indicators['macd_hist'].iloc[idx]
            macd_hist_prev = indicators['macd_hist'].iloc[idx-1]
            
            atr = indicators['atr'].iloc[idx]
            volume = indicators['volume'].iloc[idx]
            volume_ma = indicators['volume_ma'].iloc[idx]
            
            # 初始化评分
            score = 0
            reasons = []
            direction = 0
            
            # ==================== 1. ADX强度评分 (2分) ====================
            if adx >= self.adx_strong_threshold:
                score += 2
                reasons.append(f"✓ ADX强趋势({adx:.1f}) +2分")
            elif adx >= self.adx_min_threshold:
                score += 1
                reasons.append(f"• ADX中等趋势({adx:.1f}) +1分")
            else:
                reasons.append(f"✗ ADX弱势({adx:.1f})")
                return self._no_signal("ADX低于阈值")
            
            # ==================== 2. DI差异评分 (2分) ====================
            di_diff = abs(plus_di - minus_di)
            if di_diff >= self.di_diff_threshold * 1.5:
                score += 2
                reasons.append(f"✓ DI差异大({di_diff:.1f}) +2分")
            elif di_diff >= self.di_diff_threshold:
                score += 1
                reasons.append(f"• DI差异中等({di_diff:.1f}) +1分")
            else:
                reasons.append(f"✗ DI差异不足({di_diff:.1f})")
                return self._no_signal("DI差异不足")
            
            # 确定基础方向
            if plus_di > minus_di:
                direction = 1  # 做多
                # 做多要求更高的信号质量（历史表现较差）
                required_score_long = 7  # 做多需要7分
            else:
                direction = -1  # 做空
                required_score_long = 5  # 做空需要5分
            
            # ==================== 3. EMA趋势评分 (2分) ====================
            if direction == 1:  # 做多
                if close > ema_fast > ema_slow > ema_trend:
                    score += 2
                    reasons.append(f"✓ EMA完美多头排列 +2分")
                elif close > ema_fast and ema_fast > ema_slow:
                    score += 1
                    reasons.append(f"• EMA短期多头 +1分")
                elif close < ema_trend:
                    reasons.append(f"✗ 价格低于长期趋势")
                    return self._no_signal("价格低于主趋势")
            else:  # 做空
                if close < ema_fast < ema_slow < ema_trend:
                    score += 2
                    reasons.append(f"✓ EMA完美空头排列 +2分")
                elif close < ema_fast and ema_fast < ema_slow:
                    score += 1
                    reasons.append(f"• EMA短期空头 +1分")
                elif close > ema_trend:
                    reasons.append(f"✗ 价格高于长期趋势")
                    return self._no_signal("价格高于主趋势")
            
            # ==================== 4. RSI位置评分 (1分) ====================
            if direction == 1:  # 做多
                if 40 <= rsi <= 60:
                    score += 1
                    reasons.append(f"✓ RSI健康区域({rsi:.1f}) +1分")
                elif rsi > self.rsi_overbought:
                    reasons.append(f"⚠ RSI超买({rsi:.1f})")
            else:  # 做空
                if 40 <= rsi <= 60:
                    score += 1
                    reasons.append(f"✓ RSI健康区域({rsi:.1f}) +1分")
                elif rsi < self.rsi_oversold:
                    reasons.append(f"⚠ RSI超卖({rsi:.1f})")
            
            # ==================== 5. MACD确认评分 (1分) ====================
            if direction == 1:  # 做多
                if macd > macd_signal and macd_hist > 0:
                    score += 1
                    reasons.append(f"✓ MACD金叉 +1分")
            else:  # 做空
                if macd < macd_signal and macd_hist < 0:
                    score += 1
                    reasons.append(f"✓ MACD死叉 +1分")
            
            # ==================== 6. ADX动量评分 (1分) ====================
            if adx > adx_prev:
                score += 1
                reasons.append(f"✓ ADX上升 +1分")
            else:
                reasons.append(f"⚠ ADX下降")
            
            # ==================== 7. 成交量确认评分 (1分) ====================
            vol_ratio = volume / volume_ma if volume_ma > 0 else 1
            if vol_ratio >= self.volume_threshold:
                score += 1
                reasons.append(f"✓ 成交量放大({vol_ratio:.2f}x) +1分")
            else:
                reasons.append(f"• 成交量正常({vol_ratio:.2f}x)")
            
            # ==================== 计算动态止损止盈 ====================
            if self.use_dynamic_stops and atr > 0:
                stop_loss_price = close - atr * self.atr_multiplier_sl if direction == 1 else close + atr * self.atr_multiplier_sl
                take_profit_price = close + atr * self.atr_multiplier_tp if direction == 1 else close - atr * self.atr_multiplier_tp
                
                stop_loss_pct = abs(stop_loss_price - close) / close
                take_profit_pct = abs(take_profit_price - close) / close
            else:
                stop_loss_pct = self.stop_loss_pct
                take_profit_pct = self.take_profit_pct
            
            # ==================== 最终判断 ====================
            final_signal = 0
            if score >= required_score_long:
                final_signal = direction
            elif direction == 1:
                reasons.append(f"✗ 做多信号分数不足({score}/{required_score_long})")
            
            return {
                'signal': final_signal,
                'score': score,
                'max_score': 10,
                'direction_text': '做多' if direction == 1 else '做空',
                'reasons': reasons,
                'stop_loss_pct': stop_loss_pct,
                'take_profit_pct': take_profit_pct,
                'adx': adx,
                'di_diff': di_diff,
                'rsi': rsi,
                'vol_ratio': vol_ratio
            }
            
        except Exception as e:
            self.logger.error(f"分析信号质量出错: {str(e)}")
            return self._no_signal("计算出错")
    
    def _no_signal(self, reason: str) -> dict:
        """返回无信号结果"""
        return {
            'signal': 0,
            'score': 0,
            'max_score': 10,
            'direction_text': '观望',
            'reasons': [reason],
            'stop_loss_pct': self.stop_loss_pct,
            'take_profit_pct': self.take_profit_pct,
            'adx': 0,
            'di_diff': 0,
            'rsi': 0,
            'vol_ratio': 0
        }
    
    def generate_signal(self, klines=None) -> int:
        """生成交易信号"""
        try:
            # 获取K线
            if klines is None:
                klines = self.trader.get_klines(
                    symbol=self.trader.symbol,
                    interval=self.kline_interval,
                    limit=self.lookback_period
                )
            
            if not klines or len(klines) < 50:
                self.logger.warning(f"K线数据不足: {len(klines) if klines else 0}")
                return 0
            
            # 转DataFrame
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 计算指标
            indicators = self.calculate_indicators(df)
            if indicators is None:
                return 0
            
            # 分析信号质量
            analysis = self.analyze_signal_quality(indicators, idx=-2)  # 使用-2避免未完成K线
            
            # 打印分析报告
            self._print_analysis_report(analysis)
            
            # 更新止损止盈
            if analysis['signal'] != 0:
                self.current_stop_loss_pct = analysis['stop_loss_pct']
                self.current_take_profit_pct = analysis['take_profit_pct']
                self.last_signal = analysis['signal']
                self.last_signal_time = time.time()
            
            return analysis['signal']
            
        except Exception as e:
            self.logger.error(f"生成信号出错: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return 0
    
    def _print_analysis_report(self, analysis: dict):
        """打印分析报告"""
        signal_emoji = {1: "🟢 做多", -1: "🔴 做空", 0: "⚪ 观望"}
        
        self.logger.info("=" * 70)
        self.logger.info("【Enhanced ADX-DI Strategy 信号分析】")
        self.logger.info("=" * 70)
        
        # 信号和评分
        self.logger.info(f"🎯 信号: {signal_emoji.get(analysis['signal'], '未知')}")
        self.logger.info(f"📊 评分: {analysis['score']:.1f}/{analysis['max_score']} 分")
        
        if analysis['signal'] != 0:
            self.logger.info(f"📈 方向: {analysis['direction_text']}")
            self.logger.info(f"🛑 止损: {analysis['stop_loss_pct']:.2%}")
            self.logger.info(f"🎯 止盈: {analysis['take_profit_pct']:.2%}")
        
        # 关键指标
        self.logger.info("-" * 70)
        self.logger.info(f"关键指标:")
        self.logger.info(f"  ADX: {analysis['adx']:.1f} | DI差异: {analysis['di_diff']:.1f}")
        self.logger.info(f"  RSI: {analysis['rsi']:.1f} | 成交量倍数: {analysis['vol_ratio']:.2f}x")
        
        # 评分详情
        self.logger.info("-" * 70)
        self.logger.info("评分详情:")
        for reason in analysis['reasons']:
            self.logger.info(f"  {reason}")
        
        self.logger.info("=" * 70)
    
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
                    trade_pct = symbol_config.get('trade_amount_percent', 100)
                    trade_amount = (available * trade_pct / 100) / current_price
                    
                    if signal == 1:
                        self.trader.open_long(amount=trade_amount)
                        self.logger.info(
                            f"✅ 开多仓 | 数量: {trade_amount:.6f} | 价格: {current_price} | "
                            f"止损: {self.current_stop_loss_pct:.2%} | 止盈: {self.current_take_profit_pct:.2%}"
                        )
                    elif signal == -1:
                        self.trader.open_short(amount=trade_amount)
                        self.logger.info(
                            f"✅ 开空仓 | 数量: {trade_amount:.6f} | 价格: {current_price} | "
                            f"止损: {self.current_stop_loss_pct:.2%} | 止盈: {self.current_take_profit_pct:.2%}"
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
            stop_loss = getattr(self, 'current_stop_loss_pct', self.stop_loss_pct)
            if pnl_pct <= -stop_loss:
                self.logger.info(f"🛑 止损触发 | 盈亏: {pnl_pct:.2%}")
                self.trader.close_position()
                return
            
            # 检查止盈
            take_profit = getattr(self, 'current_take_profit_pct', self.take_profit_pct)
            if pnl_pct >= take_profit:
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
            
            # 检查趋势反转
            signal = self.generate_signal()
            if (side == "long" and signal == -1) or (side == "short" and signal == 1):
                self.logger.info(f"🔄 趋势反转 | 盈亏: {pnl_pct:.2%}")
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
