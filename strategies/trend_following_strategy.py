"""
趋势跟踪交易策略
结合 Supertrend、多EMA系统、ADX趋势强度过滤

File: strategies/trend_following_strategy.py
"""
import numpy as np
import pandas as pd
import pandas_ta_classic as ta
from datetime import datetime
import time
import logging
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from strategies.base_strategy import BaseStrategy


@dataclass
class TrendSignal:
    """趋势信号数据类"""
    signal: int  # 1=做多, -1=做空, 0=观望
    trend_direction: str  # 'up', 'down', 'sideways'
    trend_strength: str  # 'strong', 'moderate', 'weak'
    confidence: float
    supertrend_signal: int
    ema_signal: int
    adx_value: float
    reasoning: str


class TrendFollowingStrategy(BaseStrategy):
    """
    趋势跟踪策略
    
    核心逻辑：
    1. Supertrend 确定趋势方向
    2. 三重EMA系统确认趋势
    3. ADX过滤弱趋势市场
    4. ATR动态止损止盈
    5. 成交量确认入场
    
    入场条件（做多）：
    - Supertrend翻多
    - 价格在EMA21上方
    - EMA8 > EMA21 > EMA55
    - ADX > 20（有趋势）
    - 成交量放大（可选）
    
    出场条件：
    - Supertrend翻空
    - 价格跌破EMA21
    - ATR追踪止损触发
    """
    
    def __init__(self, trader, interval='15m'):
        """初始化策略"""
        super().__init__(trader)
        self.logger = self.get_logger()
        
        # ==================== 时间配置 ====================
        self.kline_interval = '15m'
        self.check_interval = 300  # 5分钟检查
        self.lookback_period = 150
        self.training_lookback = 150
        
        # ==================== Supertrend 参数 ====================
        self.supertrend_period = 10
        self.supertrend_multiplier = 3.0
        
        # ==================== EMA 参数 ====================
        self.ema_fast = 8
        self.ema_mid = 21
        self.ema_slow = 55
        
        # ==================== ADX 参数 ====================
        self.adx_period = 14
        self.adx_threshold = 20  # 最小趋势强度
        self.adx_strong_threshold = 30  # 强趋势
        
        # ==================== ATR 参数 ====================
        self.atr_period = 14
        self.atr_stop_multiplier = 2.0  # 止损倍数
        self.atr_profit_multiplier = 3.0  # 止盈倍数
        
        # ==================== 成交量参数 ====================
        self.volume_ma_period = 20
        self.volume_threshold = 1.2  # 放量阈值
        
        # ==================== 信号确认参数 ====================
        self.require_volume_confirm = False  # 是否需要成交量确认
        self.min_ema_separation = 0.001  # EMA最小间距（0.1%）
        
        # ==================== 仓位管理 ====================
        self.stop_loss_pct = 0.025  # 2.5% 止损
        self.take_profit_pct = 0.075  # 7.5% 止盈
        self.max_hold_time = 1440  # 最大持仓24小时（分钟）
        
        # 追踪止损
        self.trailing_stop_enabled = True
        self.trailing_activation = 0.03  # 3%激活
        self.trailing_distance = 0.015  # 1.5%距离
        
        # ==================== 状态追踪 ====================
        self.position_entry_time = None
        self.position_entry_price = None
        self.max_profit_reached = 0
        self.trailing_stop_price = None
        self.last_supertrend_direction = 0
        self.last_signal: Optional[TrendSignal] = None
        
        self.logger.info("=" * 60)
        self.logger.info("趋势跟踪策略初始化完成")
        self.logger.info(f"时间周期: {self.kline_interval}")
        self.logger.info(f"Supertrend: 周期={self.supertrend_period}, 乘数={self.supertrend_multiplier}")
        self.logger.info(f"EMA系统: {self.ema_fast}/{self.ema_mid}/{self.ema_slow}")
        self.logger.info(f"ADX阈值: 最小={self.adx_threshold}, 强势={self.adx_strong_threshold}")
        self.logger.info("=" * 60)
    
    # ==================== Supertrend 计算 ====================
    
    def calculate_supertrend(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算 Supertrend 指标
        
        Returns:
            DataFrame with supertrend columns
        """
        try:
            high = df['high']
            low = df['low']
            close = df['close']
            
            # 计算ATR
            atr = ta.atr(high, low, close, length=self.supertrend_period)
            
            # 计算基础线
            hl2 = (high + low) / 2
            
            # 上轨和下轨
            upper_band = hl2 + (self.supertrend_multiplier * atr)
            lower_band = hl2 - (self.supertrend_multiplier * atr)
            
            # 初始化Supertrend
            supertrend = pd.Series(index=df.index, dtype=float)
            direction = pd.Series(index=df.index, dtype=int)
            
            # 第一个值
            supertrend.iloc[0] = upper_band.iloc[0]
            direction.iloc[0] = -1
            
            for i in range(1, len(df)):
                # 调整上下轨
                if lower_band.iloc[i] > lower_band.iloc[i-1] or close.iloc[i-1] < lower_band.iloc[i-1]:
                    lower_band.iloc[i] = lower_band.iloc[i]
                else:
                    lower_band.iloc[i] = lower_band.iloc[i-1]
                
                if upper_band.iloc[i] < upper_band.iloc[i-1] or close.iloc[i-1] > upper_band.iloc[i-1]:
                    upper_band.iloc[i] = upper_band.iloc[i]
                else:
                    upper_band.iloc[i] = upper_band.iloc[i-1]
                
                # 确定方向
                if supertrend.iloc[i-1] == upper_band.iloc[i-1]:
                    if close.iloc[i] > upper_band.iloc[i]:
                        supertrend.iloc[i] = lower_band.iloc[i]
                        direction.iloc[i] = 1  # 多头
                    else:
                        supertrend.iloc[i] = upper_band.iloc[i]
                        direction.iloc[i] = -1  # 空头
                else:
                    if close.iloc[i] < lower_band.iloc[i]:
                        supertrend.iloc[i] = upper_band.iloc[i]
                        direction.iloc[i] = -1  # 空头
                    else:
                        supertrend.iloc[i] = lower_band.iloc[i]
                        direction.iloc[i] = 1  # 多头
            
            df['supertrend'] = supertrend
            df['supertrend_direction'] = direction
            df['supertrend_upper'] = upper_band
            df['supertrend_lower'] = lower_band
            
            return df
            
        except Exception as e:
            self.logger.error(f"计算Supertrend出错: {str(e)}")
            return df
    
    # ==================== 技术指标计算 ====================
    
    def calculate_indicators(self, df: pd.DataFrame) -> Optional[Dict]:
        """计算所有技术指标"""
        try:
            high = df['high']
            low = df['low']
            close = df['close']
            volume = df['volume']
            
            indicators = {}
            
            # ---------- Supertrend ----------
            df = self.calculate_supertrend(df)
            indicators['supertrend'] = df['supertrend']
            indicators['supertrend_direction'] = df['supertrend_direction']
            
            # ---------- EMA系统 ----------
            indicators['ema_fast'] = ta.ema(close, length=self.ema_fast)
            indicators['ema_mid'] = ta.ema(close, length=self.ema_mid)
            indicators['ema_slow'] = ta.ema(close, length=self.ema_slow)
            
            # ---------- ADX + DI ----------
            adx_df = ta.adx(high, low, close, length=self.adx_period)
            indicators['adx'] = adx_df[f'ADX_{self.adx_period}']
            indicators['plus_di'] = adx_df[f'DMP_{self.adx_period}']
            indicators['minus_di'] = adx_df[f'DMN_{self.adx_period}']
            
            # ---------- ATR ----------
            indicators['atr'] = ta.atr(high, low, close, length=self.atr_period)
            
            # ---------- 成交量 ----------
            indicators['volume'] = volume
            indicators['volume_ma'] = ta.sma(volume, length=self.volume_ma_period)
            
            # ---------- 价格数据 ----------
            indicators['close'] = close
            indicators['high'] = high
            indicators['low'] = low
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"计算指标出错: {str(e)}")
            return None
    
    # ==================== 信号分析 ====================
    
    def analyze_supertrend(self, indicators: Dict) -> Tuple[int, str]:
        """
        分析Supertrend信号
        
        Returns:
            (信号, 描述)
        """
        try:
            direction = indicators['supertrend_direction'].iloc[-1]
            prev_direction = indicators['supertrend_direction'].iloc[-2]
            supertrend = indicators['supertrend'].iloc[-1]
            close = indicators['close'].iloc[-1]
            
            signal = 0
            desc = ""
            
            # 检测方向变化
            if direction == 1 and prev_direction == -1:
                signal = 1
                desc = f"Supertrend翻多 (ST={supertrend:.4f})"
            elif direction == -1 and prev_direction == 1:
                signal = -1
                desc = f"Supertrend翻空 (ST={supertrend:.4f})"
            elif direction == 1:
                signal = 1
                desc = f"Supertrend多头持续 (价格在ST上方)"
            elif direction == -1:
                signal = -1
                desc = f"Supertrend空头持续 (价格在ST下方)"
            
            return signal, desc
            
        except Exception as e:
            self.logger.error(f"分析Supertrend出错: {str(e)}")
            return 0, "分析出错"
    
    def analyze_ema_system(self, indicators: Dict) -> Tuple[int, float, str]:
        """
        分析EMA系统
        
        Returns:
            (信号, 强度, 描述)
        """
        try:
            close = indicators['close'].iloc[-1]
            ema_fast = indicators['ema_fast'].iloc[-1]
            ema_mid = indicators['ema_mid'].iloc[-1]
            ema_slow = indicators['ema_slow'].iloc[-1]
            
            signal = 0
            strength = 0.0
            reasons = []
            
            # 检查EMA排列
            if ema_fast > ema_mid > ema_slow:
                signal = 1
                strength += 0.4
                reasons.append("EMA多头排列")
            elif ema_fast < ema_mid < ema_slow:
                signal = -1
                strength += 0.4
                reasons.append("EMA空头排列")
            
            # 检查价格与EMA关系
            if close > ema_mid:
                if signal >= 0:
                    signal = max(signal, 1)
                    strength += 0.3
                reasons.append("价格在EMA21上方")
            elif close < ema_mid:
                if signal <= 0:
                    signal = min(signal, -1)
                    strength += 0.3
                reasons.append("价格在EMA21下方")
            
            # 检查EMA间距
            if ema_slow > 0:
                fast_mid_sep = abs(ema_fast - ema_mid) / ema_slow
                mid_slow_sep = abs(ema_mid - ema_slow) / ema_slow
                
                if fast_mid_sep > self.min_ema_separation:
                    strength += 0.15
                if mid_slow_sep > self.min_ema_separation:
                    strength += 0.15
            
            desc = " | ".join(reasons) if reasons else "EMA无明显信号"
            
            return signal, min(strength, 1.0), desc
            
        except Exception as e:
            self.logger.error(f"分析EMA系统出错: {str(e)}")
            return 0, 0.0, "分析出错"
    
    def analyze_adx(self, indicators: Dict) -> Tuple[str, float, str]:
        """
        分析ADX趋势强度
        
        Returns:
            (趋势强度级别, ADX值, 描述)
        """
        try:
            adx = indicators['adx'].iloc[-1]
            plus_di = indicators['plus_di'].iloc[-1]
            minus_di = indicators['minus_di'].iloc[-1]
            
            # 判断趋势强度
            if adx >= self.adx_strong_threshold:
                strength = 'strong'
                desc = f"强趋势 ADX={adx:.1f}"
            elif adx >= self.adx_threshold:
                strength = 'moderate'
                desc = f"中等趋势 ADX={adx:.1f}"
            else:
                strength = 'weak'
                desc = f"弱趋势/震荡 ADX={adx:.1f}"
            
            # DI方向
            if plus_di > minus_di:
                desc += f" (+DI>{'-'}DI)"
            else:
                desc += f" ({'-'}DI>+DI)"
            
            return strength, adx, desc
            
        except Exception as e:
            self.logger.error(f"分析ADX出错: {str(e)}")
            return 'weak', 0.0, "分析出错"
    
    def analyze_volume(self, indicators: Dict) -> Tuple[bool, float, str]:
        """
        分析成交量
        
        Returns:
            (是否放量, 量比, 描述)
        """
        try:
            volume = indicators['volume'].iloc[-1]
            volume_ma = indicators['volume_ma'].iloc[-1]
            
            if volume_ma > 0:
                volume_ratio = volume / volume_ma
            else:
                volume_ratio = 1.0
            
            is_high_volume = volume_ratio >= self.volume_threshold
            
            if volume_ratio >= 1.5:
                desc = f"明显放量 (量比={volume_ratio:.2f})"
            elif volume_ratio >= self.volume_threshold:
                desc = f"温和放量 (量比={volume_ratio:.2f})"
            elif volume_ratio < 0.7:
                desc = f"缩量 (量比={volume_ratio:.2f})"
            else:
                desc = f"成交量正常 (量比={volume_ratio:.2f})"
            
            return is_high_volume, volume_ratio, desc
            
        except Exception as e:
            self.logger.error(f"分析成交量出错: {str(e)}")
            return False, 1.0, "分析出错"
    
    # ==================== 综合信号生成 ====================
    
    def generate_trend_signal(self, df: pd.DataFrame) -> TrendSignal:
        """生成趋势交易信号"""
        try:
            # 计算指标
            indicators = self.calculate_indicators(df)
            if indicators is None:
                return self._default_signal()
            
            # 分析各维度
            st_signal, st_desc = self.analyze_supertrend(indicators)
            ema_signal, ema_strength, ema_desc = self.analyze_ema_system(indicators)
            trend_strength, adx_value, adx_desc = self.analyze_adx(indicators)
            is_high_vol, vol_ratio, vol_desc = self.analyze_volume(indicators)
            
            # ---------- 综合判断 ----------
            signal = 0
            confidence = 0.0
            reasons = []
            
            # 核心条件：Supertrend方向
            if st_signal != 0:
                reasons.append(st_desc)
            
            # 条件1：Supertrend和EMA同向
            if st_signal == ema_signal and st_signal != 0:
                signal = st_signal
                confidence = 0.5 + ema_strength * 0.3
                reasons.append(ema_desc)
            
            # 条件2：ADX确认趋势存在
            if trend_strength == 'weak':
                # 弱趋势，降低信号强度或观望
                if signal != 0:
                    confidence *= 0.5
                    reasons.append(f"⚠️ {adx_desc}，信号减弱")
                else:
                    reasons.append(adx_desc)
            else:
                if signal != 0:
                    confidence += 0.1 if trend_strength == 'moderate' else 0.2
                reasons.append(adx_desc)
            
            # 条件3：成交量确认（可选）
            if self.require_volume_confirm:
                if is_high_vol and signal != 0:
                    confidence += 0.1
                    reasons.append(vol_desc)
                elif not is_high_vol and signal != 0:
                    confidence *= 0.8
                    reasons.append(f"⚠️ 成交量不足: {vol_desc}")
            else:
                reasons.append(vol_desc)
            
            # 最终过滤
            if trend_strength == 'weak' and adx_value < 15:
                signal = 0
                reasons.append("❌ ADX过低，不适合趋势交易")
            
            # 确定趋势方向
            if st_signal == 1:
                trend_direction = 'up'
            elif st_signal == -1:
                trend_direction = 'down'
            else:
                trend_direction = 'sideways'
            
            # 构建推理
            reasoning = self._build_reasoning(
                signal, confidence, trend_direction, trend_strength,
                st_desc, ema_desc, adx_desc, vol_desc
            )
            
            result = TrendSignal(
                signal=signal,
                trend_direction=trend_direction,
                trend_strength=trend_strength,
                confidence=float(min(confidence, 1.0)),
                supertrend_signal=st_signal,
                ema_signal=ema_signal,
                adx_value=float(adx_value),
                reasoning=reasoning
            )
            
            self.last_signal = result
            return result
            
        except Exception as e:
            self.logger.error(f"生成趋势信号出错: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return self._default_signal()
    
    def _build_reasoning(
        self,
        signal: int,
        confidence: float,
        trend_direction: str,
        trend_strength: str,
        st_desc: str,
        ema_desc: str,
        adx_desc: str,
        vol_desc: str
    ) -> str:
        """构建推理说明"""
        
        action = {1: "🟢 做多", -1: "🔴 做空", 0: "⚪ 观望"}.get(signal, "观望")
        direction_cn = {'up': '上升', 'down': '下降', 'sideways': '横盘'}.get(trend_direction, '未知')
        strength_cn = {'strong': '强', 'moderate': '中', 'weak': '弱'}.get(trend_strength, '未知')
        
        reasoning = f"""
【交易建议】{action} | 置信度: {confidence:.1%}

【趋势状态】{direction_cn}趋势 | 强度: {strength_cn}

【Supertrend】{st_desc}

【EMA系统】{ema_desc}

【ADX分析】{adx_desc}

【成交量】{vol_desc}
"""
        return reasoning.strip()
    
    def _default_signal(self) -> TrendSignal:
        """返回默认信号"""
        return TrendSignal(
            signal=0,
            trend_direction='sideways',
            trend_strength='weak',
            confidence=0.0,
            supertrend_signal=0,
            ema_signal=0,
            adx_value=0.0,
            reasoning="数据不足或计算出错，建议观望"
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
            
            if not klines or len(klines) < 50:
                self.logger.warning(f"K线数据不足: {len(klines) if klines else 0}")
                return 0
            
            df = pd.DataFrame(
                klines,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            signal = self.generate_trend_signal(df)
            self._print_analysis_report(signal)
            
            return signal.signal
            
        except Exception as e:
            self.logger.error(f"生成信号出错: {str(e)}")
            return 0
    
    def _print_analysis_report(self, signal: TrendSignal):
        """打印分析报告"""
        self.logger.info("=" * 70)
        self.logger.info("【趋势跟踪策略分析报告】")
        self.logger.info("=" * 70)
        
        # 趋势状态
        direction_icon = {'up': '📈', 'down': '📉', 'sideways': '➡️'}
        self.logger.info(f"{direction_icon.get(signal.trend_direction, '❓')} 趋势方向: {signal.trend_direction}")
        self.logger.info(f"💪 趋势强度: {signal.trend_strength} (ADX={signal.adx_value:.1f})")
        self.logger.info(f"📊 置信度: {signal.confidence:.1%}")
        
        # 指标信号
        self.logger.info(f"🔹 Supertrend: {signal.supertrend_signal}")
        self.logger.info(f"🔹 EMA系统: {signal.ema_signal}")
        
        # 交易信号
        signal_text = {1: "🟢 做多", -1: "🔴 做空", 0: "⚪ 观望"}
        self.logger.info(f"🎯 交易信号: {signal_text.get(signal.signal, '未知')}")
        
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
            
            # 检查Supertrend反转
            signal = self.generate_signal()
            if (side == "long" and signal == -1) or (side == "short" and signal == 1):
                self.logger.info(f"🔄 趋势反转 | 盈亏: {pnl_pct:.2%}")
                self.trader.close_position()
                return
            
            self.logger.debug(f"📊 持仓状态 | 方向: {side} | 盈亏: {pnl_pct:.2%}")
            
        except Exception as e:
            self.logger.error(f"管理仓位出错: {str(e)}")
