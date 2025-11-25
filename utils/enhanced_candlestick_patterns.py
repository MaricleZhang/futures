"""
增强版K线形态识别工具库
识别更多经典K线形态并返回形态强度和方向概率

File: utils/enhanced_candlestick_patterns.py
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class PatternDirection(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


@dataclass
class PatternResult:
    """形态识别结果"""
    name: str
    detected: bool
    strength: float  # 0-1
    direction: PatternDirection
    confidence: float  # 0-1
    description: str


class EnhancedCandlestickPattern:
    """增强版K线形态识别类"""
    
    def __init__(self):
        """初始化形态识别器"""
        # 形态权重配置
        self.pattern_weights = {
            # 单根K线形态
            'hammer': 0.75,
            'inverted_hammer': 0.70,
            'hanging_man': 0.75,
            'shooting_star': 0.75,
            'doji': 0.50,
            'dragonfly_doji': 0.65,
            'gravestone_doji': 0.65,
            'spinning_top': 0.40,
            'marubozu': 0.80,
            
            # 双根K线形态
            'engulfing': 0.85,
            'piercing': 0.80,
            'dark_cloud': 0.80,
            'harami': 0.70,
            'tweezer_top': 0.65,
            'tweezer_bottom': 0.65,
            
            # 三根K线形态
            'morning_star': 0.90,
            'evening_star': 0.90,
            'three_white_soldiers': 0.85,
            'three_black_crows': 0.85,
            'three_inside_up': 0.80,
            'three_inside_down': 0.80,
            'three_outside_up': 0.80,
            'three_outside_down': 0.80,
        }
        
    # ==================== 辅助函数 ====================
    
    @staticmethod
    def _body_size(open_price: float, close: float) -> float:
        """计算K线实体大小"""
        return abs(close - open_price)
    
    @staticmethod
    def _upper_shadow(high: float, open_price: float, close: float) -> float:
        """计算上影线长度"""
        return high - max(open_price, close)
    
    @staticmethod
    def _lower_shadow(low: float, open_price: float, close: float) -> float:
        """计算下影线长度"""
        return min(open_price, close) - low
    
    @staticmethod
    def _total_range(high: float, low: float) -> float:
        """计算K线总范围"""
        return high - low
    
    @staticmethod
    def _is_bullish(open_price: float, close: float) -> bool:
        """判断是否为阳线"""
        return close > open_price
    
    @staticmethod
    def _is_bearish(open_price: float, close: float) -> bool:
        """判断是否为阴线"""
        return close < open_price
    
    def _get_trend(self, df: pd.DataFrame, idx: int, lookback: int = 5) -> str:
        """
        判断趋势方向
        返回: 'up', 'down', 'sideways'
        """
        if idx < lookback:
            return 'sideways'
        
        closes = df.iloc[idx-lookback:idx]['close'].values
        
        # 计算趋势
        up_count = sum(1 for i in range(1, len(closes)) if closes[i] > closes[i-1])
        down_count = sum(1 for i in range(1, len(closes)) if closes[i] < closes[i-1])
        
        if up_count >= lookback * 0.7:
            return 'up'
        elif down_count >= lookback * 0.7:
            return 'down'
        return 'sideways'
    
    def _get_avg_body_size(self, df: pd.DataFrame, idx: int, lookback: int = 10) -> float:
        """获取平均实体大小"""
        if idx < lookback:
            return 0
        
        bodies = []
        for i in range(idx - lookback, idx):
            o, c = df.iloc[i][['open', 'close']]
            bodies.append(abs(c - o))
        
        return np.mean(bodies) if bodies else 0
    
    # ==================== 单根K线形态 ====================
    
    def detect_hammer(self, df: pd.DataFrame, idx: int) -> PatternResult:
        """
        检测锤子线（看涨反转）
        特征：小实体在上部，长下影线，几乎无上影线
        """
        if idx < 5:
            return PatternResult('hammer', False, 0, PatternDirection.NEUTRAL, 0, '')
        
        o, h, l, c = df.iloc[idx][['open', 'high', 'low', 'close']]
        
        body = self._body_size(o, c)
        total = self._total_range(h, l)
        lower_shadow = self._lower_shadow(l, o, c)
        upper_shadow = self._upper_shadow(h, o, c)
        
        if total == 0:
            return PatternResult('hammer', False, 0, PatternDirection.NEUTRAL, 0, '')
        
        # 锤子线条件
        is_hammer = (
            lower_shadow >= body * 2 and
            upper_shadow <= body * 0.3 and
            body / total <= 0.35
        )
        
        if not is_hammer:
            return PatternResult('hammer', False, 0, PatternDirection.NEUTRAL, 0, '')
        
        # 检查下跌趋势
        trend = self._get_trend(df, idx)
        
        if trend != 'down':
            return PatternResult('hammer', True, 0.3, PatternDirection.NEUTRAL, 0.3, 
                               '锤子线形成但非下跌趋势')
        
        # 计算强度
        shadow_ratio = min(lower_shadow / body, 4) / 4 if body > 0 else 0.5
        strength = shadow_ratio * self.pattern_weights['hammer']
        confidence = 0.7 + shadow_ratio * 0.2
        
        return PatternResult(
            'hammer', True, strength, PatternDirection.BULLISH, confidence,
            f'锤子线：下影线是实体的{lower_shadow/body:.1f}倍，看涨反转信号'
        )
    
    def detect_shooting_star(self, df: pd.DataFrame, idx: int) -> PatternResult:
        """
        检测射击之星（看跌反转）
        特征：小实体在下部，长上影线，几乎无下影线
        """
        if idx < 5:
            return PatternResult('shooting_star', False, 0, PatternDirection.NEUTRAL, 0, '')
        
        o, h, l, c = df.iloc[idx][['open', 'high', 'low', 'close']]
        
        body = self._body_size(o, c)
        total = self._total_range(h, l)
        lower_shadow = self._lower_shadow(l, o, c)
        upper_shadow = self._upper_shadow(h, o, c)
        
        if total == 0:
            return PatternResult('shooting_star', False, 0, PatternDirection.NEUTRAL, 0, '')
        
        is_shooting_star = (
            upper_shadow >= body * 2 and
            lower_shadow <= body * 0.3 and
            body / total <= 0.35
        )
        
        if not is_shooting_star:
            return PatternResult('shooting_star', False, 0, PatternDirection.NEUTRAL, 0, '')
        
        trend = self._get_trend(df, idx)
        
        if trend != 'up':
            return PatternResult('shooting_star', True, 0.3, PatternDirection.NEUTRAL, 0.3,
                               '射击之星形成但非上升趋势')
        
        shadow_ratio = min(upper_shadow / body, 4) / 4 if body > 0 else 0.5
        strength = shadow_ratio * self.pattern_weights['shooting_star']
        confidence = 0.7 + shadow_ratio * 0.2
        
        return PatternResult(
            'shooting_star', True, strength, PatternDirection.BEARISH, confidence,
            f'射击之星：上影线是实体的{upper_shadow/body:.1f}倍，看跌反转信号'
        )
    
    def detect_doji(self, df: pd.DataFrame, idx: int) -> PatternResult:
        """
        检测十字星（犹豫信号）
        特征：开盘价≈收盘价
        """
        o, h, l, c = df.iloc[idx][['open', 'high', 'low', 'close']]
        
        body = self._body_size(o, c)
        total = self._total_range(h, l)
        
        if total == 0:
            return PatternResult('doji', False, 0, PatternDirection.NEUTRAL, 0, '')
        
        # 十字星：实体非常小
        is_doji = body / total <= 0.1
        
        if not is_doji:
            return PatternResult('doji', False, 0, PatternDirection.NEUTRAL, 0, '')
        
        upper_shadow = self._upper_shadow(h, o, c)
        lower_shadow = self._lower_shadow(l, o, c)
        
        # 判断十字星类型
        if lower_shadow > upper_shadow * 2:
            # 蜻蜓十字星（看涨）
            return PatternResult(
                'dragonfly_doji', True, 
                self.pattern_weights['dragonfly_doji'],
                PatternDirection.BULLISH, 0.65,
                '蜻蜓十字星：长下影线，潜在看涨反转'
            )
        elif upper_shadow > lower_shadow * 2:
            # 墓碑十字星（看跌）
            return PatternResult(
                'gravestone_doji', True,
                self.pattern_weights['gravestone_doji'],
                PatternDirection.BEARISH, 0.65,
                '墓碑十字星：长上影线，潜在看跌反转'
            )
        
        # 普通十字星
        strength = self.pattern_weights['doji']
        return PatternResult(
            'doji', True, strength, PatternDirection.NEUTRAL, 0.5,
            '十字星：市场犹豫，可能变盘'
        )
    
    def detect_marubozu(self, df: pd.DataFrame, idx: int) -> PatternResult:
        """
        检测光头光脚（强趋势延续）
        特征：无影线或影线很短
        """
        o, h, l, c = df.iloc[idx][['open', 'high', 'low', 'close']]
        
        body = self._body_size(o, c)
        total = self._total_range(h, l)
        upper_shadow = self._upper_shadow(h, o, c)
        lower_shadow = self._lower_shadow(l, o, c)
        
        if total == 0 or body == 0:
            return PatternResult('marubozu', False, 0, PatternDirection.NEUTRAL, 0, '')
        
        # 光头光脚：影线很短
        is_marubozu = (
            upper_shadow <= body * 0.05 and
            lower_shadow <= body * 0.05 and
            body / total >= 0.9
        )
        
        if not is_marubozu:
            return PatternResult('marubozu', False, 0, PatternDirection.NEUTRAL, 0, '')
        
        direction = PatternDirection.BULLISH if self._is_bullish(o, c) else PatternDirection.BEARISH
        strength = self.pattern_weights['marubozu']
        
        desc = '阳线光头光脚：强烈看涨' if direction == PatternDirection.BULLISH else '阴线光头光脚：强烈看跌'
        
        return PatternResult('marubozu', True, strength, direction, 0.8, desc)
    
    # ==================== 双根K线形态 ====================
    
    def detect_engulfing(self, df: pd.DataFrame, idx: int) -> PatternResult:
        """
        检测吞没形态（强反转信号）
        看涨吞没：阳线完全吞没前一阴线
        看跌吞没：阴线完全吞没前一阳线
        """
        if idx < 1:
            return PatternResult('engulfing', False, 0, PatternDirection.NEUTRAL, 0, '')
        
        # 当前K线
        o1, h1, l1, c1 = df.iloc[idx][['open', 'high', 'low', 'close']]
        # 前一根K线
        o0, h0, l0, c0 = df.iloc[idx-1][['open', 'high', 'low', 'close']]
        
        current_body = self._body_size(o1, c1)
        prev_body = self._body_size(o0, c0)
        
        # 看涨吞没
        bullish_engulfing = (
            self._is_bearish(o0, c0) and
            self._is_bullish(o1, c1) and
            o1 <= c0 and
            c1 >= o0 and
            current_body > prev_body
        )
        
        # 看跌吞没
        bearish_engulfing = (
            self._is_bullish(o0, c0) and
            self._is_bearish(o1, c1) and
            o1 >= c0 and
            c1 <= o0 and
            current_body > prev_body
        )
        
        if bullish_engulfing:
            engulf_ratio = current_body / prev_body if prev_body > 0 else 1
            strength = min(engulf_ratio / 2, 1) * self.pattern_weights['engulfing']
            return PatternResult(
                'bullish_engulfing', True, strength, PatternDirection.BULLISH, 0.8,
                f'看涨吞没：当前阳线吞没前阴线{engulf_ratio:.1f}倍'
            )
        
        if bearish_engulfing:
            engulf_ratio = current_body / prev_body if prev_body > 0 else 1
            strength = min(engulf_ratio / 2, 1) * self.pattern_weights['engulfing']
            return PatternResult(
                'bearish_engulfing', True, strength, PatternDirection.BEARISH, 0.8,
                f'看跌吞没：当前阴线吞没前阳线{engulf_ratio:.1f}倍'
            )
        
        return PatternResult('engulfing', False, 0, PatternDirection.NEUTRAL, 0, '')
    
    def detect_harami(self, df: pd.DataFrame, idx: int) -> PatternResult:
        """
        检测孕线形态（潜在反转）
        当前K线实体完全在前一根K线实体内
        """
        if idx < 1:
            return PatternResult('harami', False, 0, PatternDirection.NEUTRAL, 0, '')
        
        o1, h1, l1, c1 = df.iloc[idx][['open', 'high', 'low', 'close']]
        o0, h0, l0, c0 = df.iloc[idx-1][['open', 'high', 'low', 'close']]
        
        current_body = self._body_size(o1, c1)
        prev_body = self._body_size(o0, c0)
        
        # 孕线条件：当前实体在前一实体内
        prev_body_high = max(o0, c0)
        prev_body_low = min(o0, c0)
        curr_body_high = max(o1, c1)
        curr_body_low = min(o1, c1)
        
        is_harami = (
            curr_body_high <= prev_body_high and
            curr_body_low >= prev_body_low and
            current_body < prev_body * 0.5
        )
        
        if not is_harami:
            return PatternResult('harami', False, 0, PatternDirection.NEUTRAL, 0, '')
        
        # 判断方向
        if self._is_bearish(o0, c0) and self._is_bullish(o1, c1):
            return PatternResult(
                'bullish_harami', True, self.pattern_weights['harami'],
                PatternDirection.BULLISH, 0.65,
                '看涨孕线：阳线孕于阴线内，潜在反转'
            )
        elif self._is_bullish(o0, c0) and self._is_bearish(o1, c1):
            return PatternResult(
                'bearish_harami', True, self.pattern_weights['harami'],
                PatternDirection.BEARISH, 0.65,
                '看跌孕线：阴线孕于阳线内，潜在反转'
            )
        
        return PatternResult('harami', True, self.pattern_weights['harami'] * 0.5,
                           PatternDirection.NEUTRAL, 0.5, '孕线形态，方向待确认')
    
    # ==================== 三根K线形态 ====================
    
    def detect_morning_star(self, df: pd.DataFrame, idx: int) -> PatternResult:
        """
        检测早晨之星（强看涨反转）
        第一根：大阴线
        第二根：小实体（缺口向下）
        第三根：大阳线（收在第一根实体中部以上）
        """
        if idx < 2:
            return PatternResult('morning_star', False, 0, PatternDirection.NEUTRAL, 0, '')
        
        # 三根K线
        o0, h0, l0, c0 = df.iloc[idx-2][['open', 'high', 'low', 'close']]
        o1, h1, l1, c1 = df.iloc[idx-1][['open', 'high', 'low', 'close']]
        o2, h2, l2, c2 = df.iloc[idx][['open', 'high', 'low', 'close']]
        
        body0 = self._body_size(o0, c0)
        body1 = self._body_size(o1, c1)
        body2 = self._body_size(o2, c2)
        
        avg_body = self._get_avg_body_size(df, idx)
        
        is_morning_star = (
            self._is_bearish(o0, c0) and  # 第一根阴线
            body0 > avg_body * 0.8 and     # 大阴线
            body1 < body0 * 0.3 and        # 小实体
            self._is_bullish(o2, c2) and   # 第三根阳线
            body2 > avg_body * 0.8 and     # 大阳线
            c2 > (o0 + c0) / 2             # 收在第一根中部以上
        )
        
        if not is_morning_star:
            return PatternResult('morning_star', False, 0, PatternDirection.NEUTRAL, 0, '')
        
        # 计算强度
        recovery_ratio = (c2 - c0) / body0 if body0 > 0 else 0
        strength = min(recovery_ratio, 1) * self.pattern_weights['morning_star']
        
        return PatternResult(
            'morning_star', True, strength, PatternDirection.BULLISH, 0.85,
            f'早晨之星：强烈看涨反转，恢复率{recovery_ratio:.1%}'
        )
    
    def detect_evening_star(self, df: pd.DataFrame, idx: int) -> PatternResult:
        """
        检测黄昏之星（强看跌反转）
        """
        if idx < 2:
            return PatternResult('evening_star', False, 0, PatternDirection.NEUTRAL, 0, '')
        
        o0, h0, l0, c0 = df.iloc[idx-2][['open', 'high', 'low', 'close']]
        o1, h1, l1, c1 = df.iloc[idx-1][['open', 'high', 'low', 'close']]
        o2, h2, l2, c2 = df.iloc[idx][['open', 'high', 'low', 'close']]
        
        body0 = self._body_size(o0, c0)
        body1 = self._body_size(o1, c1)
        body2 = self._body_size(o2, c2)
        
        avg_body = self._get_avg_body_size(df, idx)
        
        is_evening_star = (
            self._is_bullish(o0, c0) and
            body0 > avg_body * 0.8 and
            body1 < body0 * 0.3 and
            self._is_bearish(o2, c2) and
            body2 > avg_body * 0.8 and
            c2 < (o0 + c0) / 2
        )
        
        if not is_evening_star:
            return PatternResult('evening_star', False, 0, PatternDirection.NEUTRAL, 0, '')
        
        decline_ratio = (c0 - c2) / body0 if body0 > 0 else 0
        strength = min(decline_ratio, 1) * self.pattern_weights['evening_star']
        
        return PatternResult(
            'evening_star', True, strength, PatternDirection.BEARISH, 0.85,
            f'黄昏之星：强烈看跌反转，下跌率{decline_ratio:.1%}'
        )
    
    def detect_three_white_soldiers(self, df: pd.DataFrame, idx: int) -> PatternResult:
        """检测三白兵（强看涨延续）"""
        if idx < 2:
            return PatternResult('three_white_soldiers', False, 0, PatternDirection.NEUTRAL, 0, '')
        
        candles = df.iloc[idx-2:idx+1][['open', 'high', 'low', 'close']]
        
        # 检查是否都是阳线
        all_bullish = all(
            candles.iloc[i]['close'] > candles.iloc[i]['open'] 
            for i in range(3)
        )
        
        if not all_bullish:
            return PatternResult('three_white_soldiers', False, 0, PatternDirection.NEUTRAL, 0, '')
        
        # 检查逐步上升
        closes = candles['close'].values
        opens = candles['open'].values
        ascending = closes[0] < closes[1] < closes[2]
        
        # 每根开盘价在前一根实体内
        proper_opens = (
            opens[1] > opens[0] and opens[1] < closes[0] and
            opens[2] > opens[1] and opens[2] < closes[1]
        )
        
        if not (ascending and proper_opens):
            return PatternResult('three_white_soldiers', False, 0, PatternDirection.NEUTRAL, 0, '')
        
        # 计算强度
        bodies = [abs(candles.iloc[i]['close'] - candles.iloc[i]['open']) for i in range(3)]
        consistency = 1 - np.std(bodies) / np.mean(bodies) if np.mean(bodies) > 0 else 0
        strength = consistency * self.pattern_weights['three_white_soldiers']
        
        return PatternResult(
            'three_white_soldiers', True, strength, PatternDirection.BULLISH, 0.8,
            '三白兵：连续三根上涨阳线，强看涨'
        )
    
    def detect_three_black_crows(self, df: pd.DataFrame, idx: int) -> PatternResult:
        """检测三黑鸦（强看跌延续）"""
        if idx < 2:
            return PatternResult('three_black_crows', False, 0, PatternDirection.NEUTRAL, 0, '')
        
        candles = df.iloc[idx-2:idx+1][['open', 'high', 'low', 'close']]
        
        all_bearish = all(
            candles.iloc[i]['close'] < candles.iloc[i]['open']
            for i in range(3)
        )
        
        if not all_bearish:
            return PatternResult('three_black_crows', False, 0, PatternDirection.NEUTRAL, 0, '')
        
        closes = candles['close'].values
        opens = candles['open'].values
        descending = closes[0] > closes[1] > closes[2]
        
        proper_opens = (
            opens[1] < opens[0] and opens[1] > closes[0] and
            opens[2] < opens[1] and opens[2] > closes[1]
        )
        
        if not (descending and proper_opens):
            return PatternResult('three_black_crows', False, 0, PatternDirection.NEUTRAL, 0, '')
        
        bodies = [abs(candles.iloc[i]['close'] - candles.iloc[i]['open']) for i in range(3)]
        consistency = 1 - np.std(bodies) / np.mean(bodies) if np.mean(bodies) > 0 else 0
        strength = consistency * self.pattern_weights['three_black_crows']
        
        return PatternResult(
            'three_black_crows', True, strength, PatternDirection.BEARISH, 0.8,
            '三黑鸦：连续三根下跌阴线，强看跌'
        )
    
    # ==================== 综合检测 ====================
    
    def detect_all_patterns(self, df: pd.DataFrame, idx: int = -1) -> Dict[str, PatternResult]:
        """
        检测所有K线形态
        
        Returns:
            字典，包含所有形态的检测结果
        """
        if idx == -1:
            idx = len(df) - 1
        
        results = {}
        
        # 单根形态
        results['hammer'] = self.detect_hammer(df, idx)
        results['shooting_star'] = self.detect_shooting_star(df, idx)
        results['doji'] = self.detect_doji(df, idx)
        results['marubozu'] = self.detect_marubozu(df, idx)
        
        # 双根形态
        results['engulfing'] = self.detect_engulfing(df, idx)
        results['harami'] = self.detect_harami(df, idx)
        
        # 三根形态
        results['morning_star'] = self.detect_morning_star(df, idx)
        results['evening_star'] = self.detect_evening_star(df, idx)
        results['three_white_soldiers'] = self.detect_three_white_soldiers(df, idx)
        results['three_black_crows'] = self.detect_three_black_crows(df, idx)
        
        return results
    
    def get_pattern_summary(self, patterns: Dict[str, PatternResult]) -> Dict:
        """
        汇总形态检测结果
        
        Returns:
            {
                'bullish_patterns': [(name, strength, confidence), ...],
                'bearish_patterns': [(name, strength, confidence), ...],
                'neutral_patterns': [(name, strength, confidence), ...],
                'total_bullish_score': float,
                'total_bearish_score': float,
                'dominant_direction': str,
                'strongest_pattern': PatternResult
            }
        """
        bullish = []
        bearish = []
        neutral = []
        strongest = None
        max_strength = 0
        
        for name, result in patterns.items():
            if result.detected:
                item = (result.name, result.strength, result.confidence, result.description)
                
                if result.direction == PatternDirection.BULLISH:
                    bullish.append(item)
                elif result.direction == PatternDirection.BEARISH:
                    bearish.append(item)
                else:
                    neutral.append(item)
                
                if result.strength > max_strength:
                    max_strength = result.strength
                    strongest = result
        
        # 按强度排序
        bullish.sort(key=lambda x: x[1], reverse=True)
        bearish.sort(key=lambda x: x[1], reverse=True)
        
        total_bullish = sum(s for _, s, _, _ in bullish)
        total_bearish = sum(s for _, s, _, _ in bearish)
        
        # 确定主导方向
        if total_bullish > total_bearish * 1.2:
            dominant = 'bullish'
        elif total_bearish > total_bullish * 1.2:
            dominant = 'bearish'
        else:
            dominant = 'neutral'
        
        return {
            'bullish_patterns': bullish,
            'bearish_patterns': bearish,
            'neutral_patterns': neutral,
            'total_bullish_score': total_bullish,
            'total_bearish_score': total_bearish,
            'dominant_direction': dominant,
            'strongest_pattern': strongest
        }
    
    def calculate_direction_probability(self, df: pd.DataFrame, idx: int = -1) -> Dict[str, float]:
        """
        计算交易方向概率
        
        Returns:
            {
                'long_prob': float,  # 做多概率
                'short_prob': float, # 做空概率
                'hold_prob': float,  # 观望概率
                'confidence': float  # 整体置信度
            }
        """
        patterns = self.detect_all_patterns(df, idx)
        summary = self.get_pattern_summary(patterns)
        
        bullish_score = summary['total_bullish_score']
        bearish_score = summary['total_bearish_score']
        
        # 基础观望概率
        base_hold = 0.3
        
        # 计算原始概率
        total_score = bullish_score + bearish_score + base_hold
        
        if total_score == 0:
            return {
                'long_prob': 0.0,
                'short_prob': 0.0,
                'hold_prob': 1.0,
                'confidence': 0.0
            }
        
        long_prob = bullish_score / total_score
        short_prob = bearish_score / total_score
        hold_prob = base_hold / total_score
        
        # 计算置信度（基于最强形态）
        confidence = 0.0
        if summary['strongest_pattern']:
            confidence = summary['strongest_pattern'].confidence
        
        return {
            'long_prob': float(long_prob),
            'short_prob': float(short_prob),
            'hold_prob': float(hold_prob),
            'confidence': float(confidence)
        }
