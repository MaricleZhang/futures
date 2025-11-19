"""
K线形态识别工具库
识别经典的K线形态并返回形态强度分数
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional


class CandlestickPattern:
    """K线形态识别类"""
    
    def __init__(self):
        """初始化形态识别器"""
        # 形态权重配置（可在config中覆盖）
        self.pattern_weights = {
            'hammer': 0.8,
            'inverted_hammer': 0.75,
            'hanging_man': 0.8,
            'shooting_star': 0.75,
            'engulfing': 0.9,
            'piercing': 0.85,
            'dark_cloud': 0.85,
            'morning_star': 0.9,
            'evening_star': 0.9,
            'three_white_soldiers': 0.85,
            'three_black_crows': 0.85,
            'doji': 0.5,
            'spinning_top': 0.4,
        }
        
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
    
    def detect_hammer(self, df: pd.DataFrame, idx: int) -> Tuple[bool, float, str]:
        """
        检测锤子线（看涨反转）
        特征：
        - 实体小，位于K线上端
        - 下影线长度至少是实体的2倍
        - 上影线很短或没有
        - 应出现在下跌趋势中
        
        Returns:
            (是否检测到, 强度分数0-1, 信号方向'bullish'/'bearish'/'neutral')
        """
        if idx < 3:  # 需要前3根K线确认趋势
            return False, 0.0, 'neutral'
        
        o, h, l, c = df.iloc[idx][['open', 'high', 'low', 'close']]
        
        body = self._body_size(o, c)
        total = self._total_range(h, l)
        lower_shadow = self._lower_shadow(l, o, c)
        upper_shadow = self._upper_shadow(h, o, c)
        
        # 避免除零
        if total == 0:
            return False, 0.0, 'neutral'
        
        # 锤子线特征检查
        is_hammer = (
            lower_shadow >= body * 2 and  # 下影线至少是实体2倍
            upper_shadow <= body * 0.1 and  # 上影线很短
            body / total <= 0.3  # 实体占比小
        )
        
        if not is_hammer:
            return False, 0.0, 'neutral'
        
        # 检查前趋势（应该在下跌趋势中）
        prev_closes = df.iloc[idx-3:idx]['close'].values
        is_downtrend = all(prev_closes[i] > prev_closes[i+1] for i in range(len(prev_closes)-1))
        
        # 计算强度分数
        strength = 0.0
        if is_downtrend:
            # 下影线越长，强度越大
            shadow_ratio = min(lower_shadow / body, 5) / 5  # 归一化到0-1
            # 实体越小，强度越大
            body_ratio = 1 - min(body / total, 0.5) / 0.5
            strength = (shadow_ratio * 0.6 + body_ratio * 0.4) * self.pattern_weights['hammer']
        
        return True, strength, 'bullish' if is_downtrend else 'neutral'
    
    def detect_inverted_hammer(self, df: pd.DataFrame, idx: int) -> Tuple[bool, float, str]:
        """
        检测倒锤子线（看涨反转）
        特征与锤子线相反，上影线长
        """
        if idx < 3:
            return False, 0.0, 'neutral'
        
        o, h, l, c = df.iloc[idx][['open', 'high', 'low', 'close']]
        
        body = self._body_size(o, c)
        total = self._total_range(h, l)
        lower_shadow = self._lower_shadow(l, o, c)
        upper_shadow = self._upper_shadow(h, o, c)
        
        if total == 0:
            return False, 0.0, 'neutral'
        
        is_inverted_hammer = (
            upper_shadow >= body * 2 and
            lower_shadow <= body * 0.1 and
            body / total <= 0.3
        )
        
        if not is_inverted_hammer:
            return False, 0.0, 'neutral'
        
        # 检查下跌趋势
        prev_closes = df.iloc[idx-3:idx]['close'].values
        is_downtrend = all(prev_closes[i] > prev_closes[i+1] for i in range(len(prev_closes)-1))
        
        strength = 0.0
        if is_downtrend:
            shadow_ratio = min(upper_shadow / body, 5) / 5
            body_ratio = 1 - min(body / total, 0.5) / 0.5
            strength = (shadow_ratio * 0.6 + body_ratio * 0.4) * self.pattern_weights['inverted_hammer']
        
        return True, strength, 'bullish' if is_downtrend else 'neutral'
    
    def detect_shooting_star(self, df: pd.DataFrame, idx: int) -> Tuple[bool, float, str]:
        """
        检测射击之星（看跌反转）
        与倒锤子相似，但出现在上升趋势顶部
        """
        if idx < 3:
            return False, 0.0, 'neutral'
        
        o, h, l, c = df.iloc[idx][['open', 'high', 'low', 'close']]
        
        body = self._body_size(o, c)
        total = self._total_range(h, l)
        lower_shadow = self._lower_shadow(l, o, c)
        upper_shadow = self._upper_shadow(h, o, c)
        
        if total == 0:
            return False, 0.0, 'neutral'
        
        is_shooting_star = (
            upper_shadow >= body * 2 and
            lower_shadow <= body * 0.1 and
            body / total <= 0.3
        )
        
        if not is_shooting_star:
            return False, 0.0, 'neutral'
        
        # 检查上升趋势
        prev_closes = df.iloc[idx-3:idx]['close'].values
        is_uptrend = all(prev_closes[i] < prev_closes[i+1] for i in range(len(prev_closes)-1))
        
        strength = 0.0
        if is_uptrend:
            shadow_ratio = min(upper_shadow / body, 5) / 5
            body_ratio = 1 - min(body / total, 0.5) / 0.5
            strength = (shadow_ratio * 0.6 + body_ratio * 0.4) * self.pattern_weights['shooting_star']
        
        return True, strength, 'bearish' if is_uptrend else 'neutral'
    
    def detect_hanging_man(self, df: pd.DataFrame, idx: int) -> Tuple[bool, float, str]:
        """
        检测上吊线（看跌反转）
        与锤子相似，但出现在上升趋势顶部
        """
        if idx < 3:
            return False, 0.0, 'neutral'
        
        o, h, l, c = df.iloc[idx][['open', 'high', 'low', 'close']]
        
        body = self._body_size(o, c)
        total = self._total_range(h, l)
        lower_shadow = self._lower_shadow(l, o, c)
        upper_shadow = self._upper_shadow(h, o, c)
        
        if total == 0:
            return False, 0.0, 'neutral'
        
        is_hanging_man = (
            lower_shadow >= body * 2 and
            upper_shadow <= body * 0.1 and
            body / total <= 0.3
        )
        
        if not is_hanging_man:
            return False, 0.0, 'neutral'
        
        # 检查上升趋势
        prev_closes = df.iloc[idx-3:idx]['close'].values
        is_uptrend = all(prev_closes[i] < prev_closes[i+1] for i in range(len(prev_closes)-1))
        
        strength = 0.0
        if is_uptrend:
            shadow_ratio = min(lower_shadow / body, 5) / 5
            body_ratio = 1 - min(body / total, 0.5) / 0.5
            strength = (shadow_ratio * 0.6 + body_ratio * 0.4) * self.pattern_weights['hanging_man']
        
        return True, strength, 'bearish' if is_uptrend else 'neutral'
    
    def detect_engulfing(self, df: pd.DataFrame, idx: int) -> Tuple[bool, float, str]:
        """
        检测吞没形态（强反转信号）
        看涨吞没：阳线完全吞没前一根阴线
        看跌吞没：阴线完全吞没前一根阳线
        """
        if idx < 1:
            return False, 0.0, 'neutral'
        
        # 当前K线
        o1, h1, l1, c1 = df.iloc[idx][['open', 'high', 'low', 'close']]
        # 前一根K线
        o0, h0, l0, c0 = df.iloc[idx-1][['open', 'high', 'low', 'close']]
        
        # 看涨吞没：当前阳线吞没前一阴线
        bullish_engulfing = (
            c0 < o0 and  # 前一根是阴线
            c1 > o1 and  # 当前是阳线
            o1 <= c0 and  # 当前开盘价低于或等于前收盘价
            c1 >= o0  # 当前收盘价高于或等于前开盘价
        )
        
        # 看跌吞没：当前阴线吞没前一阳线
        bearish_engulfing = (
            c0 > o0 and  # 前一根是阳线
            c1 < o1 and  # 当前是阴线
            o1 >= c0 and  # 当前开盘价高于或等于前收盘价
            c1 <= o0  # 当前收盘价低于或等于前开盘价
        )
        
        if bullish_engulfing:
            # 吞没程度（当前实体相对前实体的大小）
            current_body = abs(c1 - o1)
            prev_body = abs(c0 - o0)
            engulf_ratio = min(current_body / prev_body if prev_body > 0 else 1, 2) / 2
            strength = engulf_ratio * self.pattern_weights['engulfing']
            return True, strength, 'bullish'
        
        elif bearish_engulfing:
            current_body = abs(c1 - o1)
            prev_body = abs(c0 - o0)
            engulf_ratio = min(current_body / prev_body if prev_body > 0 else 1, 2) / 2
            strength = engulf_ratio * self.pattern_weights['engulfing']
            return True, strength, 'bearish'
        
        return False, 0.0, 'neutral'
    
    def detect_doji(self, df: pd.DataFrame, idx: int) -> Tuple[bool, float, str]:
        """
        检测十字星（犹豫信号）
        特征：开盘价和收盘价非常接近，上下影线较长
        """
        o, h, l, c = df.iloc[idx][['open', 'high', 'low', 'close']]
        
        body = self._body_size(o, c)
        total = self._total_range(h, l)
        
        if total == 0:
            return False, 0.0, 'neutral'
        
        # 十字星：实体非常小（占总范围的5%以内）
        is_doji = body / total <= 0.05
        
        if not is_doji:
            return False, 0.0, 'neutral'
        
        # 十字星强度基于影线长度
        upper_shadow = self._upper_shadow(h, o, c)
        lower_shadow = self._lower_shadow(l, o, c)
        shadow_balance = 1 - abs(upper_shadow - lower_shadow) / total  # 影线越平衡强度越高
        
        strength = shadow_balance * self.pattern_weights['doji']
        
        return True, strength, 'neutral'
    
    def detect_three_white_soldiers(self, df: pd.DataFrame, idx: int) -> Tuple[bool, float, str]:
        """
        检测三白兵（强看涨信号）
        三根连续的阳线，每根都有实体，逐步上升
        """
        if idx < 2:
            return False, 0.0, 'neutral'
        
        # 获取三根K线
        candles = df.iloc[idx-2:idx+1][['open', 'high', 'low', 'close']]
        
        # 检查是否都是阳线
        all_bullish = all(candles.iloc[i]['close'] > candles.iloc[i]['open'] for i in range(3))
        if not all_bullish:
            return False, 0.0, 'neutral'
        
        # 检查是否逐步上升
        closes = candles['close'].values
        ascending = closes[0] < closes[1] < closes[2]
        
        if not ascending:
            return False, 0.0, 'neutral'
        
        # 计算强度：基于三根K线的一致性
        bodies = [abs(candles.iloc[i]['close'] - candles.iloc[i]['open']) for i in range(3)]
        avg_body = np.mean(bodies)
        body_consistency = 1 - np.std(bodies) / avg_body if avg_body > 0 else 0
        
        strength = body_consistency * self.pattern_weights['three_white_soldiers']
        
        return True, strength, 'bullish'
    
    def detect_three_black_crows(self, df: pd.DataFrame, idx: int) -> Tuple[bool, float, str]:
        """
        检测三黑鸦（强看跌信号）
        三根连续的阴线，每根都有实体，逐步下降
        """
        if idx < 2:
            return False, 0.0, 'neutral'
        
        candles = df.iloc[idx-2:idx+1][['open', 'high', 'low', 'close']]
        
        # 检查是否都是阴线
        all_bearish = all(candles.iloc[i]['close'] < candles.iloc[i]['open'] for i in range(3))
        if not all_bearish:
            return False, 0.0, 'neutral'
        
        # 检查是否逐步下降
        closes = candles['close'].values
        descending = closes[0] > closes[1] > closes[2]
        
        if not descending:
            return False, 0.0, 'neutral'
        
        # 计算强度
        bodies = [abs(candles.iloc[i]['close'] - candles.iloc[i]['open']) for i in range(3)]
        avg_body = np.mean(bodies)
        body_consistency = 1 - np.std(bodies) / avg_body if avg_body > 0 else 0

        
        strength = body_consistency * self.pattern_weights['three_black_crows']
        
        return True, strength, 'bearish'
    
    def detect_all_patterns(self, df: pd.DataFrame, idx: int = -1) -> Dict[str, Tuple[bool, float, str]]:
        """
        检测指定位置的所有K线形态
        
        Args:
            df: K线数据DataFrame
            idx: 要检测的K线索引，默认-1（最新K线）
        
        Returns:
            字典，包含所有形态的检测结果
            格式: {pattern_name: (detected, strength, direction)}
        """
        if idx == -1:
            idx = len(df) - 1
        
        results = {
            'hammer': self.detect_hammer(df, idx),
            'inverted_hammer': self.detect_inverted_hammer(df, idx),
            'shooting_star': self.detect_shooting_star(df, idx),
            'hanging_man': self.detect_hanging_man(df, idx),
            'engulfing': self.detect_engulfing(df, idx),
            'doji': self.detect_doji(df, idx),
            'three_white_soldiers': self.detect_three_white_soldiers(df, idx),
            'three_black_crows': self.detect_three_black_crows(df, idx),
        }
        
        return results
    
    def get_pattern_summary(self, patterns: Dict[str, Tuple[bool, float, str]]) -> Dict:
        """
        汇总形态检测结果
        
        Returns:
            {
                'bullish_patterns': [(name, strength), ...],
                'bearish_patterns': [(name, strength), ...],
                'neutral_patterns': [(name, strength), ...],
                'total_bullish_strength': float,
                'total_bearish_strength': float,
            }
        """
        bullish = []
        bearish = []
        neutral = []
        
        for name, (detected, strength, direction) in patterns.items():
            if detected:
                if direction == 'bullish':
                    bullish.append((name, strength))
                elif direction == 'bearish':
                    bearish.append((name, strength))
                else:
                    neutral.append((name, strength))
        
        # 按强度排序
        bullish.sort(key=lambda x: x[1], reverse=True)
        bearish.sort(key=lambda x: x[1], reverse=True)
        neutral.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'bullish_patterns': bullish,
            'bearish_patterns': bearish,
            'neutral_patterns': neutral,
            'total_bullish_strength': sum(s for _, s in bullish),
            'total_bearish_strength': sum(s for _, s in bearish),
            'dominant_direction': 'bullish' if sum(s for _, s in bullish) > sum(s for _, s in bearish) else 'bearish' if sum(s for _, s in bearish) > 0 else 'neutral'
        }
