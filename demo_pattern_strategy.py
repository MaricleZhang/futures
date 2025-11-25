"""
K线形态概率交易策略演示脚本
展示如何分析K线形态并输出做多/做空/观望的概率

Usage:
    python demo_pattern_strategy.py
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.enhanced_candlestick_patterns import EnhancedCandlestickPattern, PatternDirection


def generate_sample_klines(pattern_type: str = "random", count: int = 100) -> pd.DataFrame:
    """
    生成示例K线数据
    
    Args:
        pattern_type: 要生成的形态类型
            - "random": 随机数据
            - "uptrend": 上升趋势
            - "downtrend": 下降趋势
            - "hammer": 锤子线
            - "engulfing_bullish": 看涨吞没
            - "morning_star": 早晨之星
        count: K线数量
    
    Returns:
        DataFrame with OHLCV data
    """
    np.random.seed(42)
    
    base_price = 100
    timestamps = [datetime.now() - timedelta(minutes=15*i) for i in range(count, 0, -1)]
    
    data = []
    
    if pattern_type == "uptrend":
        # 生成上升趋势
        for i in range(count):
            trend = i * 0.5
            noise = np.random.randn() * 0.5
            
            open_price = base_price + trend + noise
            close_price = open_price + np.random.uniform(0.1, 1.0)
            high_price = max(open_price, close_price) + np.random.uniform(0, 0.5)
            low_price = min(open_price, close_price) - np.random.uniform(0, 0.3)
            volume = np.random.uniform(1000, 5000)
            
            data.append([timestamps[i], open_price, high_price, low_price, close_price, volume])
    
    elif pattern_type == "downtrend":
        # 生成下降趋势
        for i in range(count):
            trend = -i * 0.5
            noise = np.random.randn() * 0.5
            
            open_price = base_price + trend + noise
            close_price = open_price - np.random.uniform(0.1, 1.0)
            high_price = max(open_price, close_price) + np.random.uniform(0, 0.3)
            low_price = min(open_price, close_price) - np.random.uniform(0, 0.5)
            volume = np.random.uniform(1000, 5000)
            
            data.append([timestamps[i], open_price, high_price, low_price, close_price, volume])
    
    elif pattern_type == "hammer":
        # 先下跌，最后一根是锤子线
        for i in range(count - 1):
            trend = -i * 0.3
            noise = np.random.randn() * 0.3
            
            open_price = base_price + trend + noise
            close_price = open_price - np.random.uniform(0.2, 0.8)
            high_price = max(open_price, close_price) + np.random.uniform(0, 0.2)
            low_price = min(open_price, close_price) - np.random.uniform(0.1, 0.5)
            volume = np.random.uniform(1000, 5000)
            
            data.append([timestamps[i], open_price, high_price, low_price, close_price, volume])
        
        # 最后一根是锤子线
        last_close = data[-1][4]
        open_price = last_close - 0.5
        close_price = open_price + 0.3  # 小阳线
        high_price = close_price + 0.1  # 几乎没有上影线
        low_price = open_price - 2.0    # 长下影线
        volume = 6000  # 放量
        
        data.append([timestamps[-1], open_price, high_price, low_price, close_price, volume])
    
    elif pattern_type == "engulfing_bullish":
        # 先下跌，最后形成看涨吞没
        for i in range(count - 2):
            trend = -i * 0.3
            noise = np.random.randn() * 0.3
            
            open_price = base_price + trend + noise
            close_price = open_price - np.random.uniform(0.2, 0.8)
            high_price = max(open_price, close_price) + np.random.uniform(0, 0.2)
            low_price = min(open_price, close_price) - np.random.uniform(0.1, 0.3)
            volume = np.random.uniform(1000, 5000)
            
            data.append([timestamps[i], open_price, high_price, low_price, close_price, volume])
        
        # 倒数第二根是阴线
        last_close = data[-1][4]
        open_price = last_close
        close_price = open_price - 1.0
        high_price = open_price + 0.1
        low_price = close_price - 0.1
        volume = 4000
        
        data.append([timestamps[-2], open_price, high_price, low_price, close_price, volume])
        
        # 最后一根是大阳线（吞没前一根）
        prev_open = open_price
        prev_close = close_price
        open_price = prev_close - 0.2
        close_price = prev_open + 0.5
        high_price = close_price + 0.2
        low_price = open_price - 0.1
        volume = 8000  # 放量
        
        data.append([timestamps[-1], open_price, high_price, low_price, close_price, volume])
    
    elif pattern_type == "morning_star":
        # 先下跌，最后形成早晨之星
        for i in range(count - 3):
            trend = -i * 0.3
            noise = np.random.randn() * 0.3
            
            open_price = base_price + trend + noise
            close_price = open_price - np.random.uniform(0.3, 1.0)
            high_price = max(open_price, close_price) + np.random.uniform(0, 0.2)
            low_price = min(open_price, close_price) - np.random.uniform(0.1, 0.3)
            volume = np.random.uniform(1000, 5000)
            
            data.append([timestamps[i], open_price, high_price, low_price, close_price, volume])
        
        # 第一根：大阴线
        last_close = data[-1][4]
        open_price = last_close
        close_price = open_price - 2.0
        high_price = open_price + 0.1
        low_price = close_price - 0.1
        volume = 5000
        
        data.append([timestamps[-3], open_price, high_price, low_price, close_price, volume])
        
        # 第二根：小实体（跳空向下）
        open_price = close_price - 0.3
        close_price = open_price + 0.2
        high_price = close_price + 0.1
        low_price = open_price - 0.1
        volume = 3000
        
        data.append([timestamps[-2], open_price, high_price, low_price, close_price, volume])
        
        # 第三根：大阳线
        open_price = close_price + 0.2
        close_price = data[-2][1] + 0.5  # 收在第一根中部以上
        high_price = close_price + 0.2
        low_price = open_price - 0.1
        volume = 8000
        
        data.append([timestamps[-1], open_price, high_price, low_price, close_price, volume])
    
    else:  # random
        for i in range(count):
            noise = np.random.randn() * 2
            
            open_price = base_price + noise
            change = np.random.uniform(-1, 1)
            close_price = open_price + change
            high_price = max(open_price, close_price) + np.random.uniform(0, 0.5)
            low_price = min(open_price, close_price) - np.random.uniform(0, 0.5)
            volume = np.random.uniform(1000, 5000)
            
            data.append([timestamps[i], open_price, high_price, low_price, close_price, volume])
    
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    return df


def demo_pattern_detection():
    """演示K线形态检测"""
    print("\n" + "=" * 80)
    print("🕯️  K线形态识别演示")
    print("=" * 80)
    
    detector = EnhancedCandlestickPattern()
    
    # 测试不同的形态
    test_cases = [
        ("hammer", "锤子线（下跌后反转信号）"),
        ("engulfing_bullish", "看涨吞没（强反转信号）"),
        ("morning_star", "早晨之星（强反转信号）"),
        ("uptrend", "上升趋势"),
        ("downtrend", "下降趋势"),
    ]
    
    for pattern_type, description in test_cases:
        print(f"\n{'─' * 60}")
        print(f"📊 测试场景: {description}")
        print(f"{'─' * 60}")
        
        df = generate_sample_klines(pattern_type, 100)
        
        # 检测所有形态
        patterns = detector.detect_all_patterns(df)
        summary = detector.get_pattern_summary(patterns)
        
        # 输出结果
        print(f"\n检测到的形态:")
        
        if summary['bullish_patterns']:
            print(f"\n  🟢 看涨形态:")
            for name, strength, confidence, desc in summary['bullish_patterns']:
                print(f"     • {name}: 强度={strength:.2f}, 置信度={confidence:.2f}")
                print(f"       {desc}")
        
        if summary['bearish_patterns']:
            print(f"\n  🔴 看跌形态:")
            for name, strength, confidence, desc in summary['bearish_patterns']:
                print(f"     • {name}: 强度={strength:.2f}, 置信度={confidence:.2f}")
                print(f"       {desc}")
        
        if summary['neutral_patterns']:
            print(f"\n  ⚪ 中性形态:")
            for name, strength, confidence, desc in summary['neutral_patterns']:
                print(f"     • {name}: 强度={strength:.2f}")
        
        # 计算方向概率
        probs = detector.calculate_direction_probability(df)
        
        print(f"\n  📈 方向概率:")
        print(f"     做多概率: {probs['long_prob']:.1%}")
        print(f"     做空概率: {probs['short_prob']:.1%}")
        print(f"     观望概率: {probs['hold_prob']:.1%}")
        print(f"     置信度: {probs['confidence']:.1%}")
        
        # 主导方向
        print(f"\n  🎯 主导方向: {summary['dominant_direction'].upper()}")


def demo_probability_calculation():
    """演示概率计算"""
    print("\n" + "=" * 80)
    print("📊  交易概率计算演示")
    print("=" * 80)
    
    detector = EnhancedCandlestickPattern()
    
    # 生成不同市场状态的数据
    scenarios = [
        ("uptrend", "强势上涨行情"),
        ("downtrend", "强势下跌行情"),
        ("hammer", "底部锤子线反转"),
        ("engulfing_bullish", "看涨吞没反转"),
        ("random", "震荡行情"),
    ]
    
    print("\n" + "─" * 80)
    print(f"{'场景':<20} {'做多%':>10} {'做空%':>10} {'观望%':>10} {'置信度%':>10} {'建议':>10}")
    print("─" * 80)
    
    for pattern_type, description in scenarios:
        df = generate_sample_klines(pattern_type, 100)
        probs = detector.calculate_direction_probability(df)
        
        # 确定建议
        if probs['long_prob'] > 0.45 and probs['confidence'] > 0.5:
            advice = "做多"
            advice_color = "🟢"
        elif probs['short_prob'] > 0.45 and probs['confidence'] > 0.5:
            advice = "做空"
            advice_color = "🔴"
        else:
            advice = "观望"
            advice_color = "⚪"
        
        print(f"{description:<20} {probs['long_prob']*100:>9.1f}% {probs['short_prob']*100:>9.1f}% "
              f"{probs['hold_prob']*100:>9.1f}% {probs['confidence']*100:>9.1f}% {advice_color} {advice:>8}")
    
    print("─" * 80)


def demo_realtime_analysis():
    """模拟实时分析"""
    print("\n" + "=" * 80)
    print("⚡  实时分析演示")
    print("=" * 80)
    
    detector = EnhancedCandlestickPattern()
    
    # 模拟市场从下跌转为反转
    print("\n模拟市场从下跌到反转的过程...")
    print("─" * 60)
    
    # 生成基础下跌数据
    df = generate_sample_klines("downtrend", 95)
    
    # 逐步添加反转信号
    for i in range(5):
        # 模拟新K线
        last_row = df.iloc[-1]
        
        if i < 2:
            # 继续下跌
            new_open = last_row['close'] - 0.2
            new_close = new_open - 0.5
        elif i == 2:
            # 出现锤子线
            new_open = last_row['close'] - 0.3
            new_close = new_open + 0.2
            new_low = new_open - 1.5  # 长下影线
        elif i == 3:
            # 确认反转
            new_open = last_row['close'] + 0.1
            new_close = new_open + 1.0
        else:
            # 继续上涨
            new_open = last_row['close'] + 0.2
            new_close = new_open + 0.8
        
        new_high = max(new_open, new_close) + 0.2
        if i != 2:
            new_low = min(new_open, new_close) - 0.2
        
        new_row = pd.DataFrame([{
            'timestamp': datetime.now(),
            'open': new_open,
            'high': new_high,
            'low': new_low,
            'close': new_close,
            'volume': 5000 + i * 1000
        }])
        
        df = pd.concat([df, new_row], ignore_index=True)
        
        # 分析当前状态
        probs = detector.calculate_direction_probability(df)
        patterns = detector.detect_all_patterns(df)
        summary = detector.get_pattern_summary(patterns)
        
        detected = [p[0] for p in summary['bullish_patterns'] + summary['bearish_patterns']]
        
        print(f"\n第 {i+1} 根K线:")
        print(f"  价格: {new_close:.2f}")
        print(f"  检测到形态: {', '.join(detected) if detected else '无'}")
        print(f"  做多概率: {probs['long_prob']:.1%} | 做空概率: {probs['short_prob']:.1%} | 观望: {probs['hold_prob']:.1%}")
        
        if probs['long_prob'] > probs['short_prob'] and probs['long_prob'] > 0.4:
            print(f"  🟢 建议: 做多 (置信度: {probs['confidence']:.1%})")
        elif probs['short_prob'] > probs['long_prob'] and probs['short_prob'] > 0.4:
            print(f"  🔴 建议: 做空 (置信度: {probs['confidence']:.1%})")
        else:
            print(f"  ⚪ 建议: 观望")


def main():
    """主函数"""
    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + "K线形态概率交易策略 - 演示程序".center(78) + "║")
    print("╚" + "═" * 78 + "╝")
    
    # 演示形态检测
    demo_pattern_detection()
    
    # 演示概率计算
    demo_probability_calculation()
    
    # 演示实时分析
    demo_realtime_analysis()
    
    print("\n" + "=" * 80)
    print("演示完成！")
    print("=" * 80)
    print("\n策略特点:")
    print("  1. 识别20+种经典K线形态（锤子、吞没、早晨之星等）")
    print("  2. 结合趋势、动量、波动率、成交量四维分析")
    print("  3. 输出做多/做空/观望的概率及置信度")
    print("  4. 支持实时分析和信号生成")
    print("\n使用方法:")
    print("  在 config.py 中设置 STRATEGY_TYPE = 'pattern_probability'")
    print("  然后运行 python main.py")


if __name__ == "__main__":
    main()
