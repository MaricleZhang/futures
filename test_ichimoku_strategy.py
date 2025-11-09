#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 Ichimoku 策略的基本功能
"""

import sys
import pandas as pd
import numpy as np

try:
    # 测试导入策略
    print("正在测试导入策略...")
    from strategies.ichimoku_short_strategy import IchimokuShortStrategy
    print("✓ 策略导入成功")

    # 创建模拟的trader对象
    class MockTrader:
        def __init__(self):
            self.symbol = 'BTCUSDT'

        def get_logger(self):
            import logging
            return logging.getLogger('test')

    # 测试策略实例化
    print("\n正在测试策略实例化...")
    mock_trader = MockTrader()
    strategy = IchimokuShortStrategy(mock_trader)
    print("✓ 策略实例化成功")

    # 测试Ichimoku指标计算
    print("\n正在测试 Ichimoku 指标计算...")

    # 创建模拟K线数据
    np.random.seed(42)
    n = 100
    close_prices = 50000 + np.cumsum(np.random.randn(n) * 100)
    high_prices = close_prices + np.random.rand(n) * 200
    low_prices = close_prices - np.random.rand(n) * 200

    df = pd.DataFrame({
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': np.random.rand(n) * 1000
    })

    indicators = strategy.calculate_ichimoku_indicators(df)
    print("✓ Ichimoku 指标计算成功")

    # 验证指标数据
    print("\n验证指标数据...")
    required_indicators = ['tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 'chikou_span', 'atr', 'close']

    for indicator in required_indicators:
        if indicator in indicators:
            print(f"  ✓ {indicator}: 长度={len(indicators[indicator])}, 最后值={indicators[indicator][-1]:.2f}")
        else:
            print(f"  ✗ {indicator}: 缺失")

    # 测试信号分析
    print("\n正在测试信号分析...")
    signal, confidence, details = strategy.analyze_ichimoku_signal(indicators)
    print(f"✓ 信号分析成功")
    print(f"  信号: {signal} (1=看涨, -1=看跌, 0=观望)")
    print(f"  置信度: {confidence:.1f}/5")
    print(f"  详情: {details}")

    # 测试止损止盈计算
    print("\n正在测试止损止盈计算...")
    entry_price = 50000
    atr_value = 500

    # 测试做多
    stop_loss, take_profit = strategy.calculate_stop_loss_take_profit(entry_price, 1, atr_value)
    print(f"✓ 做多止损止盈计算成功")
    print(f"  入场价: {entry_price}")
    print(f"  止损价: {stop_loss:.2f} (风险: {((entry_price - stop_loss) / entry_price * 100):.2f}%)")
    print(f"  止盈价: {take_profit:.2f} (收益: {((take_profit - entry_price) / entry_price * 100):.2f}%)")

    # 测试做空
    stop_loss, take_profit = strategy.calculate_stop_loss_take_profit(entry_price, -1, atr_value)
    print(f"\n✓ 做空止损止盈计算成功")
    print(f"  入场价: {entry_price}")
    print(f"  止损价: {stop_loss:.2f} (风险: {((stop_loss - entry_price) / entry_price * 100):.2f}%)")
    print(f"  止盈价: {take_profit:.2f} (收益: {((entry_price - take_profit) / entry_price * 100):.2f}%)")

    print("\n" + "="*60)
    print("所有测试通过！✓")
    print("="*60)

    # 打印策略配置信息
    print("\n策略配置信息:")
    print(f"  K线周期: {strategy.kline_interval}")
    print(f"  检查间隔: {strategy.check_interval}秒")
    print(f"  Tenkan周期: {strategy.tenkan_period}")
    print(f"  Kijun周期: {strategy.kijun_period}")
    print(f"  Senkou B周期: {strategy.senkou_span_b_period}")
    print(f"  位移周期: {strategy.displacement}")
    print(f"  ATR倍数: {strategy.atr_multiplier}")
    print(f"  风险收益比: {strategy.risk_reward_ratio}")
    print(f"  最大仓位: {strategy.max_position_size * 100}%")
    print(f"  追踪止损激活: {strategy.trailing_stop_activation * 100}%")

except Exception as e:
    print(f"\n✗ 测试失败: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
