import numpy as np
import pandas as pd
import talib
from strategies.base_strategy import BaseStrategy

class MAStrategy(BaseStrategy):
    """双均线交易策略"""
    
    def __init__(self, trader):
        """初始化策略
        Args:
            trader (BinanceFuturesTrader): 交易者实例
        """
        super(MAStrategy, self).__init__(trader)
        
        # 策略参数
        self.fast_period = 5  # 快线周期
        self.slow_period = 10  # 慢线周期
        self.kline_interval = '1m'  # K线周期
        self.lookback = 20  # 回看的K线数量
        self.check_interval = 60  # 检查间隔(秒)
        self.training_lookback = 100  # 用于计算指标的历史数据长度
        
    def generate_signal(self, klines=None):
        """生成交易信号"""
        try:
            if klines is None:
                klines = self.trader.get_klines(interval=self.kline_interval, limit=self.lookback)
            if len(klines) < self.lookback:
                return None
                
            # 转换为DataFrame
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df = df.astype({
                'open': 'float64',
                'high': 'float64',
                'low': 'float64',
                'close': 'float64',
                'volume': 'float64'
            })
            
            # 计算均线
            ma_fast = talib.MA(df['close'].values, timeperiod=self.fast_period)
            ma_slow = talib.MA(df['close'].values, timeperiod=self.slow_period)
            
            # 获取最新的均线值
            current_fast = ma_fast[-1]
            current_slow = ma_slow[-1]
            prev_fast = ma_fast[-2]
            prev_slow = ma_slow[-2]
            
            # 检查持仓状态
            position = self.trader.get_position()
            current_position = 0
            if position and isinstance(position, dict):
                current_position = position.get('size', 0)
            
            # 记录市场状态
            self.logger.info(
                f"MA{self.fast_period}: {current_fast:.2f} "
                f"MA{self.slow_period}: {current_slow:.2f} "
                f"当前价格: {df['close'].iloc[-1]:.2f} "
                f"当前持仓: {current_position}"
            )
            
            # 生成信号
            if current_fast > current_slow:  # 快线在上，上涨趋势
                self.logger.info(f"金叉开多: MA{self.fast_period}上穿MA{self.slow_period}")
                return 1
            else:  # 快线在下，下跌趋势
                self.logger.info(f"死叉开空: MA{self.fast_period}下穿MA{self.slow_period}")
                return -1
            
            return None
            
        except Exception as e:
            self.logger.error(f"生成信号出错: {str(e)}")
            return None
