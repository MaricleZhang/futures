import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from strategies.base_strategy import BaseStrategy

class TrendStrategy(BaseStrategy):
    """
    基于趋势的交易策略
    使用快慢均线的相对强度来判断趋势，生成交易信号
    """
    
    def __init__(self, trader):
        """
        初始化趋势策略
        Args:
            trader: 交易器实例
        """
        super().__init__(trader)
        self.logger = self.get_logger()
        
        # 趋势参数
        self.fast_window = 5      # 快速均线窗口
        self.slow_window = 20     # 慢速均线窗口
        self.std_window = 20      # 标准差窗口
        self.threshold_factor = 1.5  # 信号阈值系数
        
        # 模型状态
        self.last_signal = 0      # 最后一次信号（-1:卖出, 0:观望, 1:买入）
        self.last_update_time = None
        self.retraining_interval = timedelta(hours=4)  # 每4小时更新一次趋势状态
        
    def prepare_features(self, klines):
        """准备技术指标特征"""
        try:
            # 将K线数据转换为DataFrame
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # 将所有列转换为数值类型
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 计算移动平均线
            df['SMA_fast'] = df['close'].rolling(window=self.fast_window).mean()
            df['SMA_slow'] = df['close'].rolling(window=self.slow_window).mean()
            
            # 计算趋势强度
            df['trend_strength'] = (df['SMA_fast'] - df['SMA_slow']) / df['SMA_slow']
            
            # 计算趋势强度的标准差
            df['trend_std'] = df['trend_strength'].rolling(window=self.std_window).std()
            
            # 删除包含NaN的行
            df = df.dropna()
            
            return df
            
        except Exception as e:
            self.logger.error(f"准备特征失败: {str(e)}")
            return None
    
    def generate_signals(self, klines):
        """生成交易信号"""
        try:
            current_time = datetime.now()
            
            # 检查K线数据
            if not klines or len(klines) == 0:
                self.logger.warning("K线数据为空")
                return self.last_signal
            
            # 准备特征数据
            df = self.prepare_features(klines)
            if df is None or len(df) == 0:
                return self.last_signal
            
            # 获取最新的趋势状态
            latest = df.iloc[-1]
            trend_strength = latest['trend_strength']
            trend_std = latest['trend_std']
            
            # 生成信号
            if trend_strength > self.threshold_factor * trend_std:
                signal = 1  # 买入信号
            elif trend_strength < -self.threshold_factor * trend_std:
                signal = -1  # 卖出信号
            else:
                signal = 0  # 观望信号
            
            # 更新状态
            self.last_signal = signal
            self.last_update_time = current_time
            
            # 记录信号
            signal_type = "买入" if signal == 1 else "卖出" if signal == -1 else "观望"
            self.logger.info(f"生成信号: {signal_type}, 趋势强度: {trend_strength:.4f}, "
                           f"趋势标准差: {trend_std:.4f}")
            
            return signal
            
        except Exception as e:
            self.logger.error(f"生成信号失败: {str(e)}")
            return self.last_signal
    
    def update_trend_state(self, klines):
        """更新趋势状态"""
        try:
            current_time = datetime.now()
            
            # 如果距离上次更新时间不足retraining_interval，直接返回
            if (self.last_update_time is not None and 
                current_time - self.last_update_time < self.retraining_interval):
                return
            
            # 检查K线数据
            if not klines or len(klines) == 0:
                self.logger.warning("K线数据为空")
                return
            
            # 准备特征数据
            df = self.prepare_features(klines)
            if df is None or len(df) == 0:
                return
            
            # 统计趋势分布
            trend_strength = df['trend_strength']
            trend_std = df['trend_std']
            
            up_trend = sum(trend_strength > self.threshold_factor * trend_std)
            down_trend = sum(trend_strength < -self.threshold_factor * trend_std)
            neutral = len(trend_strength) - up_trend - down_trend
            
            total = len(trend_strength)
            self.logger.info(f"趋势分布: 上涨={up_trend/total:.1%}, "
                           f"下跌={down_trend/total:.1%}, "
                           f"震荡={neutral/total:.1%}")
            
            # 更新时间
            self.last_update_time = current_time
            
        except Exception as e:
            self.logger.error(f"更新趋势状态失败: {str(e)}")
    
    def get_logger(self):
        """获取日志记录器"""
        logger = logging.getLogger(f"[{self.__class__.__name__}]")
        logger.setLevel(logging.INFO)
        return logger
