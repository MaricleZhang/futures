"""
随机森林模型训练脚本

使用方法:
    python train_random_forest.py --symbol BTCUSDT --interval 15m

防过拟合措施:
1. 时间序列交叉验证
2. 限制树深度和最小样本数
3. OOB评分验证
4. 训练/测试集分离
5. 特征重要性分析
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging
import sys

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_historical_data(symbol: str, interval: str, days: int = 180, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """从Binance加载历史数据
    
    Args:
        symbol: 交易对
        interval: K线周期
        days: 数据天数 (当start_date/end_date未指定时使用)
        start_date: 开始日期 (格式: YYYY-MM-DD)
        end_date: 结束日期 (格式: YYYY-MM-DD)
    """
    try:
        import ccxt
        
        exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })
        
        # 计算时间范围
        if start_date and end_date:
            start_time = datetime.strptime(start_date, '%Y-%m-%d')
            end_time = datetime.strptime(end_date, '%Y-%m-%d')
            logger.info(f"📥 加载 {symbol} {interval} 数据 ({start_date} ~ {end_date})...")
        else:
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)
            logger.info(f"📥 加载 {symbol} {interval} 数据 (最近{days}天)...")
        
        since = int(start_time.timestamp() * 1000)
        end_ts = int(end_time.timestamp() * 1000)
        
        all_klines = []
        while since < end_ts:
            klines = exchange.fetch_ohlcv(symbol, interval, since=since, limit=1000)
            if not klines:
                break
            all_klines.extend(klines)
            since = klines[-1][0] + 1
            
        df = pd.DataFrame(all_klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        logger.info(f"✅ 加载完成: {len(df)} 条数据")
        return df
        
    except Exception as e:
        logger.error(f"加载数据失败: {str(e)}")
        return None


def train_model(symbol: str, interval: str, days: int = 180, start_date: str = None, end_date: str = None):
    """训练随机森林模型
    
    Args:
        symbol: 交易对
        interval: K线周期
        days: 训练数据天数 (当start_date/end_date未指定时使用)
        start_date: 开始日期 (格式: YYYY-MM-DD)
        end_date: 结束日期 (格式: YYYY-MM-DD)
    """
    
    # 加载数据
    df = load_historical_data(symbol, interval, days, start_date, end_date)
    if df is None or len(df) < 1000:
        logger.error("数据不足，无法训练")
        return
    
    # 创建模拟trader
    class MockTrader:
        def __init__(self, symbol):
            self.symbol = symbol
    
    # 导入策略
    from strategies.random_forest_strategy import RandomForestStrategy
    
    # 创建策略实例
    trader = MockTrader(symbol)
    strategy = RandomForestStrategy(trader, interval=interval, symbol=symbol)
    
    # 转换数据格式
    klines = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].values.tolist()
    df_train = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df_train[col] = pd.to_numeric(df_train[col], errors='coerce')
    
    # 训练模型
    result = strategy.train_model(df_train)
    
    if result['success']:
        logger.info("=" * 70)
        logger.info("🎉 训练完成!")
        logger.info("=" * 70)
        logger.info(f"训练集准确率: {result['train_accuracy']:.2%} | F1: {result.get('train_f1', 0):.2%}")
        logger.info(f"验证集准确率: {result.get('val_accuracy', 0):.2%}")
        logger.info(f"测试集准确率: {result['test_accuracy']:.2%} | F1: {result.get('test_f1', 0):.2%}")
        logger.info(f"交叉验证准确率: {result['cv_accuracy']:.2%} (+/- {result['cv_std']*2:.2%})")
        logger.info(f"OOB准确率: {result['oob_accuracy']:.2%}")
        logger.info(f"过拟合差距: {result['overfit_gap']:.2%}")
        
        if 'selected_features' in result:
            logger.info(f"选择特征数: {len(result['selected_features'])}")
        
        # 检查是否达到65%准确率
        if result['test_accuracy'] >= 0.65:
            logger.info("✅ 测试集准确率达到65%目标!")
        elif result['test_accuracy'] >= 0.60:
            logger.info(f"📊 测试集准确率接近目标 (当前: {result['test_accuracy']:.2%})")
        else:
            logger.warning(f"⚠️ 测试集准确率未达到65%目标 (当前: {result['test_accuracy']:.2%})")
            logger.info("建议: 尝试增加训练数据天数或使用更长K线周期")
    else:
        logger.error(f"训练失败: {result.get('error')}")


def main():
    parser = argparse.ArgumentParser(description='训练随机森林交易模型')
    parser.add_argument('--symbol', type=str, default='BTC/USDT', help='交易对')
    parser.add_argument('--interval', type=str, default='15m', help='K线周期')
    parser.add_argument('--days', type=int, default=180, help='训练数据天数')
    parser.add_argument('--start', type=str, default=None, help='开始日期 (格式: YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=None, help='结束日期 (格式: YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("🌲 随机森林模型训练")
    logger.info("=" * 70)
    logger.info(f"交易对: {args.symbol}")
    logger.info(f"K线周期: {args.interval}")
    if args.start and args.end:
        logger.info(f"时间范围: {args.start} ~ {args.end}")
    else:
        logger.info(f"数据天数: {args.days}")
    logger.info("=" * 70)
    
    train_model(args.symbol, args.interval, args.days, args.start, args.end)


if __name__ == '__main__':
    main()
