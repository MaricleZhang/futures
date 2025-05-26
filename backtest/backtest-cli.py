"""
回测系统命令行接口
"""
import argparse
import logging
import sys
import os
from datetime import datetime

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import Logger
from backtest.backtest_system import Backtester, MockTrader

# 导入所有策略
from strategies.trend_strategy import SimpleTrendStrategy15m
from strategies.multi_timeframe_di_adx_strategy import MultiTimeframeDIADXStrategy

# 策略字典
STRATEGIES = {
    'simple_trend': SimpleTrendStrategy15m,
    'multi_timeframe_di_adx': MultiTimeframeDIADXStrategy,
}

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='加密货币交易策略回测工具')
    
    # 必选参数
    parser.add_argument('--symbol', type=str, required=True, help='交易对，例如 BTCUSDT')
    parser.add_argument('--strategy', type=str, required=True, choices=STRATEGIES.keys(), help='策略名称')
    
    # 可选参数
    parser.add_argument('--start', type=str, default=(datetime.now().replace(day=1)).strftime('%Y-%m-%d'),
                      help='开始日期，格式 YYYY-MM-DD，默认为当月1日')
    parser.add_argument('--end', type=str, default=datetime.now().strftime('%Y-%m-%d'),
                      help='结束日期，格式 YYYY-MM-DD，默认为今天')
    parser.add_argument('--timeframe', type=str, default='5m',
                      help='K线周期，可选值：1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d, 默认为5m')
    parser.add_argument('--balance', type=float, default=10000,
                      help='初始资金，默认为10000 USDT')
    parser.add_argument('--leverage', type=int, default=5,
                      help='杠杆倍数，默认为5倍')
    parser.add_argument('--reload', action='store_true',
                      help='强制重新下载历史数据，即使本地已有数据文件')
    
    return parser.parse_args()

def main():
    """主函数"""
    # 初始化日志记录器
    logger = Logger.get_logger()
    
    # 解析命令行参数
    args = parse_arguments()
    
    # 获取策略类
    strategy_class = STRATEGIES[args.strategy]
    
    # 初始化回测器
    backtester = Backtester(
        symbol=args.symbol,
        strategy_class=strategy_class,
        start_date=args.start,
        end_date=args.end,
        timeframe=args.timeframe,
        initial_balance=args.balance,
        leverage=args.leverage
    )
    
    # 如果需要强制重新下载数据
    if args.reload:
        backtester.fetch_historical_data()
    
    # 运行回测
    logger.info(f"开始回测 {args.symbol} ({args.start} - {args.end}), 策略: {args.strategy}, K线周期: {args.timeframe}")
    results = backtester.run()
    
    # 打印回测结果
    if results:
        logger.info("=" * 50)
        logger.info(f"回测结果: {args.symbol} ({args.start} - {args.end})")
        logger.info(f"策略: {strategy_class.__name__}")
        logger.info(f"初始资金: {args.balance} USDT")
        logger.info(f"最终资金: {results['final_balance']:.2f} USDT")
        logger.info(f"利润: {results['profit']:.2f} USDT ({results['profit_percent']:.2f}%)")
        logger.info(f"交易次数: {results['trade_count']}")
        logger.info(f"胜率: {results.get('win_rate', 0):.2f}%")
        logger.info(f"平均收益: {results.get('avg_profit', 0):.2f} USDT")
        logger.info(f"最大盈利: {results.get('max_profit', 0):.2f} USDT")
        logger.info(f"最大亏损: {results.get('max_loss', 0):.2f} USDT")
        logger.info(f"最大回撤: {results.get('max_drawdown_percent', 0):.2f}%")
        logger.info(f"夏普比率: {results.get('sharpe_ratio', 0):.2f}")
        logger.info(f"卡玛比率: {results.get('calmar_ratio', 0):.2f}")
        logger.info("=" * 50)
    else:
        logger.error("回测失败")

if __name__ == "__main__":
    main()
