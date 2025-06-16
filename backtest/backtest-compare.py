"""
回测结果比较工具
"""
import argparse
import logging
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from utils.logger import Logger
from backtest import Backtester

# 导入所有策略
from strategies.strategy_module import DirectionalIndexStrategy15m
from strategies.multi_timeframe_di_adx_strategy import MultiTimeframeDIADXStrategy


# 策略字典
STRATEGIES = {
    'di_sco': DirectionalIndexStrategy15m,
    'multi_timeframe_di_adx': MultiTimeframeDIADXStrategy,
}

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='加密货币交易策略回测比较工具')
    
    # 必选参数
    parser.add_argument('--symbol', type=str, required=True, help='交易对，例如 BTCUSDT')
    parser.add_argument('--strategies', type=str, required=True, nargs='+',
                      choices=STRATEGIES.keys(), help='要比较的策略列表')
    
    # 可选参数
    parser.add_argument('--start', type=str, default=(datetime.now().replace(day=1)).strftime('%Y-%m-%d'),
                      help='开始日期，格式 YYYY-MM-DD，默认为当月1日')
    parser.add_argument('--end', type=str, default=datetime.now().strftime('%Y-%m-%d'),
                      help='结束日期，格式 YYYY-MM-DD，默认为今天')
    parser.add_argument('--timeframe', type=str, default='5m',
                      help='K线周期，可选值：1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d，默认为5m')
    parser.add_argument('--balance', type=float, default=10000,
                      help='初始资金，默认为10000 USDT')
    parser.add_argument('--leverage', type=int, default=5,
                      help='杠杆倍数，默认为5倍')
    parser.add_argument('--reload', action='store_true',
                      help='强制重新下载历史数据，即使本地已有数据文件')
    
    return parser.parse_args()

def run_backtest(args, strategy_name):
    """运行单个策略的回测"""
    logger = logging.getLogger()
    
    strategy_class = STRATEGIES[strategy_name]
    
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
    
    # 如果需要强制重新下载数据，并且是第一个策略
    if args.reload and strategy_name == args.strategies[0]:
        backtester.fetch_historical_data()
    
    # 运行回测
    logger.info(f"开始回测 {args.symbol} ({args.start} - {args.end}), 策略: {strategy_name}, K线周期: {args.timeframe}")
    results = backtester.run()
    
    if results:
        logger.info(f"回测完成: {strategy_name}, 利润: {results['profit']:.2f} USDT ({results['profit_percent']:.2f}%)")
    else:
        logger.error(f"回测失败: {strategy_name}")
    
    return results

def compare_results(results_list, strategy_names):
    """比较多个策略的回测结果"""
    logger = logging.getLogger()
    
    if not results_list or len(results_list) == 0:
        logger.error("没有有效的回测结果可以比较")
        return
    
    # 创建比较数据框
    compare_data = []
    
    for i, results in enumerate(results_list):
        if results:
            data = {
                'Strategy': strategy_names[i],
                'Final Balance': results['final_balance'],
                'Profit': results['profit'],
                'Profit %': results['profit_percent'],
                'Trade Count': results['trade_count'],
                'Win Rate %': results.get('win_rate', 0),
                'Avg Profit': results.get('avg_profit', 0),
                'Max Profit': results.get('max_profit', 0),
                'Max Loss': results.get('max_loss', 0),
                'Max Drawdown %': results.get('max_drawdown_percent', 0),
                'Sharpe Ratio': results.get('sharpe_ratio', 0),
                'Calmar Ratio': results.get('calmar_ratio', 0),
            }
            compare_data.append(data)
    
    # 创建DataFrame
    df = pd.DataFrame(compare_data)
    
    # 创建结果目录
    os.makedirs('comparison_results', exist_ok=True)
    
    # 保存比较结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = f"comparison_results/comparison_{timestamp}.csv"
    df.to_csv(csv_file, index=False)
    
    # 打印比较结果
    logger.info("=" * 80)
    logger.info("策略比较结果")
    logger.info("=" * 80)
    logger.info("\n" + df.to_string())
    logger.info("=" * 80)
    logger.info(f"比较结果已保存到 {csv_file}")
    
    # 绘制比较图表
    plot_comparison(df, results_list, strategy_names, timestamp)

def plot_comparison(df, results_list, strategy_names, timestamp):
    """绘制比较图表"""
    logger = logging.getLogger()
    
    try:
        # 创建图表目录
        os.makedirs('comparison_charts', exist_ok=True)
        
        # 图表文件名
        chart_file = f"comparison_charts/comparison_{timestamp}.png"
        
        # 创建图表
        fig = plt.figure(figsize=(20, 15))
        fig.suptitle("策略比较", fontsize=16)
        
        # 1. 权益曲线比较
        ax1 = plt.subplot(3, 2, 1)
        ax1.set_title("权益曲线")
        ax1.set_ylabel("USDT")
        
        for i, results in enumerate(results_list):
            if results and 'equity_curve' in results and len(results['equity_curve']) > 0:
                equity_data = pd.DataFrame(results['equity_curve'])
                if not equity_data.empty and 'timestamp' in equity_data.columns and 'equity' in equity_data.columns:
                    ax1.plot(range(len(equity_data)), equity_data['equity'], label=strategy_names[i])
        
        ax1.axhline(y=df['Final Balance'].iloc[0], color='r', linestyle='--', label='初始资金')
        ax1.grid(True)
        ax1.legend()
        
        # 2. 利润百分比
        ax2 = plt.subplot(3, 2, 2)
        ax2.set_title("利润百分比")
        ax2.bar(df['Strategy'], df['Profit %'])
        ax2.set_ylabel("%")
        ax2.grid(True)
        
        # 3. 胜率
        ax3 = plt.subplot(3, 2, 3)
        ax3.set_title("胜率")
        ax3.bar(df['Strategy'], df['Win Rate %'])
        ax3.set_ylabel("%")
        ax3.grid(True)
        
        # 4. 交易次数
        ax4 = plt.subplot(3, 2, 4)
        ax4.set_title("交易次数")
        ax4.bar(df['Strategy'], df['Trade Count'])
        ax4.grid(True)
        
        # 5. 最大回撤
        ax5 = plt.subplot(3, 2, 5)
        ax5.set_title("最大回撤")
        ax5.bar(df['Strategy'], df['Max Drawdown %'])
        ax5.set_ylabel("%")
        ax5.grid(True)
        
        # 6. 夏普比率和卡玛比率
        ax6 = plt.subplot(3, 2, 6)
        ax6.set_title("夏普比率和卡玛比率")
        
        x = range(len(df['Strategy']))
        width = 0.35
        
        ax6.bar([i - width/2 for i in x], df['Sharpe Ratio'], width, label='夏普比率')
        ax6.bar([i + width/2 for i in x], df['Calmar Ratio'], width, label='卡玛比率')
        
        ax6.set_xticks(x)
        ax6.set_xticklabels(df['Strategy'])
        ax6.legend()
        ax6.grid(True)
        
        # 保存图表
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(chart_file)
        logger.info(f"比较图表已保存到 {chart_file}")
        
    except Exception as e:
        logger.error(f"绘制比较图表失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

def main():
    """主函数"""
    # 初始化日志记录器
    logger = Logger.get_logger()
    
    # 解析命令行参数
    args = parse_arguments()
    
    # 运行每个策略的回测
    results_list = []
    
    for strategy_name in args.strategies:
        results = run_backtest(args, strategy_name)
        results_list.append(results)
    
    # 比较结果
    compare_results(results_list, args.strategies)

if __name__ == "__main__":
    main()
