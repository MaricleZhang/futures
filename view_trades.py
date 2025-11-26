#!/usr/bin/env python3
"""
交易记录查看工具
用于查询和分析历史交易记录
"""
import sys
from pathlib import Path
from datetime import datetime
from utils.trade_recorder import TradeRecorder
from tabulate import tabulate

def format_datetime(dt_str):
    """格式化日期时间字符串"""
    if not dt_str:
        return "N/A"
    try:
        dt = datetime.fromisoformat(dt_str)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return dt_str

def view_trades(symbol=None, limit=50, strategy=None):
    """查看交易记录
    
    Args:
        symbol: 交易对，如果为None则显示所有交易对
        limit: 显示记录数量
        strategy: 策略名称，如果为None则显示所有策略
    """
    try:
        recorder = TradeRecorder()
        
        # 获取交易历史
        trades = recorder.get_trade_history(symbol, limit, strategy)
        
        if not trades:
            print("没有找到交易记录")
            return
        
        # 准备表格数据
        table_data = []
        for trade in trades:
            table_data.append([
                trade['trade_id'],
                trade['symbol'],
                trade['side'],
                f"{trade['open_amount']:.4f}",
                f"{trade['open_price']:.2f}",
                f"{trade['open_value']:.2f}",
                format_datetime(trade['open_time']),
                f"{trade['close_price']:.2f}" if trade['close_price'] else "N/A",
                format_datetime(trade['close_time']) if trade['close_time'] else "N/A",
                f"{trade['profit_loss']:.2f}" if trade['profit_loss'] is not None else "N/A",
                f"{trade['profit_rate']:.2f}%" if trade['profit_rate'] is not None else "N/A",
                trade.get('strategy', 'N/A') or 'N/A',
                trade['status']
            ])
        
        # 显示表格
        headers = ['ID', '交易对', '方向', '数量', '开仓价', '开仓金额', 
                  '开仓时间', '平仓价', '平仓时间', '盈亏(USDT)', '收益率', '策略', '状态']
        print("\n" + "="*150)
        title = f"交易记录 {'- ' + symbol if symbol else '- 所有交易对'}"
        if strategy:
            title += f" - 策略: {strategy}"
        print(title)
        print("="*150)
        print(tabulate(table_data, headers=headers, tablefmt='grid'))
        
        # 显示统计信息
        stats = recorder.get_statistics(symbol, strategy)
        if stats['total_trades'] > 0:
            print("\n" + "="*80)
            print("交易统计")
            print("="*80)
            print(f"总交易次数: {stats['total_trades']}")
            print(f"盈利次数: {stats['winning_trades']}")
            print(f"亏损次数: {stats['losing_trades']}")
            print(f"胜率: {stats['win_rate']:.2f}%")
            print(f"累计盈亏: {stats['total_profit']:.2f} USDT")
            print(f"平均收益率: {stats['avg_profit_rate']:.2f}%")
            print(f"最大单笔盈利: {stats['max_profit']:.2f} USDT")
            print(f"最大单笔亏损: {stats['max_loss']:.2f} USDT")
            print("="*80 + "\n")
        
    except Exception as e:
        print(f"查看交易记录失败: {str(e)}")
        import traceback
        traceback.print_exc()

def view_trades_by_date(date=None, start_date=None, end_date=None, symbol=None, strategy=None):
    """按日期查看交易记录
    
    Args:
        date: 查询指定日期，格式: YYYY-MM-DD
        start_date: 开始日期，格式: YYYY-MM-DD
        end_date: 结束日期，格式: YYYY-MM-DD
        symbol: 交易对，如果为None则显示所有交易对
        strategy: 策略名称，如果为None则显示所有策略
    """
    try:
        recorder = TradeRecorder()
        
        # 获取交易记录
        trades = recorder.get_trades_by_date(date=date, start_date=start_date, 
                                            end_date=end_date, symbol=symbol, strategy=strategy)
        
        if not trades:
            print("没有找到交易记录")
            return
        
        # 准备表格数据
        table_data = []
        total_profit = 0
        for trade in trades:
            table_data.append([
                trade['trade_id'],
                trade['symbol'],
                trade['side'],
                f"{trade['open_amount']:.4f}",
                f"{trade['open_price']:.2f}",
                f"{trade['open_value']:.2f}",
                format_datetime(trade['open_time']),
                f"{trade['close_price']:.2f}" if trade['close_price'] else "N/A",
                format_datetime(trade['close_time']) if trade['close_time'] else "N/A",
                f"{trade['profit_loss']:.2f}" if trade['profit_loss'] is not None else "N/A",
                f"{trade['profit_rate']:.2f}%" if trade['profit_rate'] is not None else "N/A",
                trade.get('strategy', 'N/A') or 'N/A',
                trade['status']
            ])
            if trade['profit_loss'] is not None:
                total_profit += trade['profit_loss']
        
        # 显示表格
        headers = ['ID', '交易对', '方向', '数量', '开仓价', '开仓金额', 
                  '开仓时间', '平仓价', '平仓时间', '盈亏(USDT)', '收益率', '策略', '状态']
        
        date_range_str = ""
        if date:
            date_range_str = f" - {date}"
        elif start_date and end_date:
            date_range_str = f" - {start_date} 到 {end_date}"
        elif start_date:
            date_range_str = f" - 从 {start_date}"
        elif end_date:
            date_range_str = f" - 到 {end_date}"
        
        symbol_str = f" - {symbol}" if symbol else " - 所有交易对"
        strategy_str = f" - 策略: {strategy}" if strategy else ""
        
        print("\n" + "="*150)
        print(f"交易记录{symbol_str}{strategy_str}{date_range_str}")
        print("="*150)
        print(tabulate(table_data, headers=headers, tablefmt='grid'))
        
        # 显示汇总信息
        closed_trades = [t for t in trades if t['status'] == 'CLOSED']
        if closed_trades:
            winning_trades = len([t for t in closed_trades if t['profit_loss'] and t['profit_loss'] > 0])
            print("\n" + "="*80)
            print("汇总统计")
            print("="*80)
            print(f"总交易次数: {len(closed_trades)}")
            print(f"盈利次数: {winning_trades}")
            print(f"亏损次数: {len(closed_trades) - winning_trades}")
            print(f"胜率: {(winning_trades/len(closed_trades)*100):.2f}%" if closed_trades else "N/A")
            print(f"累计盈亏: {total_profit:.2f} USDT")
            print("="*80 + "\n")
        
    except Exception as e:
        print(f"按日期查看交易记录失败: {str(e)}")
        import traceback
        traceback.print_exc()

def view_open_positions(symbol=None):
    """查看未平仓记录
    
    Args:
        symbol: 交易对，如果为None则显示所有交易对
    """
    try:
        recorder = TradeRecorder()
        
        # 获取未平仓记录
        positions = recorder.get_open_positions(symbol)
        
        if not positions:
            print("没有未平仓记录")
            return
        
        # 准备表格数据
        table_data = []
        for pos in positions:
            table_data.append([
                pos['trade_id'],
                pos['symbol'],
                pos['side'],
                f"{pos['open_amount']:.4f}",
                f"{pos['open_price']:.2f}",
                f"{pos['open_value']:.2f}",
                format_datetime(pos['open_time']),
                pos['leverage'],
                pos.get('strategy', 'N/A') or 'N/A'
            ])
        
        # 显示表格
        headers = ['ID', '交易对', '方向', '数量', '开仓价', '开仓金额', '开仓时间', '杠杆', '策略']
        print("\n" + "="*120)
        print(f"未平仓记录 {'- ' + symbol if symbol else '- 所有交易对'}")
        print("="*120)
        print(tabulate(table_data, headers=headers, tablefmt='grid'))
        print()
        
    except Exception as e:
        print(f"查看未平仓记录失败: {str(e)}")
        import traceback
        traceback.print_exc()

def export_to_csv(symbol=None, output_file='trades_export.csv'):
    """导出交易记录到CSV文件
    
    Args:
        symbol: 交易对，如果为None则导出所有交易对
        output_file: 输出文件名
    """
    try:
        import csv
        recorder = TradeRecorder()
        
        # 获取所有交易记录
        trades = recorder.get_trade_history(symbol, limit=10000)
        
        if not trades:
            print("没有找到交易记录")
            return
        
        # 写入CSV文件
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=trades[0].keys())
            writer.writeheader()
            writer.writerows(trades)
        
        print(f"成功导出 {len(trades)} 条记录到 {output_file}")
        
    except Exception as e:
        print(f"导出失败: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    """主函数"""
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == 'open':
            # 查看未平仓记录
            symbol = sys.argv[2] if len(sys.argv) > 2 else None
            view_open_positions(symbol)
        elif command == 'export':
            # 导出到CSV
            symbol = sys.argv[2] if len(sys.argv) > 2 else None
            output_file = sys.argv[3] if len(sys.argv) > 3 else 'trades_export.csv'
            export_to_csv(symbol, output_file)
        elif command == 'history':
            # 查看历史记录
            symbol = sys.argv[2] if len(sys.argv) > 2 else None
            limit = int(sys.argv[3]) if len(sys.argv) > 3 else 50
            view_trades(symbol, limit)
        elif command == 'date':
            # 按日期查询
            # python view_trades.py date 2025-11-27
            # python view_trades.py date 2025-11-20 2025-11-27
            # python view_trades.py date 2025-11-27 ETHUSDC
            if len(sys.argv) == 3:
                # 单个日期
                date_arg = sys.argv[2]
                view_trades_by_date(date=date_arg)
            elif len(sys.argv) == 4:
                # 可能是日期范围或日期+交易对
                arg2 = sys.argv[2]
                arg3 = sys.argv[3]
                # 判断arg3是否是日期格式
                if '-' in arg3 and len(arg3) == 10:
                    # 日期范围
                    view_trades_by_date(start_date=arg2, end_date=arg3)
                else:
                    # 日期+交易对
                    view_trades_by_date(date=arg2, symbol=arg3)
            elif len(sys.argv) == 5:
                # 日期范围+交易对
                view_trades_by_date(start_date=sys.argv[2], end_date=sys.argv[3], symbol=sys.argv[4])
            else:
                print("日期查询参数错误")
                print_usage()
        elif command == 'strategy':
            # 按策略查询
            # python view_trades.py strategy deepseek
            # python view_trades.py strategy deepseek ETHUSDC
            if len(sys.argv) >= 3:
                strategy = sys.argv[2]
                symbol = sys.argv[3] if len(sys.argv) > 3 else None
                view_trades(symbol=symbol, strategy=strategy)
            else:
                print("策略查询参数错误")
                print_usage()
        else:
            print("未知命令")
            print_usage()
    else:
        # 默认显示最近50条记录
        view_trades(limit=50)

def print_usage():
    """打印使用说明"""
    print("""
使用方法:
    python view_trades.py                           # 查看最近50条交易记录
    python view_trades.py history [symbol] [limit]  # 查看历史记录
    python view_trades.py open [symbol]             # 查看未平仓记录
    python view_trades.py export [symbol] [output_file]  # 导出到CSV
    python view_trades.py date <date> [symbol]      # 查看指定日期的记录
    python view_trades.py date <start_date> <end_date> [symbol]  # 查看日期范围的记录
    python view_trades.py strategy <strategy_name> [symbol]  # 查看指定策略的记录

示例:
    python view_trades.py history ETHUSDC 100       # 查看ETHUSDC最近100条记录
    python view_trades.py open ETHUSDC              # 查看ETHUSDC未平仓记录
    python view_trades.py export ETHUSDC trades.csv # 导出ETHUSDC记录到trades.csv
    python view_trades.py date 2025-11-27           # 查看2025-11-27的所有记录
    python view_trades.py date 2025-11-27 ETHUSDC   # 查看ETHUSDC在2025-11-27的记录
    python view_trades.py date 2025-11-20 2025-11-27  # 查看2025-11-20到2025-11-27的记录
    python view_trades.py date 2025-11-20 2025-11-27 ETHUSDC  # 查看ETHUSDC在该日期范围的记录
    python view_trades.py strategy deepseek         # 查看deepseek策略的所有记录
    python view_trades.py strategy deepseek ETHUSDC # 查看deepseek策略在ETHUSDC的记录
    """)


if __name__ == '__main__':
    main()
