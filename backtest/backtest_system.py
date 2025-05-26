"""
交易策略回测系统
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging
import os
import json
from tqdm import tqdm
import ccxt
import time
from dotenv import load_dotenv
import config
from utils.logger import Logger

# 模拟交易者类，模拟Trader类的接口
class MockTrader:
    def __init__(self, symbol, backtest_data, balance=10000, leverage=1):
        """初始化模拟交易者
        
        Args:
            symbol: 交易对名称
            backtest_data: 回测数据，pd.DataFrame格式，包含timestamp, open, high, low, close, volume列
            balance: 初始资金，默认10000 USDT
            leverage: 杠杆倍数
        """
        self.symbol = symbol
        self.symbol_config = config.SYMBOL_CONFIGS.get(symbol, {})
        self.data = backtest_data
        self.current_index = 0
        self.balance = balance
        self.initial_balance = balance
        self.leverage = leverage
        
        # 当前持仓
        self.position = None
        self.position_entry_time = None
        self.position_entry_price = None
        self.position_amount = 0
        
        # 交易历史
        self.trade_history = []
        
        # 回测结果
        self.equity_curve = []
        
        # 设置日志
        self.logger = logging.getLogger(symbol)
    
    def get_klines(self, symbol=None, interval='1m', limit=100):
        """获取K线数据"""
        # 修改逻辑：如果请求的数据量大于当前索引，则返回从0开始的数据
        # 这样在初始训练时可以获取足够的数据
        if self.current_index == 0 or limit > self.current_index:
            # 在初始训练阶段，返回尽可能多的数据，最多返回limit条
            return self.data[:min(limit, len(self.data))].values.tolist()
        else:
            # 正常回测过程中，返回当前索引前limit条数据
            return self.data[self.current_index-limit+1:self.current_index+1].values.tolist()
    
    def get_market_price(self, symbol=None):
        """获取当前市场价格"""
        return float(self.data.iloc[self.current_index]['close'])
    
    def get_balance(self):
        """获取账户余额"""
        # 计算当前权益，包括未实现盈亏
        equity = self.balance
        if self.position:
            current_price = self.get_market_price()
            if self.position_amount > 0:  # 多仓
                equity += self.position_amount * (current_price - self.position_entry_price)
            else:  # 空仓
                equity += abs(self.position_amount) * (self.position_entry_price - current_price)
        
        return {
            'free': self.balance,
            'used': 0 if not self.position else abs(self.position_amount) * self.position_entry_price / self.leverage,
            'total': equity
        }
    
    def get_position(self, symbol=None):
        """获取当前持仓"""
        if not self.position or self.position_amount == 0:
            return None
        
        current_price = self.get_market_price()
        
        # 计算未实现盈亏
        if self.position_amount > 0:  # 多仓
            unrealized_pnl = self.position_amount * (current_price - self.position_entry_price)
        else:  # 空仓
            unrealized_pnl = abs(self.position_amount) * (self.position_entry_price - current_price)
        
        # 计算ROE
        position_value = abs(self.position_amount) * self.position_entry_price / self.leverage
        roe = unrealized_pnl / position_value if position_value > 0 else 0
        
        position_info = {
            'info': {
                'symbol': self.symbol,
                'positionAmt': str(self.position_amount),
                'entryPrice': str(self.position_entry_price),
                'unRealizedProfit': str(unrealized_pnl),
                'leverage': str(self.leverage),
                'marginType': 'isolated',
                'isolatedMargin': str(position_value),
                'positionSide': 'BOTH'
            },
            'symbol': self.symbol,
            'notional': abs(self.position_amount) * current_price,
            'unrealizedPnl': unrealized_pnl,
            'leverage': self.leverage,
            'entryPrice': self.position_entry_price,
            'side': 'long' if self.position_amount > 0 else 'short'
        }
        
        return position_info
    
    def place_order(self, symbol=None, side=None, amount=None, order_type=None, price=None, **kwargs):
        """下单"""
        current_price = self.get_market_price()
        timestamp = self.data.iloc[self.current_index]['timestamp']
        
        # 模拟交易执行
        if side.upper() == 'BUY':
            # 检查是否已有空仓，如果有则先平仓
            if self.position and self.position_amount < 0:
                self._close_position()
            
            # 计算可用资金
            available_margin = self.balance * self.leverage
            
            # 检查是否有足够的资金
            if amount * current_price > available_margin:
                self.logger.warning(f"资金不足，可用资金: {available_margin}，需要: {amount * current_price}")
                return None
            
            # 更新余额
            margin_required = amount * current_price / self.leverage
            self.balance -= margin_required
            
            # 更新持仓
            self.position = True
            self.position_entry_time = timestamp
            self.position_entry_price = current_price
            self.position_amount = amount
            
            # 记录交易
            trade = {
                'timestamp': timestamp,
                'side': 'BUY',
                'price': current_price,
                'amount': amount,
                'cost': amount * current_price,
                'realized_pnl': 0
            }
            self.trade_history.append(trade)
            
            self.logger.info(f"开多仓: 价格={current_price}, 数量={amount}, 保证金={margin_required}")
            
            return {'id': len(self.trade_history), 'info': trade}
            
        elif side.upper() == 'SELL':
            # 检查是否已有多仓，如果有则先平仓
            if self.position and self.position_amount > 0:
                self._close_position()
            
            # 计算可用资金
            available_margin = self.balance * self.leverage
            
            # 检查是否有足够的资金
            if amount * current_price > available_margin:
                self.logger.warning(f"资金不足，可用资金: {available_margin}，需要: {amount * current_price}")
                return None
            
            # 更新余额
            margin_required = amount * current_price / self.leverage
            self.balance -= margin_required
            
            # 更新持仓
            self.position = True
            self.position_entry_time = timestamp
            self.position_entry_price = current_price
            self.position_amount = -amount  # 负数表示空仓
            
            # 记录交易
            trade = {
                'timestamp': timestamp,
                'side': 'SELL',
                'price': current_price,
                'amount': amount,
                'cost': amount * current_price,
                'realized_pnl': 0
            }
            self.trade_history.append(trade)
            
            self.logger.info(f"开空仓: 价格={current_price}, 数量={amount}, 保证金={margin_required}")
            
            return {'id': len(self.trade_history), 'info': trade}
        
        return None
    
    def _close_position(self):
        """平仓内部函数"""
        if not self.position or self.position_amount == 0:
            return None
        
        current_price = self.get_market_price()
        timestamp = self.data.iloc[self.current_index]['timestamp']
        
        # 计算已实现盈亏
        if self.position_amount > 0:  # 平多仓
            realized_pnl = self.position_amount * (current_price - self.position_entry_price)
            side = 'SELL'
        else:  # 平空仓
            realized_pnl = abs(self.position_amount) * (self.position_entry_price - current_price)
            side = 'BUY'
        
        # 更新余额
        margin_used = abs(self.position_amount) * self.position_entry_price / self.leverage
        self.balance += margin_used + realized_pnl
        
        # 记录交易
        trade = {
            'timestamp': timestamp,
            'side': side,
            'price': current_price,
            'amount': abs(self.position_amount),
            'cost': abs(self.position_amount) * current_price,
            'realized_pnl': realized_pnl
        }
        self.trade_history.append(trade)
        
        # 记录权益曲线点
        self.equity_curve.append({
            'timestamp': timestamp,
            'equity': self.balance
        })
        
        self.logger.info(f"平仓: 价格={current_price}, 数量={abs(self.position_amount)}, 盈亏={realized_pnl}")
        
        # 清空持仓
        self.position = None
        self.position_entry_time = None
        self.position_entry_price = None
        self.position_amount = 0
        
        return {'id': len(self.trade_history), 'info': trade}
    
    def close_position(self, symbol=None):
        """平仓"""
        return self._close_position()
    
    def open_long(self, symbol=None, amount=None):
        """开多仓"""
        return self.place_order(symbol, 'BUY', amount)
    
    def open_short(self, symbol=None, amount=None):
        """开空仓"""
        return self.place_order(symbol, 'SELL', amount)
    
    def close_long(self, symbol=None, amount=None):
        """平多仓"""
        if not amount:
            amount = self.position_amount if self.position_amount > 0 else 0
        return self.place_order(symbol, 'SELL', amount)
    
    def close_short(self, symbol=None, amount=None):
        """平空仓"""
        if not amount:
            amount = abs(self.position_amount) if self.position_amount < 0 else 0
        return self.place_order(symbol, 'BUY', amount)
    
    def step(self):
        """推进一个时间步"""
        self.current_index += 1
        if self.current_index >= len(self.data):
            return False
        
        # 记录权益曲线
        equity = self.get_balance()['total']
        timestamp = self.data.iloc[self.current_index]['timestamp']
        self.equity_curve.append({
            'timestamp': timestamp,
            'equity': equity
        })
        
        return True
    
    def set_leverage(self, leverage):
        """设置杠杆"""
        self.leverage = leverage
        self.logger.info(f"设置杠杆倍数: {leverage}")
        return True
    
    def set_margin_type(self, margin_type):
        """设置保证金模式"""
        self.logger.info(f"设置保证金模式: {margin_type}")
        return True
    
    def set_position_mode(self, dual_side_position=False):
        """设置持仓模式"""
        self.logger.info(f"设置{'双向' if dual_side_position else '单向'}持仓模式")
        return True
    
    def cancel_all_orders(self, symbol=None):
        """取消所有订单"""
        self.logger.info("取消所有订单")
        return True


class Backtester:
    def __init__(self, symbol, strategy_class, start_date=None, end_date=None, timeframe='1m', initial_balance=10000, leverage=5):
        """初始化回测器
        
        Args:
            symbol: 交易对名称
            strategy_class: 策略类
            start_date: 开始日期，格式"YYYY-MM-DD"
            end_date: 结束日期，格式"YYYY-MM-DD"
            timeframe: K线周期
            initial_balance: 初始资金
            leverage: 杠杆倍数
        """
        self.symbol = symbol
        self.strategy_class = strategy_class
        self.start_date = start_date
        self.end_date = end_date
        self.timeframe = timeframe
        self.initial_balance = initial_balance
        self.leverage = leverage
        
        # 初始化日志
        self.logger = logging.getLogger(f"Backtest_{symbol}")
        
        # 加载环境变量
        load_dotenv()
    
    def fetch_historical_data(self):
        """获取历史数据"""
        try:
            self.logger.info(f"获取 {self.symbol} 历史数据")
            
            # 初始化交易所
            exchange = ccxt.binanceusdm({
                'apiKey': os.getenv('BINANCE_API_KEY'),
                'secret': os.getenv('BINANCE_SECRET_KEY'),
                'enableRateLimit': True,
                'options': {'defaultType': 'future'},
                'proxies': {
                    'http': config.PROXY_URL,
                    'https': config.PROXY_URL
                } if config.USE_PROXY else None,
            })
            
            all_data = []
            current_date = datetime.strptime(self.start_date, "%Y-%m-%d")
            end_date = datetime.strptime(self.end_date, "%Y-%m-%d")
            
            while current_date <= end_date:
                try:
                    since = exchange.parse8601(current_date.strftime("%Y-%m-%d") + "T00:00:00Z")
                    self.logger.info(f"获取 {current_date.strftime('%Y-%m-%d')} 的数据")
                    
                    # 获取当天的K线数据
                    ohlcv = exchange.fetch_ohlcv(
                        symbol=self.symbol,
                        timeframe=self.timeframe,
                        since=since,
                        limit=1500  # CCXT的最大限制
                    )
                    
                    if ohlcv and len(ohlcv) > 0:
                        all_data.extend(ohlcv)
                        self.logger.info(f"获取到 {len(ohlcv)} 条数据")
                        
                        # 获取下一天的时间
                        last_timestamp = ohlcv[-1][0]
                        next_date = datetime.fromtimestamp(last_timestamp / 1000) + timedelta(days=1)
                        next_date = next_date.replace(hour=0, minute=0, second=0, microsecond=0)
                        current_date = next_date
                    else:
                        self.logger.warning(f"当天没有数据，移动到下一天")
                        current_date += timedelta(days=1)
                    
                except Exception as e:
                    self.logger.error(f"获取数据出错: {str(e)}")
                    time.sleep(1)  # 等待一秒再重试
                    current_date += timedelta(days=1)
            
            # 创建DataFrame
            df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # 转换时间戳为日期时间
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            self.logger.info(f"共获取到 {len(df)} 条历史数据")
            
            # 保存数据到本地文件
            os.makedirs('data', exist_ok=True)
            filename = f"data/{self.symbol}_{self.timeframe}_{self.start_date}_{self.end_date}.csv"
            df.to_csv(filename, index=False)
            self.logger.info(f"数据已保存到 {filename}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"获取历史数据失败: {str(e)}")
            return None
    
    def load_data_from_file(self, filename=None):
        """从本地文件加载数据"""
        if not filename:
            filename = f"data/{self.symbol}_{self.timeframe}_{self.start_date}_{self.end_date}.csv"
        
        try:
            df = pd.read_csv(filename)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            self.logger.info(f"从 {filename} 加载了 {len(df)} 条历史数据")
            return df
        except Exception as e:
            self.logger.error(f"加载数据失败: {str(e)}")
            return None
    
    def run(self):
        """运行回测"""
        # 获取历史数据
        try:
            # 尝试从文件加载数据
            filename = f"data/{self.symbol}_{self.timeframe}_{self.start_date}_{self.end_date}.csv"
            if os.path.exists(filename):
                df = self.load_data_from_file(filename)
            else:
                df = self.fetch_historical_data()
            
            if df is None or len(df) == 0:
                self.logger.error("没有历史数据，无法进行回测")
                return None
            
            # 初始化模拟交易者
            trader = MockTrader(self.symbol, df, self.initial_balance, self.leverage)
            
            # 初始化策略
            strategy = self.strategy_class(trader)
            
            # 设置杠杆和保证金模式
            trader.set_leverage(self.leverage)
            trader.set_margin_type('ISOLATED')
            
            # 运行回测
            self.logger.info(f"开始回测: {self.symbol}, 从 {self.start_date} 到 {self.end_date}")
            
            # 进度条
            progress_bar = tqdm(total=len(df), desc="回测进度")
            
            # 运行策略
            while trader.current_index < len(df) - 1:
                # 检查策略是否需要重新训练
                if hasattr(strategy, 'should_retrain') and callable(getattr(strategy, 'should_retrain')):
                    if strategy.should_retrain():
                        # 获取需要的K线数据
                        lookback = strategy.training_lookback if hasattr(strategy, 'training_lookback') else 500
                        klines = trader.get_klines(limit=lookback)
                        
                        # 训练模型
                        if hasattr(strategy, 'train_model') and callable(getattr(strategy, 'train_model')):
                            strategy.train_model(klines)
                
                # 运行策略的监控持仓逻辑
                if hasattr(strategy, 'monitor_position') and callable(getattr(strategy, 'monitor_position')):
                    strategy.monitor_position()
                
                # 推进到下一个时间步
                trader.step()
                progress_bar.update(1)
            
            progress_bar.close()
            
            # 强制平仓最后的持仓
            trader.close_position()
            
            # 计算回测结果
            results = self.calculate_results(trader)
            
            # 保存回测结果
            self.save_results(results)
            
            # 绘制回测结果
            self.plot_results(results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"回测失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def calculate_results(self, trader):
        """计算回测结果"""
        # 基本回测信息
        results = {
            'symbol': self.symbol,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'timeframe': self.timeframe,
            'initial_balance': self.initial_balance,
            'final_balance': trader.balance,
            'profit': trader.balance - self.initial_balance,
            'profit_percent': (trader.balance / self.initial_balance - 1) * 100,
            'leverage': self.leverage,
            'trade_count': len(trader.trade_history),
            'strategy': self.strategy_class.__name__
        }
        
        # 交易统计
        if len(trader.trade_history) > 0:
            # 计算胜率
            profitable_trades = [t for t in trader.trade_history if t.get('realized_pnl', 0) > 0]
            results['win_rate'] = len(profitable_trades) / len(trader.trade_history) * 100
            
            # 计算平均收益
            total_pnl = sum(t.get('realized_pnl', 0) for t in trader.trade_history)
            results['avg_profit'] = total_pnl / len(trader.trade_history)
            
            # 计算最大盈利和最大亏损
            max_profit = max([t.get('realized_pnl', 0) for t in trader.trade_history], default=0)
            max_loss = min([t.get('realized_pnl', 0) for t in trader.trade_history], default=0)
            results['max_profit'] = max_profit
            results['max_loss'] = max_loss
            
            # 计算收益风险比
            if max_loss != 0:
                results['profit_risk_ratio'] = abs(max_profit / max_loss)
            else:
                results['profit_risk_ratio'] = float('inf')
            
            # 计算夏普比率
            if len(trader.equity_curve) > 1:
                equity_series = pd.Series([e['equity'] for e in trader.equity_curve])
                returns = equity_series.pct_change().dropna()
                
                if len(returns) > 0:
                    sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0
                    results['sharpe_ratio'] = sharpe_ratio
                else:
                    results['sharpe_ratio'] = 0
            else:
                results['sharpe_ratio'] = 0
            
            # 计算最大回撤
            if len(trader.equity_curve) > 0:
                equity = np.array([e['equity'] for e in trader.equity_curve])
                max_drawdown, max_drawdown_percent = self.calculate_max_drawdown(equity)
                results['max_drawdown'] = max_drawdown
                results['max_drawdown_percent'] = max_drawdown_percent * 100  # 百分比
            else:
                results['max_drawdown'] = 0
                results['max_drawdown_percent'] = 0
            
            # 计算卡玛比率
            if results['max_drawdown_percent'] > 0:
                results['calmar_ratio'] = results['profit_percent'] / results['max_drawdown_percent']
            else:
                results['calmar_ratio'] = float('inf')
        
        else:
            # 没有交易的默认值
            results.update({
                'win_rate': 0,
                'avg_profit': 0,
                'max_profit': 0,
                'max_loss': 0,
                'profit_risk_ratio': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'max_drawdown_percent': 0,
                'calmar_ratio': 0
            })
        
        # 添加策略相关信息
        results['strategy_name'] = self.strategy_class.__name__
        
        return results
    
    def calculate_max_drawdown(self, equity):
        """计算最大回撤"""
        max_dd = 0
        max_dd_percent = 0
        peak = equity[0]
        
        for value in equity:
            if value > peak:
                peak = value
            
            dd = peak - value
            dd_percent = dd / peak if peak > 0 else 0
            
            if dd > max_dd:
                max_dd = dd
                max_dd_percent = dd_percent
        
        return max_dd, max_dd_percent
    
    def save_results(self, results):
        """保存回测结果"""
        # 创建结果目录
        os.makedirs('backtest_results', exist_ok=True)
        
        # 文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"backtest_results/{self.symbol}_{self.strategy_class.__name__}_{timestamp}.json"
        
        # 保存为JSON文件
        with open(filename, 'w') as f:
            json.dump(results, f, indent=4)
        
        self.logger.info(f"回测结果已保存到 {filename}")
    
    def plot_results(self, results):
        """绘制回测结果"""
        try:
            # 创建图表目录
            os.makedirs('backtest_charts', exist_ok=True)
            
            # 文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"backtest_charts/{self.symbol}_{self.strategy_class.__name__}_{timestamp}.png"
            
            # 创建图表
            plt.figure(figsize=(12, 10))
            
            # 创建子图
            plt.subplot(2, 1, 1)
            plt.title(f"回测结果: {self.symbol} ({self.start_date} - {self.end_date})")
            
            # 绘制权益曲线
            equity_data = pd.DataFrame(results.get('equity_curve', []))
            if not equity_data.empty and 'timestamp' in equity_data.columns and 'equity' in equity_data.columns:
                plt.plot(equity_data['timestamp'], equity_data['equity'], label='权益曲线')
                plt.axhline(y=self.initial_balance, color='r', linestyle='-', label='初始资金')
                plt.grid(True)
                plt.legend()
                plt.ylabel('权益 (USDT)')
            
            # 绘制回撤
            plt.subplot(2, 1, 2)
            plt.title('回撤')
            
            if not equity_data.empty and 'timestamp' in equity_data.columns and 'equity' in equity_data.columns:
                equity = equity_data['equity'].values
                timestamps = equity_data['timestamp'].values
                
                # 计算回撤
                drawdowns = []
                peak = equity[0]
                
                for i, value in enumerate(equity):
                    if value > peak:
                        peak = value
                    
                    dd_percent = (peak - value) / peak * 100 if peak > 0 else 0
                    drawdowns.append(dd_percent)
                
                plt.fill_between(timestamps, drawdowns, 0, color='red', alpha=0.3, label='回撤 (%)')
                plt.grid(True)
                plt.legend()
                plt.ylabel('回撤 (%)')
                plt.xlabel('日期')
            
            # 添加回测结果摘要
            textstr = '\n'.join((
                f"初始资金: {results['initial_balance']} USDT",
                f"最终资金: {results['final_balance']:.2f} USDT",
                f"利润: {results['profit']:.2f} USDT ({results['profit_percent']:.2f}%)",
                f"交易次数: {results['trade_count']}",
                f"胜率: {results.get('win_rate', 0):.2f}%",
                f"平均收益: {results.get('avg_profit', 0):.2f} USDT",
                f"最大盈利: {results.get('max_profit', 0):.2f} USDT",
                f"最大亏损: {results.get('max_loss', 0):.2f} USDT",
                f"最大回撤: {results.get('max_drawdown_percent', 0):.2f}%",
                f"夏普比率: {results.get('sharpe_ratio', 0):.2f}",
                f"卡玛比率: {results.get('calmar_ratio', 0):.2f}",
            ))
            
            plt.figtext(0.15, 0.01, textstr, fontsize=10, va="bottom", ha="left",
                      bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5})
            
            # 保存图表
            plt.tight_layout(rect=[0, 0.1, 1, 0.95])
            plt.savefig(filename)
            
            self.logger.info(f"回测图表已保存到 {filename}")
            
        except Exception as e:
            self.logger.error(f"绘制回测结果失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())


if __name__ == "__main__":
    # 初始化日志记录器
    logger = Logger.get_logger()
    
    # 导入策略类
from strategies.trend_strategy import SimpleTrendStrategy15m

def main():
    """回测系统入口函数"""
    # 初始化日志记录器
    logger = Logger.get_logger()
    
    # 设置回测参数
    symbol = 'PIPPINUSDT'  # 交易对
    strategy_class = SimpleTrendStrategy15m  # 策略类
    start_date = '2024-02-01'  # 开始日期
    end_date = '2024-02-15'    # 结束日期
    timeframe = '5m'           # K线周期
    initial_balance = 10000    # 初始资金
    leverage = 5               # 杠杆倍数
    
    # 初始化回测器
    backtester = Backtester(
        symbol=symbol,
        strategy_class=strategy_class,
        start_date=start_date,
        end_date=end_date,
        timeframe=timeframe,
        initial_balance=initial_balance,
        leverage=leverage
    )
    
    # 运行回测
    results = backtester.run()
    
    # 打印回测结果
    if results:
        logger.info("=" * 50)
        logger.info(f"回测结果: {symbol} ({start_date} - {end_date})")
        logger.info(f"策略: {strategy_class.__name__}")
        logger.info(f"初始资金: {initial_balance} USDT")
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