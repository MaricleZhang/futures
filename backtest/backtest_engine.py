"""
Main backtest engine
Coordinates data loading, strategy execution, and performance analysis
"""

import pandas as pd
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import importlib
import sys
from pathlib import Path

from backtest.data_loader import DataLoader
from backtest.backtest_trader import BacktestTrader
from backtest.performance_analyzer import PerformanceAnalyzer


class BacktestEngine:
    """Main backtest engine"""
    
    def __init__(
        self,
        strategy_name: str,
        symbol: str,
        start_date: str,
        end_date: str,
        initial_capital: float = 1000.0,
        leverage: int = 1,
        fee_rate: float = 0.0004,
        slippage_rate: float = 0.0001,
        interval: str = '15m',
        base_interval: str = '1m',
        cache_dir: str = None,
        results_dir: str = None
    ):
        """Initialize backtest engine
        
        Args:
            strategy_name: Name of strategy to backtest
            symbol: Trading symbol (e.g., 'ETH/USDC')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            initial_capital: Initial capital in USDT
            leverage: Leverage multiplier
            fee_rate: Trading fee rate
            slippage_rate: Slippage rate
            slippage_rate: Slippage rate
            interval: Strategy Kline interval (e.g., '15m', '1h')
            base_interval: Base data interval for simulation (e.g., '1m')
            cache_dir: Directory for data cache
            results_dir: Directory for results
        """
        self.logger = logging.getLogger(__name__)
        
        # Parameters
        self.strategy_name = strategy_name
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.leverage = leverage
        self.fee_rate = fee_rate
        self.slippage_rate = slippage_rate
        self.interval = interval
        self.base_interval = base_interval
        
        # Directories
        if cache_dir is None:
            cache_dir = Path(__file__).parent.parent / 'data' / 'backtest'
        if results_dir is None:
            results_dir = Path(__file__).parent.parent / 'results' / 'backtest'
        
        self.cache_dir = Path(cache_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Components
        self.data_loader = DataLoader(cache_dir=cache_dir)
        self.trader = None
        self.strategy = None
        self.analyzer = PerformanceAnalyzer()
        
        # Results
        self.data = None
        self.metrics = None
        
        self.logger.info(f"Backtest engine initialized: {strategy_name} on {symbol}")
        self.logger.info(f"Period: {start_date} to {end_date}")
        self.logger.info(f"Capital: {initial_capital} USDT, Leverage: {leverage}x")
        self.logger.info(f"Fee: {fee_rate:.2%}, Slippage: {slippage_rate:.2%}")
    
    def load_strategy(self):
        """Load strategy class dynamically"""
        self.logger.info(f"Loading strategy: {self.strategy_name}")
        
        # Map strategy names to class names (use actual class names from files)
        strategy_map = {
            'pattern_probability': 'PatternProbabilityStrategy',
            'simple_adx_di': 'SimpleADXDIStrategy15m',  # Note: ADX, DI are caps, has 15m suffix
            'trend_following': 'TrendFollowingStrategy',
            'deepseek': 'DeepSeekTradingStrategy',
            'qwen': 'QwenTradingStrategy',
            'kimi': 'KimiTradingStrategy',
            'dl_lstm': 'DLLSTMStrategy',
        }
        
        strategy_files = {
            'simple_adx_di': 'simple_adx_di_15m_strategy',
            'deepseek': 'deepseek_trading_strategy',
            'qwen': 'qwen_trading_strategy',
            'kimi': 'kimi_trading_strategy',
            'dl_lstm': 'dl_lstm_strategy',
        }
        
        strategy_name_lower = self.strategy_name.lower().replace('_strategy', '').replace('-', '_')
        
        if strategy_name_lower not in strategy_map:
            raise ValueError(f"Unknown strategy: {self.strategy_name}. "
                           f"Available: {list(strategy_map.keys())}")
        
        class_name = strategy_map[strategy_name_lower]
        file_name = strategy_files[strategy_name_lower]
        
        try:
            # Import strategy module
            module = importlib.import_module(f'strategies.{file_name}')
            strategy_class = getattr(module, class_name)
            
            # Initialize strategy with backtest trader and interval
            self.strategy = strategy_class(self.trader, interval=self.interval)
            
            self.logger.info(f"Strategy loaded: {class_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to load strategy: {str(e)}")
            raise
    
    def run(self) -> Dict[str, Any]:
        """Run backtest
        
        Returns:
            Dictionary with backtest results and metrics
        """
        self.logger.info("=" * 70)
        self.logger.info("STARTING BACKTEST")
        self.logger.info("=" * 70)
        
        # Step 1: Load historical data
        self.logger.info("Step 1: Loading historical data...")
        self.data = self.data_loader.load_data(
            symbol=self.symbol,
            interval=self.base_interval,
            start_date=self.start_date,
            end_date=self.end_date
        )
        
        if self.data is None or len(self.data) == 0:
            raise ValueError("No data loaded")
        
        self.logger.info(f"Loaded {len(self.data)} candles")
        
        # Validate data
        if not self.data_loader.validate_data(self.data):
            raise ValueError("Data validation failed")
        
        # Step 2: Initialize backtest trader
        self.logger.info("Step 2: Initializing backtest trader...")
        self.trader = BacktestTrader(
            symbol=self.symbol,
            initial_capital=self.initial_capital,
            leverage=self.leverage,
            fee_rate=self.fee_rate,
            slippage_rate=self.slippage_rate,
            data=self.data,
            base_interval=self.base_interval
        )
        
        # Step 3: Load strategy
        self.logger.info("Step 3: Loading strategy...")
        self.load_strategy()
        
        # Step 4: Run backtest loop
        self.logger.info("Step 4: Running backtest simulation...")
        self._run_backtest_loop()
        
        # Step 5: Analyze results
        self.logger.info("Step 5: Analyzing results...")
        equity_curve = self.trader.get_equity_curve()
        trades = self.trader.get_trades()
        
        self.metrics = self.analyzer.analyze(
            equity_curve=equity_curve,
            trades=trades,
            initial_capital=self.initial_capital,
            start_date=self.start_date,
            end_date=self.end_date
        )
        
        # Step 6: Print summary
        summary = self.analyzer.generate_summary(self.metrics)
        print("\n" + summary + "\n")
        
        self.logger.info("=" * 70)
        self.logger.info("BACKTEST COMPLETED")
        self.logger.info("=" * 70)
        
        return {
            'metrics': self.metrics,
            'equity_curve': equity_curve,
            'trades': trades
        }
    
    def _run_backtest_loop(self):
        """Main backtest simulation loop"""
        total_bars = len(self.data)
        check_interval = getattr(self.strategy, 'check_interval', 300) // 60  # Convert seconds to minutes
        
        # Determine interval in minutes
        interval_minutes = {
            '1m': 1, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '4h': 240, '1d': 1440
        }.get(self.base_interval, 1)
        
        # Skip bars based on check interval
        skip_bars = max(1, check_interval // interval_minutes)
        
        self.logger.info(f"Total bars: {total_bars}, Check every {skip_bars} bars")
        
        # Start from a reasonable index to have enough history
        # Use strategy's lookback_period if available, otherwise default to 100
        lookback = getattr(self.strategy, 'lookback_period', 100)
        start_index = lookback
        
        if total_bars <= start_index:
            error_msg = (f"Insufficient data for backtest: {total_bars} bars loaded, but strategy requires {start_index} bars for warmup. "
                        "Please increase the date range or use a smaller timeframe interval.")
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        for i in range(start_index, total_bars, skip_bars):
            self.trader.set_current_index(i)
            
            # Record equity
            self.trader.record_equity()
            
            # Get strategy signal (strategy will call trader.get_klines internally)
            try:
                # Let strategy monitor position (handles both entry and exit)
                if hasattr(self.strategy, 'monitor_position'):
                    self.strategy.monitor_position()
                else:
                    # Fallback: call generate_signal
                    signal = self.strategy.generate_signal()
                    
                    # Basic position management
                    position = self.trader.get_position()
                    
                    if position is None:
                        # No position, check for entry
                        if signal == 1:  # Buy signal
                            balance = self.trader.get_balance()
                            price = self.trader.get_market_price()
                            amount = (balance['free'] * 0.95) / price  # Use 95% of balance
                            self.trader.open_long(amount=amount)
                        elif signal == -1:  # Sell signal
                            balance = self.trader.get_balance()
                            price = self.trader.get_market_price()
                            amount = (balance['free'] * 0.95) / price
                            self.trader.open_short(amount=amount)
                    else:
                        # Has position, check for exit
                        pos_side = position['info']['positionSide']
                        if (pos_side == 'LONG' and signal == -1) or \
                           (pos_side == 'SHORT' and signal == 1) or \
                           signal == 0:
                            self.trader.close_position()
                
            except Exception as e:
                self.logger.error(f"Error at index {i}: {str(e)}")
                continue
            
            # Progress logging
            if i % (skip_bars * 10) == 0:
                progress = (i - start_index) / (total_bars - start_index) * 100
                current_date = self.data.iloc[i]['timestamp']
                equity = self.trader.equity
                self.logger.info(f"Progress: {progress:.1f}% | {current_date} | Equity: {equity:.2f} USDT")
        
        # Close any remaining position at the end
        if self.trader.get_position() is not None:
            self.logger.info("Closing final position...")
            self.trader.close_position()
        
        # Final equity record
        self.trader.record_equity()
    
    def save_results(self, output_dir: str = None):
        """Save backtest results to files
        
        Args:
            output_dir: Directory to save results (default: results/backtest/{strategy}_{symbol}_{date})
        """
        if output_dir is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = self.results_dir / f"{self.strategy_name}_{self.symbol.replace('/', '_')}_{timestamp}"
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save equity curve
        equity_curve = self.trader.get_equity_curve()
        equity_curve.to_csv(output_path / 'equity_curve.csv', index=False)
        
        # Save trades
        trades_df = pd.DataFrame(self.trader.get_trades())
        if len(trades_df) > 0:
            trades_df.to_csv(output_path / 'trades.csv', index=False)
        
        # Save metrics
        metrics_df = pd.DataFrame([self.metrics])
        metrics_df.to_csv(output_path / 'metrics.csv', index=False)
        
        # Generate visualizations
        try:
            from backtest.visualizer import Visualizer
            visualizer = Visualizer()
            visualizer.create_all_plots(
                equity_curve=equity_curve,
                trades=self.trader.get_trades(),
                initial_capital=self.initial_capital,
                output_dir=str(output_path)
            )
        except Exception as e:
            self.logger.warning(f"Failed to generate visualizations: {str(e)}")
        
        self.logger.info(f"Results saved to: {output_path}")
        
        return str(output_path)

