"""
Backtest runner script
Run backtests on strategies with specified parameters

Usage:
    python run_backtest.py --strategy kama_roc_adx --symbol ETHUSDC --start_date 2025-10-01 --end_date 2025-11-27
"""

import argparse
import sys
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from backtest.backtest_engine import BacktestEngine
from utils.logger import Logger
import config


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run backtest on futures trading strategy')
    
    parser.add_argument('--strategy', type=str, required=True,
                       help='Strategy name (e.g., kama_roc_adx, pattern_probability)')
    parser.add_argument('--symbol', type=str, default='ETHUSDC',
                       help='Trading symbol (default: ETHUSDC)')
    parser.add_argument('--start_date', type=str, required=True,
                       help='Start date in YYYY-MM-DD format')
    parser.add_argument('--end_date', type=str, required=True,
                       help='End date in YYYY-MM-DD format')
    parser.add_argument('--initial_capital', type=float, default=1000,
                       help='Initial capital in USDT (default: 1000)')
    parser.add_argument('--leverage', type=int, default=10,
                       help='Leverage multiplier (default: 10)')
    parser.add_argument('--fee_rate', type=float, default=0.0004,
                       help='Trading fee rate (default: 0.0004 = 0.04%%)')
    parser.add_argument('--slippage', type=float, default=0.0001,
                       help='Slippage rate (default: 0.0001 = 0.01%%)')
    parser.add_argument('--interval', type=str, default='15m',
                       help='Kline interval (default: 15m)')
    parser.add_argument('--base_interval', type=str, default='1m',
                       help='Base data interval for simulation (default: 1m)')
    parser.add_argument('--no_cache', action='store_true',
                       help='Disable data caching')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for results')
    
    return parser.parse_args()


def main():
    """Main entry point"""
    # Parse arguments
    args = parse_args()
    
    # Initialize logger
    logger = Logger.get_logger()
    
    # Convert symbol format if needed (ETH/USDC for ccxt)
    if '/' not in args.symbol:
        # Convert ETHUSDC to ETH/USDC
        if args.symbol.endswith('USDT'):
            base = args.symbol[:-4]
            quote = 'USDT'
        elif args.symbol.endswith('USDC'):
            base = args.symbol[:-4]
            quote = 'USDC'
        else:
            raise ValueError(f"Unsupported symbol format: {args.symbol}")
        
        symbol = f"{base}/{quote}"
    else:
        symbol = args.symbol
    
    logger.info("=" * 80)
    logger.info("BACKTEST CONFIGURATION")
    logger.info("=" * 80)
    logger.info(f"Strategy:        {args.strategy}")
    logger.info(f"Symbol:          {symbol}")
    logger.info(f"Period:          {args.start_date} to {args.end_date}")
    logger.info(f"Initial Capital: {args.initial_capital:,.2f} USDT")
    logger.info(f"Leverage:        {args.leverage}x")
    logger.info(f"Fee Rate:        {args.fee_rate:.4f} ({args.fee_rate * 100:.2f}%)")
    logger.info(f"Slippage:        {args.slippage:.4f} ({args.slippage * 100:.2f}%)")
    logger.info(f"Interval:        {args.interval}")
    logger.info(f"Base Interval:   {args.base_interval}")
    logger.info("=" * 80)
    
    try:
        # Create backtest engine
        engine = BacktestEngine(
            strategy_name=args.strategy,
            symbol=symbol,
            start_date=args.start_date,
            end_date=args.end_date,
            initial_capital=args.initial_capital,
            leverage=args.leverage,
            fee_rate=args.fee_rate,
            slippage_rate=args.slippage,
            interval=args.interval,
            base_interval=args.base_interval
        )
        
        # Run backtest
        results = engine.run()
        
        # Save results
        output_path = engine.save_results(args.output_dir)
        logger.info(f"\n✅ Backtest completed successfully!")
        logger.info(f"📊 Results saved to: {output_path}")
        
        # Print key metrics
        metrics = results['metrics']
        logger.info("\n" + "=" * 80)
        logger.info("KEY METRICS")
        logger.info("=" * 80)
        logger.info(f"Total Return:     {metrics['total_return']:.2f}%")
        logger.info(f"Max Drawdown:     {metrics['max_drawdown']:.2f}%")
        logger.info(f"Sharpe Ratio:     {metrics['sharpe_ratio']:.2f}")
        logger.info(f"Win Rate:         {metrics['win_rate']:.2f}%")
        logger.info(f"Total Trades:     {metrics['total_trades']}")
        logger.info("=" * 80)
        
        return 0
        
    except Exception as e:
        logger.error(f"\n❌ Backtest failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == '__main__':
    sys.exit(main())
