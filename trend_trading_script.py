import os
import logging
import time
from trader import Trader
from scanner.trend_scanner import scan_for_trends, setup_trader
from trading_manager import TradingManager
from utils.logger import Logger
import config

# Initialize logging
logger = Logger.get_logger()
logger.info("Starting Trend Trading System")

def main():
    try:
        # Step 1: Initialize a base trader for market operations
        base_trader = setup_trader()
        if not base_trader:
            logger.error("Failed to initialize trader")
            return
        
        logger.info("Base trader initialized successfully")
        
        # Step 2: Scan the market for trending pairs
        logger.info("Scanning market for trending pairs...")
        trending_pairs = scan_for_trends(base_trader, min_consecutive_days=1, max_consecutive_days=3)
        
        if not trending_pairs:
            logger.warning("No trending pairs found")
            return
        
        # Step 3: Filter and select top trending pairs
        # Sort by strength (percentage change)
        trending_pairs.sort(key=lambda x: x['strength'], reverse=True)
        
        # Separate up and down trends
        up_trends = [p for p in trending_pairs if p['direction'] == 'up']
        down_trends = [p for p in trending_pairs if p['direction'] == 'down']
        
        # Select top pairs from each direction
        top_n_per_direction = 2  # Number of pairs to select from each direction
        selected_pairs = up_trends[:top_n_per_direction] + down_trends[:top_n_per_direction]
        
        # Sort again by strength
        selected_pairs.sort(key=lambda x: x['strength'], reverse=True)
        selected_pairs = selected_pairs[:5]  # Limit to top 5
        
        if not selected_pairs:
            logger.warning("No suitable trending pairs found")
            return
        
        # Log the selected pairs
        logger.info(f"Selected {len(selected_pairs)} top trending pairs for trading:")
        for pair in selected_pairs:
            direction = "UP" if pair['direction'] == 'up' else "DOWN"
            logger.info(f"  - {pair['symbol']}: {direction} trend, strength: {pair['strength']:.2f}%, duration: {pair['consecutive_days']} days")
        
        # Step 4: Update config with selected pairs
        config.SYMBOLS = [pair['symbol'] for pair in selected_pairs]
        
        # Initialize symbol configs
        for i, symbol in enumerate(config.SYMBOLS):
            pair_info = selected_pairs[i]
            if symbol not in config.SYMBOL_CONFIGS:
                config.SYMBOL_CONFIGS[symbol] = {
                    'leverage': config.DEFAULT_LEVERAGE,
                    'min_notional': 20,
                    'trade_amount_percent': 50,
                    'check_interval': 60,
                }
        
        # Step 5: Initialize trading manager with selected pairs
        trading_manager = TradingManager()
        
        # Step 6: Start trading
        trading_manager.start_trading()
        logger.info(f"Trading system started with {len(config.SYMBOLS)} pairs")
        
        # Step 7: Keep the main thread running
        try:
            while True:
                time.sleep(60)
                # You could add periodic status updates here
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt detected, stopping trading...")
            trading_manager.stop_trading()
    
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
