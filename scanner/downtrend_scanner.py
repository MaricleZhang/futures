"""
Binance 15m Downtrend Scanner
Identifies the top 5 trading pairs with the strongest downward trends in 15-minute timeframe.
"""
import os
import time
import pandas as pd
from datetime import datetime
import logging
from trader import Trader
from utils.logger import Logger

# Initialize logger
logger = Logger.get_logger()
logger.info("Starting 15m Downtrend Scanner")

def get_downtrend_pairs(min_consecutive_periods=3, min_volume_usdt=500000):
    """
    Scans all tradable pairs and identifies those with downward trends in 15m timeframe
    
    Args:
        min_consecutive_periods: Minimum consecutive 15m periods with lower closes
        min_volume_usdt: Minimum 24h volume in USDT to consider
        
    Returns:
        list: Trading pairs with downtrend information, sorted by strength
    """
    try:
        # Initialize trader without specific symbol to access all pairs
        trader = Trader()
        logger.info("Trader initialized successfully")
        
        # Get all USDT futures trading pairs
        all_symbols = trader.get_all_symbols()
        logger.info(f"Found {len(all_symbols)} USDT futures trading pairs")
        
        # Filter by volume
        try:
            all_tickers = trader.exchange.fetch_tickers()
            valid_symbols = []
            
            for symbol in all_symbols:
                if symbol in all_tickers:
                    volume_usd = all_tickers[symbol]['quoteVolume']
                    if volume_usd >= min_volume_usdt:
                        valid_symbols.append(symbol)
                        
            logger.info(f"Found {len(valid_symbols)} pairs with 24h volume >= {min_volume_usdt} USDT")
        except Exception as e:
            logger.error(f"Failed to filter by volume: {str(e)}")
            valid_symbols = all_symbols  # Use all symbols if filtering fails
        
        # Store downtrend results
        downtrend_pairs = []
        
        # Process each symbol
        total = len(valid_symbols)
        for i, symbol in enumerate(valid_symbols):
            try:
                logger.info(f"[{i+1}/{total}] Analyzing {symbol}...")
                
                # Get 15m klines
                klines = trader.get_klines(
                    symbol=symbol,
                    interval='15m',
                    limit=50  # Get enough data for analysis
                )
                
                if not klines or len(klines) < min_consecutive_periods + 1:
                    logger.debug(f"Not enough data for {symbol}")
                    continue
                
                # Find the longest consecutive downtrend
                consecutive_down = 0
                max_consecutive = 0
                start_idx = 0
                
                for i in range(1, len(klines)):
                    if float(klines[i][4]) < float(klines[i-1][4]):  # Compare closes
                        consecutive_down += 1
                        if consecutive_down > max_consecutive:
                            max_consecutive = consecutive_down
                            start_idx = i - consecutive_down
                    else:
                        consecutive_down = 0
                
                # If we found a downtrend of at least min_consecutive_periods
                if max_consecutive >= min_consecutive_periods:
                    start_price = float(klines[start_idx][4])
                    end_price = float(klines[start_idx + max_consecutive][4])
                    percent_change = ((end_price - start_price) / start_price) * 100
                    
                    downtrend_pairs.append({
                        'symbol': symbol,
                        'consecutive_periods': max_consecutive,
                        'start_price': start_price,
                        'end_price': end_price,
                        'percent_change': percent_change,
                        'strength': abs(percent_change)
                    })
                    
                    logger.info(f"Found {max_consecutive}-period downtrend for {symbol}: ↓ {percent_change:.2f}%")
                
                # Sleep to avoid API rate limits
                time.sleep(0.2)
                
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {str(e)}")
        
        # Sort by strength and get top pairs
        downtrend_pairs.sort(key=lambda x: x['strength'], reverse=True)
        
        return downtrend_pairs
    
    except Exception as e:
        logger.error(f"Error scanning for downtrends: {str(e)}")
        return []

def save_results(downtrend_pairs, max_results=5):
    """
    Save and display the top downtrend pairs
    
    Args:
        downtrend_pairs: List of pairs with downtrend information
        max_results: Maximum number of results to save/display
    """
    if not downtrend_pairs:
        logger.info("No downtrend pairs found")
        return
    
    # Create results folder if it doesn't exist
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # Limit to top results
    top_pairs = downtrend_pairs[:max_results]
    
    # Create dataframe
    df = pd.DataFrame(top_pairs)
    
    # Format timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save to CSV
    filename = f"results/downtrend_pairs_15m_{timestamp}.csv"
    df.to_csv(filename, index=False)
    
    logger.info(f"Saved {len(top_pairs)} downtrend pairs to {filename}")
    
    # Print summary
    logger.info("=" * 60)
    logger.info(f"TOP {max_results} DOWNTREND PAIRS (15m Timeframe)")
    logger.info("=" * 60)
    
    for i, trend in enumerate(top_pairs, 1):
        logger.info(f"{i}. {trend['symbol']}: ↓ {trend['percent_change']:.2f}% over {trend['consecutive_periods']} periods")
    
    logger.info("=" * 60)
    
    return top_pairs

def main():
    """Main execution function"""
    try:
        logger.info("Scanning for 15m downtrends...")
        
        # Get downtrend pairs
        downtrend_pairs = get_downtrend_pairs(min_consecutive_periods=3, min_volume_usdt=500000)
        
        # Save and display results
        top_pairs = save_results(downtrend_pairs, max_results=5)
        
        logger.info("15m downtrend scanning completed successfully")
        
        return top_pairs
        
    except Exception as e:
        logger.error(f"Error during downtrend scanning: {str(e)}")
        return []

if __name__ == "__main__":
    main()
