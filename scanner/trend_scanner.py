import os
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
import logging
import time
from trader import Trader

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('trend_scanner.log')
    ]
)
logger = logging.getLogger()

# Load environment variables
load_dotenv()

def setup_trader():
    """Initialize the trader instance"""
    try:
        trader = Trader()
        logger.info("Trader initialized successfully")
        return trader
    except Exception as e:
        logger.error(f"Failed to initialize trader: {str(e)}")
        return None

def get_all_symbols(trader):
    """Get all USDT futures trading pairs"""
    try:
        symbols = trader.get_all_symbols()
        logger.info(f"Found {len(symbols)} USDT futures trading pairs")
        return symbols
    except Exception as e:
        logger.error(f"Failed to get symbols: {str(e)}")
        return []

def analyze_trend(klines, consecutive_days=3):
    """
    Analyze if a symbol has a consistent trend
    
    Args:
        klines: List of klines data [timestamp, open, high, low, close, volume]
        consecutive_days: Number of consecutive days required to identify a trend
        
    Returns:
        dict: Trend information with direction and strength
    """
    if not klines or len(klines) < consecutive_days + 1:
        return {'has_trend': False}
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['close'] = pd.to_numeric(df['close'])
    
    # Get the last n days
    recent_closes = df['close'].iloc[-consecutive_days-1:].values
    
    # Check for uptrend (each day closes higher than previous)
    is_uptrend = all(recent_closes[i] < recent_closes[i+1] for i in range(consecutive_days))
    
    # Check for downtrend (each day closes lower than previous)
    is_downtrend = all(recent_closes[i] > recent_closes[i+1] for i in range(consecutive_days))
    
    # Calculate trend strength (percentage change)
    if is_uptrend or is_downtrend:
        start_price = recent_closes[0]
        end_price = recent_closes[-1]
        percent_change = ((end_price - start_price) / start_price) * 100
        
        return {
            'has_trend': True,
            'direction': 'up' if is_uptrend else 'down',
            'strength': abs(percent_change),
            'consecutive_days': consecutive_days,
            'start_price': start_price,
            'end_price': end_price,
            'percent_change': percent_change
        }
    
    return {'has_trend': False}

def scan_for_trends(trader, min_consecutive_days=3, max_consecutive_days=7):
    """
    Scan all trading pairs for trends
    
    Args:
        trader: Trader instance
        min_consecutive_days: Minimum number of consecutive days to check
        max_consecutive_days: Maximum number of consecutive days to check
        
    Returns:
        list: List of trading pairs with trend information
    """
    symbols = get_all_symbols(trader)
    trending_pairs = []
    
    total = len(symbols)
    logger.info(f"Scanning {total} trading pairs for trends")
    
    for i, symbol in enumerate(symbols):
        try:
            logger.info(f"[{i+1}/{total}] Analyzing {symbol}...")
            
            # Get daily klines for the symbol
            klines = trader.exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe='15m',
                limit=30  # Get enough data for analysis
            )
            
            if not klines or len(klines) < max_consecutive_days + 1:
                logger.warning(f"Not enough data for {symbol}")
                continue
            
            # Convert klines to expected format
            formatted_klines = []
            for k in klines:
                formatted_klines.append([
                    int(k[0]),  # timestamp
                    float(k[1]),  # open
                    float(k[2]),  # high
                    float(k[3]),  # low
                    float(k[4]),  # close
                    float(k[5])   # volume
                ])
            
            # Try different consecutive day counts
            best_trend = {'has_trend': False, 'strength': 0}
            
            for days in range(min_consecutive_days, max_consecutive_days + 1):
                trend_info = analyze_trend(formatted_klines, days)
                
                if trend_info['has_trend'] and trend_info['strength'] > best_trend.get('strength', 0):
                    best_trend = trend_info
            
            if best_trend['has_trend']:
                # Add symbol information to the trend
                best_trend['symbol'] = symbol
                trending_pairs.append(best_trend)
                
                # Log the found trend
                direction_str = "↑" if best_trend['direction'] == 'up' else "↓"
                logger.info(f"Found {best_trend['consecutive_days']}-day {best_trend['direction']} trend for {symbol}: {direction_str} {best_trend['percent_change']:.2f}%")
            
            # Sleep to avoid hitting rate limits
            time.sleep(0.5)
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {str(e)}")
    
    # Sort by trend strength
    trending_pairs.sort(key=lambda x: x['strength'], reverse=True)
    
    return trending_pairs

def save_results(trending_pairs):
    """Save trending pairs to CSV file"""
    if not trending_pairs:
        logger.info("No trending pairs found")
        return
    
    # Create results folder if it doesn't exist
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # Create dataframe from trending pairs
    df = pd.DataFrame(trending_pairs)
    
    # Format timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save to CSV
    filename = f"results/trending_pairs_{timestamp}.csv"
    df.to_csv(filename, index=False)
    
    # Also save separate files for uptrends and downtrends
    up_trends = df[df['direction'] == 'up']
    down_trends = df[df['direction'] == 'down']
    
    if not up_trends.empty:
        up_filename = f"results/uptrends_{timestamp}.csv"
        up_trends.to_csv(up_filename, index=False)
        logger.info(f"Saved {len(up_trends)} uptrends to {up_filename}")
    
    if not down_trends.empty:
        down_filename = f"results/downtrends_{timestamp}.csv"
        down_trends.to_csv(down_filename, index=False)
        logger.info(f"Saved {len(down_trends)} downtrends to {down_filename}")
    
    logger.info(f"Saved {len(trending_pairs)} trending pairs to {filename}")
    
    # Print summary
    print_summary(trending_pairs)

def print_summary(trending_pairs):
    """Print summary of trending pairs"""
    if not trending_pairs:
        logger.info("No trending pairs found")
        return
    
    # Count of uptrends and downtrends
    uptrends = [pair for pair in trending_pairs if pair['direction'] == 'up']
    downtrends = [pair for pair in trending_pairs if pair['direction'] == 'down']
    
    logger.info("=" * 50)
    logger.info(f"TREND ANALYSIS SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Total trending pairs: {len(trending_pairs)}")
    logger.info(f"Uptrends: {len(uptrends)}")
    logger.info(f"Downtrends: {len(downtrends)}")
    logger.info("=" * 50)
    
    # Print top 5 uptrends
    if uptrends:
        logger.info("TOP 5 UPTRENDS:")
        for i, trend in enumerate(uptrends[:5]):
            logger.info(f"{i+1}. {trend['symbol']}: ↑ {trend['percent_change']:.2f}% over {trend['consecutive_days']} days")
    
    # Print top 5 downtrends
    if downtrends:
        logger.info("TOP 5 DOWNTRENDS:")
        for i, trend in enumerate(downtrends[:5]):
            logger.info(f"{i+1}. {trend['symbol']}: ↓ {trend['percent_change']:.2f}% over {trend['consecutive_days']} days")
    
    logger.info("=" * 50)

def main():
    """Main function"""
    logger.info("Starting Binance Futures Trend Scanner")
    
    # Initialize trader
    trader = setup_trader()
    if not trader:
        return
    
    try:
        # Scan for trends
        trending_pairs = scan_for_trends(trader, min_consecutive_days=3, max_consecutive_days=7)
        
        # Save results
        save_results(trending_pairs)
        
        logger.info("Trend scanning completed successfully")
        
    except Exception as e:
        logger.error(f"Error during trend scanning: {str(e)}")
    
if __name__ == "__main__":
    main()
