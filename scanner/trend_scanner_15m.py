"""
Trend scanner module - identifies trending pairs without excessive volume
"""
import logging
import pandas as pd
import numpy as np
import talib
import traceback

def setup_trader():
    """Initialize a trader instance for market scanning"""
    try:
        # Initialize without specific symbol
        from trader import Trader
        trader = Trader()
        return trader
    except Exception as e:
        logging.error(f"Failed to initialize trader: {str(e)}")
        return None

def analyze_trend(klines, period=15):
    """
    Analyze if a trading pair has a trend based on klines data
    
    Args:
        klines: K-line data [[timestamp, open, high, low, close, volume], ...]
        period: Number of recent candles to analyze
        
    Returns:
        dict: Trend information including direction, strength, etc.
    """
    try:
        # Convert to DataFrame
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])
        
        # Calculate indicators
        # 1. Moving Averages
        df['ema_short'] = talib.EMA(df['close'].values, timeperiod=8)
        df['ema_medium'] = talib.EMA(df['close'].values, timeperiod=21)
        df['ema_long'] = talib.EMA(df['close'].values, timeperiod=50)
        
        # 2. RSI
        df['rsi'] = talib.RSI(df['close'].values, timeperiod=14)
        
        # 3. ADX (Trend Strength)
        df['adx'] = talib.ADX(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
        df['plus_di'] = talib.PLUS_DI(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
        df['minus_di'] = talib.MINUS_DI(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
        
        # 4. MACD
        macd, signal, hist = talib.MACD(df['close'].values, fastperiod=12, slowperiod=26, signalperiod=9)
        df['macd'] = macd
        df['macd_signal'] = signal
        df['macd_hist'] = hist
        
        # 5. Volume analysis
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Analyze recent candles
        recent = df.tail(period)
        
        # Calculate percentage change
        price_change = (recent['close'].iloc[-1] / recent['close'].iloc[0] - 1) * 100
        
        # Determine trend direction based on DI+ and DI-
        current_plus_di = recent['plus_di'].iloc[-1]
        current_minus_di = recent['minus_di'].iloc[-1]
        
        direction = "neutral"
        if current_plus_di > current_minus_di and current_plus_di > 20 and price_change > 1.0:
            direction = "up"
        elif current_minus_di > current_plus_di and current_minus_di > 20 and price_change < -1.0:
            direction = "down"
        
        # Check the alignment of EMAs for trend confirmation
        ema_aligned = False
        if direction == "up" and (recent['ema_short'] > recent['ema_medium']).all() and (recent['ema_medium'] > recent['ema_long']).all():
            ema_aligned = True
        elif direction == "down" and (recent['ema_short'] < recent['ema_medium']).all() and (recent['ema_medium'] < recent['ema_long']).all():
            ema_aligned = True
        
        # Check consistency of trend
        up_candles = (recent['close'] > recent['open']).sum()
        down_candles = (recent['close'] < recent['open']).sum()
        consistency = max(up_candles, down_candles) / len(recent)
        
        # Check volume (looking for normal volume, not excessive)
        recent_volume_ratio = recent['volume_ratio'].iloc[-5:].mean()  # Average of last 5 candles
        max_volume_ratio = recent['volume_ratio'].iloc[-5:].max()      # Maximum of last 5 candles
        
        # Define normal volume as:
        # 1. Average ratio below 1.5 (less than 150% of 20-period average)
        # 2. Maximum ratio below 2.5 (no single candle with more than 250% volume)
        is_volume_normal = recent_volume_ratio < 1.5 and max_volume_ratio < 2.5
        
        # Calculate trend strength
        strength = abs(price_change)
        adx_strength = recent['adx'].iloc[-1]
        
        # Is it a valid trend without excessive volume?
        is_valid_trend = (
            direction != "neutral" and 
            consistency > 0.6 and 
            is_volume_normal and 
            adx_strength > 20 and
            ema_aligned
        )
        
        return {
            "direction": direction,
            "strength": strength,
            "consistency": consistency,
            "adx": adx_strength,
            "volume_ratio": recent_volume_ratio,
            "max_volume_ratio": max_volume_ratio,
            "is_valid": is_valid_trend,
            "ema_aligned": ema_aligned
        }
        
    except Exception as e:
        logging.error(f"Error analyzing trend: {str(e)}")
        logging.error(traceback.format_exc())
        return {"direction": "neutral", "is_valid": False}

def scan_for_trends(trader, interval='15m', limit=100, min_strength=1.0, max_volume_ratio=1.5):
    """
    Scan all trading pairs for trends without excessive volume
    
    Args:
        trader: Trader instance
        interval: K-line interval (e.g., '15m')
        limit: Number of k-lines to analyze
        min_strength: Minimum percentage change to consider
        max_volume_ratio: Maximum acceptable volume ratio (to filter out pairs with excessive volume)
        
    Returns:
        list: Trading pairs with trends
    """
    try:
        logger = logging.getLogger()
        logger.info("Starting market scan for trend trading pairs without excessive volume...")
        
        # Get all trading pairs
        all_symbols = trader.get_all_symbols()
        logger.info(f"Found {len(all_symbols)} trading pairs")
        
        trending_pairs = []
        
        # Analyze each pair
        for i, symbol in enumerate(all_symbols):
            try:
                logger.info(f"Analyzing {symbol} ({i+1}/{len(all_symbols)})")
                
                # Get k-line data
                klines = trader.get_klines(symbol=symbol, interval=interval, limit=limit)
                
                if not klines or len(klines) < limit * 0.9:  # Ensure we have enough data
                    logger.warning(f"Insufficient data for {symbol}")
                    continue
                
                # Analyze trend
                trend_info = analyze_trend(klines)
                
                # If valid trend with sufficient strength and normal volume
                if (trend_info["is_valid"] and 
                    trend_info["strength"] >= min_strength and 
                    trend_info["volume_ratio"] <= max_volume_ratio):
                    
                    pair_info = {
                        "symbol": symbol,
                        "direction": trend_info["direction"],
                        "strength": trend_info["strength"],
                        "adx": trend_info["adx"],
                        "volume_ratio": trend_info["volume_ratio"],
                        "consistency": trend_info["consistency"]
                    }
                    trending_pairs.append(pair_info)
                    logger.info(f"Found trending pair: {symbol} - {trend_info['direction']} trend, "
                               f"strength: {trend_info['strength']:.2f}%, ADX: {trend_info['adx']:.2f}, "
                               f"Volume ratio: {trend_info['volume_ratio']:.2f}")
            
            except Exception as e:
                logger.error(f"Error processing {symbol}: {str(e)}")
                continue
        
        logger.info(f"Scan complete. Found {len(trending_pairs)} trending pairs")
        return trending_pairs
        
    except Exception as e:
        logger.error(f"Market scan failed: {str(e)}")
        logger.error(traceback.format_exc())
        return []
