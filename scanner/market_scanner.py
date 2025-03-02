import os
import time
import pandas as pd
import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from trader import Trader
from utils.logger import Logger
import talib

# Initialize logging
logger = Logger.get_logger()
logger.info("Starting Market Scanner for High Confidence Trading Opportunities")

class MarketScanner:
    def __init__(self, top_n=5, min_volume_usdt=500000, kline_interval='1h', lookback=72):
        """
        Initialize the market scanner.
        
        Args:
            top_n: Number of top trading pairs to return
            min_volume_usdt: Minimum 24h volume in USDT
            kline_interval: Kline interval to use for analysis
            lookback: Number of klines to look back for analysis
        """
        self.top_n = top_n
        self.min_volume_usdt = min_volume_usdt
        self.kline_interval = kline_interval
        self.lookback = lookback
        self.trader = Trader()  # Base trader for market operations
        self.logger = logging.getLogger("MarketScanner")
        
    def get_all_tradable_symbols(self):
        """Get all USDT futures trading pairs that meet volume requirements"""
        try:
            # Get all USDT futures pairs
            all_symbols = self.trader.get_all_symbols()
            self.logger.info(f"Found {len(all_symbols)} USDT futures trading pairs")
            
            # Get 24h ticker for all symbols
            exchange = self.trader.exchange
            tickers = exchange.fetch_tickers()
            
            # Filter by volume
            valid_symbols = []
            for symbol in all_symbols:
                if symbol in tickers:
                    volume_usd = tickers[symbol]['quoteVolume']
                    if volume_usd >= self.min_volume_usdt:
                        valid_symbols.append(symbol)
                        
            self.logger.info(f"Found {len(valid_symbols)} pairs with 24h volume >= {self.min_volume_usdt} USDT")
            return valid_symbols
        except Exception as e:
            self.logger.error(f"Error getting tradable symbols: {str(e)}")
            return []
            
    def analyze_symbol(self, symbol):
        """
        Analyze a single symbol and return its confidence score.
        
        Returns:
            dict: Symbol analysis results with confidence score
        """
        try:
            # Create a Trader instance for this symbol
            symbol_trader = Trader(symbol)
            
            # Get historical klines
            klines = symbol_trader.get_klines(
                interval=self.kline_interval,
                limit=self.lookback
            )
            
            if not klines or len(klines) < 30:
                self.logger.warning(f"{symbol}: Not enough kline data")
                return None
                
            # Convert to dataframe
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Calculate technical indicators
            result = self.calculate_indicators(df, symbol)
            
            # Wait a bit to avoid API rate limits
            time.sleep(0.1)
            
            return result
        except Exception as e:
            self.logger.error(f"Error analyzing {symbol}: {str(e)}")
            return None
            
    def calculate_indicators(self, df, symbol):
        """
        Calculate technical indicators and return prediction with confidence
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Trading pair symbol
            
        Returns:
            dict: Analysis results
        """
        # Extract price data
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values
        
        # Calculate indicators
        # RSI (14)
        rsi = talib.RSI(close, timeperiod=14)
        
        # MACD
        macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20)
        
        # Stochastic
        slowk, slowd = talib.STOCH(high, low, close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
        
        # ADX
        adx = talib.ADX(high, low, close, timeperiod=14)
        
        # ATR
        atr = talib.ATR(high, low, close, timeperiod=14)
        
        # Calculate percentage distance from price to BBands
        price = close[-1]
        bb_position = (price - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1]) if bb_upper[-1] != bb_lower[-1] else 0.5
        
        # Momentum & trend indicators
        price_sma10 = talib.SMA(close, timeperiod=10)
        price_sma50 = talib.SMA(close, timeperiod=50)
        price_ema10 = talib.EMA(close, timeperiod=10)
        price_ema50 = talib.EMA(close, timeperiod=50)
        
        # Calculate price change rates
        price_change_1d = (close[-1] / close[-2] - 1) * 100 if len(close) > 1 else 0
        price_change_3d = (close[-1] / close[-4] - 1) * 100 if len(close) > 3 else 0
        price_change_7d = (close[-1] / close[-8] - 1) * 100 if len(close) > 7 else 0
        
        # Calculate volatility
        returns = np.diff(np.log(close))
        volatility = np.std(returns) * 100  # Scaled to percentage
        
        # Volume profile
        volume_sma20 = talib.SMA(volume, timeperiod=20)
        volume_ratio = volume[-1] / volume_sma20[-1] if volume_sma20[-1] > 0 else 1
        
        # *************** PREDICTION LOGIC ***************
        
        # Trend signals (bullish / bearish)
        trend_signals = [
            price_ema10[-1] > price_ema50[-1],  # EMA 10 > EMA 50
            macd[-1] > macd_signal[-1],         # MACD > Signal
            rsi[-1] > 50,                        # RSI > 50
            slowk[-1] > slowd[-1],              # Stoch %K > %D
            price > bb_middle[-1]               # Price > BB middle
        ]
        
        # Count bullish signals
        bullish_count = sum(trend_signals)
        
        # Momentum and reversal signals
        overbought = rsi[-1] > 70 or bb_position > 0.9 or slowk[-1] > 80
        oversold = rsi[-1] < 30 or bb_position < 0.1 or slowk[-1] < 20
        
        # Volatility condition
        high_volatility = volatility > 3  # 3% daily volatility is relatively high
        
        # Volume confirmation
        strong_volume = volume_ratio > 1.5
        
        # Trend strength
        strong_trend = adx[-1] > 25
        
        # Determine prediction and confidence
        confidence = 0
        prediction = "NEUTRAL"
        
        # Base confidence on trend signals (0-5 scale)
        if bullish_count >= 4:
            base_confidence = 0.7
            prediction = "UP"
        elif bullish_count <= 1:
            base_confidence = 0.7
            prediction = "DOWN"
        else:
            # 2-3 signals are less certain
            base_confidence = 0.3
            prediction = "UP" if bullish_count > 2.5 else "DOWN"
        
        # Adjust confidence based on other factors
        
        # Reversal patterns - reduce confidence if against trend
        if prediction == "UP" and overbought:
            base_confidence *= 0.7  # Reduce confidence if predicting UP but overbought
        elif prediction == "DOWN" and oversold:
            base_confidence *= 0.7  # Reduce confidence if predicting DOWN but oversold
            
        # Strong reversal patterns - can change prediction
        if prediction == "UP" and bb_position > 0.95 and rsi[-1] > 75:
            prediction = "DOWN"
            base_confidence = 0.6
        elif prediction == "DOWN" and bb_position < 0.05 and rsi[-1] < 25:
            prediction = "UP"
            base_confidence = 0.6
        
        # Increase confidence if trend strength and volume confirm
        if strong_trend:
            base_confidence *= 1.2
            
        if strong_volume:
            base_confidence *= 1.15
        
        # Adjust based on price momentum
        if prediction == "UP" and price_change_3d > 3:
            base_confidence *= 1.1
        elif prediction == "DOWN" and price_change_3d < -3:
            base_confidence *= 1.1
        
        # Cap confidence at 0.95
        confidence = min(base_confidence, 0.95)
        
        # Format confidence as percentage
        confidence_pct = confidence * 100
        
        # Return the analysis result
        return {
            'symbol': symbol,
            'prediction': prediction,
            'confidence': confidence_pct,
            'current_price': price,
            'rsi': rsi[-1],
            'macd_hist': macd_hist[-1],
            'bb_position': bb_position,
            'adx': adx[-1],
            'price_change_1d': price_change_1d,
            'price_change_3d': price_change_3d,
            'volume_ratio': volume_ratio,
            'volatility': volatility
        }
        
    def scan_market(self, max_workers=10):
        """
        Scan all tradable symbols and return top opportunities.
        
        Args:
            max_workers: Maximum number of threads to use
            
        Returns:
            DataFrame: Top opportunities with confidence scores
        """
        # Get all tradable symbols
        symbols = self.get_all_tradable_symbols()
        
        if not symbols:
            self.logger.error("No valid symbols found")
            return pd.DataFrame()
            
        # Analyze all symbols in parallel
        results = []
        
        self.logger.info(f"Analyzing {len(symbols)} symbols with {max_workers} workers")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_symbol = {executor.submit(self.analyze_symbol, symbol): symbol for symbol in symbols}
            
            # Process results as they complete
            for i, future in enumerate(as_completed(future_to_symbol)):
                symbol = future_to_symbol[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                    
                    # Log progress
                    if (i+1) % 10 == 0 or i+1 == len(symbols):
                        self.logger.info(f"Analyzed {i+1}/{len(symbols)} symbols")
                        
                except Exception as e:
                    self.logger.error(f"Error processing {symbol}: {str(e)}")
        
        # Convert to DataFrame
        if not results:
            self.logger.warning("No valid results found")
            return pd.DataFrame()
            
        df_results = pd.DataFrame(results)
        
        # Get top opportunities
        up_opportunities = df_results[df_results['prediction'] == 'UP'].sort_values('confidence', ascending=False).head(self.top_n)
        down_opportunities = df_results[df_results['prediction'] == 'DOWN'].sort_values('confidence', ascending=False).head(self.top_n)
        
        # Combine and sort
        top_opportunities = pd.concat([up_opportunities, down_opportunities])
        top_opportunities = top_opportunities.sort_values('confidence', ascending=False)
        
        return top_opportunities
        
def main():
    # Load environment variables
    load_dotenv()
    
    # Create scanner
    scanner = MarketScanner(
        top_n=5,
        min_volume_usdt=1000000,  # 1M USDT daily volume
        kline_interval='1h',
        lookback=48  # 48 hours of data
    )
    
    # Scan market
    logger.info("Scanning market for high confidence trading opportunities...")
    opportunities = scanner.scan_market(max_workers=5)
    
    if opportunities.empty:
        logger.error("No opportunities found")
        return
    
    # Display results
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    
    print("\n" + "="*80)
    print("TOP TRADING OPPORTUNITIES WITH HIGHEST CONFIDENCE")
    print("="*80)
    
    # Format and display the results
    display_cols = ['symbol', 'prediction', 'confidence', 'current_price', 
                   'rsi', 'adx', 'price_change_1d', 'volume_ratio']
    
    formatted_df = opportunities[display_cols].copy()
    formatted_df['confidence'] = formatted_df['confidence'].round(2).astype(str) + '%'
    formatted_df['rsi'] = formatted_df['rsi'].round(2)
    formatted_df['adx'] = formatted_df['adx'].round(2)
    formatted_df['price_change_1d'] = formatted_df['price_change_1d'].round(2).astype(str) + '%'
    formatted_df['volume_ratio'] = formatted_df['volume_ratio'].round(2)
    
    print(formatted_df.to_string(index=False))
    print("\n" + "="*80)
    
    # Save results to CSV
    output_dir = 'analysis'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    file_path = os.path.join(output_dir, f'trading_opportunities_{timestamp}.csv')
    opportunities.to_csv(file_path, index=False)
    
    print(f"\nResults saved to {file_path}")
    
if __name__ == "__main__":
    main()
