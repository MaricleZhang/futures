"""
Historical data loader for backtesting
Downloads and caches historical kline data from Binance
"""

import os
import ccxt
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List
import logging
import time


class DataLoader:
    """Historical data loader with caching support"""
    
    def __init__(self, cache_dir: str = None):
        """Initialize data loader
        
        Args:
            cache_dir: Directory to cache CSV files
        """
        self.logger = logging.getLogger(__name__)
        
        if cache_dir is None:
            cache_dir = Path(__file__).parent.parent / 'data' / 'backtest'
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize exchange
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })
        
        self.logger.info(f"Data loader initialized, cache dir: {self.cache_dir}")
    
    def get_cache_path(self, symbol: str, interval: str) -> Path:
        """Get cache file path for symbol and interval"""
        filename = f"{symbol.replace('/', '_')}_{interval}.csv"
        return self.cache_dir / filename
    
    def load_data(
        self,
        symbol: str,
        interval: str,
        start_date: str,
        end_date: str,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """Load historical kline data
        
        Args:
            symbol: Trading symbol (e.g., 'ETH/USDC')
            interval: Timeframe (e.g., '15m', '1h')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            use_cache: Whether to use cached data
            
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        cache_path = self.get_cache_path(symbol, interval)
        
        # Try to load from cache
        if use_cache and cache_path.exists():
            self.logger.info(f"Loading cached data from {cache_path}")
            df = pd.read_csv(cache_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Check if cache covers the requested period
            cache_start = df['timestamp'].min()
            cache_end = df['timestamp'].max()
            
            req_start = pd.to_datetime(start_date)
            req_end = pd.to_datetime(end_date)
            
            if cache_start <= req_start and cache_end >= req_end:
                # Cache covers requested period
                mask = (df['timestamp'] >= req_start) & (df['timestamp'] <= req_end)
                result = df[mask].copy()
                self.logger.info(f"Loaded {len(result)} rows from cache")
                return result
            else:
                self.logger.info("Cache doesn't cover requested period, downloading...")
        
        # Download from exchange
        df = self._download_data(symbol, interval, start_date, end_date)
        
        # Save to cache
        if use_cache:
            df.to_csv(cache_path, index=False)
            self.logger.info(f"Saved data to cache: {cache_path}")
        
        return df
    
    def _download_data(
        self,
        symbol: str,
        interval: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Download historical data from exchange
        
        Args:
            symbol: Trading symbol
            interval: Timeframe
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            
        Returns:
            DataFrame with kline data
        """
        self.logger.info(f"Downloading {symbol} {interval} data from {start_date} to {end_date}")
        
        start_ts = int(pd.to_datetime(start_date).timestamp() * 1000)
        end_ts = int(pd.to_datetime(end_date).timestamp() * 1000)
        
        all_klines = []
        current_ts = start_ts
        
        # Binance limit is 1500 candles per request
        limit = 1500
        
        while current_ts < end_ts:
            try:
                klines = self.exchange.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=interval,
                    since=current_ts,
                    limit=limit
                )
                
                if not klines:
                    break
                
                all_klines.extend(klines)
                current_ts = klines[-1][0] + 1
                
                self.logger.info(f"Downloaded {len(klines)} candles, total: {len(all_klines)}")
                
                # Rate limiting
                time.sleep(0.5)
                
                # Stop if we've reached the end date
                if klines[-1][0] >= end_ts:
                    break
                    
            except Exception as e:
                self.logger.error(f"Error downloading data: {str(e)}")
                raise
        
        # Convert to DataFrame
        df = pd.DataFrame(
            all_klines,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Filter to requested date range
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        df = df[(df['timestamp'] >= start_dt) & (df['timestamp'] <= end_dt)].copy()
        
        # Convert numeric columns
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove duplicates and sort
        df = df.drop_duplicates(subset='timestamp').sort_values('timestamp').reset_index(drop=True)
        
        self.logger.info(f"Downloaded {len(df)} rows of data")
        
        return df
    
    def update_cache(self, symbol: str, interval: str) -> None:
        """Update cache with latest data
        
        Args:
            symbol: Trading symbol
            interval: Timeframe
        """
        cache_path = self.get_cache_path(symbol, interval)
        
        if cache_path.exists():
            # Load existing cache
            df = pd.read_csv(cache_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Get the last timestamp
            last_ts = df['timestamp'].max()
            
            # Download data from last timestamp to now
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (last_ts + timedelta(days=1)).strftime('%Y-%m-%d')
            
            if start_date < end_date:
                new_df = self._download_data(symbol, interval, start_date, end_date)
                
                # Append new data
                df = pd.concat([df, new_df], ignore_index=True)
                df = df.drop_duplicates(subset='timestamp').sort_values('timestamp').reset_index(drop=True)
                
                # Save updated cache
                df.to_csv(cache_path, index=False)
                self.logger.info(f"Updated cache with {len(new_df)} new rows")
        else:
            self.logger.warning(f"Cache file not found: {cache_path}")
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate data quality
        
        Args:
            df: DataFrame to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        if df is None or len(df) == 0:
            self.logger.error("Empty dataframe")
            return False
        
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                self.logger.error(f"Missing column: {col}")
                return False
        
        # Check for NaN values
        if df[required_columns].isnull().any().any():
            self.logger.warning("Data contains NaN values")
            return False
        
        # Check for invalid OHLC relationships
        invalid_ohlc = (df['high'] < df['low']) | (df['high'] < df['open']) | \
                      (df['high'] < df['close']) | (df['low'] > df['open']) | \
                      (df['low'] > df['close'])
        
        if invalid_ohlc.any():
            self.logger.warning(f"Found {invalid_ohlc.sum()} rows with invalid OHLC relationships")
            return False
        
        self.logger.info("Data validation passed")
        return True
