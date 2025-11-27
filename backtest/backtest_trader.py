"""
Backtest trader that simulates trading without making real orders
Implements the same interface as Trader for strategy compatibility
"""

import pandas as pd
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
import copy


class BacktestTrader:
    """Simulated trader for backtesting"""
    
    def __init__(
        self,
        symbol: str,
        initial_capital: float = 1000.0,
        leverage: int = 1,
        fee_rate: float = 0.0004,  # 0.04% Binance futures taker fee
        slippage_rate: float = 0.0001,  # 0.01% slippage
        data: pd.DataFrame = None
    ):
        """Initialize backtest trader
        
        Args:
            symbol: Trading symbol
            initial_capital: Initial capital in USDT
            leverage: Leverage multiplier
            fee_rate: Trading fee rate
            slippage_rate: Slippage rate for market orders
            data: Historical kline data
        """
        self.logger = logging.getLogger(f"BacktestTrader.{symbol}")
        self.symbol = symbol
        self.initial_capital = initial_capital
        self.leverage = leverage
        self.fee_rate = fee_rate
        self.slippage_rate = slippage_rate
        
        # Account state
        self.balance = initial_capital
        self.equity = initial_capital
        self.position = None  # Current position
        
        # Historical data
        self.data = data
        self.current_index = 0
        
        # Trading records
        self.trades = []
        self.equity_curve = []
        
        # Symbol config (for compatibility with strategies)
        self.symbol_config = {
            'leverage': leverage,
            'min_notional': 20,
            'trade_amount_percent': 100,
            'check_interval': 60
        }
        
        self.logger.info(f"Backtest trader initialized: {symbol}, " 
                        f"Capital={initial_capital}, Leverage={leverage}x, "
                        f"Fee={fee_rate:.2%}, Slippage={slippage_rate:.2%}")
    
    def set_data(self, data: pd.DataFrame) -> None:
        """Set historical data for backtesting"""
        self.data = data
        self.current_index = 0
    
    def set_current_index(self, index: int) -> None:
        """Set current time index"""
        self.current_index = index
    
    def get_current_price(self) -> float:
        """Get current market price"""
        if self.data is None or self.current_index >= len(self.data):
            raise ValueError("No data available or invalid index")
        return float(self.data.iloc[self.current_index]['close'])
    
    def get_market_price(self, symbol: str = None) -> float:
        """Get current market price (compatibility method)"""
        return self.get_current_price()
    
    def get_klines(self, symbol: str = None, interval: str = '15m', limit: int = 100) -> List[List]:
        """Get historical klines up to current time
        
        Args:
            symbol: Trading symbol (ignored, uses self.symbol)
            interval: Interval (ignored, uses loaded data)
            limit: Number of candles to return
            
        Returns:
            List of klines in format: [timestamp, open, high, low, close, volume]
        """
        if self.data is None:
            return []
        
        # Get data up to current index
        end_idx = self.current_index + 1
        
        # If requested interval matches data interval, return as is
        # We assume data interval is 1m if not specified, or we can infer it
        # For now, we'll check if resampling is needed based on the requested interval
        
        # Simple check: if we have high frequency data (e.g. 1m) and request is for 15m
        # We need to resample.
        
        # Get the slice of data up to current time
        df_slice = self.data.iloc[:end_idx].copy()
        
        if df_slice.empty:
            return []
            
        # If interval is different from base data interval (assumed 1m if we are here), resample
        # Note: This is a simplified check. Ideally we should know the base interval.
        # But since we are implementing "Option 2", we assume base is lower than requested.
        
        if interval and interval != '1m': # Assuming base is 1m for now, or at least lower
             try:
                # Resample
                # Ensure index is datetime
                if not isinstance(df_slice.index, pd.DatetimeIndex):
                    df_slice.set_index('timestamp', inplace=True)
                
                # Map interval string to pandas offset alias
                offset_map = {
                    '1m': '1T', '3m': '3T', '5m': '5T', '15m': '15T', '30m': '30T',
                    '1h': '1H', '2h': '2H', '4h': '4H', '6h': '6H', '8h': '8H', '12h': '12H',
                    '1d': '1D', '3d': '3D', '1w': '1W', '1M': '1M'
                }
                rule = offset_map.get(interval, interval)
                
                # Resample logic:
                # Open: first
                # High: max
                # Low: min
                # Close: last
                # Volume: sum
                resampled = df_slice.resample(rule).agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                })
                
                # Drop incomplete last bin if it doesn't cover the full period?
                # Actually, for backtesting, we want the *forming* candle as well if we are "inside" it?
                # But usually strategies work on *closed* candles.
                # If we are at 10:07 (1m data), and we ask for 15m candles.
                # We have 10:00-10:15 candle forming.
                # The strategy usually expects the last *completed* candle + maybe current forming.
                # Let's keep all, but drop NaN (which happens if no data in bin)
                resampled.dropna(inplace=True)
                
                # Slice the last 'limit' candles
                if len(resampled) > limit:
                    resampled = resampled.iloc[-limit:]
                
                # Convert to list format
                klines = []
                for time_idx, row in resampled.iterrows():
                    klines.append([
                        int(time_idx.timestamp() * 1000),
                        float(row['open']),
                        float(row['high']),
                        float(row['low']),
                        float(row['close']),
                        float(row['volume'])
                    ])
                return klines
                
             except Exception as e:
                 self.logger.error(f"Resampling failed: {e}")
                 # Fallback to raw data if resampling fails
                 pass

        # Fallback / No resampling needed (or failed)
        # Just take the last 'limit' rows
        start_idx = max(0, end_idx - limit)
        df_slice = self.data.iloc[start_idx:end_idx]
        
        # Convert to list format
        klines = []
        for _, row in df_slice.iterrows():
            klines.append([
                int(row['timestamp'].timestamp() * 1000) if isinstance(row['timestamp'], pd.Timestamp) else int(row['timestamp']),
                float(row['open']),
                float(row['high']),
                float(row['low']),
                float(row['close']),
                float(row['volume'])
            ])
        
        return klines
    
    def get_balance(self, symbol: str = None) -> Dict[str, float]:
        """Get account balance
        
        Returns:
            Dictionary with 'free' and 'total' balance
        """
        return {
            'free': self.balance,
            'total': self.equity,
            'used': self.equity - self.balance
        }
    
    def get_position(self, symbol: str = None) -> Optional[Dict[str, Any]]:
        """Get current position
        
        Returns:
            Position info dict or None if no position
        """
        if self.position is None:
            return None
        
        # Format similar to ccxt position structure
        # Ensure positionAmt is available at top level for strategy compatibility
        position_amt = self.position['amount'] if self.position['side'] == 'LONG' else -self.position['amount']
        
        return {
            'symbol': self.symbol,
            'side': self.position['side'],
            'contracts': self.position['amount'],
            'contractSize': 1,
            'entryPrice': self.position['entry_price'],
            'markPrice': self.get_current_price(),
            'notional': self.position['amount'] * self.get_current_price(),
            'leverage': self.leverage,
            'unrealizedPnl': self._calculate_unrealized_pnl(),
            'positionAmt': position_amt,  # Added for compatibility
            'info': {
                'positionAmt': str(position_amt),
                'entryPrice': str(self.position['entry_price']),
                'positionSide': self.position['side']
            }
        }
    
    def _calculate_unrealized_pnl(self) -> float:
        """Calculate unrealized profit/loss"""
        if self.position is None:
            return 0.0
        
        current_price = self.get_current_price()
        entry_price = self.position['entry_price']
        amount = self.position['amount']
        side = self.position['side']
        
        if side == 'LONG':
            pnl = amount * (current_price - entry_price)
        else:  # SHORT
            pnl = amount * (entry_price - current_price)
        
        return pnl
    
    def place_order(
        self,
        symbol: str = None,
        side: str = None,
        amount: float = None,
        order_type: str = 'market',
        price: float = None,
        stop_loss: float = None,
        take_profit: float = None,
        reduce_only: bool = False
    ) -> Dict[str, Any]:
        """Simulate placing an order
        
        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            amount: Order amount
            order_type: Order type (only 'market' supported)
            price: Limit price (not used for market orders)
            stop_loss: Stop loss price (not implemented)
            take_profit: Take profit price (not implemented)
            reduce_only: If True, do not check/deduct margin (for closing positions)
            
        Returns:
            Simulated order info
        """
        if side is None or amount is None:
            raise ValueError("Side and amount are required")
            
        if amount <= 0:
            raise ValueError("Amount must be positive")
        
        current_price = self.get_current_price()
        
        # Apply slippage for market orders
        if side.lower() == 'buy':
            execution_price = current_price * (1 + self.slippage_rate)
        else:
            execution_price = current_price * (1 - self.slippage_rate)
        
        # Calculate order value
        order_value = amount * execution_price
        
        # Calculate fee
        fee = order_value * self.fee_rate
        
        # Update balance only if NOT reduce_only
        if not reduce_only:
            required_margin = order_value / self.leverage
            if required_margin + fee > self.balance:
                raise ValueError(f"Insufficient balance: need {required_margin + fee:.2f}, have {self.balance:.2f}")
            self.balance -= (required_margin + fee)
        
        timestamp = self.data.iloc[self.current_index]['timestamp']
        
        order = {
            'id': f"backtest_{len(self.trades)}",
            'symbol': symbol or self.symbol,
            'type': order_type,
            'side': side.upper(),
            'price': execution_price,
            'amount': amount,
            'cost': order_value,
            'fee': {'cost': fee, 'currency': 'USDT'},
            'timestamp': timestamp,
            'datetime': str(timestamp)
        }
        
        self.logger.info(f"Order executed: {side.upper()} {amount:.6f} @ {execution_price:.2f}, "
                        f"Fee: {fee:.4f} USDT")
        
        return order
    
    def open_long(self, symbol: str = None, amount: float = None) -> Dict[str, Any]:
        """Open long position"""
        if self.position is not None:
            raise ValueError("Position already exists, close it first")
        
        order = self.place_order(symbol, 'buy', amount)
        
        self.position = {
            'side': 'LONG',
            'amount': amount,
            'entry_price': order['price'],
            'entry_time': order['timestamp'],
            'entry_fee': order['fee']['cost'],
            'margin': (amount * order['price']) / self.leverage
        }
        
        self.logger.info(f"Opened LONG position: {amount:.6f} @ {order['price']:.2f}")
        return order
    
    def open_short(self, symbol: str = None, amount: float = None) -> Dict[str, Any]:
        """Open short position"""
        if self.position is not None:
            raise ValueError("Position already exists, close it first")
        
        order = self.place_order(symbol, 'sell', amount)
        
        self.position = {
            'side': 'SHORT',
            'amount': amount,
            'entry_price': order['price'],
            'entry_time': order['timestamp'],
            'entry_fee': order['fee']['cost'],
            'margin': (amount * order['price']) / self.leverage
        }
        
        self.logger.info(f"Opened SHORT position: {amount:.6f} @ {order['price']:.2f}")
        return order
    
    def close_position(self, symbol: str = None) -> Dict[str, Any]:
        """Close current position"""
        if self.position is None:
            self.logger.warning("No position to close")
            return None
        
        # Determine close side
        if self.position['side'] == 'LONG':
            close_side = 'sell'
        else:
            close_side = 'buy'
        
        amount = self.position['amount']
        # Use reduce_only=True to avoid margin check/deduction
        order = self.place_order(symbol, close_side, amount, reduce_only=True)
        
        # Calculate profit/loss
        entry_price = self.position['entry_price']
        exit_price = order['price']
        
        if self.position['side'] == 'LONG':
            pnl = amount * (exit_price - entry_price)
        else:  # SHORT
            pnl = amount * (entry_price - exit_price)
        
        # Fees
        entry_fee = self.position['entry_fee']
        exit_fee = order['fee']['cost']
        total_fee = entry_fee + exit_fee
        net_pnl = pnl - total_fee
        
        # Return margin to balance
        self.balance += self.position['margin']
        
        # Add PnL (excluding fees)
        # We already paid entry_fee. We need to pay exit_fee.
        # Balance change = PnL - exit_fee
        self.balance += (pnl - exit_fee)
        
        # Update equity
        self.equity = self.balance
        
        # Calculate return rate
        initial_value = self.position['margin']
        return_rate = (net_pnl / initial_value) * 100 if initial_value > 0 else 0
        
        # Record trade
        trade = {
            'symbol': symbol or self.symbol,
            'side': self.position['side'],
            'entry_time': self.position['entry_time'],
            'exit_time': order['timestamp'],
            'entry_price': entry_price,
            'exit_price': exit_price,
            'amount': amount,
            'leverage': self.leverage,
            'pnl': pnl,
            'fee': total_fee,
            'net_pnl': net_pnl,
            'return_rate': return_rate,
            'balance_after': self.balance,
            'equity_after': self.equity
        }
        
        self.trades.append(trade)
        
        self.logger.info(f"Closed {self.position['side']} position: "
                        f"Entry={entry_price:.2f}, Exit={exit_price:.2f}, "
                        f"PNL={net_pnl:.2f} USDT ({return_rate:.2f}%)")
        
        # Clear position
        self.position = None
        
        return order

    
    def record_equity(self) -> None:
        """Record current equity for equity curve"""
        # Calculate equity (balance + unrealized PnL)
        unrealized_pnl = self._calculate_unrealized_pnl()
        current_equity = self.balance + unrealized_pnl
        
        # Add margin back if in position
        if self.position is not None:
            current_equity += self.position['margin']
        
        self.equity = current_equity
        
        timestamp = self.data.iloc[self.current_index]['timestamp']
        self.equity_curve.append({
            'timestamp': timestamp,
            'equity': current_equity,
            'balance': self.balance,
            'unrealized_pnl': unrealized_pnl
        })
    
    def get_equity_curve(self) -> pd.DataFrame:
        """Get equity curve as DataFrame"""
        return pd.DataFrame(self.equity_curve)
    
    def get_trades(self) -> List[Dict[str, Any]]:
        """Get all completed trades"""
        return self.trades
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get trading statistics"""
        if not self.trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'total_return': 0,
                'avg_return': 0
            }
        
        winning_trades = [t for t in self.trades if t['net_pnl'] > 0]
        losing_trades = [t for t in self.trades if t['net_pnl'] <= 0]
        
        total_pnl = sum(t['net_pnl'] for t in self.trades)
        total_return = ((self.equity - self.initial_capital) / self.initial_capital) * 100
        
        return {
            'total_trades': len(self.trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': (len(winning_trades) / len(self.trades)) * 100,
            'total_pnl': total_pnl,
            'total_return': total_return,
            'avg_return': total_return / len(self.trades) if self.trades else 0,
            'total_fees': sum(t['fee'] for t in self.trades)
        }
    
    # Compatibility methods for strategies
    
    def set_position_mode(self, dual_side_position: bool = False) -> None:
        """Set position mode (no-op for backtest)"""
        pass
    
    def set_leverage(self, leverage: int = None) -> None:
        """Set leverage (updates internal leverage)"""
        if leverage is not None:
            self.leverage = leverage
            self.symbol_config['leverage'] = leverage
    
    def set_margin_type(self, margin_type: str = None) -> None:
        """Set margin type (no-op for backtest)"""
        pass
    
    def cancel_all_orders(self, symbol: str = None) -> None:
        """Cancel all orders (no-op for backtest)"""
        pass
    
    def check_order_amount(self, symbol: str = None, amount: float = None) -> float:
        """Check and adjust order amount"""
        if amount is None:
            raise ValueError("Amount not specified")
        # For backtest, we don't have min amount constraints
        return amount
    
    def get_market_info(self, symbol: str = None) -> Dict[str, Any]:
        """Get market info (mock)"""
        return {
            'limits': {
                'amount': {'min': 0.001},
                'cost': {'min': 20}
            }
        }
    
    def calculate_trade_amount(self, symbol: str = None) -> float:
        """Calculate trade amount based on available balance and configuration
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Trade amount in base currency
        """
        try:
            balance = self.get_balance()
            available = balance['free']
            current_price = self.get_market_price()
            
            # Get trade amount percentage from config (default 100%)
            trade_pct = self.symbol_config.get('trade_amount_percent', 100)
            
            # Calculate position value
            position_value = (available * trade_pct / 100) * self.leverage
            
            # Calculate amount in base currency
            amount = position_value / current_price
            
            # Ensure we have enough for fees and margin
            margin_needed = position_value / self.leverage
            fee_estimate = position_value * self.fee_rate * 2  # Entry + exit fees
            
            if margin_needed + fee_estimate > available:
                # Reduce amount to fit available balance
                usable_balance = available * 0.95  # Use 95% to leave buffer
                position_value = usable_balance * self.leverage
                amount = position_value / current_price
            
            return amount
            
        except Exception as e:
            self.logger.error(f"Error calculating trade amount: {str(e)}")
            return 0
