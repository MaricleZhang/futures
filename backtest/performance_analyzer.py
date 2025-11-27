"""
Performance analyzer for backtest results
Calculates comprehensive trading metrics and risk indicators
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime, timedelta


class PerformanceAnalyzer:
    """Analyze backtest performance and calculate metrics"""
    
    def __init__(self):
        """Initialize performance analyzer"""
        self.logger = logging.getLogger(__name__)
    
    def analyze(
        self,
        equity_curve: pd.DataFrame,
        trades: List[Dict[str, Any]],
        initial_capital: float,
        start_date: str,
        end_date: str
    ) -> Dict[str, Any]:
        """Analyze backtest results
        
        Args:
            equity_curve: DataFrame with timestamp and equity columns
            trades: List of trade dictionaries
            initial_capital: Initial capital
            start_date: Backtest start date
            end_date: Backtest end date
            
        Returns:
            Dictionary with all performance metrics
        """
        self.logger.info("Analyzing backtest performance...")
        
        if len(equity_curve) == 0:
            return self._empty_results()
        
        # Basic metrics
        final_equity = equity_curve['equity'].iloc[-1]
        total_return = ((final_equity - initial_capital) / initial_capital) * 100
        
        # Time metrics
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        days = (end_dt - start_dt).days
        years = days / 365.25
        
        # Annualized return
        if years > 0:
            annualized_return = ((final_equity / initial_capital) ** (1 / years) - 1) * 100
        else:
            annualized_return = 0
        
        # Calculate returns
        equity_curve['returns'] = equity_curve['equity'].pct_change()
        
        # Risk metrics
        max_drawdown, max_dd_duration = self._calculate_max_drawdown(equity_curve)
        volatility = self._calculate_volatility(equity_curve)
        sharpe_ratio = self._calculate_sharpe_ratio(equity_curve)
        sortino_ratio = self._calculate_sortino_ratio(equity_curve)
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Trade metrics
        trade_stats = self._calculate_trade_stats(trades)
        
        # Combine all metrics
        results = {
            # Return metrics
            'initial_capital': initial_capital,
            'final_equity': final_equity,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'total_pnl': final_equity - initial_capital,
            
            # Risk metrics
            'max_drawdown': max_drawdown,
            'max_drawdown_duration': max_dd_duration,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            
            # Time metrics
            'start_date': start_date,
            'end_date': end_date,
            'trading_days': days,
            'trading_years': years,
            
            # Trade metrics
            **trade_stats
        }
        
        self.logger.info(f"Analysis complete: Return={total_return:.2f}%, "
                        f"MaxDD={max_drawdown:.2f}%, Sharpe={sharpe_ratio:.2f}")
        
        return results
    
    def _calculate_max_drawdown(self, equity_curve: pd.DataFrame) -> tuple:
        """Calculate maximum drawdown and its duration
        
        Returns:
            (max_drawdown_pct, max_drawdown_duration_days)
        """
        equity = equity_curve['equity'].values
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(equity)
        
        # Calculate drawdown
        drawdown = (equity - running_max) / running_max * 100
        
        max_drawdown = drawdown.min()
        
        # Calculate drawdown duration
        max_dd_duration = 0
        current_dd_duration = 0
        
        for dd in drawdown:
            if dd < 0:
                current_dd_duration += 1
                max_dd_duration = max(max_dd_duration, current_dd_duration)
            else:
                current_dd_duration = 0
        
        return max_drawdown, max_dd_duration
    
    def _calculate_volatility(self, equity_curve: pd.DataFrame) -> float:
        """Calculate annualized volatility"""
        returns = equity_curve['returns'].dropna()
        
        if len(returns) == 0:
            return 0
        
        # Annualized volatility (assuming daily data)
        volatility = returns.std() * np.sqrt(252) * 100
        
        return volatility
    
    def _calculate_sharpe_ratio(self, equity_curve: pd.DataFrame, risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio
        
        Args:
            equity_curve: Equity curve DataFrame
            risk_free_rate: Risk-free rate (annual)
            
        Returns:
            Sharpe ratio
        """
        returns = equity_curve['returns'].dropna()
        
        if len(returns) == 0 or returns.std() == 0:
            return 0
        
        # Annualized Sharpe ratio
        avg_return = returns.mean()
        std_return = returns.std()
        
        sharpe = (avg_return - risk_free_rate / 252) / std_return * np.sqrt(252)
        
        return sharpe
    
    def _calculate_sortino_ratio(self, equity_curve: pd.DataFrame, risk_free_rate: float = 0.0) -> float:
        """Calculate Sortino ratio (focuses on downside volatility)
        
        Args:
            equity_curve: Equity curve DataFrame
            risk_free_rate: Risk-free rate (annual)
            
        Returns:
            Sortino ratio
        """
        returns = equity_curve['returns'].dropna()
        
        if len(returns) == 0:
            return 0
        
        # Only consider negative returns for downside deviation
        negative_returns = returns[returns < 0]
        
        if len(negative_returns) == 0:
            return 0
        
        downside_std = negative_returns.std()
        
        if downside_std == 0:
            return 0
        
        avg_return = returns.mean()
        sortino = (avg_return - risk_free_rate / 252) / downside_std * np.sqrt(252)
        
        return sortino
    
    def _calculate_trade_stats(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate trade statistics
        
        Args:
            trades: List of trade dictionaries
            
        Returns:
            Dictionary with trade statistics
        """
        if not trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'largest_win': 0,
                'largest_loss': 0,
                'profit_factor': 0,
                'avg_trade_duration': 0,
                'total_fees': 0
            }
        
        # Separate winning and losing trades
        winning_trades = [t for t in trades if t['net_pnl'] > 0]
        losing_trades = [t for t in trades if t['net_pnl'] <= 0]
        
        # Win rate
        win_rate = (len(winning_trades) / len(trades)) * 100 if trades else 0
        
        # Average win/loss
        avg_win = np.mean([t['net_pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['net_pnl'] for t in losing_trades]) if losing_trades else 0
        
        # Largest win/loss
        largest_win = max([t['net_pnl'] for t in winning_trades]) if winning_trades else 0
        largest_loss = min([t['net_pnl'] for t in losing_trades]) if losing_trades else 0
        
        # Profit factor
        total_wins = sum([t['net_pnl'] for t in winning_trades])
        total_losses = abs(sum([t['net_pnl'] for t in losing_trades]))
        profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        # Average trade duration
        durations = []
        for t in trades:
            if isinstance(t['entry_time'], pd.Timestamp) and isinstance(t['exit_time'], pd.Timestamp):
                duration = (t['exit_time'] - t['entry_time']).total_seconds() / 3600
                durations.append(duration)
        
        avg_duration = np.mean(durations) if durations else 0
        
        # Total fees
        total_fees = sum([t['fee'] for t in trades])
        
        # Long vs Short stats
        long_trades = [t for t in trades if t['side'] == 'LONG']
        short_trades = [t for t in trades if t['side'] == 'SHORT']
        
        long_win_rate = (len([t for t in long_trades if t['net_pnl'] > 0]) / len(long_trades)) * 100 if long_trades else 0
        short_win_rate = (len([t for t in short_trades if t['net_pnl'] > 0]) / len(short_trades)) * 100 if short_trades else 0
        
        return {
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'profit_factor': profit_factor,
            'avg_trade_duration_hours': avg_duration,
            'total_fees': total_fees,
            'long_trades': len(long_trades),
            'short_trades': len(short_trades),
            'long_win_rate': long_win_rate,
            'short_win_rate': short_win_rate
        }
    
    def _empty_results(self) -> Dict[str, Any]:
        """Return empty results when no data"""
        return {
            'initial_capital': 0,
            'final_equity': 0,
            'total_return': 0,
            'annualized_return': 0,
            'total_pnl': 0,
            'max_drawdown': 0,
            'max_drawdown_duration': 0,
            'volatility': 0,
            'sharpe_ratio': 0,
            'sortino_ratio': 0,
            'calmar_ratio': 0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'profit_factor': 0
        }
    
    def generate_summary(self, metrics: Dict[str, Any]) -> str:
        """Generate a text summary of performance
        
        Args:
            metrics: Dictionary with performance metrics
            
        Returns:
            Formatted text summary
        """
        summary = f"""
╔══════════════════════════════════════════════════════════════╗
║                    BACKTEST PERFORMANCE SUMMARY               ║
╚══════════════════════════════════════════════════════════════╝

【收益指标】
  初始资金:        {metrics['initial_capital']:,.2f} USDT
  最终权益:        {metrics['final_equity']:,.2f} USDT
  总收益:          {metrics['total_pnl']:,.2f} USDT
  收益率:          {metrics['total_return']:.2f}%
  年化收益率:      {metrics['annualized_return']:.2f}%

【风险指标】
  最大回撤:        {metrics['max_drawdown']:.2f}%
  回撤持续:        {metrics.get('max_drawdown_duration', 0)} 天
  波动率:          {metrics['volatility']:.2f}%
  夏普比率:        {metrics['sharpe_ratio']:.2f}
  索提诺比率:      {metrics['sortino_ratio']:.2f}
  卡玛比率:        {metrics['calmar_ratio']:.2f}

【交易统计】
  交易次数:        {metrics['total_trades']}
  盈利次数:        {metrics['winning_trades']}
  亏损次数:        {metrics['losing_trades']}
  胜率:            {metrics['win_rate']:.2f}%
  平均盈利:        {metrics.get('avg_win', 0):.2f} USDT
  平均亏损:        {metrics.get('avg_loss', 0):.2f} USDT
  最大盈利:        {metrics.get('largest_win', 0):.2f} USDT
  最大亏损:        {metrics.get('largest_loss', 0):.2f} USDT
  盈亏比:          {metrics.get('profit_factor', 0):.2f}
  
【多空统计】
  做多次数:        {metrics.get('long_trades', 0)}
  做空次数:        {metrics.get('short_trades', 0)}
  做多胜率:        {metrics.get('long_win_rate', 0):.2f}%
  做空胜率:        {metrics.get('short_win_rate', 0):.2f}%

【费用统计】
  总手续费:        {metrics.get('total_fees', 0):.2f} USDT

【时间范围】
  开始日期:        {metrics.get('start_date', 'N/A')}
  结束日期:        {metrics.get('end_date', 'N/A')}
  交易天数:        {metrics.get('trading_days', 0)} 天

╚══════════════════════════════════════════════════════════════╝
        """.strip()
        
        return summary
