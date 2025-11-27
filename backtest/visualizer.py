"""
Visualizer for backtest results
Creates charts and graphs for analysis
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
import logging


class Visualizer:
    """Create visualizations for backtest results"""
    
    def __init__(self):
        """Initialize visualizer"""
        self.logger = logging.getLogger(__name__)
        
        # Set style
        sns.set_style('darkgrid')
        plt.rcParams['figure.figsize'] = (12, 6)
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    
    def plot_equity_curve(
        self,
        equity_curve: pd.DataFrame,
        output_path: str,
        initial_capital: float
    ):
        """Plot equity curve
        
        Args:
            equity_curve: DataFrame with timestamp and equity columns
            output_path: Path to save figure
            initial_capital: Initial capital for reference line
        """
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Plot equity
        ax.plot(equity_curve['timestamp'], equity_curve['equity'], 
               label='Equity', color='#2E86DE', linewidth=2)
        
        # Plot initial capital line
        ax.axhline(y=initial_capital, color='gray', linestyle='--', 
                  label=f'Initial Capital ({initial_capital:.0f} USDT)', alpha=0.7)
        
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Equity (USDT)', fontsize=12)
        ax.set_title('Equity Curve', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Equity curve saved to: {output_path}")
    
    def plot_drawdown(
        self,
        equity_curve: pd.DataFrame,
        output_path: str
    ):
        """Plot drawdown curve
        
        Args:
            equity_curve: DataFrame with timestamp and equity columns
            output_path: Path to save figure
        """
        equity = equity_curve['equity'].values
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max * 100
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Plot drawdown
        ax.fill_between(equity_curve['timestamp'], drawdown, 0, 
                        color='#EE5A6F', alpha=0.5, label='Drawdown')
        ax.plot(equity_curve['timestamp'], drawdown, 
               color='#C23616', linewidth=1.5)
        
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Drawdown (%)', fontsize=12)
        ax.set_title('Drawdown Curve', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Drawdown curve saved to: {output_path}")
    
    def plot_returns_distribution(
        self,
        trades: List[Dict[str, Any]],
        output_path: str
    ):
        """Plot returns distribution histogram
        
        Args:
            trades: List of trade dictionaries
            output_path: Path to save figure
        """
        if not trades:
            self.logger.warning("No trades to plot")
            return
        
        returns = [t['return_rate'] for t in trades]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot histogram
        ax.hist(returns, bins=30, color='#2E86DE', alpha=0.7, edgecolor='black')
        
        # Plot mean line
        mean_return = np.mean(returns)
        ax.axvline(x=mean_return, color='red', linestyle='--', 
                  label=f'Mean: {mean_return:.2f}%', linewidth=2)
        
        ax.set_xlabel('Return Rate (%)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Trade Returns Distribution', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Returns distribution saved to: {output_path}")
    
    def plot_trade_analysis(
        self,
        trades: List[Dict[str, Any]],
        output_path: str
    ):
        """Plot trade analysis (wins/losses over time)
        
        Args:
            trades: List of trade dictionaries
            output_path: Path to save figure
        """
        if not trades:
            self.logger.warning("No trades to plot")
            return
        
        trades_df = pd.DataFrame(trades)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Plot 1: Cumulative PnL
        trades_df['cumulative_pnl'] = trades_df['net_pnl'].cumsum()
        
        colors = ['green' if pnl > 0 else 'red' for pnl in trades_df['net_pnl']]
        ax1.bar(range(len(trades_df)), trades_df['net_pnl'], color=colors, alpha=0.6)
        ax1.plot(range(len(trades_df)), trades_df['cumulative_pnl'], 
                color='#2E86DE', linewidth=2, label='Cumulative PnL')
        
        ax1.set_xlabel('Trade Number', fontsize=12)
        ax1.set_ylabel('PnL (USDT)', fontsize=12)
        ax1.set_title('Trade-by-Trade PnL', fontsize=14, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Win/Loss comparison
        winning_trades = [t for t in trades if t['net_pnl'] > 0]
        losing_trades = [t for t in trades if t['net_pnl'] <= 0]
        
        categories = ['Winning Trades', 'Losing Trades']
        counts = [len(winning_trades), len(losing_trades)]
        avg_pnls = [
            np.mean([t['net_pnl'] for t in winning_trades]) if winning_trades else 0,
            np.mean([t['net_pnl'] for t in losing_trades]) if losing_trades else 0
        ]
        
        x = np.arange(len(categories))
        width = 0.35
        
        ax2_twin = ax2.twinx()
        
        bars1 = ax2.bar(x - width/2, counts, width, label='Count', color='#273c75', alpha=0.7)
        bars2 = ax2_twin.bar(x + width/2, avg_pnls, width, label='Avg PnL', color='#44bd32', alpha=0.7)
        
        ax2.set_xlabel('Trade Type', fontsize=12)
        ax2.set_ylabel('Count', fontsize=12, color='#273c75')
        ax2_twin.set_ylabel('Average PnL (USDT)', fontsize=12, color='#44bd32')
        ax2.set_title('Win/Loss Analysis', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(categories)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        
        for bar in bars2:
            height = bar.get_height()
            ax2_twin.text(bar.get_x() + bar.get_width()/2., height,
                         f'{height:.2f}', ha='center', va='bottom')
        
        ax2.legend(loc='upper left')
        ax2_twin.legend(loc='upper right')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Trade analysis saved to: {output_path}")
    
    def create_all_plots(
        self,
        equity_curve: pd.DataFrame,
        trades: List[Dict[str, Any]],
        initial_capital: float,
        output_dir: str
    ):
        """Create all visualization plots
        
        Args:
            equity_curve: Equity curve DataFrame
            trades: List of trades
            initial_capital: Initial capital
            output_dir: Directory to save plots
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Creating visualizations...")
        
        # Equity curve
        self.plot_equity_curve(
            equity_curve, 
            str(output_path / 'equity_curve.png'),
            initial_capital
        )
        
        # Drawdown
        self.plot_drawdown(
            equity_curve,
            str(output_path / 'drawdown.png')
        )
        
        # Returns distribution
        if trades:
            self.plot_returns_distribution(
                trades,
                str(output_path / 'returns_distribution.png')
            )
            
            # Trade analysis
            self.plot_trade_analysis(
                trades,
                str(output_path / 'trade_analysis.png')
            )
        
        self.logger.info(f"All visualizations saved to: {output_dir}")
