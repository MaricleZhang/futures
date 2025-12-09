"""
策略模块
"""

from .base_strategy import BaseStrategy
from .xgboost_strategy import XGBoostStrategy

__all__ = ['BaseStrategy', 'XGBoostStrategy']
