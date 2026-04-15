"""
套利策略基类
与 BaseStrategy 不同，套利策略持有多个 Trader（每个交易所一个），跨所做决策。
"""
import logging
from abc import ABC, abstractmethod
from typing import Dict, Optional


class ArbitrageStrategy(ABC):
    """跨交易所套利策略基类"""

    # 状态机
    STATE_FLAT = 'FLAT'
    STATE_HEDGED = 'HEDGED'
    STATE_CLOSING = 'CLOSING'
    STATE_CRITICAL = 'CRITICAL'

    def __init__(self, traders: Dict[str, "Trader"], monitor, config_: dict):
        """
        Args:
            traders: {exchange_name: Trader}
            monitor: FundingRateMonitor
            config_: pair 对应的配置字典（ARBITRAGE_CONFIGS 中的一项）
        """
        self.traders = traders
        self.monitor = monitor
        self.cfg = config_
        self.pair_id = config_['pair_id']
        self.logger = logging.getLogger(f"arb.{self.pair_id}")
        self.state = self.STATE_FLAT
        self.open_trade_id: Optional[int] = None       # 当前开仓对应 arb_trades.id
        self.open_info: Optional[dict] = None          # 开仓快照

    @abstractmethod
    def should_open(self, diff: dict) -> Optional[dict]:
        """返回开仓计划 dict 或 None。"""

    @abstractmethod
    def should_close(self, diff: dict, position_snapshot: dict) -> Optional[str]:
        """返回 close_reason 字符串，或 None 表示继续持有。"""

    @abstractmethod
    def tick(self, executor) -> None:
        """策略状态机的一次迭代。ArbitrageManager 循环调用。"""
