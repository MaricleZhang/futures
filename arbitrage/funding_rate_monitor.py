"""
资金费率监控模块
- 同时拉取多个交易所同一 symbol 的资金费率
- 计算利差（bp / 年化）、下一次结算时间、结算周期差异
- 带 TTL 缓存，避免高频 API 调用
"""
import logging
import time
from typing import Dict


class FundingRateMonitor:
    def __init__(self, adapters: Dict[str, "ExchangeAdapter"], cache_ttl: int = 30):
        """
        Args:
            adapters: {exchange_name: ExchangeAdapter}
            cache_ttl: 单条费率的缓存 TTL 秒
        """
        self.adapters = adapters
        self.cache_ttl = cache_ttl
        self._cache = {}       # {(exchange, symbol): (info_dict, fetched_at_epoch)}
        self.logger = logging.getLogger(__name__)

    def invalidate(self):
        self._cache.clear()

    def get_rate(self, exchange: str, symbol: str) -> dict:
        """返回 adapter.fetch_funding_rate 统一结构，带缓存。"""
        adapter = self.adapters.get(exchange)
        if adapter is None:
            raise ValueError(f"未注册 adapter: {exchange}")

        key = (exchange, adapter.normalize_symbol(symbol))
        cached = self._cache.get(key)
        now = time.time()
        if cached and (now - cached[1] < self.cache_ttl):
            return cached[0]

        info = adapter.fetch_funding_rate(key[1])
        self._cache[key] = (info, now)
        return info

    def get_differential(self, symbol: str, ex_a: str, ex_b: str) -> dict:
        """计算两所资金费率差异。

        返回 {
            'a_rate', 'b_rate',                 # 原始周期费率（例如 0.0001）
            'a_interval', 'b_interval',         # 各自结算周期小时数
            'a_mark', 'b_mark',                 # mark price
            'a_next', 'b_next',                 # 下次结算毫秒时间戳
            'spread_bps': (a_rate - b_rate) * 10000,   # 按单周期 bp 计
            'sign': +1 / -1 / 0,                # a>b 正；a<b 负
            'annualized_spread': (a_annual - b_annual),
            'snapshot_ts': 毫秒
        }
        """
        a = self.get_rate(ex_a, symbol)
        b = self.get_rate(ex_b, symbol)

        a_rate = float(a.get('rate') or 0)
        b_rate = float(b.get('rate') or 0)
        spread_bps = (a_rate - b_rate) * 10000

        if abs(spread_bps) < 1e-9:
            sign = 0
        elif spread_bps > 0:
            sign = 1
        else:
            sign = -1

        a_interval = int(a.get('interval_hours') or 8) or 8
        b_interval = int(b.get('interval_hours') or 8) or 8
        a_annual = a_rate * (24 / a_interval) * 365
        b_annual = b_rate * (24 / b_interval) * 365

        return {
            'a_rate': a_rate,
            'b_rate': b_rate,
            'a_interval': a_interval,
            'b_interval': b_interval,
            'a_mark': a.get('mark_price'),
            'b_mark': b.get('mark_price'),
            'a_next': a.get('next_funding_time'),
            'b_next': b.get('next_funding_time'),
            'spread_bps': spread_bps,
            'sign': sign,
            'annualized_spread': a_annual - b_annual,
            'snapshot_ts': int(time.time() * 1000),
        }
