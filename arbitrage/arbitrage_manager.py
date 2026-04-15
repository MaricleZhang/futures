"""
套利管理器
- 按 config.ARBITRAGE_CONFIGS 为每个 pair 创建两腿 Trader + Strategy + Executor
- 每个 pair 一个工作线程：周期性调用 strategy.tick(executor)
- start() / stop() 与 TradingManager 并列
"""
import logging
import threading
import time

import config
from trader import Trader
from arbitrage.funding_rate_monitor import FundingRateMonitor
from arbitrage.arb_executor import ArbExecutor
from utils.arb_trade_recorder import ArbTradeRecorder
from strategies.funding_rate_arb_strategy import FundingRateArbStrategy


class ArbitrageManager:
    def __init__(self):
        self.logger = logging.getLogger('arb.manager')
        self.pairs = {}                  # pair_id -> dict(strategy, executor, traders, cfg, thread)
        self.recorder = ArbTradeRecorder()
        self._stop_event = threading.Event()

        for cfg in getattr(config, 'ARBITRAGE_CONFIGS', []):
            try:
                self._build_pair(cfg)
            except Exception as e:
                self.logger.error(f"初始化套利 pair {cfg.get('pair_id')} 失败: {e}", exc_info=True)

    def _build_pair(self, cfg: dict):
        pair_id = cfg['pair_id']
        ex_a, ex_b = cfg['exchanges']

        self.logger.info(f"[{pair_id}] 构建 {ex_a} / {ex_b} 两腿 Trader...")

        trader_a = Trader(symbol=cfg['symbol'], exchange_name=ex_a)
        trader_b = Trader(symbol=cfg['symbol'], exchange_name=ex_b)

        # 设置杠杆 / 保证金模式 / 持仓模式
        for tr in (trader_a, trader_b):
            try:
                tr.set_leverage(cfg['leverage'])
            except Exception as e:
                self.logger.warning(f"[{pair_id}] {tr.exchange_name} 设置杠杆失败: {e}")
            try:
                tr.set_margin_type(cfg.get('margin_type', 'CROSSED'))
            except Exception as e:
                self.logger.warning(f"[{pair_id}] {tr.exchange_name} 设置保证金模式失败: {e}")
            try:
                tr.set_position_mode(dual_side_position=False)
            except Exception as e:
                self.logger.warning(f"[{pair_id}] {tr.exchange_name} 设置单向持仓失败: {e}")

        traders = {ex_a: trader_a, ex_b: trader_b}
        adapters = {ex_a: trader_a.adapter, ex_b: trader_b.adapter}

        monitor = FundingRateMonitor(adapters, cache_ttl=30)
        strategy = FundingRateArbStrategy(traders, monitor, cfg)
        executor = ArbExecutor(traders, recorder=self.recorder)

        self.pairs[pair_id] = {
            'cfg': cfg,
            'traders': traders,
            'monitor': monitor,
            'strategy': strategy,
            'executor': executor,
            'thread': None,
        }
        self.logger.info(f"[{pair_id}] 初始化完成")

    # ---------------- 生命周期 ----------------
    def start(self):
        if not self.pairs:
            self.logger.info("没有可启动的套利 pair")
            return
        for pair_id, pair in self.pairs.items():
            t = threading.Thread(
                target=self._arb_loop,
                args=(pair_id,),
                name=f"arb-{pair_id}",
                daemon=True,
            )
            pair['thread'] = t
            t.start()
            self.logger.info(f"[{pair_id}] 套利线程启动")

    def stop(self):
        self._stop_event.set()
        self.logger.info("ArbitrageManager 停止信号已发出")

    def _arb_loop(self, pair_id: str):
        pair = self.pairs[pair_id]
        cfg = pair['cfg']
        strategy = pair['strategy']
        executor = pair['executor']
        interval = int(cfg.get('check_interval', 60))

        while not self._stop_event.is_set():
            try:
                strategy.tick(executor)
            except Exception as e:
                self.logger.error(f"[{pair_id}] tick 异常: {e}", exc_info=True)
            # 可被 stop_event 快速中断
            self._stop_event.wait(timeout=interval)

        self.logger.info(f"[{pair_id}] 套利线程退出")
