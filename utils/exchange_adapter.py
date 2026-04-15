"""
交易所适配层

目的：吸收 Binance 与 Gate 在 ccxt 之上的剩余差异
- symbol 归一化（Binance `BTC/USDT`  vs Gate `BTC/USDT:USDT`）
- 设置杠杆 / 保证金 / 持仓模式的私有 API 差异
- 资金费率字段读取（返回统一结构）
- 持仓数量字段解析（info.positionAmt vs contracts）
- 合约规格 / 最小下单量归一化

ccxt unified 方法（fetch_ticker / fetch_ohlcv / fetch_balance / create_order / cancel_order / fetch_positions）
直接由 Trader 调用 self.exchange.*，不经过 adapter。
"""
import logging
import time
import ccxt

logger = logging.getLogger(__name__)


class ExchangeAdapter:
    """交易所适配器基类"""

    name: str = ""

    def __init__(self, exchange: ccxt.Exchange):
        self.exchange = exchange

    # ---- symbol 归一化 ----
    def normalize_symbol(self, symbol: str) -> str:
        """输入可以是 'BTCUSDT' / 'BTC/USDT' / 'BTC/USDT:USDT'，返回当前交易所可用的 symbol。"""
        raise NotImplementedError

    # ---- 账户 / 合约配置 ----
    def set_leverage(self, symbol: str, leverage: int):
        raise NotImplementedError

    def set_margin_mode(self, symbol: str, margin_mode: str):
        """margin_mode: 'CROSSED' / 'ISOLATED'"""
        raise NotImplementedError

    def set_position_mode(self, dual_side: bool):
        raise NotImplementedError

    # ---- 持仓 ----
    def get_position_amt(self, position: dict) -> float:
        """从 fetch_positions 返回的 position dict 解析持仓数量（带符号：多+空-）"""
        raise NotImplementedError

    def get_position_entry_price(self, position: dict) -> float:
        raise NotImplementedError

    # ---- 资金费率 ----
    def fetch_funding_rate(self, symbol: str) -> dict:
        """统一返回 {
            'rate': float,                  # 当期费率，例如 0.0001 表示 0.01%
            'next_funding_time': int,        # 毫秒时间戳
            'interval_hours': int,           # 结算周期小时数（常见 1/4/8）
            'mark_price': float,
            'raw': dict,                     # ccxt 原始返回
        }"""
        raise NotImplementedError

    # ---- 合约规格 ----
    def normalize_amount(self, symbol: str, amount_in_base: float) -> float:
        """把「基础货币数量」转换为交易所实际下单数量（Gate 可能需要换算成张数）。"""
        market = self.exchange.market(symbol)
        try:
            precise = float(self.exchange.amount_to_precision(symbol, amount_in_base))
        except Exception:
            precise = amount_in_base
        return precise

    def contract_size(self, symbol: str) -> float:
        """单张合约对应的基础货币数量，默认 1（Binance USDT 本位合约即为 1）。"""
        market = self.exchange.market(symbol)
        return float(market.get('contractSize') or 1)


class BinanceAdapter(ExchangeAdapter):
    name = "binance"

    def normalize_symbol(self, symbol: str) -> str:
        if not symbol:
            return symbol
        # 去掉 Gate 风格尾部 settle
        if ':' in symbol:
            symbol = symbol.split(':', 1)[0]
        if '/' not in symbol and symbol.endswith('USDT'):
            symbol = symbol[:-4] + '/USDT'
        return symbol

    def set_leverage(self, symbol: str, leverage: int):
        self.exchange.fapiPrivatePostLeverage({
            'symbol': self.exchange.market(symbol)['id'],
            'leverage': leverage,
        })

    def set_margin_mode(self, symbol: str, margin_mode: str):
        try:
            self.exchange.fapiPrivatePostMarginType({
                'symbol': self.exchange.market(symbol)['id'],
                'marginType': margin_mode.upper(),
            })
        except Exception as e:
            if 'No need to change margin type' in str(e):
                logger.info(f"[binance] {symbol} 已经是 {margin_mode} 保证金模式")
                return
            raise

    def set_position_mode(self, dual_side: bool):
        try:
            self.exchange.fapiPrivatePostPositionSideDual({
                'dualSidePosition': 'true' if dual_side else 'false',
            })
        except Exception as e:
            if 'No need to change position side' in str(e):
                logger.info(f"[binance] 已经是{'双向' if dual_side else '单向'}持仓模式")
                return
            raise

    def get_position_amt(self, position: dict) -> float:
        if not position:
            return 0.0
        info = position.get('info') or {}
        if 'positionAmt' in info:
            return float(info.get('positionAmt', 0))
        contracts = position.get('contracts')
        if contracts is not None:
            amt = float(contracts)
            side = position.get('side')
            if side == 'short':
                amt = -abs(amt)
            return amt
        return 0.0

    def get_position_entry_price(self, position: dict) -> float:
        if not position:
            return 0.0
        info = position.get('info') or {}
        if info.get('entryPrice'):
            return float(info['entryPrice'])
        return float(position.get('entryPrice') or 0)

    def fetch_funding_rate(self, symbol: str) -> dict:
        raw = self.exchange.fetch_funding_rate(symbol)
        rate = raw.get('fundingRate')
        if rate is None:
            rate = raw.get('info', {}).get('lastFundingRate')
        rate = float(rate) if rate is not None else 0.0
        next_time = raw.get('fundingTimestamp') or raw.get('nextFundingTimestamp')
        mark_price = raw.get('markPrice') or raw.get('indexPrice')
        # Binance USDT-M 永续 funding 周期大部分是 8h
        interval = int(raw.get('info', {}).get('fundingIntervalHours') or 8)
        return {
            'rate': rate,
            'next_funding_time': int(next_time) if next_time else None,
            'interval_hours': interval,
            'mark_price': float(mark_price) if mark_price else None,
            'raw': raw,
        }


class GateAdapter(ExchangeAdapter):
    name = "gate"

    def normalize_symbol(self, symbol: str) -> str:
        if not symbol:
            return symbol
        # 输入示例: 'BTCUSDT' / 'BTC/USDT' / 'BTC/USDT:USDT'
        if ':' in symbol:
            return symbol
        if '/' not in symbol and symbol.endswith('USDT'):
            symbol = symbol[:-4] + '/USDT'
        if '/' in symbol:
            quote = symbol.split('/', 1)[1]
            return f"{symbol}:{quote}"
        return symbol

    def set_leverage(self, symbol: str, leverage: int):
        # ccxt 统一接口
        self.exchange.set_leverage(int(leverage), symbol)

    def set_margin_mode(self, symbol: str, margin_mode: str):
        # Gate: 'cross' / 'isolated'
        target = 'cross' if margin_mode.upper() == 'CROSSED' else 'isolated'
        try:
            self.exchange.set_margin_mode(target, symbol)
        except Exception as e:
            msg = str(e)
            if 'no change' in msg.lower() or 'already' in msg.lower():
                logger.info(f"[gate] {symbol} 已经是 {margin_mode} 保证金模式")
                return
            raise

    def set_position_mode(self, dual_side: bool):
        # Gate USDT 永续：dual_mode true=双向 / false=单向
        try:
            self.exchange.set_position_mode(dual_side)
        except Exception as e:
            msg = str(e).lower()
            if 'no change' in msg or 'already' in msg or 'same' in msg:
                logger.info(f"[gate] 已经是{'双向' if dual_side else '单向'}持仓模式")
                return
            logger.warning(f"[gate] set_position_mode 失败: {e}")

    def get_position_amt(self, position: dict) -> float:
        if not position:
            return 0.0
        info = position.get('info') or {}
        # Gate 的 size 字段代表张数（可正可负）
        if 'size' in info and info['size'] is not None:
            size = float(info['size'])
        else:
            contracts = position.get('contracts') or 0
            size = float(contracts)
            if position.get('side') == 'short':
                size = -abs(size)
        # 换算为基础货币数量
        sym = position.get('symbol')
        try:
            cs = self.contract_size(sym) if sym else 1.0
        except Exception:
            cs = 1.0
        return size * cs

    def get_position_entry_price(self, position: dict) -> float:
        if not position:
            return 0.0
        info = position.get('info') or {}
        for key in ('entry_price', 'entryPrice'):
            if info.get(key):
                return float(info[key])
        return float(position.get('entryPrice') or 0)

    def fetch_funding_rate(self, symbol: str) -> dict:
        raw = self.exchange.fetch_funding_rate(symbol)
        rate = raw.get('fundingRate')
        if rate is None:
            rate = raw.get('info', {}).get('funding_rate')
        rate = float(rate) if rate is not None else 0.0
        next_time = raw.get('fundingTimestamp') or raw.get('nextFundingTimestamp')
        mark_price = raw.get('markPrice') or raw.get('indexPrice')
        interval = raw.get('info', {}).get('funding_interval')
        if interval:
            # Gate `funding_interval` 单位通常是秒
            try:
                interval_hours = int(float(interval) / 3600)
                if interval_hours <= 0:
                    interval_hours = 8
            except Exception:
                interval_hours = 8
        else:
            interval_hours = 8
        return {
            'rate': rate,
            'next_funding_time': int(next_time) if next_time else None,
            'interval_hours': interval_hours,
            'mark_price': float(mark_price) if mark_price else None,
            'raw': raw,
        }

    def normalize_amount(self, symbol: str, amount_in_base: float) -> float:
        """Gate USDT 永续下单单位是『张数』，需要按 contractSize 换算。"""
        cs = self.contract_size(symbol) or 1.0
        contracts = amount_in_base / cs
        try:
            precise = float(self.exchange.amount_to_precision(symbol, contracts))
        except Exception:
            precise = contracts
        return precise


def make_adapter(exchange_name: str, exchange: ccxt.Exchange) -> ExchangeAdapter:
    name = (exchange_name or '').lower()
    if name == 'binance':
        return BinanceAdapter(exchange)
    if name == 'gate':
        return GateAdapter(exchange)
    raise ValueError(f"不支持的交易所: {exchange_name}")
