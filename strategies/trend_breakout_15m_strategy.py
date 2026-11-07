"""15-minute Donchian breakout strategy with an EMA trend filter.

The Donchian entry/exit rules are implemented here; the EMA trend filter uses
the same general approach found in public community strategies such as
Freqtrade's TrendRider. This implementation is self-contained and uses only
completed candles.

Signals: 1 = long, -1 = short, 0 = no active trend.
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional

import pandas as pd

from strategies.base_strategy import BaseStrategy


class TrendBreakout15mStrategy(BaseStrategy):
    """Trade channel breakouts confirmed by EMA and directional strength."""

    interval_minutes = 15
    entry_period = 20
    exit_period = 10
    trend_ema_period = 100
    adx_period = 14
    adx_threshold = 20
    position_fraction = 0.40
    check_interval = 900

    # BacktestEngine uses this as a 1-minute warm-up and checks every 15 bars.
    # 899 aligns the first check to the close of a 15-minute candle.
    lookback_period = 899
    kline_limit = 1000

    def __init__(self, trader, interval: str = "15m"):
        super().__init__(trader)
        if interval != "15m":
            raise ValueError("TrendBreakout15mStrategy requires interval='15m'")
        self.interval = interval
        self.kline_interval = interval
        self.logger = logging.getLogger(
            getattr(trader, "symbol", None) or self.__class__.__name__
        )
        self._state_initialized = False
        self._signal_state = 0
        self._last_processed_timestamp = None

    @staticmethod
    def _to_frame(klines: Iterable) -> pd.DataFrame:
        if isinstance(klines, pd.DataFrame):
            frame = klines.copy()
            if "timestamp" not in frame.columns and isinstance(frame.index, pd.DatetimeIndex):
                frame = frame.reset_index().rename(columns={"index": "timestamp"})
        else:
            frame = pd.DataFrame(
                list(klines or []),
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )

        required = {"timestamp", "open", "high", "low", "close"}
        if not required.issubset(frame.columns):
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close"])

        if pd.api.types.is_numeric_dtype(frame["timestamp"]):
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
        else:
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)

        for column in ("open", "high", "low", "close"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

        return (
            frame.dropna(subset=["timestamp", "open", "high", "low", "close"])
            .sort_values("timestamp")
            .drop_duplicates("timestamp", keep="last")
            .reset_index(drop=True)
        )

    def _get_closed_klines(self, klines=None) -> pd.DataFrame:
        if klines is None:
            klines = self.trader.get_klines(
                symbol=getattr(self.trader, "symbol", None),
                interval=self.interval,
                limit=self.kline_limit,
            )

        frame = self._to_frame(klines)
        if frame.empty:
            return frame

        # BacktestTrader resamples 1-minute bars and includes the forming
        # 15-minute candle. Keep it only after its final 1-minute bar is closed.
        if getattr(self.trader, "is_backtest", False):
            data = getattr(self.trader, "data", None)
            index = getattr(self.trader, "current_index", None)
            if data is not None and index is not None and len(data) > index:
                current_ts = pd.Timestamp(data.iloc[index]["timestamp"])
                if current_ts.minute % self.interval_minutes != self.interval_minutes - 1:
                    frame = frame.iloc[:-1]
        else:
            # Exchange OHLCV APIs can include the still-forming final candle.
            now = pd.Timestamp.now(tz="UTC")
            if frame.iloc[-1]["timestamp"] + pd.Timedelta(minutes=self.interval_minutes) > now:
                frame = frame.iloc[:-1]

        return frame.reset_index(drop=True)

    def _signal_series(self, frame: pd.DataFrame) -> pd.Series:
        """Update the active trend state from candles not seen on prior calls."""
        close = frame["close"]
        high = frame["high"]
        low = frame["low"]

        entry_high = high.shift(1).rolling(self.entry_period).max()
        entry_low = low.shift(1).rolling(self.entry_period).min()
        exit_high = high.shift(1).rolling(self.exit_period).max()
        exit_low = low.shift(1).rolling(self.exit_period).min()
        trend_ema = close.ewm(span=self.trend_ema_period, adjust=False).mean()

        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
        true_range = pd.concat(
            [
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        alpha = 1 / self.adx_period
        atr = true_range.ewm(
            alpha=alpha, adjust=False, min_periods=self.adx_period
        ).mean()
        plus_di = (
            100
            * plus_dm.ewm(alpha=alpha, adjust=False, min_periods=self.adx_period).mean()
            / atr.replace(0, float("nan"))
        )
        minus_di = (
            100
            * minus_dm.ewm(alpha=alpha, adjust=False, min_periods=self.adx_period).mean()
            / atr.replace(0, float("nan"))
        )
        dx = (
            100
            * (plus_di - minus_di).abs()
            / (plus_di + minus_di).replace(0, float("nan"))
        )
        adx = dx.ewm(
            alpha=alpha, adjust=False, min_periods=self.adx_period
        ).mean()

        state = self._signal_state if self._state_initialized else 0
        start_index = 0
        if self._state_initialized:
            previous = frame.index[frame["timestamp"] == self._last_processed_timestamp]
            if len(previous):
                start_index = int(previous[-1]) + 1
            else:
                # Recover cleanly if the caller supplied a disjoint history window.
                state = 0

        signals = [state] * len(frame)
        min_history = max(self.entry_period, self.exit_period, self.trend_ema_period)
        for index in range(start_index, len(frame)):
            if index + 1 < min_history:
                signals[index] = state
                continue

            price = close.iloc[index]
            strong_trend = adx.iloc[index] >= self.adx_threshold
            if (
                price > entry_high.iloc[index]
                and price > trend_ema.iloc[index]
                and strong_trend
                and plus_di.iloc[index] > minus_di.iloc[index]
            ):
                state = 1
            elif (
                price < entry_low.iloc[index]
                and price < trend_ema.iloc[index]
                and strong_trend
                and minus_di.iloc[index] > plus_di.iloc[index]
            ):
                state = -1
            elif state == 1 and price < exit_low.iloc[index]:
                state = 0
            elif state == -1 and price > exit_high.iloc[index]:
                state = 0

            signals[index] = state

        self._signal_state = state
        self._last_processed_timestamp = frame["timestamp"].iloc[-1]
        self._state_initialized = True
        return pd.Series(signals, index=frame.index, dtype="int64")

    def generate_signal(self, klines=None) -> int:
        """Return the current completed-candle direction: 1, -1, or 0."""
        frame = self._get_closed_klines(klines)
        if len(frame) < self.trend_ema_period:
            return 0
        if (
            self._state_initialized
            and frame["timestamp"].iloc[-1] <= self._last_processed_timestamp
        ):
            return int(self._signal_state)
        return int(self._signal_series(frame).iloc[-1])

    @staticmethod
    def _position_side(position: Optional[dict]) -> Optional[str]:
        if not position:
            return None
        side = str(position.get("side") or "").upper()
        if side in {"LONG", "SHORT"}:
            return side
        info = position.get("info", {})
        position_side = str(info.get("positionSide") or "").upper()
        if position_side in {"LONG", "SHORT"}:
            return position_side
        amount = float(info.get("positionAmt", position.get("positionAmt", 0)) or 0)
        if amount > 0:
            return "LONG"
        if amount < 0:
            return "SHORT"
        return None

    def monitor_position(self):
        """Align the position with the trend state, closing on a neutral signal."""
        signal = self.generate_signal()
        symbol = getattr(self.trader, "symbol", None)
        position = self.trader.get_position(symbol)
        current_side = self._position_side(position)
        desired_side = {1: "LONG", -1: "SHORT"}.get(signal)

        if current_side == desired_side:
            return
        if current_side is not None:
            self.trader.close_position(symbol)
        if desired_side is None:
            return

        balance = self.trader.get_balance(symbol)
        available = float(balance.get("free", 0) or 0)
        price = float(self.trader.get_market_price(symbol))
        configured_leverage = getattr(self.trader, "leverage", None)
        if configured_leverage is None:
            configured_leverage = getattr(self.trader, "symbol_config", {}).get("leverage", 1)
        amount = available * self.position_fraction * float(configured_leverage) / price
        if amount <= 0:
            self.logger.warning("Skipping signal: available balance or price is invalid")
            return

        if signal == 1:
            self.trader.open_long(symbol, amount)
        else:
            self.trader.open_short(symbol, amount)
