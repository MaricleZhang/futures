"""
套利交易记录器
- 新增 SQLite 表 arb_trades（不改动 trades 表）
- 记录两腿价格、利差、funding_collected、net_pnl、close_reason
- pending_unwind 表用于异常兜底（单腿回滚失败的情况）
"""
import sqlite3
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Optional


class ArbTradeRecorder:
    def __init__(self, db_path: str = None):
        self.logger = logging.getLogger(__name__)

        if db_path is None:
            db_dir = Path(__file__).parent.parent / 'data'
            db_dir.mkdir(exist_ok=True)
            db_path = db_dir / 'arb_trades.db'

        self.db_path = str(db_path)
        self.logger.info(f"套利交易记录数据库路径: {self.db_path}")
        self._create_tables()

    def _conn(self):
        return sqlite3.connect(self.db_path)

    def _create_tables(self):
        conn = self._conn()
        cur = conn.cursor()
        cur.execute('''
            CREATE TABLE IF NOT EXISTS arb_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pair_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                long_exchange TEXT NOT NULL,
                short_exchange TEXT NOT NULL,
                long_price REAL,
                short_price REAL,
                amount REAL,
                notional_usdt REAL,
                entry_spread_bps REAL,
                entry_rate_long REAL,
                entry_rate_short REAL,
                open_time TEXT NOT NULL,
                close_long_price REAL,
                close_short_price REAL,
                close_spread_bps REAL,
                close_time TEXT,
                funding_collected REAL DEFAULT 0,
                gross_pnl REAL,
                net_pnl REAL,
                fees REAL,
                close_reason TEXT,
                status TEXT NOT NULL DEFAULT 'OPEN',
                dry_run INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        ''')
        cur.execute('CREATE INDEX IF NOT EXISTS idx_arb_pair ON arb_trades(pair_id)')
        cur.execute('CREATE INDEX IF NOT EXISTS idx_arb_status ON arb_trades(status)')

        cur.execute('''
            CREATE TABLE IF NOT EXISTS pending_unwind (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pair_id TEXT NOT NULL,
                exchange TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,            -- 需要执行的平仓方向 buy/sell
                amount REAL NOT NULL,
                reason TEXT,
                payload TEXT,                   -- JSON 附加信息
                resolved INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        ''')
        cur.execute('CREATE INDEX IF NOT EXISTS idx_pending_pair ON pending_unwind(pair_id)')
        cur.execute('CREATE INDEX IF NOT EXISTS idx_pending_resolved ON pending_unwind(resolved)')

        conn.commit()
        conn.close()

    # ---------- arb_trades ----------
    def record_open(self, *, pair_id, symbol, long_exchange, short_exchange,
                    long_price, short_price, amount, notional_usdt,
                    entry_spread_bps, entry_rate_long, entry_rate_short, dry_run=False):
        now = datetime.utcnow().isoformat()
        conn = self._conn()
        cur = conn.cursor()
        cur.execute('''
            INSERT INTO arb_trades (
                pair_id, symbol, long_exchange, short_exchange,
                long_price, short_price, amount, notional_usdt,
                entry_spread_bps, entry_rate_long, entry_rate_short,
                open_time, status, dry_run, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (pair_id, symbol, long_exchange, short_exchange,
              long_price, short_price, amount, notional_usdt,
              entry_spread_bps, entry_rate_long, entry_rate_short,
              now, ('DRY_RUN' if dry_run else 'OPEN'), 1 if dry_run else 0, now, now))
        trade_id = cur.lastrowid
        conn.commit()
        conn.close()
        self.logger.info(f"[arb_trade #{trade_id}] open pair={pair_id} symbol={symbol} "
                         f"long={long_exchange}@{long_price} short={short_exchange}@{short_price} "
                         f"amount={amount} spread_bps={entry_spread_bps:.2f} dry_run={dry_run}")
        return trade_id

    def record_close(self, trade_id, *, close_long_price, close_short_price,
                     close_spread_bps, gross_pnl, fees, net_pnl,
                     close_reason, funding_collected=None):
        now = datetime.utcnow().isoformat()
        conn = self._conn()
        cur = conn.cursor()
        if funding_collected is None:
            cur.execute('''
                UPDATE arb_trades
                SET close_long_price=?, close_short_price=?, close_spread_bps=?,
                    close_time=?, gross_pnl=?, fees=?, net_pnl=?,
                    close_reason=?, status='CLOSED', updated_at=?
                WHERE id=?
            ''', (close_long_price, close_short_price, close_spread_bps,
                  now, gross_pnl, fees, net_pnl, close_reason, now, trade_id))
        else:
            cur.execute('''
                UPDATE arb_trades
                SET close_long_price=?, close_short_price=?, close_spread_bps=?,
                    close_time=?, funding_collected=?, gross_pnl=?, fees=?, net_pnl=?,
                    close_reason=?, status='CLOSED', updated_at=?
                WHERE id=?
            ''', (close_long_price, close_short_price, close_spread_bps,
                  now, funding_collected, gross_pnl, fees, net_pnl,
                  close_reason, now, trade_id))
        conn.commit()
        conn.close()
        self.logger.info(f"[arb_trade #{trade_id}] close spread_bps={close_spread_bps:.2f} "
                         f"net_pnl={net_pnl:.4f} reason={close_reason}")

    def add_funding(self, trade_id, amount: float):
        now = datetime.utcnow().isoformat()
        conn = self._conn()
        cur = conn.cursor()
        cur.execute('''
            UPDATE arb_trades
            SET funding_collected = COALESCE(funding_collected, 0) + ?, updated_at=?
            WHERE id=?
        ''', (amount, now, trade_id))
        conn.commit()
        conn.close()

    def get_open_trade(self, pair_id: str) -> Optional[dict]:
        conn = self._conn()
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT * FROM arb_trades WHERE pair_id=? AND status='OPEN' ORDER BY id DESC LIMIT 1",
                    (pair_id,))
        row = cur.fetchone()
        conn.close()
        return dict(row) if row else None

    def get_statistics(self, pair_id: Optional[str] = None) -> dict:
        conn = self._conn()
        cur = conn.cursor()
        if pair_id:
            cur.execute("SELECT COUNT(*), SUM(net_pnl), AVG(net_pnl) FROM arb_trades "
                        "WHERE status='CLOSED' AND pair_id=?", (pair_id,))
        else:
            cur.execute("SELECT COUNT(*), SUM(net_pnl), AVG(net_pnl) FROM arb_trades "
                        "WHERE status='CLOSED'")
        total, sum_pnl, avg_pnl = cur.fetchone()
        conn.close()
        return {
            'total_trades': total or 0,
            'total_net_pnl': float(sum_pnl) if sum_pnl else 0.0,
            'avg_net_pnl': float(avg_pnl) if avg_pnl else 0.0,
        }

    # ---------- pending_unwind ----------
    def record_pending_unwind(self, *, pair_id, exchange, symbol, side, amount,
                              reason: str, payload: Optional[dict] = None):
        now = datetime.utcnow().isoformat()
        conn = self._conn()
        cur = conn.cursor()
        cur.execute('''
            INSERT INTO pending_unwind (pair_id, exchange, symbol, side, amount,
                                        reason, payload, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (pair_id, exchange, symbol, side, amount, reason,
              json.dumps(payload or {}), now, now))
        conn.commit()
        conn.close()
        self.logger.error(f"[pending_unwind] pair={pair_id} ex={exchange} {symbol} {side} {amount} reason={reason}")

    def resolve_pending(self, pending_id: int):
        now = datetime.utcnow().isoformat()
        conn = self._conn()
        cur = conn.cursor()
        cur.execute("UPDATE pending_unwind SET resolved=1, updated_at=? WHERE id=?", (now, pending_id))
        conn.commit()
        conn.close()
