"""
交易记录模块
用于记录和管理所有交易的详细信息
"""
import sqlite3
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

class TradeRecorder:
    def __init__(self, db_path: str = None):
        """初始化交易记录器
        
        Args:
            db_path: 数据库文件路径，默认为 data/trades.db
        """
        self.logger = logging.getLogger(__name__)
        
        # 设置数据库路径
        if db_path is None:
            db_dir = Path(__file__).parent.parent / 'data'
            db_dir.mkdir(exist_ok=True)
            db_path = db_dir / 'trades.db'
        
        self.db_path = str(db_path)
        self.logger.info(f"交易记录数据库路径: {self.db_path}")
        
        # 初始化数据库
        self.create_table()
    
    def get_connection(self):
        """获取数据库连接"""
        return sqlite3.connect(self.db_path)
    
    def create_table(self):
        """创建交易记录表"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    open_amount REAL NOT NULL,
                    open_price REAL NOT NULL,
                    open_value REAL NOT NULL,
                    open_time TEXT NOT NULL,
                    close_price REAL,
                    close_time TEXT,
                    profit_loss REAL,
                    profit_rate REAL,
                    leverage INTEGER,
                    status TEXT NOT NULL DEFAULT 'OPEN',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            ''')
            
            # 创建索引以提高查询性能
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_symbol ON trades(symbol)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_status ON trades(status)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_open_time ON trades(open_time)
            ''')
            
            conn.commit()
            conn.close()
            self.logger.info("交易记录表创建成功")
        except Exception as e:
            self.logger.error(f"创建交易记录表失败: {str(e)}")
            raise
    
    def record_open_position(self, symbol: str, side: str, amount: float, 
                            price: float, leverage: int = 1) -> int:
        """记录开仓信息
        
        Args:
            symbol: 交易对
            side: 方向 (LONG/SHORT)
            amount: 开仓数量
            price: 开仓价格
            leverage: 杠杆倍数
            
        Returns:
            trade_id: 交易记录ID
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            open_value = amount * price
            now = datetime.now().isoformat()
            
            cursor.execute('''
                INSERT INTO trades (
                    symbol, side, open_amount, open_price, open_value,
                    open_time, leverage, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (symbol, side, amount, price, open_value, now, leverage, 
                  'OPEN', now, now))
            
            trade_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            self.logger.info(f"记录开仓成功 [ID:{trade_id}] {symbol} {side} "
                           f"数量:{amount} 价格:{price} 金额:{open_value:.2f}")
            return trade_id
            
        except Exception as e:
            self.logger.error(f"记录开仓失败: {str(e)}")
            raise
    
    def record_close_position(self, symbol: str, side: str, close_price: float) -> Optional[Dict[str, Any]]:
        """记录平仓信息并计算盈亏
        
        Args:
            symbol: 交易对
            side: 方向 (LONG/SHORT)
            close_price: 平仓价格
            
        Returns:
            trade_info: 包含盈亏信息的字典，如果没有找到对应的开仓记录则返回None
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # 查找最近的未平仓记录
            cursor.execute('''
                SELECT id, open_amount, open_price, open_value, open_time
                FROM trades
                WHERE symbol = ? AND side = ? AND status = 'OPEN'
                ORDER BY open_time DESC
                LIMIT 1
            ''', (symbol, side))
            
            result = cursor.fetchone()
            if not result:
                self.logger.warning(f"未找到对应的开仓记录: {symbol} {side}")
                conn.close()
                return None
            
            trade_id, open_amount, open_price, open_value, open_time = result
            
            # 计算盈亏
            # 多仓: 盈亏 = 数量 × (平仓价 - 开仓价)
            # 空仓: 盈亏 = 数量 × (开仓价 - 平仓价)
            if side == 'LONG':
                profit_loss = open_amount * (close_price - open_price)
            else:  # SHORT
                profit_loss = open_amount * (open_price - close_price)
            
            # 计算收益率
            profit_rate = (profit_loss / open_value) * 100 if open_value > 0 else 0
            
            # 更新记录
            now = datetime.now().isoformat()
            cursor.execute('''
                UPDATE trades
                SET close_price = ?, close_time = ?, profit_loss = ?,
                    profit_rate = ?, status = 'CLOSED', updated_at = ?
                WHERE id = ?
            ''', (close_price, now, profit_loss, profit_rate, now, trade_id))
            
            conn.commit()
            conn.close()
            
            trade_info = {
                'trade_id': trade_id,
                'symbol': symbol,
                'side': side,
                'open_amount': open_amount,
                'open_price': open_price,
                'open_value': open_value,
                'open_time': open_time,
                'close_price': close_price,
                'close_time': now,
                'profit_loss': profit_loss,
                'profit_rate': profit_rate
            }
            
            self.logger.info(f"记录平仓成功 [ID:{trade_id}] {symbol} {side} "
                           f"开仓价:{open_price} 平仓价:{close_price} "
                           f"盈亏:{profit_loss:.2f} USDT ({profit_rate:.2f}%)")
            
            return trade_info
            
        except Exception as e:
            self.logger.error(f"记录平仓失败: {str(e)}")
            raise
    
    def get_open_positions(self, symbol: str = None) -> List[Dict[str, Any]]:
        """获取所有未平仓的记录
        
        Args:
            symbol: 交易对，如果为None则返回所有交易对的未平仓记录
            
        Returns:
            未平仓记录列表
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            if symbol:
                cursor.execute('''
                    SELECT id, symbol, side, open_amount, open_price, open_value,
                           open_time, leverage
                    FROM trades
                    WHERE symbol = ? AND status = 'OPEN'
                    ORDER BY open_time DESC
                ''', (symbol,))
            else:
                cursor.execute('''
                    SELECT id, symbol, side, open_amount, open_price, open_value,
                           open_time, leverage
                    FROM trades
                    WHERE status = 'OPEN'
                    ORDER BY open_time DESC
                ''')
            
            rows = cursor.fetchall()
            conn.close()
            
            positions = []
            for row in rows:
                positions.append({
                    'trade_id': row[0],
                    'symbol': row[1],
                    'side': row[2],
                    'open_amount': row[3],
                    'open_price': row[4],
                    'open_value': row[5],
                    'open_time': row[6],
                    'leverage': row[7]
                })
            
            return positions
            
        except Exception as e:
            self.logger.error(f"获取未平仓记录失败: {str(e)}")
            raise
    
    def get_trade_history(self, symbol: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """获取历史交易记录
        
        Args:
            symbol: 交易对，如果为None则返回所有交易对的记录
            limit: 返回记录数量限制
            
        Returns:
            交易记录列表
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            if symbol:
                cursor.execute('''
                    SELECT id, symbol, side, open_amount, open_price, open_value,
                           open_time, close_price, close_time, profit_loss,
                           profit_rate, leverage, status
                    FROM trades
                    WHERE symbol = ?
                    ORDER BY open_time DESC
                    LIMIT ?
                ''', (symbol, limit))
            else:
                cursor.execute('''
                    SELECT id, symbol, side, open_amount, open_price, open_value,
                           open_time, close_price, close_time, profit_loss,
                           profit_rate, leverage, status
                    FROM trades
                    ORDER BY open_time DESC
                    LIMIT ?
                ''', (limit,))
            
            rows = cursor.fetchall()
            conn.close()
            
            trades = []
            for row in rows:
                trades.append({
                    'trade_id': row[0],
                    'symbol': row[1],
                    'side': row[2],
                    'open_amount': row[3],
                    'open_price': row[4],
                    'open_value': row[5],
                    'open_time': row[6],
                    'close_price': row[7],
                    'close_time': row[8],
                    'profit_loss': row[9],
                    'profit_rate': row[10],
                    'leverage': row[11],
                    'status': row[12]
                })
            
            return trades
            
        except Exception as e:
            self.logger.error(f"获取交易历史失败: {str(e)}")
            raise
    
    def get_statistics(self, symbol: str = None) -> Dict[str, Any]:
        """获取交易统计信息
        
        Args:
            symbol: 交易对，如果为None则返回所有交易对的统计
            
        Returns:
            统计信息字典
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # 基础查询条件
            where_clause = "WHERE status = 'CLOSED'"
            params = []
            if symbol:
                where_clause += " AND symbol = ?"
                params.append(symbol)
            
            # 总交易次数
            cursor.execute(f'''
                SELECT COUNT(*) FROM trades {where_clause}
            ''', params)
            total_trades = cursor.fetchone()[0]
            
            # 盈利交易次数
            cursor.execute(f'''
                SELECT COUNT(*) FROM trades {where_clause} AND profit_loss > 0
            ''', params)
            winning_trades = cursor.fetchone()[0]
            
            # 总盈亏
            cursor.execute(f'''
                SELECT SUM(profit_loss) FROM trades {where_clause}
            ''', params)
            total_profit = cursor.fetchone()[0] or 0
            
            # 平均收益率
            cursor.execute(f'''
                SELECT AVG(profit_rate) FROM trades {where_clause}
            ''', params)
            avg_profit_rate = cursor.fetchone()[0] or 0
            
            # 最大单笔盈利
            cursor.execute(f'''
                SELECT MAX(profit_loss) FROM trades {where_clause}
            ''', params)
            max_profit = cursor.fetchone()[0] or 0
            
            # 最大单笔亏损
            cursor.execute(f'''
                SELECT MIN(profit_loss) FROM trades {where_clause}
            ''', params)
            max_loss = cursor.fetchone()[0] or 0
            
            conn.close()
            
            # 计算胜率
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            statistics = {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': total_trades - winning_trades,
                'win_rate': win_rate,
                'total_profit': total_profit,
                'avg_profit_rate': avg_profit_rate,
                'max_profit': max_profit,
                'max_loss': max_loss
            }
            
            return statistics
            
        except Exception as e:
            self.logger.error(f"获取统计信息失败: {str(e)}")
            raise
