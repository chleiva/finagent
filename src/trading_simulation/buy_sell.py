import sqlite3
import uuid
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
import pytz

# Configuration constants
TRADING_BUDGET_PCT = 0.5  # 5% of available cash per trade
BUY_SLIPPAGE_PCT = 0.001   # 0.1% slippage on buy orders
SELL_SLIPPAGE_PCT = 0.001  # 0.1% slippage on sell orders
COMMISSION_PER_TRADE = 1.00  # $1 commission per trade
LATENCY_SECONDS = 2  # 2 seconds execution latency
MARKET_CLOSE_BUFFER_MINUTES = 15  # Sell 15 minutes before market close

# Get timezone objects
try:
    from zoneinfo import ZoneInfo
    NY_TZ = ZoneInfo("America/New_York")
    UTC_TZ = ZoneInfo("UTC")
except ImportError:
    NY_TZ = pytz.timezone("America/New_York")
    UTC_TZ = pytz.UTC

class PositionManager:
    def __init__(self, db_path: str = 'database/realtime_market_data.db'):
        """
        Initialize the position manager with database connection.
        
        Args:
            db_path (str): Path to the SQLite database
        """
        self.db_path = db_path
        self._create_positions_table()
    
    def _create_positions_table(self):
        """Create the positions table if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS positions (
                position_id TEXT PRIMARY KEY,
                simulation_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                current_price REAL NOT NULL,
                buy_price REAL NOT NULL,
                stop_loss_price REAL NOT NULL,
                take_profit_price REAL NOT NULL,
                actual_sell_price REAL,
                buy_timestamp TEXT NOT NULL,
                sell_timestamp TEXT,
                buy_sell_duration_minutes REAL,
                total_cost REAL NOT NULL,
                total_proceeds REAL,
                net_pnl REAL,
                commission_paid REAL NOT NULL,
                status TEXT NOT NULL DEFAULT 'OPEN',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create index for faster lookups
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_positions_simulation_symbol_status 
            ON positions(simulation_id, symbol, status)
        ''')
        
        conn.commit()
        conn.close()
    
    def buy(self, simulation_id: str, symbol: str, current_price: float, 
            features: Dict[str, Any], simulation_time_utc: datetime, 
            current_cash: float) -> float:
        """
        Execute a buy order and create a new position.
        
        Args:
            simulation_id (str): Unique simulation identifier
            symbol (str): Stock symbol
            current_price (float): Current market price
            features (Dict): Features dictionary containing stop loss and target prices
            simulation_time_utc (datetime): Current simulation time in UTC
            current_cash (float): Available cash balance
            
        Returns:
            float: New cash balance after the purchase
        """
        try:
            # Check if there's already an open position for this symbol
            existing_positions = self._get_open_positions(simulation_id, symbol)
            if existing_positions:
                print(f"⚠️  SKIPPING BUY: Open position already exists for {symbol}")
                return current_cash
            
            # Calculate position size based on trading budget
            trade_amount = current_cash * TRADING_BUDGET_PCT
            
            # Apply buy slippage
            buy_price = current_price * (1 + BUY_SLIPPAGE_PCT)
            
            # Calculate quantity (whole shares only)
            quantity = int(trade_amount / buy_price)
            
            if quantity <= 0:
                print(f"WARNING: Insufficient cash to buy {symbol} at ${buy_price:.2f}")
                return current_cash
            
            # Calculate actual cost
            total_cost = (quantity * buy_price) + COMMISSION_PER_TRADE
            
            if total_cost > current_cash:
                print(f"WARNING: Total cost ${total_cost:.2f} exceeds available cash ${current_cash:.2f}")
                return current_cash
            
            # Extract stop loss and take profit from features
            stop_loss_price = features.get('1R_Stop_Loss', buy_price * 0.98)  # 2% default stop
            take_profit_price = features.get('2R_Target', buy_price * 1.04)   # 4% default target
            
            # Calculate buy timestamp with latency
            buy_timestamp = simulation_time_utc + timedelta(seconds=LATENCY_SECONDS)
            
            # Generate unique position ID
            position_id = str(uuid.uuid4())
            
            # Insert position into database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO positions (
                    position_id, simulation_id, symbol, quantity, current_price, buy_price,
                    stop_loss_price, take_profit_price, buy_timestamp,
                    total_cost, commission_paid, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN')
            ''', (
                position_id, simulation_id, symbol, quantity, current_price, buy_price,
                stop_loss_price, take_profit_price, buy_timestamp.isoformat(),
                total_cost, COMMISSION_PER_TRADE
            ))
            
            conn.commit()
            conn.close()
            
            # Calculate new cash balance
            new_cash = current_cash - total_cost
            
            print(f"✅ BUY ORDER EXECUTED:")
            print(f"   Symbol: {symbol}")
            print(f"   Quantity: {quantity} shares")
            print(f"   Buy Price: ${buy_price:.2f} (slippage: {BUY_SLIPPAGE_PCT*100:.1f}%)")
            print(f"   Stop Loss: ${stop_loss_price:.2f}")
            print(f"   Take Profit: ${take_profit_price:.2f}")
            print(f"   Total Cost: ${total_cost:.2f}")
            print(f"   Remaining Cash: ${new_cash:.2f}")
            print(f"   Position ID: {position_id}")
            
            return new_cash
            
        except Exception as e:
            print(f"❌ ERROR in buy function: {e}")
            return current_cash
    
    def sell(self, symbol: str, simulation_time_utc: datetime, 
             current_cash: float, simulation_id: str, 
             current_price: float) -> float:
        """
        Check for sell conditions and execute sell orders if triggered.
        
        Args:
            symbol (str): Stock symbol
            simulation_time_utc (datetime): Current simulation time in UTC
            current_cash (float): Current cash balance
            simulation_id (str): Unique simulation identifier
            current_price (float): Current market price
            
        Returns:
            float: New cash balance after any sales
        """
        try:
            # Check for open positions
            open_positions = self._get_open_positions(simulation_id, symbol)
            
            if not open_positions:
                return current_cash  # No open positions
            
            new_cash = current_cash
            
            for position in open_positions:
                sell_reason = self._check_sell_conditions(position, current_price, simulation_time_utc)
                
                if sell_reason:
                    new_cash = self._execute_sell(position, current_price, simulation_time_utc, 
                                                new_cash, sell_reason)
            
            return new_cash
            
        except Exception as e:
            print(f"❌ ERROR in sell function: {e}")
            return current_cash
    
    def _get_open_positions(self, simulation_id: str, symbol: str) -> list:
        """Get all open positions for a symbol in the current simulation."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT position_id, simulation_id, symbol, quantity, buy_price,
                   stop_loss_price, take_profit_price, buy_timestamp, total_cost
            FROM positions
            WHERE simulation_id = ? AND symbol = ? AND status = 'OPEN'
        ''', (simulation_id, symbol))
        
        rows = cursor.fetchall()
        conn.close()
        
        positions = []
        for row in rows:
            positions.append({
                'position_id': row[0],
                'simulation_id': row[1],
                'symbol': row[2],
                'quantity': row[3],
                'buy_price': row[4],
                'stop_loss_price': row[5],
                'take_profit_price': row[6],
                'buy_timestamp': row[7],
                'total_cost': row[8]
            })
        
        return positions
    
    def _check_sell_conditions(self, position: Dict[str, Any], current_price: float, 
                              simulation_time_utc: datetime) -> Optional[str]:
        """
        Check if any sell conditions are met for a position.
        
        Returns:
            str: Sell reason if conditions are met, None otherwise
        """
        # Check stop loss
        if current_price <= position['stop_loss_price']:
            return "STOP_LOSS"
        
        # Check take profit
        if current_price >= position['take_profit_price']:
            return "TAKE_PROFIT"
        
        # Check if approaching market close
        if self._is_near_market_close(simulation_time_utc):
            return "MARKET_CLOSE"
        
        return None
    
    def _is_near_market_close(self, simulation_time_utc: datetime) -> bool:
        """Check if current time is within 15 minutes of market close."""
        # Convert to NY time
        ny_time = simulation_time_utc.astimezone(NY_TZ)
        
        # Check if it's a weekday
        if ny_time.weekday() >= 5:  # Weekend
            return False
        
        # Market closes at 4:00 PM ET
        market_close = ny_time.replace(hour=16, minute=0, second=0, microsecond=0)
        close_warning_time = market_close - timedelta(minutes=MARKET_CLOSE_BUFFER_MINUTES)
        
        return ny_time >= close_warning_time
    
    def _execute_sell(self, position: Dict[str, Any], current_price: float, 
                     simulation_time_utc: datetime, current_cash: float, 
                     sell_reason: str) -> float:
        """Execute a sell order for a position."""
        try:
            # Apply sell slippage
            sell_price = current_price * (1 - SELL_SLIPPAGE_PCT)
            
            # Calculate proceeds
            total_proceeds = (position['quantity'] * sell_price) - COMMISSION_PER_TRADE
            
            # Calculate sell timestamp with latency
            sell_timestamp = simulation_time_utc + timedelta(seconds=LATENCY_SECONDS)
            
            # Calculate trade duration
            buy_time = datetime.fromisoformat(position['buy_timestamp'])
            duration_minutes = (sell_timestamp - buy_time).total_seconds() / 60
            
            # Calculate P&L
            net_pnl = total_proceeds - position['total_cost']
            
            # Update position in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE positions SET
                    actual_sell_price = ?,
                    sell_timestamp = ?,
                    buy_sell_duration_minutes = ?,
                    total_proceeds = ?,
                    net_pnl = ?,
                    commission_paid = ?,
                    status = 'CLOSED'
                WHERE position_id = ?
            ''', (
                sell_price, sell_timestamp.isoformat(), duration_minutes,
                total_proceeds, net_pnl, COMMISSION_PER_TRADE * 2,  # Buy + Sell commission
                position['position_id']
            ))
            
            conn.commit()
            conn.close()
            
            # Update cash balance
            new_cash = current_cash + total_proceeds
            
            # Print sell confirmation
            pnl_pct = (net_pnl / position['total_cost']) * 100
            pnl_color = "🟢" if net_pnl >= 0 else "🔴"
            
            print(f"✅ SELL ORDER EXECUTED ({sell_reason}):")
            print(f"   Symbol: {position['symbol']}")
            print(f"   Quantity: {position['quantity']} shares")
            print(f"   Sell Price: ${sell_price:.2f} (slippage: {SELL_SLIPPAGE_PCT*100:.1f}%)")
            print(f"   Buy Price: ${position['buy_price']:.2f}")
            print(f"   Duration: {duration_minutes:.1f} minutes")
            print(f"   Total Proceeds: ${total_proceeds:.2f}")
            print(f"   {pnl_color} P&L: ${net_pnl:.2f} ({pnl_pct:+.2f}%)")
            print(f"   New Cash Balance: ${new_cash:.2f}")
            
            return new_cash
            
        except Exception as e:
            print(f"❌ ERROR executing sell order: {e}")
            return current_cash
    
    def get_position_summary(self, simulation_id: str) -> Dict[str, Any]:
        """Get a summary of all positions for a simulation."""
        conn = sqlite3.connect(self.db_path)
        
        # Get position summary
        query = '''
            SELECT 
                COUNT(*) as total_trades,
                COUNT(CASE WHEN status = 'OPEN' THEN 1 END) as open_positions,
                COUNT(CASE WHEN status = 'CLOSED' THEN 1 END) as closed_positions,
                SUM(CASE WHEN status = 'CLOSED' THEN net_pnl ELSE 0 END) as total_pnl,
                SUM(CASE WHEN status = 'CLOSED' AND net_pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                SUM(CASE WHEN status = 'CLOSED' AND net_pnl < 0 THEN 1 ELSE 0 END) as losing_trades,
                AVG(CASE WHEN status = 'CLOSED' THEN buy_sell_duration_minutes END) as avg_duration_minutes,
                SUM(commission_paid) as total_commissions
            FROM positions 
            WHERE simulation_id = ?
        '''
        
        df = pd.read_sql_query(query, conn, params=(simulation_id,))
        conn.close()
        
        if df.empty:
            summary = {
                'total_trades': 0, 'open_positions': 0, 'closed_positions': 0,
                'total_pnl': 0.0, 'winning_trades': 0, 'losing_trades': 0,
                'avg_duration_minutes': 0.0, 'total_commissions': 0.0
            }
        else:
            summary = df.iloc[0].to_dict()
            # Replace None values with 0
            for key, value in summary.items():
                if value is None:
                    summary[key] = 0.0
        
        # Calculate win rate
        closed_trades = summary['closed_positions']
        if closed_trades > 0:
            summary['win_rate'] = (summary['winning_trades'] / closed_trades) * 100
        else:
            summary['win_rate'] = 0.0
        
        return summary
        
    def _get_all_positions(self, simulation_id: str) -> list:
        """Get all positions (open and closed) for a simulation."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT position_id, simulation_id, symbol, quantity, current_price, buy_price,
                   stop_loss_price, take_profit_price, actual_sell_price,
                   buy_timestamp, sell_timestamp, buy_sell_duration_minutes,
                   total_cost, status, net_pnl
            FROM positions
            WHERE simulation_id = ?
            ORDER BY buy_timestamp DESC
        ''', (simulation_id,))
        
        rows = cursor.fetchall()
        conn.close()
        
        positions = []
        for row in rows:
            positions.append({
                'position_id': row[0],
                'simulation_id': row[1],
                'symbol': row[2],
                'quantity': row[3],
                'current_price': row[4],
                'buy_price': row[5],
                'stop_loss_price': row[6],
                'take_profit_price': row[7],
                'actual_sell_price': row[8],
                'buy_timestamp': row[9],
                'sell_timestamp': row[10],
                'buy_sell_duration_minutes': row[11],
                'total_cost': row[12],
                'status': row[13],
                'net_pnl': row[14]
            })
        
        return positions
    
    def print_position_summary(self, simulation_id: str, current_cash):
        """Print a formatted position summary with detailed position table."""
        try:
            summary = self.get_position_summary(simulation_id)
            
            # Check if summary is None
            if summary is None:
                return
            
            # Check if there are any positions for this simulation
            if summary.get('total_trades', 0) == 0:
                print("\n" + "="*50)
                print("           POSITION SUMMARY")
                print("="*50)
                print("No positions found for this simulation.")
                print("="*50 + "\n")
                return
            
            print("\n" + "="*50)
            print("           POSITION SUMMARY")
            print("="*50)
            print(f"Total Trades: {summary.get('total_trades', 0)}")
            print(f"Open Positions: {summary.get('open_positions', 0)}")
            print(f"Closed Positions: {summary.get('closed_positions', 0)}")
            print(f"Total P&L: ${summary.get('total_pnl', 0):.2f}")
            print(f"Win Rate: {summary.get('win_rate', 0):.1f}%")
            print(f"Average Duration: {summary.get('avg_duration_minutes', 0):.1f} minutes")
            print(f"Total Commissions: ${summary.get('total_commissions', 0):.2f}")
            print("="*50)
            
            # Show detailed position table
            positions = self._get_all_positions(simulation_id)
            if positions:
                print("\nPOSITION DETAILS:")
                
                from tabulate import tabulate
                
                table_data = []
                for pos in positions:
                    # Parse buy timestamp to get just time in NY timezone
                    try:
                        buy_time_utc = datetime.fromisoformat(pos['buy_timestamp'])
                        buy_time_ny = buy_time_utc.astimezone(NY_TZ)
                        time_str = buy_time_ny.strftime('%H:%M:%S')
                    except:
                        time_str = 'N/A'
                    
                    # Format sell price
                    sell_price_str = f"${pos['actual_sell_price']:.2f}" if pos['actual_sell_price'] else "OPEN"
                    
                    # Format duration
                    duration_str = f"{pos['buy_sell_duration_minutes']:.1f}" if pos['buy_sell_duration_minutes'] else "N/A"
                    
                    # Format P&L
                    if pos['status'] == 'CLOSED' and 'net_pnl' in pos:
                        pnl_str = f"${pos['net_pnl']:.2f}" if pos['net_pnl'] is not None else "N/A"
                    else:
                        pnl_str = "OPEN"
                    
                    table_data.append([
                        time_str,
                        pos['symbol'],
                        f"${pos['current_price']:.2f}",
                        f"${pos['buy_price']:.2f}",
                        f"${pos['stop_loss_price']:.2f}",
                        f"${pos['take_profit_price']:.2f}",
                        sell_price_str,
                        duration_str,
                        pnl_str
                    ])
                
                print(tabulate(table_data, 
                             headers=["Time", "Symbol", "Current Price", "Buy Price", "-1R", "2R", "Sell Price", "Duration (min)", "P&L"], 
                             tablefmt="grid"))
            
            print("\n")
            
            # Add current balance info
            print(f"💰 Current Cash Balance: ${current_cash:.2f}")
            print("="*50 + "\n")
            
        except Exception as e:
            print(f"❌ Error in position summary: {e}")
            print("="*50 + "\n")