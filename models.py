"""
Data models for portfolio strategy analysis.
Optimized for strategy-focused analytics with 25k+ trades.
"""

import sqlite3
import pandas as pd
import os
from datetime import datetime
from typing import List, Dict, Optional
import numpy as np
import csv
import re

class DatabaseManager:
    """Handles SQLite database operations."""
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            # Get the directory where this file is located
            base_dir = os.path.dirname(os.path.abspath(__file__))
            db_path = os.path.join(base_dir, "data", "portfolio.db")
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database with strategy-focused schema."""
        conn = sqlite3.connect(self.db_path)
        
        # Trades table optimized for strategy analysis
        conn.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date_opened DATE NOT NULL,
                time_opened TIME,
                date_closed DATE NOT NULL,
                time_closed TIME,
                strategy TEXT NOT NULL,
                pnl REAL NOT NULL,
                funds_at_close REAL,
                opening_price REAL,
                closing_price REAL,
                position_size INTEGER,
                trade_type TEXT,  -- 'LONG', 'SHORT', 'NEUTRAL'
                upload_batch TEXT,  -- Track which upload this came from
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Strategy metadata table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS strategies (
                name TEXT PRIMARY KEY,
                strategy_type TEXT,  -- 'BULLISH', 'BEARISH', 'NEUTRAL'
                description TEXT,
                target_allocation REAL,  -- Target % of portfolio
                risk_level TEXT,  -- 'LOW', 'MEDIUM', 'HIGH'
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes for performance
        conn.execute('CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_trades_date ON trades(date_opened)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_trades_pnl ON trades(pnl)')
        
        conn.close()
    
    def get_connection(self):
        """Get database connection."""
        return sqlite3.connect(self.db_path)

class Trade:
    """Represents a single trade with strategy context."""
    
    def __init__(self, **kwargs):
        self.date_opened = kwargs.get('date_opened')
        self.time_opened = kwargs.get('time_opened')
        self.date_closed = kwargs.get('date_closed')
        self.time_closed = kwargs.get('time_closed')
        self.strategy = kwargs.get('strategy')
        self.pnl = float(kwargs.get('pnl', 0))
        self.funds_at_close = kwargs.get('funds_at_close')
        self.trade_type = kwargs.get('trade_type', 'UNKNOWN')
        self.contracts = int(kwargs.get('contracts', 1))
        self.opening_commissions = float(kwargs.get('opening_commissions', 0))
        self.closing_commissions = float(kwargs.get('closing_commissions', 0))
        self.legs = kwargs.get('legs', '')  # Add legs field
        self.margin_req = float(kwargs.get('margin_req', 0))  # Add margin requirement field
        self.reason_for_close = kwargs.get('reason_for_close', '')  # Add reason for close field
        self.opening_price = kwargs.get('opening_price')  # Add opening price field
        self.closing_price = kwargs.get('closing_price')  # Add closing price field
    
    @property
    def is_winner(self) -> bool:
        """Check if trade was profitable."""
        return self.pnl > 0
    
    @property
    def duration_days(self) -> int:
        """Calculate trade duration in days."""
        if self.date_opened and self.date_closed:
            open_date = pd.to_datetime(self.date_opened)
            close_date = pd.to_datetime(self.date_closed)
            return (close_date - open_date).days
        return 0
    
    @property
    def pnl_per_lot(self) -> float:
        """P&L per contract/lot."""
        return self.pnl / self.contracts if self.contracts > 0 else 0
    
    @property
    def total_commissions(self) -> float:
        """Total commissions for the trade."""
        return self.opening_commissions + self.closing_commissions
    
    @property
    def commissions_per_lot(self) -> float:
        """Commissions per contract/lot."""
        return self.total_commissions / self.contracts if self.contracts > 0 else 0
    
    @property
    def margin_per_contract(self) -> float:
        """Margin requirement per contract."""
        return self.margin_req / self.contracts if self.contracts > 0 else 0

class Strategy:
    """Represents a trading strategy with its performance metrics."""
    
    def __init__(self, name: str, strategy_type: str = 'NEUTRAL'):
        self.name = name
        self.strategy_type = strategy_type  # BULLISH, BEARISH, NEUTRAL
        self.trades: List[Trade] = []
    
    def add_trade(self, trade: Trade):
        """Add a trade to this strategy."""
        self.trades.append(trade)
    
    @property
    def total_pnl(self) -> float:
        """Total P&L for this strategy."""
        return sum(trade.pnl for trade in self.trades)
    
    @property
    def trade_count(self) -> int:
        """Number of trades for this strategy."""
        return len(self.trades)
    
    @property
    def win_rate(self) -> float:
        """Win rate as percentage."""
        if not self.trades:
            return 0.0
        winners = sum(1 for trade in self.trades if trade.is_winner)
        return (winners / len(self.trades)) * 100
    
    @property
    def avg_win(self) -> float:
        """Average winning trade amount."""
        winners = [trade.pnl for trade in self.trades if trade.is_winner]
        return np.mean(winners) if winners else 0.0
    
    @property
    def avg_loss(self) -> float:
        """Average losing trade amount."""
        losers = [trade.pnl for trade in self.trades if not trade.is_winner]
        return np.mean(losers) if losers else 0.0
    
    @property
    def max_win(self) -> float:
        """Largest winning trade."""
        winners = [trade.pnl for trade in self.trades if trade.is_winner]
        return max(winners) if winners else 0.0
    
    @property
    def max_loss(self) -> float:
        """Largest losing trade."""
        losers = [trade.pnl for trade in self.trades if not trade.is_winner]
        return min(losers) if losers else 0.0
    
    @property
    def total_contracts(self) -> int:
        """Total contracts traded for this strategy."""
        return sum(trade.contracts for trade in self.trades)
    
    @property
    def total_commissions(self) -> float:
        """Total commissions paid for this strategy."""
        return sum(trade.total_commissions for trade in self.trades)
    
    @property
    def avg_win_per_lot(self) -> float:
        """Average winning trade amount per lot."""
        winners = [trade.pnl_per_lot for trade in self.trades if trade.is_winner]
        return np.mean(winners) if winners else 0.0
    
    @property
    def avg_loss_per_lot(self) -> float:
        """Average losing trade amount per lot."""
        losers = [trade.pnl_per_lot for trade in self.trades if not trade.is_winner]
        return np.mean(losers) if losers else 0.0
    
    @property
    def max_win_per_lot(self) -> float:
        """Largest winning trade per lot."""
        winners = [trade.pnl_per_lot for trade in self.trades if trade.is_winner]
        return max(winners) if winners else 0.0
    
    @property
    def max_loss_per_lot(self) -> float:
        """Largest losing trade per lot."""
        losers = [trade.pnl_per_lot for trade in self.trades if not trade.is_winner]
        return min(losers) if losers else 0.0
    
    @property
    def avg_commissions_per_lot(self) -> float:
        """Average commissions per lot."""
        if self.total_contracts > 0:
            return self.total_commissions / self.total_contracts
        return 0.0
    
    @property
    def expectancy_per_lot(self) -> float:
        """Expected value per lot: probabilistic calculation based on win rate and average wins/losses"""
        if not self.trades or self.total_contracts == 0:
            return 0.0
        
        # Use probabilistic expectancy calculation: (win_rate * avg_win) - (loss_rate * avg_loss)
        win_rate_decimal = self.win_rate / 100
        loss_rate_decimal = (100 - self.win_rate) / 100
        
        expectancy = (win_rate_decimal * self.avg_win_per_lot) - (loss_rate_decimal * abs(self.avg_loss_per_lot))
        return expectancy
    
    @property
    def max_loss_streak(self) -> int:
        """Calculate the maximum consecutive losing trades."""
        if not self.trades:
            return 0
        
        # Sort trades by date to ensure chronological order
        sorted_trades = sorted(self.trades, key=lambda t: t.date_closed or t.date_opened)
        
        max_streak = 0
        current_streak = 0
        
        for trade in sorted_trades:
            if not trade.is_winner:  # Losing trade
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:  # Winning trade
                current_streak = 0
        
        return max_streak
    
    @property
    def max_win_streak(self) -> int:
        """Calculate the maximum consecutive winning trades."""
        if not self.trades:
            return 0
        
        # Sort trades by date to ensure chronological order
        sorted_trades = sorted(self.trades, key=lambda t: t.date_closed or t.date_opened)
        
        max_streak = 0
        current_streak = 0
        
        for trade in sorted_trades:
            if trade.is_winner:  # Winning trade
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:  # Losing trade
                current_streak = 0
        
        return max_streak
    
    def get_summary_stats(self) -> Dict:
        """Get comprehensive strategy statistics."""
        # Single pass through trades for better performance
        if not self.trades:
            return {
                'name': self.name,
                'strategy_type': self.strategy_type,
                'total_pnl': 0,
                'trade_count': 0,
                'wins_count': 0,
                'losses_count': 0,
                'total_contracts': 0,
                'total_commissions': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'max_win': 0,
                'max_loss': 0,
                'avg_win_per_lot': 0,
                'avg_loss_per_lot': 0,
                'max_win_per_lot': 0,
                'max_loss_per_lot': 0,
                'avg_commissions_per_lot': 0,
                'expectancy_per_lot': 0,
                'profit_factor': 0,
                'max_loss_streak': 0,
                'max_win_streak': 0
            }
        
        # Calculate everything in a single pass
        total_pnl = 0
        wins_count = 0
        losses_count = 0
        total_contracts = 0
        total_commissions = 0
        wins = []
        losses = []
        max_win = 0
        max_loss = 0
        max_win_per_lot = 0
        max_loss_per_lot = 0
        
        for trade in self.trades:
            pnl = trade.pnl
            total_pnl += pnl
            total_contracts += getattr(trade, 'contracts', 1)
            total_commissions += getattr(trade, 'opening_commissions', 0) + getattr(trade, 'closing_commissions', 0)
            
            if pnl > 0:
                wins_count += 1
                wins.append(pnl)
                max_win = max(max_win, pnl)
                max_win_per_lot = max(max_win_per_lot, pnl / getattr(trade, 'contracts', 1))
            else:
                losses_count += 1
                losses.append(abs(pnl))
                max_loss = max(max_loss, abs(pnl))
                max_loss_per_lot = max(max_loss_per_lot, abs(pnl) / getattr(trade, 'contracts', 1))
        
        # Calculate averages
        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        avg_win_per_lot = sum(wins) / total_contracts if total_contracts > 0 else 0
        avg_loss_per_lot = sum(losses) / total_contracts if total_contracts > 0 else 0
        avg_commissions_per_lot = total_commissions / total_contracts if total_contracts > 0 else 0
        win_rate = (wins_count / len(self.trades)) * 100 if self.trades else 0
        profit_factor = avg_win / avg_loss if avg_loss > 0 else 0
        
        # Calculate expectancy per lot
        expectancy_per_lot = (avg_win_per_lot * (wins_count / len(self.trades))) - (avg_loss_per_lot * (losses_count / len(self.trades))) if self.trades else 0
        
        return {
            'name': self.name,
            'strategy_type': self.strategy_type,
            'total_pnl': round(total_pnl, 2),
            'trade_count': len(self.trades),
            'wins_count': wins_count,
            'losses_count': losses_count,
            'total_contracts': total_contracts,
            'total_commissions': round(total_commissions, 2),
            'win_rate': round(win_rate, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'max_win': round(max_win, 2),
            'max_loss': round(max_loss, 2),
            'avg_win_per_lot': round(avg_win_per_lot, 2),
            'avg_loss_per_lot': round(avg_loss_per_lot, 2),
            'max_win_per_lot': round(max_win_per_lot, 2),
            'max_loss_per_lot': round(max_loss_per_lot, 2),
            'avg_commissions_per_lot': round(avg_commissions_per_lot, 2),
            'expectancy_per_lot': round(expectancy_per_lot, 2),
            'profit_factor': round(profit_factor, 2),
            'max_loss_streak': self.max_loss_streak,
            'max_win_streak': self.max_win_streak
        }

class Portfolio:
    """Container for multiple strategies with portfolio-level analytics."""
    
    def __init__(self, name: str = "Portfolio"):
        self.name = name
        self.strategies: Dict[str, Strategy] = {}
        self.db = DatabaseManager()
    
    def load_from_database(self):
        """Load all trades from database and organize by strategy."""
        conn = self.db.get_connection()
        
        query = '''
            SELECT * FROM trades 
            ORDER BY date_opened, strategy
        '''
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        # Group trades by strategy
        for _, row in df.iterrows():
            strategy_name = row['strategy']
            
            if strategy_name not in self.strategies:
                self.strategies[strategy_name] = Strategy(strategy_name)
            
            trade = Trade(**row.to_dict())
            self.strategies[strategy_name].add_trade(trade)
    
    def get_strategy_summary(self) -> List[Dict]:
        """Get summary statistics for all strategies."""
        return [strategy.get_summary_stats() for strategy in self.strategies.values()]
    
    @property
    def total_pnl(self) -> float:
        """Total portfolio P&L."""
        return sum(strategy.total_pnl for strategy in self.strategies.values())
    
    @property
    def total_trades(self) -> int:
        """Total number of trades across all strategies."""
        return sum(strategy.trade_count for strategy in self.strategies.values())
    
    def get_strategy_types_breakdown(self) -> Dict[str, Dict]:
        """Analyze portfolio by strategy type (bullish/bearish/neutral)."""
        breakdown = {'BULLISH': [], 'BEARISH': [], 'NEUTRAL': []}
        
        for strategy in self.strategies.values():
            breakdown[strategy.strategy_type].append({
                'name': strategy.name,
                'pnl': strategy.total_pnl,
                'trade_count': strategy.trade_count
            })
        
        return breakdown
    
    def get_monthly_pnl_by_strategy(self) -> pd.DataFrame:
        """Get monthly P&L breakdown by strategy for stacked charts."""
        all_trades = []
        
        for strategy_name, strategy in self.strategies.items():
            for trade in strategy.trades:
                all_trades.append({
                    'date': pd.to_datetime(trade.date_closed).to_pydatetime(),
                    'strategy': strategy_name,
                    'pnl': trade.pnl
                })
        
        if not all_trades:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_trades)
        df['month'] = df['date'].dt.to_period('M')
        
        # Group by month and strategy
        monthly_data = df.groupby(['month', 'strategy'])['pnl'].sum().unstack(fill_value=0)
        
        return monthly_data
    
    def _extract_base_filename(self, csv_file_path: str) -> str:
        """Extract base filename without timestamp for use as strategy name.
        
        Example: 'MEIC-BayouBrian-Example-heatmap-times-2-shorts_20250908_130658.csv'
                 becomes 'MEIC-BayouBrian-Example-heatmap-times-2-shorts'
        """
        import os
        import re
        
        # Get filename without directory and extension
        base_name = os.path.splitext(os.path.basename(csv_file_path))[0]
        
        # Remove timestamp pattern (underscore followed by date_time)
        # Pattern: _YYYYMMDD_HHMMSS at the end
        timestamp_pattern = r'_\d{8}_\d{6}$'
        base_name = re.sub(timestamp_pattern, '', base_name)
        
        return base_name

    def load_from_csv(self, csv_file_path: str):
        """Load trades from CSV file and organize by strategy."""
        print(f"Loading trades from {csv_file_path}...")
        
        # Extract base filename for use as fallback strategy name
        fallback_strategy = self._extract_base_filename(csv_file_path)
        fallback_strategy_used = False
        
        with open(csv_file_path, 'r', newline='', encoding='utf-8-sig') as csvfile:
            reader = csv.DictReader(csvfile)
            
            for row_num, row in enumerate(reader, 1):
                try:
                    # Parse dates - handle quoted values
                    date_opened_str = row['Date Opened'].strip('"') if row['Date Opened'] else None
                    if not date_opened_str:
                        print(f"Warning: Row {row_num} has empty Date Opened field")
                        continue
                    
                    date_closed_str = row['Date Closed'].strip('"') if row['Date Closed'] else None
                    
                    try:
                        date_opened = pd.to_datetime(date_opened_str).date()
                    except Exception as date_e:
                        print(f"Warning: Row {row_num} - Invalid date format for Date Opened '{date_opened_str}': {date_e}")
                        continue
                    
                    date_closed = None
                    if date_closed_str:
                        try:
                            date_closed = pd.to_datetime(date_closed_str).date()
                        except Exception as date_e:
                            print(f"Warning: Row {row_num} - Invalid date format for Date Closed '{date_closed_str}': {date_e}")
                            continue
                    
                    # Parse P&L and portfolio value
                    try:
                        pnl = float(row['P/L'])
                    except Exception as pnl_e:
                        print(f"Warning: Row {row_num} - Invalid P/L value '{row['P/L']}': {pnl_e}")
                        continue
                    
                    try:
                        funds_at_close = float(row['Funds at Close'])
                    except Exception as funds_e:
                        print(f"Warning: Row {row_num} - Invalid Funds at Close value '{row['Funds at Close']}': {funds_e}")
                        continue
                    
                    # Extract and classify strategy
                    strategy_raw = ""
                    if 'Strategy' in row and row['Strategy']:
                        strategy_raw = row['Strategy'].strip('"')
                    
                    if not strategy_raw:
                        # Use filename as fallback strategy when Strategy column is missing or empty
                        if not fallback_strategy_used:
                            print(f"Strategy column missing or empty - using filename-based strategy: '{fallback_strategy}'")
                            fallback_strategy_used = True
                        strategy_raw = fallback_strategy
                    
                    strategy_name = self._extract_strategy_name(strategy_raw)
                    
                    # Parse opening and closing prices for classification
                    try:
                        opening_price = float(row['Opening Price'])
                        closing_price = float(row['Closing Price'])
                    except Exception:
                        opening_price = None
                        closing_price = None
                    
                    strategy_type = self._classify_strategy_type(strategy_raw, opening_price, closing_price, pnl)
                    
                    # Create or get strategy
                    if strategy_name not in self.strategies:
                        self.strategies[strategy_name] = Strategy(strategy_name, strategy_type)
                    
                    # Parse contracts and commissions
                    try:
                        contracts = int(float(row['No. of Contracts']))
                    except Exception:
                        contracts = 1
                    
                    try:
                        opening_commissions = float(row['Opening Commissions + Fees'])
                    except Exception:
                        opening_commissions = 0
                    
                    try:
                        closing_commissions = float(row['Closing Commissions + Fees'])
                    except Exception:
                        closing_commissions = 0
                    
                    # Parse margin requirement
                    try:
                        margin_req = float(row['Margin Req.'])
                    except Exception:
                        margin_req = 0
                    
                    # Parse time opened/closed if present
                    time_opened = row.get('Time Opened', '').strip() if 'Time Opened' in row else None
                    if time_opened == '':
                        time_opened = None
                    time_closed = row.get('Time Closed', '').strip() if 'Time Closed' in row else None
                    if time_closed == '':
                        time_closed = None
                    
                    # Get legs data if present
                    legs = row.get('Legs', '').strip() if 'Legs' in row else ''
                    
                    # Get reason for close if present
                    reason_for_close = row.get('Reason For Close', '').strip() if 'Reason For Close' in row else ''
                    
                    # Create trade
                    trade = Trade(
                        date_opened=date_opened,
                        time_opened=time_opened,
                        date_closed=date_closed,
                        time_closed=time_closed,
                        strategy=strategy_name,
                        pnl=pnl,
                        funds_at_close=funds_at_close,
                        trade_type=strategy_type,
                        contracts=contracts,
                        opening_commissions=opening_commissions,
                        closing_commissions=closing_commissions,
                        legs=legs,  # Add legs to the trade
                        margin_req=margin_req,  # Add margin requirement
                        reason_for_close=reason_for_close,  # Add reason for close
                        opening_price=opening_price,  # Add opening price
                        closing_price=closing_price  # Add closing price
                    )
                    
                    self.strategies[strategy_name].add_trade(trade)
                    
                except Exception as e:
                    print(f"Warning: Unexpected error processing row {row_num}: {e}")
                    print(f"  Row data: {dict(row)}")
                    continue
        
        print(f"Loaded {self.total_trades} trades across {len(self.strategies)} strategies")
    
    def _extract_strategy_name(self, strategy_raw: str) -> str:
        """Extract clean strategy name from raw strategy string."""
        # Return the full strategy name as provided in the CSV
        # Just clean up any extra whitespace and quotes
        strategy_clean = strategy_raw.strip().strip('"')
        return strategy_clean
    
    def _classify_strategy_type(self, strategy_raw: str, opening_price: float = None, closing_price: float = None, pnl: float = None) -> str:
        """Classify strategy based on market movement and performance."""
        
        # If we don't have price data, default to NEUTRAL
        if opening_price is None or closing_price is None:
            return 'NEUTRAL'
        
        # Calculate market movement
        price_change_pct = ((closing_price - opening_price) / opening_price) * 100
        
        # Classify based on how the strategy performed relative to market movement
        # BULLISH: Strategy makes money when market goes up OR loses money when market goes down
        # BEARISH: Strategy makes money when market goes down OR loses money when market goes up  
        # NEUTRAL: Strategy performance is not strongly correlated with market direction
        
        market_up = price_change_pct > 0.5  # Market moved up more than 0.5%
        market_down = price_change_pct < -0.5  # Market moved down more than 0.5%
        strategy_profitable = pnl > 0
        
        if market_up and strategy_profitable:
            return 'BULLISH'
        elif market_down and not strategy_profitable:
            return 'BULLISH'  # Strategy that loses money when market drops is bullish
        elif market_down and strategy_profitable:
            return 'BEARISH'
        elif market_up and not strategy_profitable:
            return 'BEARISH'  # Strategy that loses money when market rises is bearish
        else:
            return 'NEUTRAL'  # Small market movements or mixed signals 