"""
Commission configuration and calculation module.
Handles both backtest data (with commissions) and live data (estimated commissions).
"""

import json
import os
from typing import Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import re
from functools import lru_cache
import time

@dataclass
class CommissionRates:
    """Commission rates for different underlyings."""
    opening_cost: float
    closing_cost: float
    exercise_fee: float = 9.00
    
    @property
    def total_round_trip(self) -> float:
        """Total commission for opening and closing a position."""
        return self.opening_cost + self.closing_cost

@dataclass
class CommissionConfig:
    """Configuration for commission calculations."""
    spx: CommissionRates = None
    qqq: CommissionRates = None
    default: CommissionRates = None
    
    def __post_init__(self):
        if self.spx is None:
            self.spx = CommissionRates(opening_cost=1.78, closing_cost=0.78)
        if self.qqq is None:
            self.qqq = CommissionRates(opening_cost=1.00, closing_cost=0.00)
        if self.default is None:
            self.default = CommissionRates(opening_cost=1.00, closing_cost=0.00)

class CommissionCalculator:
    """Calculates commissions for trades based on configuration (optimized)."""
    
    def __init__(self, config: CommissionConfig = None):
        self.config = config or CommissionConfig()
        
        # Pre-compile regex patterns for better performance
        self._spx_pattern = re.compile(r'^\d+\s+[A-Z]{3}\s+\d{1,2}\s+\d{4,5}\s+[CP]\s+(STO|BTO)')
        self._symbol_patterns = [
            re.compile(r'^(SPX|SPY|QQQ|IWM|DIA)\s'),  # Common symbols at start
            re.compile(r'^([A-Z]{2,4})\s+\d{1,2}/'),  # Symbol followed by date format MM/DD
            re.compile(r'^([A-Z]{2,5})\s+[A-Z]{3}\s+\d'),  # Symbol followed by month abbreviation
        ]
        self._strike_pattern = re.compile(r'(\d{4,5})\s+[CP]')
        self._call_pattern = re.compile(r'(\d+)\s+C\s+(STO|BTO)')
        self._put_pattern = re.compile(r'(\d+)\s+P\s+(STO|BTO)')
        self._strike_match_pattern = re.compile(r'(\d+)\s+[CP]\s+')
        
        # Cache for frequently accessed data
        self._underlying_cache = {}
        self._contracts_cache = {}
        
    def get_rates_for_underlying(self, underlying: str) -> CommissionRates:
        """Get commission rates for a specific underlying."""
        underlying_clean = underlying.upper().strip()
        
        if 'SPX' in underlying_clean:
            return self.config.spx
        elif 'QQQ' in underlying_clean or 'NASDAQ' in underlying_clean:
            return self.config.qqq
        else:
            return self.config.default
    
    def extract_underlying_from_legs(self, legs: str) -> str:
        """Extract underlying symbol from legs string (optimized with caching)."""
        if not legs:
            return 'UNKNOWN'
        
        # Check cache first
        legs_key = legs.strip()
        if legs_key in self._underlying_cache:
            return self._underlying_cache[legs_key]
        
        # For legs format like "2 Aug 23 5650 C STO 4.15", the underlying is typically SPX
        # Since the date format suggests SPX options, let's look for patterns
        legs_upper = legs.upper().strip()
        
        # Check if this looks like SPX format (number + month + day + year)
        # Pattern: "2 AUG 23 5650 C STO" or "1 JAN 24 4000 P BTO"
        if self._spx_pattern.match(legs_upper):
            result = 'SPX'
            self._underlying_cache[legs_key] = result
            return result
        
        # Look for explicit symbols at the start using pre-compiled patterns
        # Examples: "SPX 01/18/24 C4400", "QQQ 12/15/23 P320"
        for pattern in self._symbol_patterns:
            match = pattern.search(legs_upper)
            if match:
                result = match.group(1)
                self._underlying_cache[legs_key] = result
                return result
        
        # If we can't determine from legs, check if we can infer from strike levels
        # SPX typically has strikes in 3000-6000 range
        strike_match = self._strike_pattern.search(legs_upper)
        if strike_match:
            strike = int(strike_match.group(1))
            if 2000 <= strike <= 7000:  # Typical SPX range
                result = 'SPX'
            elif 200 <= strike <= 700:  # Typical QQQ range  
                result = 'QQQ'
            elif 100 <= strike <= 600:  # Typical SPY range
                result = 'SPY'
            else:
                result = 'UNKNOWN'
        else:
            result = 'UNKNOWN'
        
        # Cache the result
        self._underlying_cache[legs_key] = result
        return result
    
    def calculate_actual_contracts_from_legs(self, legs: str) -> int:
        """Calculate total contract count by parsing the legs string (optimized with caching)."""
        if not legs or legs.strip() == '':
            return 1  # Default fallback
        
        # Check cache first
        legs_key = legs.strip()
        if legs_key in self._contracts_cache:
            return self._contracts_cache[legs_key]
        
        total_contracts = 0
        
        # Split legs by | separator and process each leg
        leg_parts = legs.split('|') if '|' in legs else [legs]
        
        for leg in leg_parts:
            leg = leg.strip()
            if not leg:
                continue
                
            # Extract the number at the beginning of each leg
            # Examples: "44 Sep 22 6370 C STO 110.95", "22 Sep 22 6300 C BTO 159.35"
            parts = leg.split()
            if parts and parts[0].isdigit():
                contracts = int(parts[0])
                total_contracts += contracts
            else:
                # Fallback: if we can't parse, assume 1 contract for this leg
                total_contracts += 1
        
        result = total_contracts if total_contracts > 0 else 1
        
        # Cache the result
        self._contracts_cache[legs_key] = result
        return result
    
    def is_trade_live_data(self, trade) -> bool:
        """Determine if trade is from live data (no commission columns) or backtest data."""
        # If commission fields are 0 or missing, likely live data
        opening_comm = getattr(trade, 'opening_commissions', 0)
        closing_comm = getattr(trade, 'closing_commissions', 0)
        
        return opening_comm == 0 and closing_comm == 0
    
    def is_leg_in_the_money(self, leg_info: str, opening_price: float, closing_price: float) -> bool:
        """Determine if a leg expired in the money and will be exercised (optimized)."""
        if not leg_info:
            return False
            
        # Extract strike price and option type from leg info using pre-compiled patterns
        # Example: "44 Sep 22 6370 C STO 110.95" -> Call with strike 6370
        # Format: {contracts} {month} {day} {year} {strike} {C/P} {action} {price}
        leg_upper = leg_info.upper()
        call_match = self._call_pattern.search(leg_upper)
        put_match = self._put_pattern.search(leg_upper)
        
        # Extract strike price (the number before C or P)
        strike_match = self._strike_match_pattern.search(leg_upper)
        if not strike_match:
            return False
            
        strike = float(strike_match.group(1))
        
        if call_match:
            # For calls: ITM if closing price > strike (regardless of long/short)
            if closing_price > strike:
                return True
        elif put_match:
            # For puts: ITM if closing price < strike (regardless of long/short)
            if closing_price < strike:
                return True
            
        return False
    
    def calculate_exercise_costs(self, trade) -> tuple[float, int]:
        """Calculate exercise costs for expired in-the-money legs.
        
        Returns:
            Tuple of (total_exercise_cost, exercised_contracts)
        """
        # Only calculate exercise costs if the trade expired
        reason_for_close = getattr(trade, 'reason_for_close', '').strip().lower()
        if reason_for_close != 'expired':
            return 0.0, 0
            
        if not hasattr(trade, 'legs') or not trade.legs:
            return 0.0, 0
            
        opening_price = getattr(trade, 'opening_price', 0) or 0
        closing_price = getattr(trade, 'closing_price', 0) or 0
        
        if opening_price == 0 or closing_price == 0:
            return 0.0, 0
            
        # Split legs by | separator and check each one
        legs = trade.legs.split('|') if '|' in trade.legs else [trade.legs]
        total_exercise_cost = 0.0
        total_exercised_contracts = 0
        
        underlying = self.extract_underlying_from_legs(trade.legs)
        rates = self.get_rates_for_underlying(underlying)
        
        # Check each leg individually for exercise
        for leg in legs:
            leg_stripped = leg.strip()
            if self.is_leg_in_the_money(leg_stripped, opening_price, closing_price):
                # Extract contract count for this specific leg
                parts = leg_stripped.split()
                if parts and parts[0].isdigit():
                    leg_contracts = int(parts[0])
                    total_exercised_contracts += leg_contracts
                    total_exercise_cost += leg_contracts * rates.exercise_fee
                
        return total_exercise_cost, total_exercised_contracts
    
    
    def get_commission_summary(self, trades) -> Dict:
        """Get comprehensive commission analysis for a list of trades (optimized)."""
        # Pre-process trades for better performance
        trade_data = []
        total_opening_from_csv = 0
        total_contracts = 0
        
        # Single pass through trades to collect all data
        for trade in trades:
            is_live = self.is_trade_live_data(trade)
            legs = getattr(trade, 'legs', '')
            
            # Calculate contracts once
            if legs and legs.strip():
                contracts = self.calculate_actual_contracts_from_legs(legs)
            else:
                contracts = getattr(trade, 'contracts', 1)
            
            # Collect trade data
            trade_info = {
                'trade': trade,
                'is_live': is_live,
                'legs': legs,
                'contracts': contracts,
                'strategy': getattr(trade, 'strategy', 'Unknown'),
                'opening_comm': getattr(trade, 'opening_commissions', 0),
                'closing_comm': getattr(trade, 'closing_commissions', 0),
                'reason_for_close': getattr(trade, 'reason_for_close', '').strip().lower(),
                'opening_price': getattr(trade, 'opening_price', 0) or 0,
                'closing_price': getattr(trade, 'closing_price', 0) or 0
            }
            trade_data.append(trade_info)
            
            # Accumulate totals
            if not is_live:
                total_opening_from_csv += trade_info['opening_comm']
                total_contracts += contracts
        
        # Commission validation (only if needed)
        max_possible_opening = 1.78 * total_contracts  # Using highest commission rate
        
        if total_opening_from_csv > max_possible_opening:
            print(f"⚠️  COMMISSION VALIDATION WARNING:")
            print(f"   Actual opening commissions from CSV: ${total_opening_from_csv:,.2f}")
            print(f"   Maximum possible (1.78 × {total_contracts:,} contracts): ${max_possible_opening:,.2f}")
            print(f"   Ratio: {total_opening_from_csv/max_possible_opening:.2f}x over maximum")
            print(f"   This suggests commission data in CSV may be corrupted or incorrectly calculated")
            
            # Show sample trades for analysis (optimized)
            backtest_trades = [t for t in trade_data if not t['is_live']]
            for i, trade_info in enumerate(backtest_trades[:5]):  # Show first 5 trades
                contracts = trade_info['contracts']
                csv_contracts = getattr(trade_info['trade'], 'contracts', 1)
                print(f"   Trade {i+1}: {contracts} contracts (from legs), CSV shows: {csv_contracts}")
                if trade_info['legs']:
                    print(f"     Legs: {trade_info['legs']}")
                
                opening_comm = trade_info['opening_comm']
                closing_comm = trade_info['closing_comm']
                expected_opening = 1.78 * contracts  # Max possible
                expected_closing = 1.78 * contracts  # Max possible
                print(f"     Opening: ${opening_comm:.2f} (expected ≤${expected_opening:.2f}, ratio: {opening_comm/expected_opening:.2f}x)")
                print(f"     Closing: ${closing_comm:.2f} (expected ≤${expected_closing:.2f}, ratio: {closing_comm/expected_closing:.2f}x)")
        # Initialize counters
        total_actual_opening = 0
        total_actual_closing = 0
        total_estimated_opening = 0
        total_estimated_closing = 0
        total_exercise_costs = 0
        
        actual_contracts = 0
        actual_closing_contracts = 0  # Track contracts that actually had closing transactions
        estimated_contracts = 0
        estimated_closing_contracts = 0  # Track non-expired contracts for live data
        exercise_contracts = 0
        
        live_trades = 0
        backtest_trades = 0
        
        strategy_breakdown = {}
        
        # Process pre-collected trade data (optimized)
        for trade_info in trade_data:
            is_live = trade_info['is_live']
            strategy = trade_info['strategy']
            contracts = trade_info['contracts']
            legs = trade_info['legs']
            reason_for_close = trade_info['reason_for_close']
            is_expired = reason_for_close == 'expired'
            
            # Initialize strategy breakdown if needed
            if strategy not in strategy_breakdown:
                strategy_breakdown[strategy] = {
                    'count': 0,
                    'contracts': 0,
                    'actual_commissions': 0,
                    'estimated_commissions': 0,
                    'exercise_costs': 0,
                    'exercise_contracts': 0
                }
            
            strategy_breakdown[strategy]['count'] += 1
            strategy_breakdown[strategy]['contracts'] += contracts
            
            # Calculate exercise costs once per trade
            exercise_cost, exercised_contracts = self.calculate_exercise_costs(trade_info['trade'])
            
            if is_live:
                # Live data: calculate estimated commissions using rates
                live_trades += 1
                underlying = self.extract_underlying_from_legs(legs)
                rates = self.get_rates_for_underlying(underlying)
                
                opening_est = rates.opening_cost * contracts
                closing_est = 0 if is_expired else rates.closing_cost * contracts
                exercise_est = exercise_cost  # Already calculated above
                
                total_estimated_opening += opening_est
                total_estimated_closing += closing_est
                total_exercise_costs += exercise_est
                estimated_contracts += contracts
                if not is_expired:
                    estimated_closing_contracts += contracts
                if exercise_est > 0:
                    exercise_contracts += exercised_contracts
                
                strategy_breakdown[strategy]['estimated_commissions'] += opening_est + closing_est
                strategy_breakdown[strategy]['exercise_costs'] += exercise_est
                if exercise_est > 0:
                    strategy_breakdown[strategy]['exercise_contracts'] += exercised_contracts
            else:
                # Backtest data: use actual commission amounts from CSV
                backtest_trades += 1
                opening_actual = trade_info['opening_comm']
                closing_actual = trade_info['closing_comm']
                
                # For expired trades, closing commissions should be 0
                if is_expired:
                    closing_actual = 0  # Override CSV value for expired trades
                
                total_actual_opening += opening_actual
                total_actual_closing += closing_actual
                total_exercise_costs += exercise_cost
                actual_contracts += contracts
                if not is_expired:
                    actual_closing_contracts += contracts
                if exercise_cost > 0:
                    exercise_contracts += exercised_contracts
                
                strategy_breakdown[strategy]['actual_commissions'] += opening_actual + closing_actual
                strategy_breakdown[strategy]['exercise_costs'] += exercise_cost
                if exercise_cost > 0:
                    strategy_breakdown[strategy]['exercise_contracts'] += exercised_contracts
        
        return {
            'summary': {
                'total_trades': len(trades),
                'live_trades': live_trades,
                'backtest_trades': backtest_trades,
                'total_actual_commissions': total_actual_opening + total_actual_closing,
                'total_estimated_commissions': total_estimated_opening + total_estimated_closing,
                'total_exercise_costs': total_exercise_costs,
                'total_all_costs': total_actual_opening + total_actual_closing + total_estimated_opening + total_estimated_closing + total_exercise_costs,
                'total_contracts': actual_contracts + estimated_contracts
            },
            'breakdown': {
                'actual_opening': total_actual_opening,
                'actual_closing': total_actual_closing,
                'estimated_opening': total_estimated_opening,
                'estimated_closing': total_estimated_closing,
                'exercise_costs': total_exercise_costs,
                'actual_contracts': actual_contracts,
                'actual_closing_contracts': actual_closing_contracts,
                'estimated_contracts': estimated_contracts,
                'estimated_closing_contracts': estimated_closing_contracts,
                'exercise_contracts': exercise_contracts
            },
            'by_strategy': strategy_breakdown,
            'commission_rates': {
                'spx': asdict(self.config.spx),
                'qqq': asdict(self.config.qqq),
                'default': asdict(self.config.default)
            }
        }

class CommissionConfigManager:
    """Manages saving and loading commission configuration."""
    
    def __init__(self, config_file: str = None):
        if config_file is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            config_file = os.path.join(base_dir, "data", "commission_config.json")
        self.config_file = config_file
        
    def save_config(self, config: CommissionConfig):
        """Save commission configuration to file."""
        os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
        
        config_dict = {
            'spx': asdict(config.spx),
            'qqq': asdict(config.qqq),
            'default': asdict(config.default)
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def load_config(self) -> CommissionConfig:
        """Load commission configuration from file."""
        if not os.path.exists(self.config_file):
            return CommissionConfig()  # Return default config
            
        try:
            with open(self.config_file, 'r') as f:
                config_dict = json.load(f)
                
            return CommissionConfig(
                spx=CommissionRates(**config_dict.get('spx', {})),
                qqq=CommissionRates(**config_dict.get('qqq', {})),
                default=CommissionRates(**config_dict.get('default', {}))
            )
        except Exception as e:
            print(f"Error loading commission config: {e}")
            return CommissionConfig()  # Return default config on error
