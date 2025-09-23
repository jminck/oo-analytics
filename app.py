"""
Main Flask application for portfolio strategy analytics with authentication.
Simple but powerful analytics platform optimized for strategy comparison and portfolio balance.
"""

from flask import Flask, request, render_template, jsonify, redirect, url_for, session
from flask_login import current_user, login_required
import pandas as pd
import os
import glob
import sqlite3
from datetime import datetime
import json
import re
from collections import defaultdict
import csv
from scipy import stats
import zipfile
import shutil

from models import Portfolio, DatabaseManager
from analytics import StrategyAnalyzer, PortfolioMetrics, MonteCarloSimulator
from commission_config import CommissionCalculator, CommissionConfig, CommissionConfigManager
from charts import ChartFactory
from file_manager import FileManager
from auth import init_auth, auth_bp, guest_mode_required, get_current_data_folder
from admin import admin_bp
from config import Config
from app_insights import app_insights
from version import get_version_string, get_version_info, get_build_badge

# Configure paths for Azure App Service
if os.environ.get('WEBSITES_PORT'):
    # Azure App Service - use persistent paths for data
    DATA_DIR = "/home/site/wwwroot/data"
    INSTANCE_DIR = "/home/site/wwwroot/instance"
    LOGS_DIR = "/home/site/wwwroot/logs"
else:
    # Local development
    DATA_DIR = "data"
    INSTANCE_DIR = "instance"
    LOGS_DIR = "logs"

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(INSTANCE_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

def get_current_user_id():
    """Get the current user ID, handling both authenticated users and guests."""
    try:
        if current_user and current_user.is_authenticated:
            return str(current_user.id)
        else:
            # For guests, use the guest_id from session
            return session.get('guest_id', 'anonymous')
    except Exception:
        # Fallback for cases where current_user is not available
        return session.get('guest_id', 'anonymous')

# Create Flask app
app = Flask(__name__)
app.config.from_object(Config)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Initialize Application Insights (only in production)
app_insights.init_app(app)

# Initialize authentication
init_auth(app)

# Register authentication blueprint
app.register_blueprint(auth_bp, url_prefix='/auth')

# Register admin blueprint
app.register_blueprint(admin_bp)

# Create base data directories
os.makedirs(app.config['DATA_BASE_DIR'], exist_ok=True)
os.makedirs(app.config['GUEST_DATA_DIR'], exist_ok=True)
os.makedirs(app.config['USER_DATA_DIR'], exist_ok=True)

# Initialize file manager and database (will be updated per request)
file_manager = FileManager(app.config['DATA_BASE_DIR'])
db_manager = DatabaseManager()

# Global portfolio instance (will be reloaded per user/session)
portfolio = Portfolio()
analyzer = None
chart_factory = None
current_user_data_dir = None
current_data_type = None
current_initial_capital = None

def setup_user_context():
    """Set up file manager and portfolio for current user/guest session."""
    global file_manager, portfolio, analyzer, chart_factory, current_user_data_dir, current_data_type, current_initial_capital
    
    # Get current user's data folder
    new_data_dir = get_current_data_folder()
    
    # Update file manager to use user-specific directory
    file_manager.set_user_data_dir(new_data_dir)
    
    # Only reload portfolio if user data directory changed or portfolio is empty
    if current_user_data_dir != new_data_dir or len(portfolio.strategies) == 0:
        current_user_data_dir = new_data_dir
        
        # Clear and reload portfolio for this user
        portfolio = Portfolio()
        
        # Reset data type and initial capital when switching users/directories
        current_data_type = None
        current_initial_capital = None
        
        # Try to load any existing data for this user
        try:
            csv_files = glob.glob(os.path.join(new_data_dir, '*.csv'))
            if csv_files:
                # Load the most recent file
                latest_file = max(csv_files, key=os.path.getctime)
                portfolio.load_from_csv(latest_file)
                
                # Set current filename
                portfolio.current_filename = os.path.basename(latest_file)
                
                # Try to detect data type and initial capital from the loaded file
                filename = os.path.basename(latest_file)
                try:
                    df = pd.read_csv(latest_file)
                    if 'Initial Premium' in df.columns:
                        current_data_type = 'real_trade'
                        current_initial_capital = extract_initial_capital_from_filename(filename, default=get_last_initial_capital())
                    else:
                        current_data_type = 'backtest'
                        current_initial_capital = None
                except Exception:
                    pass  # If we can't detect, just leave as None
                
                # print(f"Loaded user data: {portfolio.total_trades} trades from {len(portfolio.strategies)} strategies")
        except Exception as e:
            # print(f"No previous data found for user: {e}")
            pass
    
    # Initialize analyzer and chart factory (these are lightweight)
    analyzer = StrategyAnalyzer(portfolio)
    chart_factory = ChartFactory(portfolio)

@app.before_request
def before_request():
    """Set up user context before each request."""
    # Skip auth setup for static files and auth routes
    if request.endpoint and (request.endpoint.startswith('static') or 
                            request.endpoint.startswith('auth.')):
        return
    
    setup_user_context()

def get_last_initial_capital():
    try:
        capital_file = os.path.join(get_current_data_folder(), 'last_initial_capital.txt')
        if os.path.exists(capital_file):
            with open(capital_file, 'r') as f:
                val = f.read().strip()
                if val.isdigit():
                    return int(val)
    except Exception:
        pass
    return 100000

def set_last_initial_capital(value):
    try:
        capital_file = os.path.join(get_current_data_folder(), 'last_initial_capital.txt')
        with open(capital_file, 'w') as f:
            f.write(str(int(value)))
    except Exception:
        pass

def transform_real_trade_to_backtest(df, initial_capital=10000):
    # Correct Funds at Close: first row is initial capital, then add cumulative P&L
    if 'Funds at Close' not in df.columns:
        df['Funds at Close'] = initial_capital + df['P/L'].cumsum().shift(fill_value=0)
    # Add missing columns with defaults
    for col, default in [
        ('Margin Req.', 0),
        ('Opening Commissions + Fees', 0),
        ('Closing Commissions + Fees', 0),
        ('Opening Short/Long Ratio', 1.0),
        ('Closing Short/Long Ratio', 1.0),
        ('Opening VIX', 20.0),
        ('Closing VIX', 20.0),
        ('Gap', 0),
        ('Movement', 0),
        ('Max Profit', df['P/L'].max() if 'P/L' in df.columns else 0),
        ('Max Loss', df['P/L'].min() if 'P/L' in df.columns else 0),
    ]:
        if col not in df.columns:
            df[col] = default
    # Add time columns if missing, using flexible parsing
    if 'Time Opened' not in df.columns and 'Date Opened' in df.columns:
        df['Time Opened'] = pd.to_datetime(df['Date Opened'], errors='coerce').dt.strftime('%H:%M:%S')
    if 'Time Closed' not in df.columns and 'Date Closed' in df.columns:
        df['Time Closed'] = pd.to_datetime(df['Date Closed'], errors='coerce').dt.strftime('%H:%M:%S')
    return df

def extract_initial_capital_from_filename(filename, default=100000):
    match = re.search(r'__capital=(\d+)', filename)
    if match:
        return int(match.group(1))
    return default

def append_initial_capital_to_filename(filename, initial_capital):
    base, ext = os.path.splitext(filename)
    # Remove any existing __capital= pattern
    base = re.sub(r'__capital=\d+', '', base)
    return f"{base}__capital={int(initial_capital)}{ext}"

# Cache for parsed legs to avoid repeated parsing
_legs_cache = {}

def parse_legs_string(legs_str):
    """
    Parse the legs string to extract individual option legs (optimized with caching).
    
    Format: "quantity expiration strike type action premium | quantity expiration strike type action premium"
    Example: "118 Jul 23 560 P STO 0.75 | 118 Jul 23 570 C STO 0.56"
    
    Returns:
        List of dictionaries with leg information
    """
    if not legs_str or legs_str.strip() == '':
        return []
    
    # Check cache first
    legs_key = legs_str.strip()
    if legs_key in _legs_cache:
        return _legs_cache[legs_key]
    
    legs = []
    # Split by pipe separator
    leg_parts = [part.strip() for part in legs_str.split('|')]
    
    for part in leg_parts:
        if not part:
            continue
            
        # Parse each leg: "quantity expiration strike type action premium"
        # Use regex to handle various formats
        pattern = r'(\d+)\s+([A-Za-z]+\s+\d+)\s+(\d+(?:\.\d+)?)\s+([PC])\s+([STBO]+)\s+([\d.]+)'
        match = re.match(pattern, part.strip())
        
        if match:
            quantity, expiration, strike, option_type, action, premium = match.groups()
            legs.append({
                'quantity': int(quantity),
                'expiration': expiration.strip(),
                'strike': float(strike),
                'type': option_type.upper(),  # P or C
                'action': action.upper(),     # STO or BTO
                'premium': float(premium)
            })
    
    # Cache the result
    _legs_cache[legs_key] = legs
    return legs

# Global variable to store the current debug log path for this session
_current_margin_log_path = None

def _get_margin_log_path() -> str:
    """Resolve the path for the margin debug CSV in the current user's debug folder."""
    global _current_margin_log_path
    
    # If we already have a path for this session, reuse it
    if _current_margin_log_path is not None:
        return _current_margin_log_path
    
    try:
        data_dir = get_current_data_folder()
    except Exception:
        # Fallback to base data dir if session context not available
        data_dir = app.config.get('DATA_BASE_DIR', '.')
    
    # Create debug subfolder
    debug_dir = os.path.join(data_dir, 'debug')
    os.makedirs(debug_dir, exist_ok=True)
    
    # Add timestamp to filename to avoid conflicts
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    _current_margin_log_path = os.path.join(debug_dir, f'margin_debug_log_{timestamp}.csv')
    return _current_margin_log_path


def reset_margin_debug_log():
    """Reset the margin debug log path to create a new log file for the next analysis."""
    global _current_margin_log_path
    _current_margin_log_path = None

def log_margin_debug(row: dict) -> None:
    """Append a row to the margin debug CSV, creating it with headers if needed."""
    log_path = _get_margin_log_path()
    # Define column order
    fieldnames = [
        'timestamp', 'source', 'strategy', 'date_opened', 'legs', 'contracts', 'margin_req',
        'overall_margin', 'margin_per_contract', 'total_contracts', 'trade_type', 'margin_breakdown'
    ]
    # Ensure keys exist
    enriched = {k: row.get(k, '') for k in fieldnames}
    enriched['timestamp'] = enriched.get('timestamp') or datetime.utcnow().isoformat()

    file_exists = os.path.isfile(log_path)
    try:
        with open(log_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(enriched)
    except Exception as e:
        print(f"Failed to write margin debug log: {e}")


def calculate_margin_from_legs(legs_str, context: dict | None = None):
    """
    Calculate margin required for a trade based on the legs string.
    
    Handles different trade types:
    - Credit spreads: (Strike Width × 100) - Credit Received
    - Debit spreads: Debit Paid (capital required)
    - Calendars: Debit Paid (back expiration) - Credit Received (front expiration)
    - Iron condors/butterflies: (Widest Spread Width × 100) - Net Credit
    
    Returns:
        Dictionary with margin information
    """
    # print(f"\n=== MARGIN CALCULATION DEBUG ===")
    # print(f"Input legs string: {legs_str}")
    
    legs = parse_legs_string(legs_str)
    # print(f"Parsed legs: {legs}")
    
    if not legs:
        # print("No legs found or unable to parse legs string")
        result = {
            'overall_margin': 0,
            'margin_per_contract': 0,
            'total_contracts': 0,
            'trade_type': 'unknown',
            'margin_breakdown': 'No legs found or unable to parse legs string'
        }
        # Log even when parsing fails
        try:
            log_margin_debug({
                'source': 'legs',
                'strategy': (context or {}).get('strategy', ''),
                'date_opened': (context or {}).get('date_opened', ''),
                'legs': legs_str,
                'contracts': (context or {}).get('contracts', ''),
                'overall_margin': result['overall_margin'],
                'margin_per_contract': result['margin_per_contract'],
                'total_contracts': result['total_contracts'],
                'trade_type': result['trade_type'],
                'margin_breakdown': result['margin_breakdown']
            })
        except Exception:
            pass
        return result
    
    # Calculate total contracts (each spread/butterfly/condor counts as 1 contract)
    # We'll determine the actual contract count based on the trade type detected
    total_contracts = sum(leg['quantity'] for leg in legs)  # This will be updated based on trade type
    # print(f"Initial total contracts (legs): {total_contracts}")
    
    # Group legs by expiration and type
    legs_by_expiration = {}
    for leg in legs:
        exp = leg['expiration']
        if exp not in legs_by_expiration:
            legs_by_expiration[exp] = {'puts': [], 'calls': []}
        if leg['type'] == 'P':
            legs_by_expiration[exp]['puts'].append(leg)
        else:
            legs_by_expiration[exp]['calls'].append(leg)
    
    # print(f"Legs by expiration: {legs_by_expiration}")
    
    # Sort expirations (assuming format like "Jul 23", "Jul 25")
    expirations = sorted(legs_by_expiration.keys())
    # print(f"Expirations: {expirations}")
    
    # Determine trade type and calculate margin
    trade_type = 'unknown'
    overall_margin = 0
    margin_breakdown = []
    
    try:
        # print(f"Number of expirations: {len(expirations)}")
        
        if len(expirations) == 1:
            # print("=== SINGLE EXPIRATION ANALYSIS ===")
            # Single expiration - likely vertical spreads
            exp = expirations[0]
            puts = legs_by_expiration[exp]['puts']
            calls = legs_by_expiration[exp]['calls']
            # print(f"Expiration: {exp}")
            # print(f"Puts: {puts}")
            # print(f"Calls: {calls}")
            
            # Check for butterfly spreads and condors first (before vertical spreads)
            # PUT BUTTERFLY/CONDOR SECTION
            if len(puts) >= 3 and len(calls) == 0:
                # print("=== CHECKING FOR PUT BUTTERFLY/CONDOR ===")
                strikes = sorted([leg['strike'] for leg in puts])
                # print(f"Put strikes: {strikes}")
                
                # Check if this is a butterfly pattern (1:2:1 ratio)
                quantities = [leg['quantity'] for leg in puts]
                # print(f"Put quantities: {quantities}")
                
                if len(puts) == 3 and len(set(quantities)) == 2 and max(quantities) == 2 and quantities.count(2) == 1:
                    # print("PUT BUTTERFLY PATTERN DETECTED")
                    trade_type = 'butterfly'
                    
                    # Calculate net credit/debit (simple sum of debits and credits)
                    total_credit = sum(leg['premium'] * leg['quantity'] for leg in puts if leg['action'] == 'STO')
                    total_debit = sum(leg['premium'] * leg['quantity'] for leg in puts if leg['action'] == 'BTO')
                    net_credit = total_credit - total_debit
                    
                    # print(f"Total credit: {total_credit}")
                    # print(f"Total debit: {total_debit}")
                    # print(f"Net credit: {net_credit}")
                    
                    # Simple margin calculation: net debit * 100
                    net_debit = total_debit - total_credit
                    butterfly_margin = net_debit * 100
                    # print(f"PUT BUTTERFLY - Margin: ({total_debit} - {total_credit}) * 100 = {butterfly_margin}")
                    margin_breakdown.append(f"Put butterfly: Net debit({net_debit:.2f}) × 100 = {butterfly_margin:.2f}")
                    
                    overall_margin = butterfly_margin
                    total_contracts = 1  # 1 butterfly = 1 contract
                
                elif len(puts) == 4:
                    # print("=== CHECKING FOR PUT CONDOR ===")
                    # Check if this is a condor pattern (1:1:1:1 ratio)
                    if len(set(quantities)) == 1 and quantities[0] == 1:
                        # print("PUT CONDOR PATTERN DETECTED")
                        trade_type = 'condor'
                        
                        # Calculate net credit/debit (simple sum of debits and credits)
                    total_credit = sum(leg['premium'] * leg['quantity'] for leg in puts if leg['action'] == 'STO')
                    total_debit = sum(leg['premium'] * leg['quantity'] for leg in puts if leg['action'] == 'BTO')
                    net_credit = total_credit - total_debit
                    
                    # print(f"Total credit: {total_credit}")
                    # print(f"Total debit: {total_debit}")
                    # print(f"Net credit: {net_credit}")
                    
                    # Simple margin calculation: net debit * 100
                    net_debit = total_debit - total_credit
                    condor_margin = net_debit * 100
                    # print(f"PUT CONDOR - Margin: ({total_debit} - {total_credit}) * 100 = {condor_margin}")
                    margin_breakdown.append(f"Put condor: Net debit({net_debit:.2f}) × 100 = {condor_margin:.2f}")
                    
                    overall_margin = condor_margin
                    total_contracts = 1  # 1 condor = 1 contract
            
            elif len(calls) >= 3 and len(puts) == 0:
                # CALL BUTTERFLY/CONDOR SECTION
                # print("=== CHECKING FOR CALL BUTTERFLY/CONDOR ===")
                strikes = sorted([leg['strike'] for leg in calls])
                # print(f"Call strikes: {strikes}")
                
                # Check if this is a butterfly pattern (1:2:1 ratio)
                quantities = [leg['quantity'] for leg in calls]
                # print(f"Call quantities: {quantities}")
                
                if len(calls) == 3 and len(set(quantities)) == 2 and max(quantities) == 2 and quantities.count(2) == 1:
                    # print("CALL BUTTERFLY PATTERN DETECTED")
                    trade_type = 'butterfly'
                    
                    # Calculate net credit/debit (simple sum of debits and credits)
                    total_credit = sum(leg['premium'] * leg['quantity'] for leg in calls if leg['action'] == 'STO')
                    total_debit = sum(leg['premium'] * leg['quantity'] for leg in calls if leg['action'] == 'BTO')
                    net_credit = total_credit - total_debit
                    
                    # print(f"Total credit: {total_credit}")
                    # print(f"Total debit: {total_debit}")
                    # print(f"Net credit: {net_credit}")
                    
                    # Simple margin calculation: net debit * 100
                    net_debit = total_debit - total_credit
                    butterfly_margin = net_debit * 100
                    # print(f"CALL BUTTERFLY - Margin: ({total_debit} - {total_credit}) * 100 = {butterfly_margin}")
                    margin_breakdown.append(f"Call butterfly: Net debit({net_debit:.2f}) × 100 = {butterfly_margin:.2f}")
                    
                    overall_margin = butterfly_margin
                    total_contracts = 1  # 1 butterfly = 1 contract
                
                elif len(calls) == 4:
                    # print("=== CHECKING FOR CALL CONDOR ===")
                    # Check if this is a condor pattern (1:1:1:1 ratio)
                    if len(set(quantities)) == 1 and quantities[0] == 1:
                        # print("CALL CONDOR PATTERN DETECTED")
                        trade_type = 'condor'
                        
                        # Calculate net credit/debit (simple sum of debits and credits)
                    total_credit = sum(leg['premium'] * leg['quantity'] for leg in calls if leg['action'] == 'STO')
                    total_debit = sum(leg['premium'] * leg['quantity'] for leg in calls if leg['action'] == 'BTO')
                    net_credit = total_credit - total_debit
                    
                    # print(f"Total credit: {total_credit}")
                    # print(f"Total debit: {total_debit}")
                    # print(f"Net credit: {net_credit}")
                    
                    # Simple margin calculation: net debit * 100
                    net_debit = total_debit - total_credit
                    condor_margin = net_debit * 100
                    # print(f"CALL CONDOR - Margin: ({total_debit} - {total_credit}) × 100 = {condor_margin}")
                    margin_breakdown.append(f"Call condor: Net debit({net_debit:.2f}) × 100 = {condor_margin:.2f}")
                    
                    overall_margin = condor_margin
                    total_contracts = 1  # 1 condor = 1 contract
            
            # Check for vertical spreads
            put_spreads = []
            call_spreads = []
            
            # Find put spreads (STO + BTO pairs)
            sto_puts = [leg for leg in puts if leg['action'] == 'STO']
            bto_puts = [leg for leg in puts if leg['action'] == 'BTO']
            for sto in sto_puts:
                for bto in bto_puts:
                    spread_width = abs(sto['strike'] - bto['strike'])
                    put_spreads.append({
                        'width': spread_width,
                        'sto_premium': sto['premium'],
                        'bto_premium': bto['premium'],
                        'sto_strike': sto['strike'],
                        'bto_strike': bto['strike']
                    })
            
            # Find call spreads (STO + BTO pairs)
            sto_calls = [leg for leg in calls if leg['action'] == 'STO']
            bto_calls = [leg for leg in calls if leg['action'] == 'BTO']
            for sto in sto_calls:
                for bto in bto_calls:
                    spread_width = abs(sto['strike'] - bto['strike'])
                    call_spreads.append({
                        'width': spread_width,
                        'sto_premium': sto['premium'],
                        'bto_premium': bto['premium'],
                        'sto_strike': sto['strike'],
                        'bto_strike': bto['strike']
                    })
            
            # Check for reverse iron butterflies and iron condors first (both spreads are debit spreads)
            if puts and calls and trade_type == 'unknown':
                # print("=== CHECKING FOR REVERSE IRON BUTTERFLIES/CONDORS ===")
                
                # Check for reverse iron butterfly and iron condor (2 puts + 2 calls)
                # Also check for mixed debit strategies (put spread + long call, long put + call spread, etc)
                if (len(puts) == 2 and len(calls) == 2) or (len(puts) + len(calls) == 3):
                    put_strikes = sorted([leg['strike'] for leg in puts])
                    call_strikes = sorted([leg['strike'] for leg in calls])
                    put_quantities = [leg['quantity'] for leg in puts]
                    call_quantities = [leg['quantity'] for leg in calls]
                    
                    # print(f"Put strikes: {put_strikes}")
                    # print(f"Call strikes: {call_strikes}")
                    # print(f"Put quantities: {put_quantities}")
                    # print(f"Call quantities: {call_quantities}")
                    
                    # Check if this is a valid debit pattern (quantities must match if both sides have legs)
                    if (len(puts) == 0 or len(set(put_quantities)) == 1) and \
                       (len(calls) == 0 or len(set(call_quantities)) == 1) and \
                       (len(puts) == 0 or len(calls) == 0 or put_quantities[0] == call_quantities[0]):
                        
                        # Check if both spreads are debit spreads (BTO long legs, STO short legs)
                        put_bto = [leg for leg in puts if leg['action'] == 'BTO']
                        put_sto = [leg for leg in puts if leg['action'] == 'STO']
                        call_bto = [leg for leg in calls if leg['action'] == 'BTO']
                        call_sto = [leg for leg in calls if leg['action'] == 'STO']
                        
                        # Check if we have valid debit patterns (either spreads or long options)
                        total_legs = len(put_bto) + len(put_sto) + len(call_bto) + len(call_sto)
                        if ((len(put_bto) >= 0 and len(put_sto) <= 1) and 
                            (len(call_bto) >= 0 and len(call_sto) <= 1) and
                            (total_legs == 3 or total_legs == 4) and
                            (len(put_bto) + len(call_bto) >= 1)):
                            
                            # print("REVERSE IRON PATTERN DETECTED - Debit strategy detected")
                            
                            # Calculate debit for put side (either spread, long, or none)
                            if len(put_bto) == 1 and len(put_sto) == 1:  # Put spread
                                put_debit = put_bto[0]['premium'] - put_sto[0]['premium']
                                put_type = "spread"
                            elif len(put_bto) == 1:  # Just long put
                                put_debit = put_bto[0]['premium']
                                put_type = "long"
                            else:  # No puts
                                put_debit = 0
                                put_type = "none"
                            
                            # Calculate debit for call side (either spread, long, or none)
                            if len(call_bto) == 1 and len(call_sto) == 1:  # Call spread
                                call_debit = call_bto[0]['premium'] - call_sto[0]['premium']
                                call_type = "spread"
                            elif len(call_bto) == 1:  # Just long call
                                call_debit = call_bto[0]['premium']
                                call_type = "long"
                            else:  # No calls
                                call_debit = 0
                                call_type = "none"
                            
                            total_debit = put_debit + call_debit
                            
                            # print(f"Put {put_type}: {put_debit}")
                            # print(f"Call {call_type}: {call_debit}")
                            # print(f"Total debit: {total_debit}")
                            
                            # Determine the specific pattern type
                            if len(put_sto) == 1 and len(call_sto) == 1:
                                # Both sides are spreads - check for iron butterfly/condor patterns
                                put_spread_width = abs(put_strikes[1] - put_strikes[0])
                                call_spread_width = abs(call_strikes[1] - call_strikes[0])
                                
                                # For reverse iron butterfly, the put spread and call spread should have the same width
                                # and the middle strike should be the same (put_strikes[1] == call_strikes[0] OR put_strikes[0] == call_strikes[1])
                                if put_spread_width == call_spread_width and (put_strikes[1] == call_strikes[0] or put_strikes[0] == call_strikes[1]):
                                    # print("REVERSE IRON BUTTERFLY PATTERN DETECTED")
                                    trade_type = 'reverse_iron_butterfly'
                                    
                                    # Reverse iron butterfly margin: sum of both debit spreads
                                    reverse_iron_butterfly_margin = total_debit * 100
                                    # print(f"REVERSE IRON BUTTERFLY - Margin: {total_debit} × 100 = {reverse_iron_butterfly_margin}")
                                    margin_breakdown.append(f"Reverse iron butterfly: Put debit({put_debit:.2f}) + Call debit({call_debit:.2f}) = {total_debit:.2f} × 100 = {reverse_iron_butterfly_margin:.2f}")
                                    
                                    overall_margin = reverse_iron_butterfly_margin
                                    total_contracts = 1  # 1 reverse iron butterfly = 1 contract
                                
                                # Check if this is a reverse iron condor (different strikes with gap)
                                # For reverse iron condor, the put spread and call spread should be at different strikes
                                # and there should be a gap between them (put_strikes[1] < call_strikes[0])
                                elif put_strikes[1] < call_strikes[0]:
                                    # print("REVERSE IRON CONDOR PATTERN DETECTED")
                                    trade_type = 'reverse_iron_condor'
                                    
                                    # Reverse iron condor margin: sum of both debit spreads
                                    reverse_iron_condor_margin = total_debit * 100
                                    # print(f"REVERSE IRON CONDOR - Margin: {total_debit} × 100 = {reverse_iron_condor_margin}")
                                    margin_breakdown.append(f"Reverse iron condor: Put debit({put_debit:.2f}) + Call debit({call_debit:.2f}) = {total_debit:.2f} × 100 = {reverse_iron_condor_margin:.2f}")
                                    
                                    overall_margin = reverse_iron_condor_margin
                                    total_contracts = 1  # 1 reverse iron condor = 1 contract
                                
                                # Mixed spread case (not a standard iron butterfly/condor)
                                else:
                                    # print("MIXED DEBIT SPREADS DETECTED")
                                    trade_type = 'mixed_debit_spreads'
                                    
                                    # Mixed spreads margin: sum of both debit spreads
                                    mixed_spreads_margin = total_debit * 100
                                    # print(f"MIXED DEBIT SPREADS - Margin: {total_debit} × 100 = {mixed_spreads_margin}")
                                    margin_breakdown.append(f"Mixed debit spreads: Put spread({put_debit:.2f}) + Call spread({call_debit:.2f}) = {total_debit:.2f} × 100 = {mixed_spreads_margin:.2f}")
                                    
                                    overall_margin = mixed_spreads_margin
                                    total_contracts = 1  # 1 mixed spread strategy = 1 contract
                            
                            # One or both sides are just long options (uncapped)
                            else:
                                 # print("MIXED DEBIT STRATEGY DETECTED (includes uncapped legs)")
                                 
                                 if len(put_sto) == 0 and len(call_sto) == 0:
                                     trade_type = 'long_strangle_straddle'
                                     strategy_desc = "Long strangle/straddle"
                                 elif len(put_sto) == 0:
                                     trade_type = 'mixed_long_put_call_spread'
                                     strategy_desc = f"Long put + call {call_type}"
                                 elif len(call_sto) == 0:
                                     trade_type = 'mixed_put_spread_long_call'
                                     strategy_desc = f"Put {put_type} + long call"
                                 else:
                                     trade_type = 'mixed_debit_long'
                                     strategy_desc = "Mixed debit strategy"
                                 
                                 # Mixed strategy margin: sum of all debit components
                                 mixed_margin = total_debit * 100
                                 # print(f"MIXED STRATEGY - Margin: {total_debit} × 100 = {mixed_margin}")
                                 margin_breakdown.append(f"{strategy_desc}: Put component({put_debit:.2f}) + Call component({call_debit:.2f}) = {total_debit:.2f} × 100 = {mixed_margin:.2f}")
                                 
                                 overall_margin = mixed_margin
                                 total_contracts = 1  # 1 mixed strategy = 1 contract
            
            # Check for regular iron butterflies and iron condors (if no reverse pattern detected)
            if puts and calls and trade_type == 'unknown':
                # print("=== CHECKING FOR REGULAR IRON BUTTERFLIES/CONDORS ===")
                
                # Check for iron butterfly and iron condor (2 puts + 2 calls)
                if len(puts) == 2 and len(calls) == 2:
                    put_strikes = sorted([leg['strike'] for leg in puts])
                    call_strikes = sorted([leg['strike'] for leg in calls])
                    put_quantities = [leg['quantity'] for leg in puts]
                    call_quantities = [leg['quantity'] for leg in calls]
                    
                    # print(f"Put strikes: {put_strikes}")
                    # print(f"Call strikes: {call_strikes}")
                    # print(f"Put quantities: {put_quantities}")
                    # print(f"Call quantities: {call_quantities}")
                    
                    # Check if this is an iron butterfly or iron condor pattern
                    if (len(set(put_quantities)) == 1 and len(set(call_quantities)) == 1 and 
                        put_quantities[0] == call_quantities[0]):
                        
                        # Check if the spreads share a common middle strike (iron butterfly)
                        put_spread_width = abs(put_strikes[1] - put_strikes[0])
                        call_spread_width = abs(call_strikes[1] - call_strikes[0])
                        
                        # For iron butterfly, the put spread and call spread should have the same width
                        # and the middle strike should be the same (put_strikes[1] == call_strikes[0] OR put_strikes[0] == call_strikes[1])
                        if put_spread_width == call_spread_width and (put_strikes[1] == call_strikes[0] or put_strikes[0] == call_strikes[1]):
                            # print("IRON BUTTERFLY PATTERN DETECTED")
                            trade_type = 'iron_butterfly'
                            
                            # Calculate net credit/debit
                            total_credit = sum(leg['premium'] * leg['quantity'] for leg in puts + calls if leg['action'] == 'STO')
                            total_debit = sum(leg['premium'] * leg['quantity'] for leg in puts + calls if leg['action'] == 'BTO')
                            net_credit = total_credit - total_debit
                            
                            # print(f"Total credit: {total_credit}")
                            # print(f"Total debit: {total_debit}")
                            # print(f"Net credit: {net_credit}")
                            
                            # Iron butterfly margin: (spread width * 100) - (net credit * 100)
                            iron_butterfly_margin = (put_spread_width * 100) - (net_credit * 100)
                            # print(f"IRON BUTTERFLY - Margin: ({put_spread_width} * 100) - ({net_credit} * 100) = {iron_butterfly_margin}")
                            margin_breakdown.append(f"Iron butterfly: (Spread width({put_spread_width}) × 100) - (Net credit({net_credit:.2f}) × 100) = {iron_butterfly_margin:.2f}")
                            
                            overall_margin = iron_butterfly_margin
                            total_contracts = 1  # 1 iron butterfly = 1 contract
                        
                        # For iron condor, the put spread and call spread should be at different strikes
                        # and there should be a gap between them (put_strikes[1] < call_strikes[0])
                        elif put_strikes[1] < call_strikes[0]:
                            # print("IRON CONDOR PATTERN DETECTED")
                            trade_type = 'iron_condor'
                            
                            # Calculate net credit/debit
                            total_credit = sum(leg['premium'] * leg['quantity'] for leg in puts + calls if leg['action'] == 'STO')
                            total_debit = sum(leg['premium'] * leg['quantity'] for leg in puts + calls if leg['action'] == 'BTO')
                            net_credit = total_credit - total_debit
                            
                            # print(f"Total credit: {total_credit}")
                            # print(f"Total debit: {total_debit}")
                            # print(f"Net credit: {net_credit}")
                            
                            # Iron condor margin: (widest spread width * 100) - (net credit * 100)
                            widest_spread = max(put_spread_width, call_spread_width)
                            iron_condor_margin = (widest_spread * 100) - (net_credit * 100)
                            # print(f"IRON CONDOR - Margin: ({widest_spread} * 100) - ({net_credit} * 100) = {iron_condor_margin}")
                            margin_breakdown.append(f"Iron condor: (Widest spread({widest_spread}) × 100) - (Net credit({net_credit:.2f}) × 100) = {iron_condor_margin:.2f}")
                            
                            overall_margin = iron_condor_margin
                            total_contracts = 1  # 1 iron condor = 1 contract
            
            # Check for strangles and straddles (if no iron butterfly/condor detected)
            if puts and calls and trade_type == 'unknown':
                # print("=== CHECKING FOR STRADDLES/STRANGLES ===")
                # Check for straddle (same strike, both put and call)
                put_strikes = set(leg['strike'] for leg in puts)
                call_strikes = set(leg['strike'] for leg in calls)
                common_strikes = put_strikes & call_strikes
                # print(f"Put strikes: {put_strikes}")
                # print(f"Call strikes: {call_strikes}")
                # print(f"Common strikes: {common_strikes}")
                
                if common_strikes:
                    # print("=== STRADDLE DETECTED ===")
                    # Straddle detected
                    strike = list(common_strikes)[0]
                    trade_type = 'straddle'
                    straddle_margin = 0
                    # print(f"Straddle strike: {strike}")
                    
                    # Check if this is a short straddle (STO positions)
                    sto_puts = [leg for leg in puts if leg['action'] == 'STO' and leg['strike'] == strike]
                    sto_calls = [leg for leg in calls if leg['action'] == 'STO' and leg['strike'] == strike]
                    # print(f"STO puts: {sto_puts}")
                    # print(f"STO calls: {sto_calls}")
                    if sto_puts and sto_calls:
                        # print("SHORT STRADDLE DETECTED - unable to calculate")
                        # Short straddle - unable to calculate margin
                        straddle_margin = 0
                        margin_breakdown.append(f"Short straddle at {strike}: unable to calculate")
                    else:
                        # print("LONG STRADDLE DETECTED")
                        # Long straddle - debit of put + debit of call
                        put_debit = sum(leg['premium'] for leg in puts if leg['action'] == 'BTO' and leg['strike'] == strike)
                        call_debit = sum(leg['premium'] for leg in calls if leg['action'] == 'BTO' and leg['strike'] == strike)
                        straddle_margin = (put_debit + call_debit) * 100
                        # print(f"Put debit: {put_debit}, Call debit: {call_debit}")
                        # print(f"Straddle margin: ({put_debit} + {call_debit}) * 100 = {straddle_margin}")
                        margin_breakdown.append(f"Long straddle at {strike}: (Put debit({put_debit:.2f}) + Call debit({call_debit:.2f})) × 100 = {straddle_margin:.2f}")
                    
                    overall_margin = straddle_margin
                    total_contracts = 1  # 1 straddle = 1 contract
                else:
                    # print("=== CHECKING FOR STRANGLE ===")
                    # Check for strangle (different strikes, both put and call)
                    if len(put_strikes) > 0 and len(call_strikes) > 0:
                        # print("STRANGLE DETECTED")
                        trade_type = 'strangle'
                        strangle_margin = 0
                        
                        # Check if this is a short strangle (STO positions)
                        sto_puts = [leg for leg in puts if leg['action'] == 'STO']
                        sto_calls = [leg for leg in calls if leg['action'] == 'STO']
                        # print(f"STO puts: {sto_puts}")
                        # print(f"STO calls: {sto_calls}")
                        if sto_puts and sto_calls:
                            # print("SHORT STRANGLE DETECTED - unable to calculate")
                            # Short strangle - unable to calculate margin
                            strangle_margin = 0
                            
                            put_strikes_str = ', '.join(map(str, sorted(put_strikes)))
                            call_strikes_str = ', '.join(map(str, sorted(call_strikes)))
                            margin_breakdown.append(f"Short strangle: Puts at {put_strikes_str}, Calls at {call_strikes_str}: unable to calculate")
                        else:
                            # print("LONG STRANGLE DETECTED")
                            # Long strangle - debit of put + debit of call
                            put_debit = sum(leg['premium'] for leg in puts if leg['action'] == 'BTO')
                            call_debit = sum(leg['premium'] for leg in calls if leg['action'] == 'BTO')
                            strangle_margin = (put_debit + call_debit) * 100
                            # print(f"Put debit: {put_debit}, Call debit: {call_debit}")
                            # print(f"Strangle margin: ({put_debit} + {call_debit}) * 100 = {strangle_margin}")
                            
                            put_strikes_str = ', '.join(map(str, sorted(put_strikes)))
                            call_strikes_str = ', '.join(map(str, sorted(call_strikes)))
                            margin_breakdown.append(f"Long strangle: Puts at {put_strikes_str}, Calls at {call_strikes_str}, (Put debit({put_debit:.2f}) + Call debit({call_debit:.2f})) × 100 = {strangle_margin:.2f}")
                        
                        overall_margin = strangle_margin
                        total_contracts = 1  # 1 strangle = 1 contract
                    else:
                        if trade_type == 'unknown':
                            # print("=== CHECKING FOR VERTICAL SPREADS ===")
                            # No strangle/straddle, check for vertical spreads
                            # Calculate margin for vertical spreads
                            total_credit = 0
                            total_debit = 0
                            widest_spread = 0
                        
                        # print(f"Put spreads: {put_spreads}")
                        # print(f"Call spreads: {call_spreads}")
                        
                        for spread in put_spreads + call_spreads:
                            widest_spread = max(widest_spread, spread['width'])
                            net_premium = spread['sto_premium'] - spread['bto_premium']
                            if net_premium > 0:
                                total_credit += net_premium
                            else:
                                total_debit += abs(net_premium)
                        
                        # print(f"Widest spread: {widest_spread}")
                        # print(f"Total credit: {total_credit}")
                        # print(f"Total debit: {total_debit}")
                        
                        if total_credit > 0 and total_debit == 0:
                            # print("CREDIT SPREAD DETECTED")
                            # Credit spread
                            trade_type = 'credit_spread'
                            overall_margin = (widest_spread * 100) - (total_credit * 100)
                            # print(f"Credit spread margin: ({widest_spread} * 100) - ({total_credit} * 100) = {overall_margin}")
                            margin_breakdown.append(f"Credit spread: ({widest_spread} × 100) - ({total_credit:.2f} × 100) = {overall_margin:.2f}")
                            total_contracts = len(put_spreads + call_spreads)  # 1 spread = 1 contract
                        elif total_debit > 0 and total_credit == 0:
                            # print("DEBIT SPREAD DETECTED")
                            # Debit spread: (Total Credit + Total Debit) * 100
                            trade_type = 'debit_spread'
                            overall_margin = (total_credit + total_debit) * 100
                            # print(f"Debit spread margin: ({total_credit} + {total_debit}) * 100 = {overall_margin}")
                            margin_breakdown.append(f"Debit spread: (Total credit {total_credit:.2f} + Total debit {total_debit:.2f}) * 100 = {overall_margin:.2f}")
                            total_contracts = len(put_spreads + call_spreads)  # 1 spread = 1 contract
                        else:
                            # Mixed or complex
                            trade_type = 'complex_vertical'
                            overall_margin = (widest_spread * 100) - (total_credit * 100) + (total_debit * 100)
                            margin_breakdown.append(f"Complex: ({widest_spread} × 100) - ({total_credit:.2f} × 100) + ({total_debit:.2f} × 100) = {overall_margin:.2f}")
                            total_contracts = len(put_spreads + call_spreads)  # 1 spread = 1 contract
            else:
                if trade_type == 'unknown':
                    # Only puts or only calls - vertical spreads
                    # Calculate margin for vertical spreads
                    total_credit = 0
                    total_debit = 0
                    widest_spread = 0
                    
                    for spread in put_spreads + call_spreads:
                        widest_spread = max(widest_spread, spread['width'])
                        net_premium = spread['sto_premium'] - spread['bto_premium']
                        if net_premium > 0:
                            total_credit += net_premium
                        else:
                            total_debit += abs(net_premium)
                    
                    if total_credit > 0 and total_debit == 0:
                        # Credit spread
                        trade_type = 'credit_spread'
                        overall_margin = (widest_spread * 100) - (total_credit * 100)
                        margin_breakdown.append(f"Credit spread: ({widest_spread} × 100) - ({total_credit:.2f} × 100) = {overall_margin:.2f}")
                        total_contracts = len(put_spreads + call_spreads)  # 1 spread = 1 contract
                    elif total_debit > 0 and total_credit == 0:
                        # Debit spread: (Total Credit + Total Debit) * 100
                        trade_type = 'debit_spread'
                        overall_margin = (total_credit + total_debit) * 100
                        margin_breakdown.append(f"Debit spread: (Total credit {total_credit:.2f} + Total debit {total_debit:.2f}) * 100 = {overall_margin:.2f}")
                        total_contracts = len(put_spreads + call_spreads)  # 1 spread = 1 contract
                    else:
                        # Mixed or complex
                        trade_type = 'complex_vertical'
                        overall_margin = (widest_spread * 100) - (total_credit * 100) + (total_debit * 100)
                        margin_breakdown.append(f"Complex: ({widest_spread} × 100) - ({total_credit:.2f} × 100) + ({total_debit:.2f} × 100) = {overall_margin:.2f}")
                        total_contracts = len(put_spreads + call_spreads)  # 1 spread = 1 contract
        
        elif len(expirations) == 2:
            # print("=== TWO EXPIRATION ANALYSIS ===")
            # Two expirations - check for calendars, strangles, straddles
            front_exp = expirations[0]  # Earlier expiration
            back_exp = expirations[1]   # Later expiration
            # print(f"Front expiration: {front_exp}")
            # print(f"Back expiration: {back_exp}")
            
            front_puts = legs_by_expiration[front_exp]['puts']
            front_calls = legs_by_expiration[front_exp]['calls']
            back_puts = legs_by_expiration[back_exp]['puts']
            back_calls = legs_by_expiration[back_exp]['calls']
                         # print(f"Front puts: {front_puts}")
             # print(f"Front calls: {front_calls}")
             # print(f"Back puts: {back_puts}")
             # print(f"Back calls: {back_calls}")
            
            # Check for calendar spreads (same strike, different expirations)
            calendar_spreads = []
            
            # Put calendars
            for front_put in front_puts:
                for back_put in back_puts:
                    if front_put['strike'] == back_put['strike']:
                        calendar_spreads.append({
                            'type': 'put',
                            'strike': front_put['strike'],
                            'front_premium': front_put['premium'],
                            'back_premium': back_put['premium'],
                            'front_action': front_put['action'],
                            'back_action': back_put['action']
                        })
            
            # Call calendars
            for front_call in front_calls:
                for back_call in back_calls:
                    if front_call['strike'] == back_call['strike']:
                        calendar_spreads.append({
                            'type': 'call',
                            'strike': front_call['strike'],
                            'front_premium': front_call['premium'],
                            'back_premium': back_call['premium'],
                            'front_action': front_call['action'],
                            'back_action': back_call['action']
                        })
            
                            # print(f"Calendar spreads found: {calendar_spreads}")
            if calendar_spreads:
                # print("=== CALENDAR ANALYSIS ===")
                # Check if this is a double calendar (both put and call calendars at same strike)
                strikes_with_calendars = {}
                for calendar in calendar_spreads:
                    strike = calendar['strike']
                    if strike not in strikes_with_calendars:
                        strikes_with_calendars[strike] = {'put': None, 'call': None}
                    strikes_with_calendars[strike][calendar['type']] = calendar
                                 # print(f"Strikes with calendars: {strikes_with_calendars}")
                
                total_calendar_margin = 0
                
                for strike, calendars in strikes_with_calendars.items():
                    put_calendar = calendars['put']
                    call_calendar = calendars['call']
                    
                    if put_calendar and call_calendar:
                        # Double calendar - check if this is a long double calendar (BTO back, STO front)
                        trade_type = 'double_calendar'
                        
                        if (put_calendar['front_action'] == 'STO' and put_calendar['back_action'] == 'BTO' and
                            call_calendar['front_action'] == 'STO' and call_calendar['back_action'] == 'BTO'):
                            
                            # Long double calendar - defined risk strategy
                            # Put calendar: back debit - front credit
                            put_net = put_calendar['back_premium'] - put_calendar['front_premium']
                            # Call calendar: back debit - front credit  
                            call_net = call_calendar['back_premium'] - call_calendar['front_premium']
                            total_net = put_net + call_net
                            
                            double_calendar_margin = total_net * 100
                            total_calendar_margin += double_calendar_margin
                            
                            margin_breakdown.append(f"Long double calendar at {strike}: Put net({put_net:.2f}) + Call net({call_net:.2f}) = {total_net:.2f} × 100 = {double_calendar_margin:.2f}")
                        else:
                            # Short double calendar (STO back, BTO front) - undefined risk
                            trade_type = 'unknown'
                            double_calendar_margin = 0
                            margin_breakdown.append(f"Short double calendar at {strike}: undefined risk - unable to calculate margin")
                        
                    elif put_calendar:
                        # Single put calendar
                        trade_type = 'calendar_spread'
                        if put_calendar['front_action'] == 'STO' and put_calendar['back_action'] == 'BTO':
                            # Long put calendar (BTO back, STO front) - defined risk
                            net_debit = put_calendar['back_premium'] - put_calendar['front_premium']
                            calendar_margin = net_debit * 100
                            total_calendar_margin += calendar_margin
                            margin_breakdown.append(f"Long put calendar at {strike}: {put_calendar['back_premium']:.2f} - {put_calendar['front_premium']:.2f} = {net_debit:.2f} × 100 = {calendar_margin:.2f}")
                        else:
                            # Short put calendar (STO back, BTO front) - undefined risk
                            trade_type = 'unknown'
                            calendar_margin = 0
                            margin_breakdown.append(f"Short put calendar at {strike}: undefined risk - unable to calculate margin")
                        
                    elif call_calendar:
                        # Single call calendar
                        trade_type = 'calendar_spread'
                        if call_calendar['front_action'] == 'STO' and call_calendar['back_action'] == 'BTO':
                            # Long call calendar (BTO back, STO front) - defined risk
                            net_debit = call_calendar['back_premium'] - call_calendar['front_premium']
                            calendar_margin = net_debit * 100
                            total_calendar_margin += calendar_margin
                            margin_breakdown.append(f"Long call calendar at {strike}: {call_calendar['back_premium']:.2f} - {call_calendar['front_premium']:.2f} = {net_debit:.2f} × 100 = {calendar_margin:.2f}")
                        else:
                            # Short call calendar (STO back, BTO front) - undefined risk
                            trade_type = 'unknown'
                            calendar_margin = 0
                            margin_breakdown.append(f"Short call calendar at {strike}: undefined risk - unable to calculate margin")
                
                overall_margin = total_calendar_margin
                total_contracts = 1  # 1 calendar strategy = 1 contract
            else:
                # Check for strangles and straddles (same expiration, different strikes or same strike)
                # Group by expiration to check for these patterns
                for exp in expirations:
                    exp_puts = legs_by_expiration[exp]['puts']
                    exp_calls = legs_by_expiration[exp]['calls']
                    
                    # Check for straddle (same strike, both put and call)
                    put_strikes = {leg['strike'] for leg in exp_puts}
                    call_strikes = {leg['strike'] for leg in exp_calls}
                    straddle_strikes = put_strikes.intersection(call_strikes)
                    
                    if straddle_strikes:
                        trade_type = 'straddle'
                        straddle_margin = 0
                        for strike in straddle_strikes:
                            put_legs = [leg for leg in exp_puts if leg['strike'] == strike]
                            call_legs = [leg for leg in exp_calls if leg['strike'] == strike]
                            
                            # Check if this is a short straddle (STO positions)
                            sto_puts = [leg for leg in put_legs if leg['action'] == 'STO']
                            sto_calls = [leg for leg in call_legs if leg['action'] == 'STO']
                            
                            if sto_puts and sto_calls:
                                # Short straddle - unable to calculate margin
                                straddle_margin = 0
                                margin_breakdown.append(f"Short straddle at {strike}: unable to calculate")
                            else:
                                # Long straddle - debit of put + debit of call
                                put_debit = sum(leg['premium'] for leg in put_legs if leg['action'] == 'BTO')
                                call_debit = sum(leg['premium'] for leg in call_legs if leg['action'] == 'BTO')
                                straddle_margin = (put_debit + call_debit) * 100
                                margin_breakdown.append(f"Long straddle at {strike}: (Put debit({put_debit:.2f}) + Call debit({call_debit:.2f})) × 100 = {straddle_margin:.2f}")
                        
                        # If no straddle found, check if this is a long straddle (BTO positions)
                        if straddle_margin == 0:
                            bto_puts = [leg for leg in put_legs if leg['action'] == 'BTO']
                            bto_calls = [leg for leg in call_legs if leg['action'] == 'BTO']
                            if bto_puts and bto_calls:
                                # Long straddle - debit of put + debit of call
                                put_debit = sum(leg['premium'] for leg in put_legs if leg['action'] == 'BTO')
                                call_debit = sum(leg['premium'] for leg in call_legs if leg['action'] == 'BTO')
                                straddle_margin = (put_debit + call_debit) * 100
                                margin_breakdown.append(f"Long straddle at {strike}: (Put debit({put_debit:.2f}) + Call debit({call_debit:.2f})) × 100 = {straddle_margin:.2f}")
                        
                        overall_margin = straddle_margin
                        total_contracts = 1  # 1 straddle = 1 contract
                        break
                    
                    # Check for strangle (different strikes, both put and call)
                    elif len(put_strikes) > 0 and len(call_strikes) > 0:
                        trade_type = 'strangle'
                        strangle_margin = 0
                        
                        # Check if this is a short strangle (STO positions)
                        sto_puts = [leg for leg in exp_puts if leg['action'] == 'STO']
                        sto_calls = [leg for leg in exp_calls if leg['action'] == 'STO']
                        
                        if sto_puts and sto_calls:
                            # Short strangle - unable to calculate margin
                            strangle_margin = 0
                            
                            put_strikes_str = ', '.join(map(str, sorted(put_strikes)))
                            call_strikes_str = ', '.join(map(str, sorted(call_strikes)))
                            margin_breakdown.append(f"Short strangle: Puts at {put_strikes_str}, Calls at {call_strikes_str}: unable to calculate")
                        else:
                            # Long strangle - debit of put + debit of call
                            put_debit = sum(leg['premium'] for leg in exp_puts if leg['action'] == 'BTO')
                            call_debit = sum(leg['premium'] for leg in exp_calls if leg['action'] == 'BTO')
                            strangle_margin = (put_debit + call_debit) * 100
                            
                            put_strikes_str = ', '.join(map(str, sorted(put_strikes)))
                            call_strikes_str = ', '.join(map(str, sorted(call_strikes)))
                            margin_breakdown.append(f"Long strangle: Puts at {put_strikes_str}, Calls at {call_strikes_str}, (Put debit({put_debit:.2f}) + Call debit({call_debit:.2f})) × 100 = {strangle_margin:.2f}")
                    
                    # If no strangle found, check if this is a long strangle (BTO positions)
                    if strangle_margin == 0:
                        bto_puts = [leg for leg in exp_puts if leg['action'] == 'BTO']
                        bto_calls = [leg for leg in exp_calls if leg['action'] == 'BTO']
                        if bto_puts and bto_calls:
                            # Long strangle - debit of put + debit of call
                            put_debit = sum(leg['premium'] for leg in exp_puts if leg['action'] == 'BTO')
                            call_debit = sum(leg['premium'] for leg in exp_calls if leg['action'] == 'BTO')
                            strangle_margin = (put_debit + call_debit) * 100
                            
                            put_strikes_str = ', '.join(map(str, sorted(put_strikes)))
                            call_strikes_str = ', '.join(map(str, sorted(call_strikes)))
                            margin_breakdown.append(f"Long strangle: Puts at {put_strikes_str}, Calls at {call_strikes_str}, (Put debit({put_debit:.2f}) + Call debit({call_debit:.2f})) × 100 = {strangle_margin:.2f}")
                        
                        overall_margin = strangle_margin
                        total_contracts = 1  # 1 strangle = 1 contract
                        break
                else:
                    # Not a recognized pattern - treat as complex multi-expiration
                    trade_type = 'complex_multi_exp'
                    overall_margin = sum(leg['premium'] * leg['quantity'] for leg in legs if leg['action'] == 'BTO')
                    margin_breakdown.append(f"Complex multi-expiration: Total debit = {overall_margin:.2f}")
        
        else:
            # More than 2 expirations or complex structure
            trade_type = 'complex_multi_exp'
            overall_margin = sum(leg['premium'] * leg['quantity'] for leg in legs if leg['action'] == 'BTO')
            margin_breakdown.append(f"Complex structure: Total debit = {overall_margin:.2f}")
        
        # Final check - if we still don't have a recognized trade type, mark as unknown
        if trade_type == 'unknown' and overall_margin > 0:
            trade_type = 'complex_multi_exp'
            margin_breakdown.append("Trade type not recognized, treating as complex multi-leg strategy")
        
        # Calculate margin per contract
        margin_per_contract = overall_margin / total_contracts if total_contracts > 0 else 0
        
        # print(f"\n=== FINAL RESULT ===")
        # print(f"Trade type: {trade_type}")
        # print(f"Overall margin: {overall_margin}")
        # print(f"Margin per contract: {margin_per_contract}")
        # print(f"Total contracts: {total_contracts}")
        # print(f"Margin breakdown: {margin_breakdown}")
        # print("=== END MARGIN CALCULATION ===\n")
        # Log the result
        try:
            log_margin_debug({
                'source': 'legs',
                'strategy': (context or {}).get('strategy', ''),
                'date_opened': (context or {}).get('date_opened', ''),
                'legs': legs_str,
                'contracts': (context or {}).get('contracts', ''),
                'overall_margin': overall_margin,
                'margin_per_contract': margin_per_contract,
                'total_contracts': total_contracts,
                'trade_type': trade_type,
                'margin_breakdown': '; '.join(margin_breakdown)
            })
        except Exception as e:
            print(f"Failed to log margin debug row: {e}")
        
    except Exception as e:
        print(f"ERROR in margin calculation: {str(e)}")
        # If any error occurs during analysis, mark as unknown
        trade_type = 'unknown'
        overall_margin = sum(leg['premium'] * leg['quantity'] for leg in legs if leg['action'] == 'BTO')
        margin_per_contract = overall_margin / total_contracts if total_contracts > 0 else 0
        margin_breakdown = [f'Error analyzing trade type: {str(e)}']
        # Log the error case
        try:
            log_margin_debug({
                'source': 'legs',
                'strategy': (context or {}).get('strategy', ''),
                'date_opened': (context or {}).get('date_opened', ''),
                'legs': legs_str,
                'contracts': (context or {}).get('contracts', ''),
                'overall_margin': overall_margin,
                'margin_per_contract': margin_per_contract,
                'total_contracts': total_contracts,
                'trade_type': trade_type,
                'margin_breakdown': '; '.join(margin_breakdown)
            })
        except Exception:
            pass
    
    return {
        'overall_margin': overall_margin,
        'margin_per_contract': margin_per_contract,
        'total_contracts': total_contracts,
        'trade_type': trade_type,
        'margin_breakdown': '; '.join(margin_breakdown)
    }

@app.route('/')
@guest_mode_required
def dashboard():
    """Main dashboard with strategy analytics."""
    # Get Application Insights key for frontend tracking
    app_insights_key = os.getenv('APPINSIGHTS_INSTRUMENTATIONKEY', '')
    
    # Get version information
    version_string = get_version_string()
    version_info = get_version_info()
    build_badge = get_build_badge()
    
    return render_template('dashboard.html', 
                         app_insights_key=app_insights_key,
                         version_string=version_string,
                         version_info=version_info,
                         build_badge=build_badge)

@app.route('/api/version')
def api_version():
    """API endpoint to get version information."""
    try:
        version_info = get_version_info()
        return jsonify({
            'success': True,
            'version': version_info
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/config')
def get_config():
    """Get application configuration as JSON."""
    try:
        config = {
            'max_trades_limit': int(os.environ.get('MAX_TRADES_LIMIT', '3000')),
            'disable_trade_limit': os.environ.get('DISABLE_TRADE_LIMIT', 'false').lower() == 'true'
        }
        return jsonify({
            'success': True,
            'data': config
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/debug_charts.html')
def debug_charts():
    """Debug page for chart buttons."""
    import os
    debug_file = os.path.join(os.getcwd(), 'debug_charts.html')
    if os.path.exists(debug_file):
        with open(debug_file, 'r') as f:
            return f.read()
    else:
        return "Debug file not found"

@app.route('/api/portfolio/overview')
def portfolio_overview():
    """Get portfolio overview metrics."""
    try:
        print("=== PORTFOLIO OVERVIEW API CALLED ===")
        
        if not portfolio.strategies:
            print("No strategies found, returning empty data")
            return jsonify({
                'success': True,
                'data': {
                    'total_pnl': 0,
                    'total_trades': 0,
                    'win_rate': 0,
                    'max_drawdown': 0,
                    'strategy_count': 0,
                    'starting_capital': 0
                }
            })
        
        print(f"Portfolio has {len(portfolio.strategies)} strategies")
        metrics = PortfolioMetrics(portfolio)
        print("PortfolioMetrics created")
        
        overview = metrics.get_overview_metrics()
        print("Overview metrics calculated")
        
        # Use the same accurate calculation logic as files overview
        global current_data_type, current_initial_capital
        
        if current_data_type == 'real_trade' and current_initial_capital is not None:
            # For real trade files, use the provided initial capital
            overview['starting_capital'] = current_initial_capital
            overview['ending_capital'] = current_initial_capital + overview.get('total_pnl', 0)
        else:
            # For backtest files, use the same accurate logic as files overview
            # Find the trade with the oldest date and time for trade closing
            earliest_trade = None
            latest_trade = None
            earliest_datetime = None
            latest_datetime = None
            
            for strategy in portfolio.strategies.values():
                for trade in strategy.trades:
                    if hasattr(trade, 'funds_at_close') and hasattr(trade, 'pnl'):
                        # Use date_closed as primary, fall back to date_opened
                        trade_datetime = trade.date_closed or trade.date_opened
                        
                        if trade_datetime:
                            # Check if this is the earliest trade
                            if earliest_datetime is None or trade_datetime < earliest_datetime:
                                earliest_datetime = trade_datetime
                                earliest_trade = trade
                            
                            # Check if this is the latest trade
                            if latest_datetime is None or trade_datetime > latest_datetime:
                                latest_datetime = trade_datetime
                                latest_trade = trade
            
            if earliest_trade and latest_trade:
                # Calculate initial capital: earliest trade's funds_at_close minus its P&L
                if (hasattr(earliest_trade, 'funds_at_close') and earliest_trade.funds_at_close is not None and
                    hasattr(earliest_trade, 'pnl') and earliest_trade.pnl is not None):
                    initial_capital = float(earliest_trade.funds_at_close - earliest_trade.pnl)
                    overview['starting_capital'] = initial_capital
                    
                    # Calculate total P&L as difference between final and initial funds
                    if (hasattr(latest_trade, 'funds_at_close') and latest_trade.funds_at_close is not None):
                        total_pnl = float(latest_trade.funds_at_close - initial_capital)
                        overview['total_pnl'] = total_pnl
                        overview['ending_capital'] = float(initial_capital + total_pnl)
                    else:
                        overview['ending_capital'] = overview.get('final_balance', 0)
                else:
                    overview['starting_capital'] = 0
                    overview['ending_capital'] = overview.get('final_balance', 0)
            else:
                overview['starting_capital'] = 0
                overview['ending_capital'] = overview.get('final_balance', 0)
        
        print("Portfolio overview completed successfully")
        print("=== PORTFOLIO OVERVIEW API COMPLETED ===")
        
        return jsonify({
            'success': True,
            'data': overview
        })
    except Exception as e:
        print(f"ERROR in portfolio_overview: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/portfolio/strategies')
def strategy_summary():
    """Get strategy summary statistics."""
    try:
        print(f"=== STRATEGY SUMMARY START ===")
        
        if not portfolio.strategies:
            print("No strategies found, returning empty data")
            return jsonify({
                'success': True,
                'data': []
            })
        
        print(f"Portfolio has {len(portfolio.strategies)} strategies")
        
        # Check if p-values are requested (default to False for performance)
        include_pvalues = request.args.get('include_pvalues', 'false').lower() == 'true'
        print(f"P-values requested: {include_pvalues}")
        
        # Get basic strategy data first (fast)
        print("Calling portfolio.get_strategy_summary()...")
        
        # Add timeout protection for the get_strategy_summary call
        import time
        import threading
        
        # Use threading-based timeout for Windows compatibility
        strategy_data = None
        timeout_error = None
        
        def get_strategy_summary_with_timeout():
            nonlocal strategy_data, timeout_error
            try:
                strategy_data = portfolio.get_strategy_summary()
            except Exception as e:
                timeout_error = e
        
        # Start the calculation in a separate thread
        thread = threading.Thread(target=get_strategy_summary_with_timeout)
        thread.daemon = True
        thread.start()
        
        # Wait for up to 30 seconds
        thread.join(timeout=30)
        
        if thread.is_alive():
            print("get_strategy_summary() timed out!")
            return jsonify({
                'success': False,
                'error': 'Strategy summary calculation timed out. The portfolio data may be too large.'
            }), 408
        
        if timeout_error:
            print(f"Error in get_strategy_summary: {timeout_error}")
            raise timeout_error
        
        print(f"get_strategy_summary() returned {len(strategy_data)} strategies")
        
        # Debug: print first few strategy names
        if strategy_data:
            print(f"First 3 strategies: {[s.get('strategy', 'N/A') for s in strategy_data[:3]]}")
        
        # Add p-value statistics for each strategy (only if requested)
        if include_pvalues:
            print(f"Calculating p-values for {len(strategy_data)} strategies...")
            for i, strategy_info in enumerate(strategy_data):
                strategy_name = strategy_info['strategy']
                print(f"Processing strategy {i+1}/{len(strategy_data)}: {strategy_name}")
                
                if strategy_name in portfolio.strategies:
                    strategy = portfolio.strategies[strategy_name]
                    if strategy.trades and len(strategy.trades) > 1:  # Need at least 2 trades for t-test
                        # Get P/L values for statistical testing (limit to reasonable size)
                        if len(strategy.trades) > 1000:
                            # For very large datasets, sample to improve performance
                            import random
                            sample_size = min(1000, len(strategy.trades))
                            sampled_trades = random.sample(strategy.trades, sample_size)
                            pnl_values = [trade.pnl for trade in sampled_trades]
                            print(f"  Sampled {sample_size} trades from {len(strategy.trades)} total")
                        else:
                            pnl_values = [trade.pnl for trade in strategy.trades]
                        
                        # Quick check: if all P/L values are the same, skip expensive calculation
                        if len(set(pnl_values)) == 1:
                            strategy_info['p_value'] = 1.0  # All values identical
                            strategy_info['t_statistic'] = 0.0
                            strategy_info['significance'] = "ns"
                            print(f"  All P/L values identical, skipping calculation")
                        else:
                            # Calculate p-value using one-sample t-test against zero
                            try:
                                t_stat, p_value = stats.ttest_1samp(pnl_values, 0)
                                strategy_info['p_value'] = round(p_value, 4)  # Round to 4 decimal places
                                strategy_info['t_statistic'] = round(t_stat, 4)
                                
                                # Add significance level interpretation
                                if p_value < 0.001:
                                    significance = "***"
                                elif p_value < 0.01:
                                    significance = "**"
                                elif p_value < 0.05:
                                    significance = "*"
                                else:
                                    significance = "ns"
                                strategy_info['significance'] = significance
                                
                                print(f"  Calculated p-value: {p_value:.4f} ({significance})")
                                
                            except Exception as e:
                                # If statistical test fails, set default values
                                strategy_info['p_value'] = None
                                strategy_info['t_statistic'] = None
                                strategy_info['significance'] = "N/A"
                                print(f"  Error calculating p-value: {e}")
                    else:
                        strategy_info['p_value'] = None
                        strategy_info['t_statistic'] = None
                        strategy_info['significance'] = "N/A"
                        print(f"  No trades or insufficient data for p-value calculation")
        
        print(f"Strategy summary completed successfully")
        print(f"=== STRATEGY SUMMARY END ===")
        return jsonify({
            'success': True,
            'data': strategy_data
        })
        
    except Exception as e:
        print(f"ERROR in strategy_summary: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/portfolio/strategies-simple')
def strategy_summary_simple():
    """Get basic strategy summary without any complex calculations."""
    try:
        print(f"=== SIMPLE STRATEGY SUMMARY START ===")
        
        if not portfolio.strategies:
            print("No strategies found, returning empty data")
            return jsonify({
                'success': True,
                'data': []
            })
        
        print(f"Portfolio has {len(portfolio.strategies)} strategies")
        
        # Just return basic strategy info without calling get_strategy_summary()
        basic_data = []
        for strategy_name, strategy in portfolio.strategies.items():
            basic_data.append({
                'strategy': strategy_name,
                'trade_count': len(strategy.trades) if strategy.trades else 0,
                'status': 'loaded'
            })
        
        print(f"Simple strategy summary completed: {len(basic_data)} strategies")
        print(f"=== SIMPLE STRATEGY SUMMARY END ===")
        
        return jsonify({
            'success': True,
            'data': basic_data
        })
        
    except Exception as e:
        print(f"ERROR in simple strategy summary: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/portfolio/balance-analysis')
def balance_analysis():
    """Get portfolio balance analysis (bullish/bearish/neutral breakdown)."""
    try:
        print("=== BALANCE ANALYSIS API CALLED ===")
        
        analyzer = StrategyAnalyzer(portfolio)
        print("StrategyAnalyzer created")
        
        balance = analyzer.get_strategy_balance_analysis()
        print("Balance analysis completed")
        
        diversification = analyzer.get_diversification_score()
        print("Diversification score calculated")
        
        print("=== BALANCE ANALYSIS API COMPLETED ===")
        
        return jsonify({
            'success': True,
            'data': {
                'balance': balance,
                'diversification': diversification
            }
        })
        
    except Exception as e:
        print(f"ERROR in balance_analysis: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/portfolio/position-sizing')
def position_sizing_suggestions():
    """Get position sizing suggestions based on strategy performance."""
    try:
        if not portfolio.strategies:
            return jsonify({
                'success': True,
                'data': []
            })
        
        analyzer = StrategyAnalyzer(portfolio)
        suggestions = analyzer.suggest_position_sizing()
        
        return jsonify({
            'success': True,
            'data': suggestions
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/charts/<chart_type>')
def get_chart(chart_type):
    """Get a specific chart by type."""
    try:
        # Convert chart type to method name
        chart_type_map = {
            'cumulative_pnl': 'cumulative_pnl',
            'monthly_pnl_stacked': 'monthly_pnl_stacked',
            'daily_pnl': 'daily_pnl',
            'drawdown': 'drawdown',
            'risk_vs_return': 'risk_vs_return',
            'correlation_matrix': 'correlation_matrix',
            'portfolio_balance': 'portfolio_balance',
            'strategy_detail': 'strategy_detail',
            'win_rates_table': 'win_rates_table',
            'margin_analysis': 'margin_analysis',
            'daily_margin_analysis': 'daily_margin_analysis',
            'meic_analysis': 'meic_analysis'
        }
        
        # Handle display names with emojis
        display_to_method = {
            '📈 Cumulative P&L': 'cumulative_pnl',
            '📊 Monthly P&L Stacked': 'monthly_pnl_stacked',
            '📉 Daily P&L': 'daily_pnl',
            '📈 Drawdown': 'drawdown',
            '🎯 Risk vs Return': 'risk_vs_return',
            '🔗 Correlation Matrix': 'correlation_matrix',
            '🥧 Portfolio Balance': 'portfolio_balance',
            '📋 Strategy Detail': 'strategy_detail',
            '📊 Win Rates Table': 'win_rates_table',
            '💰 Margin Analysis': 'margin_analysis',
            '📊 Daily Margin Analysis': 'daily_margin_analysis',
            '🎯 MEIC Analysis': 'meic_analysis'
        }
        
        method_name = display_to_method.get(chart_type, chart_type_map.get(chart_type, chart_type))
        
        if method_name == 'strategy_detail':
            # Handle strategy detail specially
            return jsonify({
                'chart_type': 'strategy_detail',
                'message': 'Strategy detail requires frontend handling'
            })
        
        if method_name == 'meic_analysis':
            # Handle MEIC analysis specially
            from analytics import MEICAnalyzer
            current_filename = session.get('current_filename')
            analyzer = MEICAnalyzer(portfolio, current_filename)
            heatmap_data = analyzer.get_time_heatmap_data()
            
            # Create the heatmap chart
            chart_generator = ChartGenerator(portfolio)
            result = chart_generator.create_meic_heatmap_chart(heatmap_data)
            return jsonify(result)
        
        chart_data = chart_factory.create_chart(method_name)
        return jsonify(chart_data)
        
    except Exception as e:
        print(f"Error generating chart {chart_type}: {e}")
        return jsonify({'error': f'Error generating chart: {str(e)}'}), 500

@app.route('/api/charts/available')
def available_charts():
    """Get list of available chart types."""
    charts = chart_factory.get_available_charts()
    return jsonify({
        'success': True,
        'data': charts
    })

@app.route('/api/monte-carlo/strategy/<path:strategy_name>', methods=['POST'])
def run_strategy_monte_carlo(strategy_name):
    """Run Monte Carlo simulation for a specific strategy."""
    try:
        # Debug: Print the received strategy name and available strategies
        print(f"DEBUG: Received strategy name: '{strategy_name}'")
        print(f"DEBUG: Available strategies: {list(portfolio.strategies.keys())}")
        
        # Try to find a matching strategy (case-insensitive and handle encoding issues)
        matching_strategy = None
        for available_strategy in portfolio.strategies.keys():
            print(f"DEBUG: Comparing '{strategy_name}' with '{available_strategy}'")
            if available_strategy == strategy_name:
                matching_strategy = available_strategy
                print(f"DEBUG: Exact match found: '{matching_strategy}'")
                break
            # Try case-insensitive match
            elif available_strategy.lower() == strategy_name.lower():
                matching_strategy = available_strategy
                print(f"DEBUG: Case-insensitive match found: '{matching_strategy}'")
                break
        
        if not matching_strategy:
            return jsonify({
                'success': False,
                'error': f'Strategy "{strategy_name}" not found. Available strategies: {list(portfolio.strategies.keys())}'
            }), 404
        
        # Get parameters from request
        data = request.get_json() or {}
        num_simulations = data.get('num_simulations', 1000)
        num_trades = data.get('num_trades', None)  # None = use historical count
        # trade_size_percent removed - using historical position sizing
        risk_free_rate = data.get('risk_free_rate', 4.0)  # Default to 4%
        
        # Validate parameters
        if not isinstance(num_simulations, int) or num_simulations < 100 or num_simulations > 10000:
            return jsonify({
                'success': False,
                'error': 'Number of simulations must be between 100 and 10,000'
            }), 400
        
        if num_trades is not None and (not isinstance(num_trades, int) or num_trades < 10):
            return jsonify({
                'success': False,
                'error': 'Number of trades must be at least 10'
            }), 400
        
        # trade_size_percent validation removed - using historical position sizing
        
        # Run simulation
        simulator = MonteCarloSimulator(portfolio)
        results = simulator.run_strategy_specific_simulation(matching_strategy, num_simulations, num_trades, risk_free_rate)
        
        return jsonify({
            'success': True,
            'data': results
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Strategy Monte Carlo simulation failed: {str(e)}'
        }), 500

@app.route('/api/charts/monte-carlo/<chart_type>', methods=['POST'])
def get_monte_carlo_chart(chart_type):
    """Get a Monte Carlo chart with simulation data."""
    try:
        if not portfolio.strategies:
            return jsonify({
                'success': False,
                'error': 'No data loaded. Please upload a CSV file first.'
            }), 400
        
        # Get simulation data from request body
        data = request.get_json()
        if not data or 'simulation_data' not in data:
            return jsonify({
                'success': False,
                'error': 'Simulation data is required'
            }), 400
        
        simulation_data = data['simulation_data']
        
        # Create the chart
        chart_data = ChartFactory.create_monte_carlo_chart(chart_type, simulation_data, portfolio)
        
        return jsonify({
            'success': True,
            'data': chart_data
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/charts/monte-carlo/available')
def available_monte_carlo_charts():
    """Get list of available Monte Carlo chart types."""
    charts = ChartFactory.get_monte_carlo_charts()
    return jsonify({
        'success': True,
        'data': charts
    })

@app.route('/api/strategy-details')
def get_strategy_details():
    """Get all trades with optional strategy filtering and pagination."""
    try:
        if not portfolio.strategies:
            return jsonify({
                'success': True,
                'data': {
                    'trades': [],
                    'strategies': [],
                    'pagination': {
                        'current_page': 1,
                        'total_pages': 0,
                        'total_trades': 0,
                        'trades_per_page': 100
                    }
                }
            })
        
        # Get strategy filter and pagination from query parameters
        selected_strategies = request.args.getlist('strategies[]')
        page = int(request.args.get('page', 1))
        trades_per_page = 100
        include_margin = request.args.get('include_margin', 'false').lower() == 'true'
        
        # Get all trades
        all_trades = []
        available_strategies = []
        
        for strategy_name, strategy in portfolio.strategies.items():
            available_strategies.append(strategy_name)
            
            # Filter by selected strategies if specified
            if not selected_strategies or strategy_name in selected_strategies:
                for trade in strategy.trades:
                    # Compose date+time strings
                    def format_datetime(date_val, time_val):
                        if date_val is None:
                            return ''
                        date_str = str(date_val)
                        if time_val:
                            # If time is already in the date (e.g. ISO), don't double-append
                            if 'T' in date_str:
                                # ISO format, just use up to seconds
                                try:
                                    dt = pd.to_datetime(date_str)
                                    return dt.strftime('%Y-%m-%d %H:%M:%S')
                                except Exception:
                                    return date_str
                            else:
                                return f"{date_str} {time_val}"
                        else:
                            # If date_str is ISO with time, format it
                            if 'T' in date_str:
                                try:
                                    dt = pd.to_datetime(date_str)
                                    return dt.strftime('%Y-%m-%d %H:%M:%S')
                                except Exception:
                                    return date_str
                            return date_str
                    # Calculate margin from legs for Trade Details table (only if requested and legs data is available)
                    margin_per_contract = 0
                    if include_margin:
                        legs_data = getattr(trade, 'legs', '')
                        if legs_data and getattr(trade, 'contracts', 0) > 0:
                            # Only calculate margin if we have meaningful legs data (not empty or just basic info)
                            if len(legs_data.strip()) > 10:  # Basic check for substantial legs data
                                try:
                                    margin_info = calculate_margin_from_legs(legs_data, context={
                                        'strategy': strategy_name,
                                        'date_opened': getattr(trade, 'date_opened', ''),
                                        'contracts': getattr(trade, 'contracts', '')
                                    })
                                    margin_per_contract = margin_info.get('margin_per_contract', 0)
                                except Exception as e:
                                    # Silently skip margin calculation on error to avoid slowing down the page
                                    margin_per_contract = 0
                    
                    all_trades.append({
                        'date_opened': format_datetime(trade.date_opened, getattr(trade, 'time_opened', None)),
                        'date_closed': format_datetime(trade.date_closed, getattr(trade, 'time_closed', None)),
                        'strategy': strategy_name,
                        'pnl': trade.pnl,
                        'funds_at_close': trade.funds_at_close,
                        'contracts': getattr(trade, 'contracts', 1),
                        'opening_commissions': getattr(trade, 'opening_commissions', 0),
                        'closing_commissions': getattr(trade, 'closing_commissions', 0),
                        'is_winner': trade.is_winner,
                        'legs': getattr(trade, 'legs', ''), # Add legs to the trade data
                        'margin_per_contract': margin_per_contract # Calculate from legs
                    })
        
        # Sort trades by date closed (most recent first)
        all_trades.sort(key=lambda x: x['date_closed'] or x['date_opened'], reverse=True)
        
        # Calculate pagination
        total_trades = len(all_trades)
        total_pages = (total_trades + trades_per_page - 1) // trades_per_page  # Ceiling division
        
        # Ensure page is within valid range
        page = max(1, min(page, total_pages)) if total_pages > 0 else 1
        
        # Get trades for current page
        start_index = (page - 1) * trades_per_page
        end_index = start_index + trades_per_page
        page_trades = all_trades[start_index:end_index]
        
        # Calculate overall statistics across all trades (not just current page)
        total_winning_trades = sum(1 for trade in all_trades if trade['is_winner'])
        total_losing_trades = total_trades - total_winning_trades
        total_win_rate = (total_winning_trades / total_trades * 100) if total_trades > 0 else 0
        total_pnl = sum(trade['pnl'] for trade in all_trades)
        total_commissions = sum(trade['opening_commissions'] + trade['closing_commissions'] for trade in all_trades)
        total_net_pnl = total_pnl - total_commissions
        
        return jsonify({
            'success': True,
            'data': {
                'trades': page_trades,
                'strategies': available_strategies,
                'pagination': {
                    'current_page': page,
                    'total_pages': total_pages,
                    'total_trades': total_trades,
                    'trades_per_page': trades_per_page
                },
                'overall_stats': {
                    'total_winning_trades': total_winning_trades,
                    'total_losing_trades': total_losing_trades,
                    'total_win_rate': total_win_rate,
                    'total_pnl': total_pnl,
                    'total_commissions': total_commissions,
                    'total_net_pnl': total_net_pnl
                }
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/upload', methods=['POST'])
def upload_csv():
    """Upload and process CSV file with trade data."""
    try:
        # Track upload attempt
        app_insights.track_event('file_upload_attempted', {
            'user_id': get_current_user_id(),
            'has_file': 'file' in request.files
        })
        
        if 'file' not in request.files:
            app_insights.track_event('file_upload_failed', {
                'user_id': get_current_user_id(),
                'error': 'no_file_provided'
            })
            return jsonify({
                'success': False,
                'error': 'No file provided'
            }), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        if not file.filename.lower().endswith('.csv'):
            return jsonify({
                'success': False,
                'error': 'Please upload a CSV file'
            }), 400
        
        # Get optional friendly name from form data
        friendly_name = request.form.get('friendly_name', '').strip()
        # Use last entered initial capital as default
        last_capital = get_last_initial_capital()
        initial_capital = request.form.get('initial_capital', '').strip()
        initial_capital = int(initial_capital) if initial_capital.isdigit() else last_capital
        
        # Save uploaded file with optional friendly name
        file_info = file_manager.save_file(file, friendly_name if friendly_name else None)
        saved_filename = file_info['filename']
        saved_path = file_manager.get_file_path(saved_filename)
        
        try:
            # Read CSV to validate columns
            df = pd.read_csv(saved_path)
            
            # Count trades and warn if too many (unless limit is disabled)
            trade_count = len(df)
            warning_message = None
            if os.environ.get('DISABLE_TRADE_LIMIT', 'false').lower() != 'true':
                max_trades = int(os.environ.get('MAX_TRADES_LIMIT', '3000'))
                print(f"DEBUG: Trade count: {trade_count}, Max trades: {max_trades}")
                if trade_count > max_trades:
                    warning_message = f'⚠️ WARNING: This file contains {trade_count:,} trades. The recommended limit is {max_trades:,} trades. Files larger than this may produce unexpected problems and slow performance. Consider splitting your data into smaller files.'
                    print(f"DEBUG: Warning message generated: {warning_message}")

            # Detect data type
            if 'Initial Premium' in df.columns:
                data_type = 'real_trade'
            else:
                data_type = 'backtest'

            # Set required columns (Strategy is now optional - can be derived from filename)
            if data_type == 'real_trade':
                required_columns = ['Date Opened', 'Date Closed', 'P/L']
            else:
                required_columns = ['Date Opened', 'Date Closed', 'P/L', 'Funds at Close']

            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                os.remove(saved_path)  # Clean up file if invalid
                return jsonify({
                    'success': False,
                    'error': f'Missing required columns: {", ".join(missing_columns)}',
                    'available_columns': list(df.columns)
                }), 400

            # --- Transform real trade data to backtest schema ---
            if data_type == 'real_trade':
                df = transform_real_trade_to_backtest(df, initial_capital=initial_capital)
                # Overwrite the saved file with the transformed DataFrame
                # Save with initial capital in filename
                new_filename = append_initial_capital_to_filename(saved_filename, initial_capital)
                new_path = file_manager.get_file_path(new_filename)
                df.to_csv(new_path, index=False)
                os.remove(saved_path)
                saved_filename = new_filename
                saved_path = new_path
                set_last_initial_capital(initial_capital)
                # print(f"Transformed real trade log to backtest schema. Columns now: {list(df.columns)}")
            
            # Clear existing data and load new data
            portfolio.strategies.clear()
            portfolio.current_filename = None
            
            # Load new data with error handling and timeout protection
            try:
                import threading
                import time
                
                # Use timeout protection for large file loading
                load_result = {'success': False, 'error': None}
                
                def load_with_timeout():
                    try:
                        portfolio.load_from_csv(saved_path)
                        load_result['success'] = True
                    except Exception as e:
                        load_result['error'] = str(e)
                
                # Start loading in a separate thread
                load_thread = threading.Thread(target=load_with_timeout)
                load_thread.daemon = True
                load_thread.start()
                
                # Wait for up to 60 seconds for large files
                load_thread.join(timeout=60)
                
                if load_thread.is_alive():
                    # Timeout occurred
                    portfolio.strategies.clear()
                    portfolio.current_filename = None
                    return jsonify({
                        'success': False,
                        'error': 'File processing timed out. The file may be too large. Please try with a smaller file or contact support.'
                    }), 408
                
                if not load_result['success']:
                    raise Exception(load_result['error'])
                    
            except Exception as load_error:
                # If loading fails, ensure portfolio is in clean state
                portfolio.strategies.clear()
                portfolio.current_filename = None
                os.remove(saved_path)  # Clean up the uploaded file
                return jsonify({
                    'success': False,
                    'error': f'Failed to load portfolio data: {str(load_error)}'
                }), 500
            
            # Set current filename and session
            portfolio.current_filename = saved_filename
            session['current_filename'] = saved_filename
            
            # Clear any cached chart data for the new portfolio
            try:
                from charts import ChartGenerator
                chart_generator = ChartGenerator(portfolio)
                chart_generator.clear_cache()
            except Exception as cache_error:
                print(f"Warning: Could not clear cache: {cache_error}")
            # Track current data type and initial capital
            global current_data_type, current_initial_capital
            current_data_type = data_type
            current_initial_capital = initial_capital if data_type == 'real_trade' else None
                    # print(f"DEBUG: After upload - {len(portfolio.strategies)} strategies loaded")
        # print(f"DEBUG: Strategy names: {list(portfolio.strategies.keys())}")
            
            # Track successful upload
            app_insights.track_event('file_upload_success', {
                'user_id': get_current_user_id(),
                'trades_count': portfolio.total_trades,
                'strategies_count': len(portfolio.strategies),
                'filename': saved_filename
            }, {
                'trades_count': portfolio.total_trades,
                'strategies_count': len(portfolio.strategies)
            })
            
            response_data = {
                'success': True,
                'message': f'Successfully uploaded {portfolio.total_trades} trades',
                'data': {
                    'trades_count': portfolio.total_trades,
                    'strategies_count': len(portfolio.strategies),
                    'strategies': list(portfolio.strategies.keys()),
                    'filename': saved_filename,
                    'friendly_name': file_info['friendly_name'],
                    'initial_capital': initial_capital if data_type == 'real_trade' else None
                }
            }
            
            # Add warning message if file exceeds recommended trade limit
            if warning_message:
                response_data['warning'] = warning_message
                print(f"DEBUG: Adding warning to response: {warning_message}")
            else:
                print("DEBUG: No warning message to add")
                
            return jsonify(response_data)
            
        except Exception as e:
            # Clean up file if error occurs during processing
            if os.path.exists(saved_path):
                os.remove(saved_path)
            raise e
        
    except Exception as e:
        # Track upload failure
        app_insights.track_exception(e, {
            'user_id': get_current_user_id(),
            'endpoint': 'upload_csv'
        })
        
        return jsonify({
            'success': False,
            'error': f'Upload failed: {str(e)}'
        }), 500

@app.route('/api/cleanup-empty-dirs', methods=['POST'])
@guest_mode_required
def cleanup_empty_directories():
    """Clean up empty directories in the guest data folder."""
    try:
        print("=== CLEANUP EMPTY DIRECTORIES START ===")
        
        # Track cleanup attempt
        app_insights.track_event('cleanup_empty_dirs_attempted', {
            'user_id': get_current_user_id()
        })
        
        # Clean up empty directories
        removed_count = file_manager.cleanup_empty_directories()
        
        # Track successful cleanup
        app_insights.track_event('cleanup_empty_dirs_success', {
            'user_id': get_current_user_id(),
            'removed_count': removed_count
        })
        
        print(f"=== CLEANUP EMPTY DIRECTORIES SUCCESS: {removed_count} directories removed ===")
        return jsonify({
            'success': True,
            'removed_count': removed_count,
            'message': f'Cleaned up {removed_count} empty directories'
        })
        
    except Exception as e:
        # Track cleanup failure
        app_insights.track_exception(e, {
            'user_id': get_current_user_id(),
            'endpoint': 'cleanup_empty_directories'
        })
        
        print(f"=== CLEANUP EMPTY DIRECTORIES ERROR: {str(e)} ===")
        return jsonify({
            'success': False,
            'error': f'Failed to cleanup directories: {str(e)}'
        }), 500

@app.route('/api/load-sample-data', methods=['POST'])
@guest_mode_required
def load_sample_data():
    """Load the sample portfolio data from Example-Portfolio.csv."""
    try:
        print("=== LOAD SAMPLE DATA START ===")
        
        # Track sample data load attempt
        app_insights.track_event('sample_data_load_attempted', {
            'user_id': get_current_user_id()
        })
        
        import os
        sample_file_path = os.path.join(os.getcwd(), 'Example-Portfolio.csv')
        
        if not os.path.exists(sample_file_path):
            app_insights.track_event('sample_data_load_failed', {
                'user_id': get_current_user_id(),
                'error': 'file_not_found'
            })
            return jsonify({
                'success': False,
                'error': 'Sample data file not found. Please ensure Example-Portfolio.csv exists in the project root.'
            }), 404
        
        # Copy the sample file to the user's data directory with a timestamp
        import shutil
        from datetime import datetime
        
        # Generate timestamped filename
        base_name = 'Example-Portfolio'
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        saved_filename = f"{base_name}_{timestamp}.csv"
        saved_path = file_manager.get_file_path(saved_filename)
        
        # Copy the sample file to the user's data directory
        shutil.copy2(sample_file_path, saved_path)
        
        # Store metadata for the copied file
        file_manager.metadata[saved_filename] = {
            'friendly_name': 'Example Portfolio',
            'upload_date': datetime.now().isoformat(),
            'original_filename': 'Example-Portfolio.csv'
        }
        
        try:
            # Read CSV to validate columns
            df = pd.read_csv(saved_path)
            
            # Detect data type
            if 'Initial Premium' in df.columns:
                data_type = 'real_trade'
            else:
                data_type = 'backtest'
            
            # Set required columns
            if data_type == 'real_trade':
                required_columns = ['Date Opened', 'Date Closed', 'P/L']
            else:
                required_columns = ['Date Opened', 'Date Closed', 'P/L', 'Funds at Close']
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                os.remove(saved_path)  # Clean up file if invalid
                return jsonify({
                    'success': False,
                    'error': f'Missing required columns: {", ".join(missing_columns)}',
                    'available_columns': list(df.columns)
                }), 400
            
            # Clear existing data and load new data
            portfolio.strategies.clear()
            portfolio.load_from_csv(saved_path)
            # Set current filename
            portfolio.current_filename = saved_filename
            # Set data type
            global current_data_type, current_initial_capital
            current_data_type = data_type
            current_initial_capital = None  # Sample data doesn't need initial capital
            
            # Track successful sample data load
            app_insights.track_event('sample_data_load_success', {
                'user_id': get_current_user_id(),
                'trades_count': portfolio.total_trades,
                'strategies_count': len(portfolio.strategies),
                'filename': saved_filename
            })
            
            print("=== LOAD SAMPLE DATA SUCCESS ===")
            return jsonify({
                'success': True,
                'data': {
                    'filename': saved_filename,
                    'friendly_name': 'Example Portfolio',
                    'trades_count': portfolio.total_trades,
                    'strategies_count': len(portfolio.strategies),
                    'data_type': data_type
                }
            })
            
        except Exception as e:
            # Clean up file if error occurs during processing
            if os.path.exists(saved_path):
                os.remove(saved_path)
            raise e
        
    except Exception as e:
        # Track sample data load failure
        app_insights.track_exception(e, {
            'user_id': get_current_user_id(),
            'endpoint': 'load_sample_data'
        })
        
        print(f"=== LOAD SAMPLE DATA ERROR: {str(e)} ===")
        return jsonify({
            'success': False,
            'error': f'Failed to load sample data: {str(e)}'
        }), 500


@app.route('/api/load-sample-meic-data', methods=['POST'])
@guest_mode_required
def load_sample_meic_data():
    """Load the sample MEIC portfolio data from Example-Portfolio-MEIC.csv."""
    try:
        print("=== LOAD SAMPLE MEIC DATA START ===")
        
        # Track sample MEIC data load attempt
        app_insights.track_event('sample_meic_data_load_attempted', {
            'user_id': get_current_user_id()
        })
        
        import os
        sample_file_path = os.path.join(os.getcwd(), 'Example-Portfolio-MEIC.csv')
        
        if not os.path.exists(sample_file_path):
            app_insights.track_event('sample_meic_data_load_failed', {
                'user_id': get_current_user_id(),
                'error': 'file_not_found'
            })
            return jsonify({
                'success': False,
                'error': 'Sample MEIC data file not found. Please ensure Example-Portfolio-MEIC.csv exists in the project root.'
            }), 404
        
        # Copy the sample file to the user's data directory with a timestamp
        import shutil
        from datetime import datetime
        
        # Generate timestamped filename
        base_name = 'Example-Portfolio-MEIC'
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        saved_filename = f"{base_name}_{timestamp}.csv"
        saved_path = file_manager.get_file_path(saved_filename)
        
        # Copy the sample file
        shutil.copy2(sample_file_path, saved_path)
        print(f"Copied sample MEIC file to: {saved_path}")
        
        # Store metadata for the copied file
        file_manager.metadata[saved_filename] = {
            'friendly_name': 'Example MEIC Portfolio',
            'upload_date': datetime.now().isoformat(),
            'original_filename': 'Example-Portfolio-MEIC.csv'
        }
        
        # Load the data using the existing portfolio loading mechanism
        # Use the global portfolio instance
        
        # Determine data type from filename
        data_type = 'real_trades'  # MEIC data is typically real trades
        
        # Load the CSV data
        try:
            import pandas as pd
            df = pd.read_csv(saved_path)
            print(f"Loaded MEIC CSV with {len(df)} rows and columns: {list(df.columns)}")
            
            # Check if required columns exist (using actual column names from MEIC data)
            required_columns = ['Date Closed', 'P/L', 'Strategy']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                return jsonify({
                    'success': False,
                    'error': f'Missing required columns: {missing_columns}. Available columns: {list(df.columns)}'
                }), 400
            
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Failed to read MEIC CSV file: {str(e)}. Available columns: {list(df.columns)}'
            }), 400
        
        # Clear existing data and load new data
        portfolio.strategies.clear()
        portfolio.load_from_csv(saved_path)
        # Set current filename
        portfolio.current_filename = saved_filename
        # Set data type
        global current_data_type, current_initial_capital
        current_data_type = data_type
        current_initial_capital = None  # Sample data doesn't need initial capital
        
        # Track successful sample MEIC data load
        app_insights.track_event('sample_meic_data_load_success', {
            'user_id': get_current_user_id(),
            'trades_count': portfolio.total_trades,
            'strategies_count': len(portfolio.strategies),
            'filename': saved_filename
        })
        
        print("=== LOAD SAMPLE MEIC DATA SUCCESS ===")
        return jsonify({
            'success': True,
            'data': {
                'filename': saved_filename,
                'friendly_name': 'Example MEIC Portfolio',
                'trades_count': portfolio.total_trades,
                'strategies_count': len(portfolio.strategies),
                'data_type': data_type
            }
        })
        
    except Exception as e:
        # Clean up file if error occurs during processing
        if os.path.exists(saved_path):
            os.remove(saved_path)
        raise e
    
    except Exception as e:
        # Track sample MEIC data load failure
        app_insights.track_exception(e, {
            'user_id': get_current_user_id(),
            'endpoint': 'load_sample_meic_data'
        })
        
        print(f"=== LOAD SAMPLE MEIC DATA ERROR: {str(e)} ===")
        return jsonify({
            'success': False,
            'error': f'Failed to load sample MEIC data: {str(e)}'
        }), 500

@app.route('/api/saved-files')
def list_saved_files():
    """Get list of saved CSV files, ordered newest to oldest."""
    try:
        print("=== SAVED FILES API CALLED ===")
        file_info = file_manager.get_file_list()
        print(f"Returning {len(file_info)} files")
        print("=== SAVED FILES API COMPLETED ===")
        
        return jsonify({
            'success': True,
            'data': file_info
        })
        
    except Exception as e:
        print(f"ERROR in saved-files: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/load-file/<filename>')
def load_saved_file(filename):
    """Load a previously saved CSV file and reload portfolio."""
    try:
        # Clear session data to prevent artifacts
        session.pop('current_filename', None)
        
        file_path = file_manager.get_file_path(filename)
        if not os.path.exists(file_path):
            return jsonify({
                'success': False,
                'error': 'File not found'
            }), 404
        
        # Check file size for large dataset warning
        file_size = os.path.getsize(file_path)
        if file_size > 50 * 1024 * 1024:  # 50MB
            print(f"Warning: Loading large file {filename} ({file_size / 1024 / 1024:.1f}MB)")
        
        # Extract initial capital from filename if present
        initial_capital = extract_initial_capital_from_filename(filename, default=get_last_initial_capital())
        set_last_initial_capital(initial_capital)
        
        # Read CSV with error handling
        try:
            df = pd.read_csv(file_path)
        except Exception as csv_error:
            return jsonify({
                'success': False,
                'error': f'Failed to read CSV file: {str(csv_error)}'
            }), 400
        
        # Count trades and warn if too many (unless limit is disabled)
        trade_count = len(df)
        warning_message = None
        if os.environ.get('DISABLE_TRADE_LIMIT', 'false').lower() != 'true':
            max_trades = int(os.environ.get('MAX_TRADES_LIMIT', '3000'))
            print(f"DEBUG: Load saved file - Trade count: {trade_count}, Max trades: {max_trades}")
            if trade_count > max_trades:
                warning_message = f'⚠️ WARNING: This file contains {trade_count:,} trades. The recommended limit is {max_trades:,} trades. Files larger than this may produce unexpected problems and slow performance. Consider splitting your data into smaller files.'
                print(f"DEBUG: Load saved file - Warning message generated: {warning_message}")
        
        # Detect data type
        if 'Initial Premium' in df.columns:
            data_type = 'real_trade'
        else:
            data_type = 'backtest'
        
        # Transform if real trade
        if data_type == 'real_trade':
            try:
                df = transform_real_trade_to_backtest(df, initial_capital=initial_capital)
                # Overwrite file with transformed data
                df.to_csv(file_path, index=False)
            except Exception as transform_error:
                return jsonify({
                    'success': False,
                    'error': f'Failed to transform real trade data: {str(transform_error)}'
                }), 400
        
        # Clear portfolio completely before loading new data
        portfolio.strategies.clear()
        portfolio.current_filename = None
        
        # Load new data with timeout protection
        try:
            import threading
            
            # Use timeout protection for large file loading
            load_result = {'success': False, 'error': None}
            
            def load_with_timeout():
                try:
                    portfolio.load_from_csv(file_path)
                    load_result['success'] = True
                except Exception as e:
                    load_result['error'] = str(e)
            
            # Start loading in a separate thread
            load_thread = threading.Thread(target=load_with_timeout)
            load_thread.daemon = True
            load_thread.start()
            
            # Wait for up to 60 seconds for large files
            load_thread.join(timeout=60)
            
            if load_thread.is_alive():
                # Timeout occurred
                portfolio.strategies.clear()
                portfolio.current_filename = None
                return jsonify({
                    'success': False,
                    'error': 'File loading timed out. The file may be too large. Please try with a smaller file or contact support.'
                }), 408
            
            if not load_result['success']:
                raise Exception(load_result['error'])
                
        except Exception as load_error:
            # If loading fails, ensure portfolio is in clean state
            portfolio.strategies.clear()
            portfolio.current_filename = None
            return jsonify({
                'success': False,
                'error': f'Failed to load portfolio data: {str(load_error)}'
            }), 500
        
        # Set current filename and session
        portfolio.current_filename = filename
        session['current_filename'] = filename
        
        # Clear any cached chart data for the new portfolio
        try:
            from charts import ChartGenerator
            chart_generator = ChartGenerator(portfolio)
            chart_generator.clear_cache()
        except Exception as cache_error:
            print(f"Warning: Could not clear chart cache: {cache_error}")
        
        # Clear Monte Carlo simulation cache for the new portfolio
        try:
            from analytics import MonteCarloSimulator
            simulator = MonteCarloSimulator(portfolio)
            simulator.clear_cache()
            print("🧹 Monte Carlo cache cleared due to data reload")
        except Exception as cache_error:
            print(f"Warning: Could not clear Monte Carlo cache: {cache_error}")
        
        # Track current data type and initial capital
        global current_data_type, current_initial_capital
        current_data_type = data_type
        current_initial_capital = initial_capital if data_type == 'real_trade' else None
        
        response_data = {
            'success': True,
            'message': f'Loaded {portfolio.total_trades} trades',
            'data': {
                'trades_count': portfolio.total_trades,
                'strategies_count': len(portfolio.strategies),
                'strategies': list(portfolio.strategies.keys()),
                'filename': filename,
                'initial_capital': initial_capital if data_type == 'real_trade' else None,
                'file_size_mb': round(file_size / 1024 / 1024, 2)
            }
        }
        
        # Add warning message if file exceeds recommended trade limit
        if warning_message:
            response_data['warning'] = warning_message
            print(f"DEBUG: Load saved file - Adding warning to response: {warning_message}")
        else:
            print("DEBUG: Load saved file - No warning message to add")
            
        return jsonify(response_data)
    except Exception as e:
        # Ensure portfolio is in clean state on any error
        portfolio.strategies.clear()
        portfolio.current_filename = None
        session.pop('current_filename', None)
        return jsonify({
            'success': False,
            'error': f'Failed to load file: {str(e)}'
        }), 500

@app.route('/api/delete-file', methods=['POST'])
def delete_file():
    """Move a file to recycle folder instead of permanently deleting it."""
    try:
        data = request.get_json()
        filename = data.get('filename')
        
        if not filename:
            return jsonify({
                'success': False,
                'error': 'No filename provided'
            }), 400
        
        # Get the file path
        file_path = file_manager.get_file_path(filename)
        
        if not os.path.exists(file_path):
            return jsonify({
                'success': False,
                'error': 'File not found'
            }), 404
        
        # Create recycle folder if it doesn't exist
        data_folder = get_current_data_folder()
        recycle_folder = os.path.join(data_folder, 'recycle')
        os.makedirs(recycle_folder, exist_ok=True)
        
        # Create zip filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = os.path.splitext(filename)[0]
        zip_filename = f"{base_name}_{timestamp}.zip"
        zip_path = os.path.join(recycle_folder, zip_filename)
        
        # Move file to recycle folder as zip
        try:
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                zipf.write(file_path, filename)
            
            # Remove the original file
            os.remove(file_path)
            
            # Remove from metadata
            if filename in file_manager.metadata:
                friendly_name = file_manager.metadata[filename].get('friendly_name', filename)
                del file_manager.metadata[filename]
                file_manager._save_metadata()
            else:
                friendly_name = filename
            
            return jsonify({
                'success': True,
                'message': f'Successfully moved "{friendly_name}" to recycle folder'
            })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Failed to move file to recycle: {str(e)}'
            }), 500
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Recycle operation failed: {str(e)}'
        }), 500

@app.route('/api/cleanup-files', methods=['POST'])
def cleanup_duplicate_files():
    """Remove duplicate CSV files, keeping only the most recent version of each base name."""
    try:
        result = file_manager.cleanup_duplicates()
        
        return jsonify({
            'success': True,
            'message': result['message'],
            'deleted_count': result['deleted_count']
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# New endpoint to update friendly names
@app.route('/api/update-friendly-name', methods=['POST'])
def update_friendly_name():
    """Update the friendly name for a file."""
    try:
        data = request.get_json()
        filename = data.get('filename')
        new_friendly_name = data.get('friendly_name', '').strip()
        
        if not filename:
            return jsonify({
                'success': False,
                'error': 'Filename is required'
            }), 400
        
        if not new_friendly_name:
            return jsonify({
                'success': False,
                'error': 'Friendly name cannot be empty'
            }), 400
        
        if file_manager.update_friendly_name(filename, new_friendly_name):
            return jsonify({
                'success': True,
                'message': f'Updated friendly name to "{new_friendly_name}"'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'File not found'
            }), 404
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/monte-carlo/portfolio', methods=['POST'])
def run_portfolio_monte_carlo():
    """Run Monte Carlo simulation for the entire portfolio."""
    try:
        if not portfolio.strategies:
            return jsonify({
                'success': False,
                'error': 'No data loaded. Please upload a CSV file first.'
            }), 400
        
        # Get parameters from request
        data = request.get_json() or {}
        num_simulations = data.get('num_simulations', 1000)
        num_trades = data.get('num_trades', None)  # None = use historical count
        risk_free_rate = data.get('risk_free_rate', 4.0)  # Default to 4%
        
        # Validate parameters
        if not isinstance(num_simulations, int) or num_simulations < 100 or num_simulations > 10000:
            return jsonify({
                'success': False,
                'error': 'Number of simulations must be between 100 and 10,000'
            }), 400
        
        if num_trades is not None and (not isinstance(num_trades, int) or num_trades < 10):
            return jsonify({
                'success': False,
                'error': 'Number of trades must be at least 10'
            }), 400
        
        # Run simulation
        simulator = MonteCarloSimulator(portfolio)
        results = simulator.run_simulation(num_simulations, num_trades, risk_free_rate)
        
        return jsonify({
            'success': True,
            'data': results
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Monte Carlo simulation failed: {str(e)}'
        }), 500

@app.route('/api/monte-carlo/all-strategies', methods=['POST'])
def run_all_strategies_monte_carlo():
    """Run Monte Carlo simulation for all strategies individually to analyze fragility."""
    try:
        if not portfolio.strategies:
            return jsonify({
                'success': False,
                'error': 'No data loaded. Please upload a CSV file first.'
            }), 400
        
        # Get parameters from request
        data = request.get_json() or {}
        num_simulations = data.get('num_simulations', 1000)
        num_trades = data.get('num_trades', None)  # None = use historical count
        # trade_size_percent removed - using historical position sizing
        risk_free_rate = data.get('risk_free_rate', 4.0)  # Default to 4%
        
        # Validate parameters
        if not isinstance(num_simulations, int) or num_simulations < 100 or num_simulations > 10000:
            return jsonify({
                'success': False,
                'error': 'Number of simulations must be between 100 and 10,000'
            }), 400
        
        if num_trades is not None and (not isinstance(num_trades, int) or num_trades < 10):
            return jsonify({
                'success': False,
                'error': 'Number of trades must be at least 10'
            }), 400
        
        # trade_size_percent validation removed - using historical position sizing
        
        # Run Monte Carlo simulation for all strategies
        simulator = MonteCarloSimulator(portfolio)
        results = simulator.run_all_strategies_simulation(num_simulations, num_trades, risk_free_rate)
        
        return jsonify({
            'success': True,
            'data': results
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'All strategies Monte Carlo simulation failed: {str(e)}'
        }), 500

@app.route('/api/monte-carlo/simulation-details/<int:simulation_id>', methods=['POST'])
def get_simulation_details(simulation_id):
    """Get detailed trades for a specific simulation run."""
    try:
        if not portfolio.strategies:
            return jsonify({
                'success': False,
                'error': 'No data loaded. Please upload a CSV file first.'
            }), 400
        
        # Get simulation data from request body
        data = request.get_json()
        if not data or 'simulation_data' not in data:
            return jsonify({
                'success': False,
                'error': 'Simulation data is required'
            }), 400
        
        simulation_data = data['simulation_data']
        simulation_results = simulation_data.get('simulation_results', [])
        
        # Find the requested simulation
        target_simulation = None
        for sim in simulation_results:
            if sim['simulation_id'] == simulation_id:
                target_simulation = sim
                break
        
        if not target_simulation:
            return jsonify({
                'success': False,
                'error': f'Simulation {simulation_id} not found'
            }), 404
        
        # Get historical trades for comparison
        all_historical_trades = []
        for strategy in portfolio.strategies.values():
            for trade in strategy.trades:
                all_historical_trades.append({
                    'strategy': strategy.name,
                    'pnl': trade.pnl,
                    'date_opened': trade.date_opened.strftime('%Y-%m-%d') if trade.date_opened else None,
                    'date_closed': trade.date_closed.strftime('%Y-%m-%d') if trade.date_closed else None
                })
        
        # Prepare detailed response
        simulated_trades = target_simulation.get('simulated_trades', [])
        account_balance = target_simulation.get('account_balance', [])
        cumulative_pnl = target_simulation.get('cumulative_pnl', [])
        
        trades_detail = []
        for i, trade_pnl in enumerate(simulated_trades):
            trades_detail.append({
                'trade_number': i + 1,
                'pnl': trade_pnl,
                'cumulative_pnl': cumulative_pnl[i] if i < len(cumulative_pnl) else 0,
                'account_balance': account_balance[i] if i < len(account_balance) else 0,
                'is_winner': trade_pnl > 0
            })
        
        return jsonify({
            'success': True,
            'data': {
                'simulation_id': simulation_id,
                'summary': {
                    'total_trades': len(simulated_trades),
                    'winning_trades': len([t for t in simulated_trades if t > 0]),
                    'losing_trades': len([t for t in simulated_trades if t < 0]),
                    'win_rate': (len([t for t in simulated_trades if t > 0]) / len(simulated_trades) * 100) if simulated_trades else 0,
                    'total_pnl': target_simulation.get('total_pnl', 0),
                    'final_balance': target_simulation.get('final_balance', 0),
                    'max_drawdown': target_simulation.get('max_drawdown', 0)
                },
                'trades': trades_detail,
                'historical_comparison': {
                    'historical_trade_count': len(all_historical_trades),
                    'historical_pnl_range': {
                        'min': min([t['pnl'] for t in all_historical_trades]) if all_historical_trades else 0,
                        'max': max([t['pnl'] for t in all_historical_trades]) if all_historical_trades else 0
                    },
                    'simulated_pnl_range': {
                        'min': min(simulated_trades) if simulated_trades else 0,
                        'max': max(simulated_trades) if simulated_trades else 0
                    }
                }
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to get simulation details: {str(e)}'
        }), 500

@app.route('/api/data/clear', methods=['DELETE'])
def clear_data():
    """Clear all trade data (for testing/reset)."""
    try:
        # Clear portfolio data
        portfolio.strategies.clear()
        
        return jsonify({
            'success': True,
            'message': 'All data cleared successfully'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/data/stats')
def data_stats():
    """Get current data statistics."""
    try:
        # Get stats from portfolio
        all_trades = []
        for strategy in portfolio.strategies.values():
            all_trades.extend(strategy.trades)
        
        date_range = [None, None]
        if all_trades:
            dates = [trade.date_opened for trade in all_trades if trade.date_opened]
            if dates:
                date_range = [min(dates).strftime('%Y-%m-%d'), max(dates).strftime('%Y-%m-%d')]
        
        return jsonify({
            'success': True,
            'data': {
                'trade_count': portfolio.total_trades,
                'strategy_count': len(portfolio.strategies),
                'strategies': list(portfolio.strategies.keys()),
                'date_range': {
                    'start': date_range[0],
                    'end': date_range[1]
                },
                'filename': getattr(portfolio, 'current_filename', None)
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/strategy-pnl', methods=['POST'])
def get_strategy_pnl():
    """Get P&L data for specific strategies."""
    try:
        if not portfolio:
            return jsonify({'error': 'No portfolio data loaded'}), 400
        
        # Get strategy names from request body
        data = request.get_json()
        if not data or 'strategies' not in data:
            return jsonify({'error': 'No strategies provided'}), 400
        
        strategy_list = data['strategies']
        # print(f"\n=== STRATEGY P&L API CALL ===")
        # print(f"Requested strategies: {strategy_list}")
        # print(f"Total strategies requested: {len(strategy_list)}")
        
        # Get current date for validation
        from datetime import datetime
        current_date = datetime.now().date()
        # print(f"Current date: {current_date}")
        
        # Group trades by strategy and calculate cumulative P&L
        strategy_data = {}
        for strategy_name in strategy_list:
            # print(f"\n--- PROCESSING STRATEGY: {strategy_name} ---")
            
            if strategy_name in portfolio.strategies:
                strategy_trades = portfolio.strategies[strategy_name].trades
                # print(f"Found {len(strategy_trades)} trades for strategy '{strategy_name}'")
                
                if strategy_trades:
                    # Sort trades by date
                    strategy_trades.sort(key=lambda t: pd.to_datetime(t.date_closed))
                    # print(f"Sorted trades by date")
                    
                    # Filter out trades with invalid future dates
                    valid_trades = []
                    for trade in strategy_trades:
                        try:
                            trade_date = pd.to_datetime(trade.date_closed).date()
                            if trade_date <= current_date:
                                valid_trades.append(trade)
                            else:
                                # print(f"WARNING: Skipping trade with future date {trade_date} for strategy '{strategy_name}'")
                                pass
                        except Exception as e:
                            # print(f"WARNING: Skipping trade with invalid date {trade.date_closed} for strategy '{strategy_name}': {e}")
                            pass
                    
                    # print(f"Valid trades after filtering: {len(valid_trades)} (from {len(strategy_trades)} total)")
                    
                    if valid_trades:
                        # Print first few trades for debugging
                        # print(f"First 3 valid trades for '{strategy_name}':")
                        # for i, trade in enumerate(valid_trades[:3]):
                        #     print(f"  {i+1}. Date: {trade.date_closed}, P&L: ${trade.pnl:,.2f}")
                        
                        # Double-check sorting and log date progression
                        # print(f"Checking date progression for '{strategy_name}':")
                        # for i in range(min(10, len(valid_trades))):
                        #     print(f"  {i+1}. {valid_trades[i].date_closed}")
                        # if len(valid_trades) > 10:
                        #     print(f"  ... (showing first 10 of {len(valid_trades)})")
                        #     print(f"  Last trade: {valid_trades[-1].date_closed}")
                        
                        cumulative_pnl = []
                        dates = []
                        running_pnl = 0
                        
                        # print(f"Calculating cumulative P&L...")
                        for i, trade in enumerate(valid_trades):
                            running_pnl += trade.pnl
                            cumulative_pnl.append(running_pnl)
                            dates.append(trade.date_closed)
                            
                            # Debug every 10th trade or last few trades
                            # if i % 10 == 0 or i >= len(valid_trades) - 3:
                            #     print(f"  Trade {i+1}: Date={trade.date_closed}, P&L=${trade.pnl:,.2f}, Running Total=${running_pnl:,.2f}")
                        
                        # Verify chronological order
                        # print(f"Verifying chronological order...")
                        is_chronological = True
                        for i in range(1, len(dates)):
                            prev_date = pd.to_datetime(dates[i-1])
                            curr_date = pd.to_datetime(dates[i])
                            if curr_date < prev_date:
                                # print(f"  ERROR: Date order issue at position {i}: {dates[i-1]} -> {dates[i]}")
                                is_chronological = False
                        
                        # if is_chronological:
                        #     print(f"  ✓ Dates are in chronological order")
                        # else:
                        #     print(f"  ❌ Dates are NOT in chronological order - this will cause zigzag lines!")
                        
                        strategy_data[strategy_name] = {
                            'dates': [pd.to_datetime(date).isoformat() for date in dates],
                            'cumulative_pnl': cumulative_pnl
                        }
                        
                        # print(f"Strategy '{strategy_name}' final stats:")
                        # print(f"  - Data points: {len(dates)}")
                        # print(f"  - Final P&L: ${running_pnl:,.2f}")
                        # print(f"  - P&L range: ${min(cumulative_pnl):,.2f} to ${max(cumulative_pnl):,.2f}")
                        # print(f"  - Date range: {min(dates)} to {max(dates)}")
                        
                        # Verify data integrity
                        # if len(dates) != len(cumulative_pnl):
                        #     print(f"ERROR: Data mismatch! {len(dates)} dates vs {len(cumulative_pnl)} P&L values")
                        # else:
                        #     print(f"✓ Data integrity check passed")
                    else:
                        # print(f"WARNING: No valid trades found for strategy '{strategy_name}' after date filtering")
                        pass
                else:
                    # print(f"Strategy '{strategy_name}' has no trades")
                    pass
            else:
                # print(f"Strategy '{strategy_name}' not found in portfolio")
                pass
        
        # print(f"\n=== FINAL SUMMARY ===")
        # print(f"Returning data for {len(strategy_data)} strategies")
        # for strategy_name, data in strategy_data.items():
        #     print(f"  {strategy_name}: {len(data['dates'])} data points")
        
        return jsonify({
            'success': True,
            'data': strategy_data
        })
        
    except Exception as e:
        # print(f"ERROR in get_strategy_pnl: {str(e)}")
        # import traceback
        # traceback.print_exc()
        return jsonify({'error': f'Error generating strategy P&L data: {str(e)}'}), 500

@app.route('/api/trade-margin', methods=['POST'])
def get_trade_margin():
    """Get margin information for a specific trade or list of trades."""
    try:
        data = request.get_json()
        legs_string = data.get('legs', '')
        
        if not legs_string:
            return jsonify({
                'error': 'No legs string provided',
                'success': False
            }), 400
        
        margin_info = calculate_margin_from_legs(legs_string)
        
        return jsonify({
            'margin_info': margin_info,
            'success': True
        })
        
    except Exception as e:
        print(f"Error calculating margin: {e}")
        return jsonify({
            'error': f'Error calculating margin: {str(e)}',
            'success': False
        }), 500

@app.route('/api/strategy-margin-summary')
def get_strategy_margin_summary():
    """Get margin summary for all strategies."""
    try:
        if not portfolio.strategies:
            return jsonify({
                'strategies': [],
                'success': True
            })
        
        strategy_margins = []
        
        for strategy_name, strategy in portfolio.strategies.items():
            total_margin = 0
            total_contracts = 0
            margin_trades = 0
            
            for trade in strategy.trades:
                # Get legs from the trade data
                # We need to access the original CSV data to get the legs
                # For now, we'll use a placeholder approach
                legs_str = getattr(trade, 'legs', '')
                if legs_str:
                    margin_info = calculate_margin_from_legs(legs_str)
                    total_margin += margin_info['overall_margin']
                    total_contracts += margin_info['total_contracts']
                    margin_trades += 1
            
            strategy_margins.append({
                'strategy': strategy_name,
                'total_margin': total_margin,
                'total_contracts': total_contracts,
                'margin_trades': margin_trades,
                'avg_margin_per_contract': total_margin / total_contracts if total_contracts > 0 else 0
            })
        
        return jsonify({
            'strategies': strategy_margins,
            'success': True
        })
        
    except Exception as e:
        print(f"Error getting strategy margin summary: {e}")
        return jsonify({
            'error': f'Error getting strategy margin summary: {str(e)}',
            'success': False
        }), 500

@app.route('/api/pnl-by-day-of-week')
def get_pnl_by_day_of_week():
    """Get P&L by day of week for all strategies."""
    try:
        if not portfolio.strategies:
            return jsonify({
                'success': True,
                'data': []
            })
        
        # Initialize days of week (excluding weekends)
        days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        
        pnl_data = []
        
        for strategy_name, strategy in portfolio.strategies.items():
            # Initialize P&L for each day of week
            day_pnl = {day: 0.0 for day in days_of_week}
            
            for trade in strategy.trades:
                if trade.date_closed:
                    # Get day of week from date_closed
                    day_of_week = trade.date_closed.strftime('%A')
                    day_pnl[day_of_week] += trade.pnl
            
            # Create row data for this strategy
            row_data = {
                'strategy': strategy_name,
                'trades': len(strategy.trades)
            }
            
            # Add P&L for each day
            for day in days_of_week:
                row_data[day] = day_pnl[day]
            
            pnl_data.append(row_data)
        
        # Sort by total P&L descending
        pnl_data.sort(key=lambda x: sum(x[day] for day in days_of_week), reverse=True)
        
        return jsonify({
            'success': True,
            'data': pnl_data,
            'days_of_week': days_of_week
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/pnl-by-day-of-week-cumulative')
def get_pnl_by_day_of_week_cumulative():
    """Get cumulative P&L by day of week for all strategies."""
    try:
        if not portfolio.strategies:
            return jsonify({
                'success': True,
                'data': []
            })
        
        # Initialize days of week (excluding weekends)
        days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        
        # Get all trades from all strategies, sorted by date
        all_trades = []
        for strategy_name, strategy in portfolio.strategies.items():
            for trade in strategy.trades:
                if trade.date_closed:
                    all_trades.append({
                        'date': trade.date_closed,
                        'pnl': trade.pnl,
                        'strategy': strategy_name,
                        'day_of_week': trade.date_closed.strftime('%A')
                    })
        
        # Sort trades by date
        all_trades.sort(key=lambda x: x['date'])
        
        # Calculate cumulative P&L by day of week
        cumulative_data = []
        day_cumulative = {day: 0.0 for day in days_of_week}
        
        for trade in all_trades:
            day_of_week = trade['day_of_week']
            if day_of_week in day_cumulative:
                day_cumulative[day_of_week] += trade['pnl']
                
                # Add data point for this trade
                cumulative_data.append({
                    'date': trade['date'].strftime('%Y-%m-%d'),
                    'day_of_week': day_of_week,
                    'pnl': trade['pnl'],
                    'cumulative_pnl': day_cumulative[day_of_week],
                    'strategy': trade['strategy']
                })
        
        return jsonify({
            'success': True,
            'data': cumulative_data,
            'days_of_week': days_of_week
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/pnl-by-month')
def get_pnl_by_month():
    """Get P&L by month for all strategies."""
    try:
        if not portfolio.strategies:
            return jsonify({
                'success': True,
                'data': [],
                'months': []
            })
        
        # Collect all unique months across all trades
        all_months = set()
        
        for strategy in portfolio.strategies.values():
            for trade in strategy.trades:
                if trade.date_closed:
                    month_key = trade.date_closed.strftime('%Y-%m')
                    all_months.add(month_key)
        
        # Sort months chronologically
        sorted_months = sorted(list(all_months))
        
        # Generate month labels for display (e.g., "Jan 2023")
        month_labels = []
        for month_key in sorted_months:
            year, month = month_key.split('-')
            month_obj = pd.to_datetime(f"{year}-{month}-01")
            month_labels.append(month_obj.strftime('%b %Y'))
        
        pnl_data = []
        monthly_totals = {month: 0.0 for month in sorted_months}
        
        for strategy_name, strategy in portfolio.strategies.items():
            # Initialize P&L for each month
            month_pnl = {month: 0.0 for month in sorted_months}
            
            for trade in strategy.trades:
                if trade.date_closed:
                    month_key = trade.date_closed.strftime('%Y-%m')
                    if month_key in month_pnl:
                        month_pnl[month_key] += trade.pnl
                        monthly_totals[month_key] += trade.pnl
            
            # Create row data for this strategy
            row_data = {
                'strategy': strategy_name,
                'trades': len(strategy.trades),
                'total_pnl': sum(month_pnl.values())
            }
            
            # Add P&L for each month
            for month in sorted_months:
                row_data[month] = month_pnl[month]
            
            pnl_data.append(row_data)
        
        # Sort by total P&L descending
        pnl_data.sort(key=lambda x: x['total_pnl'], reverse=True)
        
        return jsonify({
            'success': True,
            'data': pnl_data,
            'months': sorted_months,
            'month_labels': month_labels,
            'monthly_totals': monthly_totals
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/conflicting-legs')
@guest_mode_required
def conflicting_legs_analysis():
    """Analyze trades for conflicting legs (opposing positions on same contract) - optimized."""
    try:
        # Generate cache key based on portfolio state
        portfolio_id = getattr(portfolio, 'current_filename', 'default')
        cache_key = f"conflicting_legs_{portfolio_id}"
        
        # Try to get from cache first
        try:
            from cache_manager import cache_manager
            cached_result = cache_manager.get(cache_key)
            if cached_result:
                return jsonify(cached_result)
        except ImportError:
            pass  # Cache not available, continue with calculation
        
        if not portfolio.strategies:
            result = {
                'success': True,
                'data': {
                    'conflicts': [],
                    'summary': {
                        'total_conflicts': 0,
                        'affected_trades': 0,
                        'affected_strategies': 0
                    }
                }
            }
            # Cache the result
            try:
                from cache_manager import cache_manager
                cache_manager.set(cache_key, result)
            except ImportError:
                pass
            return jsonify(result)
        
        # Pre-process all trades with legs information (optimized)
        all_trades_with_legs = []
        for strategy_name, strategy in portfolio.strategies.items():
            for trade in strategy.trades:
                legs_str = getattr(trade, 'legs', '')
                if legs_str and legs_str.strip():
                    # Parse legs for this trade (now cached)
                    legs = parse_legs_string(legs_str)
                    if legs:
                        # Pre-process dates to avoid repeated parsing
                        date_opened = trade.date_opened
                        date_closed = trade.date_closed
                        
                        all_trades_with_legs.append({
                            'strategy': strategy_name,
                            'trade': trade,
                            'legs': legs,
                            'date_opened': date_opened,
                            'date_closed': date_closed,
                            'date_opened_ts': date_opened if date_opened else pd.Timestamp('1900-01-01'),
                            'date_closed_ts': date_closed if date_closed else pd.Timestamp('1900-01-01')
                        })
        
        if not all_trades_with_legs:
            result = {
                'success': True,
                'data': {
                    'conflicts': [],
                    'summary': {
                        'total_conflicts': 0,
                        'affected_trades': 0,
                        'affected_strategies': 0
                    }
                }
            }
            # Cache the result
            try:
                from cache_manager import cache_manager
                cache_manager.set(cache_key, result)
            except ImportError:
                pass
            return jsonify(result)
        
        # Sort trades by date opened for chronological analysis
        all_trades_with_legs.sort(key=lambda x: x['date_opened_ts'])
        
        conflicts = []
        affected_trades = set()
        affected_strategies = set()
        
        # Optimized conflict detection using contract indexing
        contract_positions = {}  # contract_key -> list of trades with that contract
        
        # First pass: index all trades by their contracts
        for trade_data in all_trades_with_legs:
            for leg in trade_data['legs']:
                contract_key = f"{leg['expiration']}_{leg['strike']}_{leg['type']}"
                if contract_key not in contract_positions:
                    contract_positions[contract_key] = []
                contract_positions[contract_key].append({
                    'trade_data': trade_data,
                    'leg': leg
                })
        
        # Second pass: find conflicts within each contract
        for contract_key, positions in contract_positions.items():
            if len(positions) < 2:
                continue
                
            # Sort positions by date opened
            positions.sort(key=lambda x: x['trade_data']['date_opened_ts'])
            
            # Check for conflicts between positions on the same contract
            for i, pos1 in enumerate(positions):
                for j, pos2 in enumerate(positions[i+1:], i+1):
                    trade1_data = pos1['trade_data']
                    trade2_data = pos2['trade_data']
                    leg1 = pos1['leg']
                    leg2 = pos2['leg']
                    
                    # Skip if same trade
                    if trade1_data['trade'] == trade2_data['trade']:
                        continue
                    
                    # Check for time overlap (trade2 opens before trade1 closes)
                    if (trade2_data['date_opened_ts'] <= trade1_data['date_closed_ts'] and
                        trade1_data['date_opened'] and trade1_data['date_closed'] and
                        trade2_data['date_opened'] and trade2_data['date_closed']):
                        
                        # Check if opposing actions (STO vs BTO)
                        if ((leg1['action'] == 'STO' and leg2['action'] == 'BTO') or
                            (leg1['action'] == 'BTO' and leg2['action'] == 'STO')):
                            
                            conflict = {
                                'conflict_id': len(conflicts) + 1,
                                'trade1': {
                                    'strategy': trade1_data['strategy'],
                                    'date_opened': trade1_data['date_opened'].strftime('%Y-%m-%d') if trade1_data['date_opened'] else 'N/A',
                                    'date_closed': trade1_data['date_closed'].strftime('%Y-%m-%d') if trade1_data['date_closed'] else 'N/A',
                                    'pnl': trade1_data['trade'].pnl,
                                    'leg': {
                                        'quantity': leg1['quantity'],
                                        'expiration': leg1['expiration'],
                                        'strike': leg1['strike'],
                                        'type': leg1['type'],
                                        'action': leg1['action'],
                                        'premium': leg1['premium']
                                    }
                                },
                                'trade2': {
                                    'strategy': trade2_data['strategy'],
                                    'date_opened': trade2_data['date_opened'].strftime('%Y-%m-%d') if trade2_data['date_opened'] else 'N/A',
                                    'date_closed': trade2_data['date_closed'].strftime('%Y-%m-%d') if trade2_data['date_closed'] else 'N/A',
                                    'pnl': trade2_data['trade'].pnl,
                                    'leg': {
                                        'quantity': leg2['quantity'],
                                        'expiration': leg2['expiration'],
                                        'strike': leg2['strike'],
                                        'type': leg2['type'],
                                        'action': leg2['action'],
                                        'premium': leg2['premium']
                                    }
                                },
                                'contract_details': {
                                    'expiration': leg1['expiration'],
                                    'strike': leg1['strike'],
                                    'type': leg1['type'],
                                    'description': f"{leg1['expiration']} {leg1['strike']} {leg1['type']}"
                                },
                                'conflict_type': 'opposing_positions',
                                'overlap_days': (min(trade1_data['date_closed_ts'], trade2_data['date_closed_ts']) - 
                                                max(trade1_data['date_opened_ts'], trade2_data['date_opened_ts'])).days,
                                'severity': 'high' if leg1['quantity'] == leg2['quantity'] else 'medium',
                                'potential_impact': 'Position cancellation' if leg1['quantity'] == leg2['quantity'] else 'Partial position reduction'
                            }
                            
                            conflicts.append(conflict)
                            affected_trades.add(f"{trade1_data['strategy']}-{trade1_data['date_opened']}")
                            affected_trades.add(f"{trade2_data['strategy']}-{trade2_data['date_opened']}")
                            affected_strategies.add(trade1_data['strategy'])
                            affected_strategies.add(trade2_data['strategy'])
        
        # Sort conflicts by date
        conflicts.sort(key=lambda x: x['trade1']['date_opened'])
        
        # Create summary statistics (optimized)
        high_severity_count = 0
        medium_severity_count = 0
        strategy_conflict_counts = {}
        
        for conflict in conflicts:
            if conflict['severity'] == 'high':
                high_severity_count += 1
            else:
                medium_severity_count += 1
                
            # Count conflicts per strategy
            strategy1 = conflict['trade1']['strategy']
            strategy2 = conflict['trade2']['strategy']
            strategy_conflict_counts[strategy1] = strategy_conflict_counts.get(strategy1, 0) + 1
            if strategy1 != strategy2:
                strategy_conflict_counts[strategy2] = strategy_conflict_counts.get(strategy2, 0) + 1
        
        summary = {
            'total_conflicts': len(conflicts),
            'affected_trades': len(affected_trades),
            'affected_strategies': len(affected_strategies),
            'high_severity_conflicts': high_severity_count,
            'medium_severity_conflicts': medium_severity_count,
            'strategy_breakdown': strategy_conflict_counts
        }
        
        result = {
            'success': True,
            'data': {
                'conflicts': conflicts,
                'summary': summary
            }
        }
        
        # Cache the result
        try:
            from cache_manager import cache_manager
            cache_manager.set(cache_key, result)
        except ImportError:
            pass
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in conflicting_legs_analysis: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/commission-analysis')
@guest_mode_required
def commission_analysis():
    """Get comprehensive commission analysis for all trades (optimized with caching)."""
    try:
        # Generate cache key based on portfolio state
        portfolio_id = getattr(portfolio, 'current_filename', 'default')
        cache_key = f"commission_analysis_{portfolio_id}"
        
        # Try to get from cache first
        try:
            from cache_manager import cache_manager
            cached_result = cache_manager.get(cache_key)
            if cached_result:
                return jsonify(cached_result)
        except ImportError:
            pass  # Cache not available, continue with calculation
        
        # Load commission configuration
        config_manager = CommissionConfigManager()
        config = config_manager.load_config()
        calculator = CommissionCalculator(config)
        
        # Collect all trades from all strategies
        all_trades = []
        for strategy in portfolio.strategies.values():
            all_trades.extend(strategy.trades)
        
        if not all_trades:
            return jsonify({'error': 'No trades available for analysis'}), 404
        
        # Get comprehensive commission analysis (optimized)
        analysis = calculator.get_commission_summary(all_trades)
        
        # Strategy breakdown is already included in the analysis
        # No need to recalculate for each strategy individually
        
        # Cache the result
        try:
            from cache_manager import cache_manager
            cache_manager.set(cache_key, analysis)
        except ImportError:
            pass  # Cache not available, continue without caching
        
        return jsonify(analysis)
        
    except Exception as e:
        print(f"Error in commission_analysis: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/commission-config', methods=['GET'])
@guest_mode_required
def get_commission_config():
    """Get current commission configuration."""
    try:
        config_manager = CommissionConfigManager()
        config = config_manager.load_config()
        
        return jsonify({
            'spx': {
                'opening_cost': config.spx.opening_cost,
                'closing_cost': config.spx.closing_cost,
                'exercise_fee': config.spx.exercise_fee
            },
            'qqq': {
                'opening_cost': config.qqq.opening_cost,
                'closing_cost': config.qqq.closing_cost,
                'exercise_fee': config.qqq.exercise_fee
            },
            'default': {
                'opening_cost': config.default.opening_cost,
                'closing_cost': config.default.closing_cost,
                'exercise_fee': config.default.exercise_fee
            }
        })
        
    except Exception as e:
        print(f"Error getting commission config: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/commission-config', methods=['POST'])
@guest_mode_required
def update_commission_config():
    """Update commission configuration."""
    try:
        data = request.get_json()
        
        # Validate input data
        for underlying in ['spx', 'qqq', 'default']:
            if underlying not in data:
                return jsonify({'error': f'Missing configuration for {underlying}'}), 400
            
            config_data = data[underlying]
            required_fields = ['opening_cost', 'closing_cost', 'exercise_fee']
            for field in required_fields:
                if field not in config_data:
                    return jsonify({'error': f'Missing {field} for {underlying}'}), 400
                try:
                    float(config_data[field])
                except ValueError:
                    return jsonify({'error': f'Invalid {field} value for {underlying}'}), 400
        
        # Create new configuration
        from commission_config import CommissionRates, CommissionConfig
        
        new_config = CommissionConfig(
            spx=CommissionRates(
                opening_cost=float(data['spx']['opening_cost']),
                closing_cost=float(data['spx']['closing_cost']),
                exercise_fee=float(data['spx']['exercise_fee'])
            ),
            qqq=CommissionRates(
                opening_cost=float(data['qqq']['opening_cost']),
                closing_cost=float(data['qqq']['closing_cost']),
                exercise_fee=float(data['qqq']['exercise_fee'])
            ),
            default=CommissionRates(
                opening_cost=float(data['default']['opening_cost']),
                closing_cost=float(data['default']['closing_cost']),
                exercise_fee=float(data['default']['exercise_fee'])
            )
        )
        
        # Save configuration
        config_manager = CommissionConfigManager()
        config_manager.save_config(new_config)
        
        return jsonify({'success': True, 'message': 'Commission configuration updated successfully'})
        
    except Exception as e:
        print(f"Error updating commission config: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    """Health check endpoint for Azure App Service."""
    try:
        # Basic health check
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'version': '1.0.0'
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/cross-file-correlation', methods=['GET', 'POST'])
@guest_mode_required
def get_cross_file_correlation():
    """Get correlation analysis across selected uploaded files."""
    try:
        from analytics import CrossFileAnalyzer
        
        # Get parameters
        if request.method == 'POST':
            data = request.get_json()
            selected_files = data.get('selected_files', [])
            max_pairs = data.get('max_pairs', 30)
        else:
            # For GET requests, get all files (backward compatibility)
            selected_files = []
            max_pairs = request.args.get('max_pairs', 30, type=int)
        
        max_pairs = min(max_pairs, 100)  # Cap at 100 for performance
        
        # Initialize cross-file analyzer
        cross_analyzer = CrossFileAnalyzer(file_manager)
        
        # Get correlation analysis
        result = cross_analyzer.get_cross_file_correlations(max_pairs=max_pairs, selected_files=selected_files)
        
        if result['success']:
            return jsonify({
                'success': True,
                'data': result['data']
            })
        else:
            return jsonify({
                'success': False,
                'error': result['error'],
                'details': {key: value for key, value in result.items() if key not in ['success', 'error']}
            }), 400
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Cross-file correlation analysis failed: {str(e)}'
        }), 500

def _generate_eom_dates():
    """Generate end of month dates from 2021 to end of following year"""
    from datetime import datetime, timedelta
    import calendar
    
    current_year = datetime.now().year
    end_year = current_year + 1  # End of following year
    
    eom_dates = []
    for year in range(2021, end_year + 1):
        for month in range(1, 13):
            # Get the last day of the month
            last_day = calendar.monthrange(year, month)[1]
            eom_date = datetime(year, month, last_day)
            
            # Skip weekends - if last day is Saturday (5) or Sunday (6), use Friday
            if eom_date.weekday() == 5:  # Saturday
                eom_date = eom_date - timedelta(days=1)
            elif eom_date.weekday() == 6:  # Sunday
                eom_date = eom_date - timedelta(days=2)
            
            eom_dates.append(eom_date.strftime('%Y-%m-%d'))
    
    return eom_dates

def _generate_eoq_dates():
    """Generate end of quarter dates from 2021 to end of following year"""
    from datetime import datetime, timedelta
    import calendar
    
    current_year = datetime.now().year
    end_year = current_year + 1  # End of following year
    
    eoq_dates = []
    for year in range(2021, end_year + 1):
        for month in [3, 6, 9, 12]:  # End of quarters
            # Get the last day of the month
            last_day = calendar.monthrange(year, month)[1]
            eoq_date = datetime(year, month, last_day)
            
            # Skip weekends - if last day is Saturday (5) or Sunday (6), use Friday
            if eoq_date.weekday() == 5:  # Saturday
                eoq_date = eoq_date - timedelta(days=1)
            elif eoq_date.weekday() == 6:  # Sunday
                eoq_date = eoq_date - timedelta(days=2)
            
            eoq_dates.append(eoq_date.strftime('%Y-%m-%d'))
    
    return eoq_dates

def _generate_fomc_dates():
    """Generate FOMC meeting dates - static list"""
    return [
        '2021-01-27', '2021-03-17', '2021-04-28', '2021-06-16', '2021-07-28', '2021-09-22', '2021-11-03', '2021-12-15',
        '2022-01-26', '2022-03-16', '2022-05-04', '2022-06-15', '2022-07-27', '2022-09-21', '2022-11-02', '2022-12-14',
        '2023-01-31', '2023-03-22', '2023-05-03', '2023-06-14', '2023-07-26', '2023-09-20', '2023-11-01', '2023-12-13',
        '2024-01-31', '2024-03-20', '2024-05-01', '2024-06-12', '2024-07-31', '2024-09-18', '2024-11-07', '2024-12-18',
        '2025-01-29', '2025-03-19', '2025-05-07', '2025-06-18', '2025-07-30', '2025-09-17', '2025-11-05', '2025-12-17',
        '2026-01-28', '2026-03-18', '2026-05-06', '2026-06-17', '2026-07-29', '2026-09-16', '2026-11-04', '2026-12-16'
    ]

def _generate_short_weeks_dates():
    """Generate stock market holidays from 2020 to end of following year"""
    from datetime import datetime, timedelta
    import calendar
    
    current_year = datetime.now().year
    end_year = current_year + 1  # End of following year
    
    short_weeks = []
    
    for year in range(2020, end_year + 1):
        # New Year's Day (January 1)
        short_weeks.append(f'{year}-01-01')
        
        # Martin Luther King Jr. Day (3rd Monday in January)
        mlk_day = _get_nth_weekday(year, 1, 0, 3)  # 3rd Monday
        short_weeks.append(mlk_day.strftime('%Y-%m-%d'))
        
        # Presidents' Day (3rd Monday in February)
        presidents_day = _get_nth_weekday(year, 2, 0, 3)  # 3rd Monday
        short_weeks.append(presidents_day.strftime('%Y-%m-%d'))
        
        # Good Friday (Friday before Easter - approximate calculation)
        easter = _calculate_easter(year)
        good_friday = easter - timedelta(days=2)
        short_weeks.append(good_friday.strftime('%Y-%m-%d'))
        
        # Memorial Day (Last Monday in May)
        memorial_day = _get_last_weekday(year, 5, 0)  # Last Monday
        short_weeks.append(memorial_day.strftime('%Y-%m-%d'))
        
        # Juneteenth (June 19, since 2021)
        if year >= 2021:
            short_weeks.append(f'{year}-06-19')
        
        # Independence Day (July 4)
        short_weeks.append(f'{year}-07-04')
        
        # Labor Day (1st Monday in September)
        labor_day = _get_nth_weekday(year, 9, 0, 1)  # 1st Monday
        short_weeks.append(labor_day.strftime('%Y-%m-%d'))
        
        # Thanksgiving Day (4th Thursday in November)
        thanksgiving = _get_nth_weekday(year, 11, 3, 4)  # 4th Thursday
        short_weeks.append(thanksgiving.strftime('%Y-%m-%d'))
        
        # Christmas Day (December 25)
        short_weeks.append(f'{year}-12-25')
    
    return short_weeks

def _get_nth_weekday(year, month, weekday, n):
    """Get the nth weekday of a given month (0=Monday, 6=Sunday)"""
    from datetime import datetime, timedelta
    
    # Find the first day of the month
    first_day = datetime(year, month, 1)
    
    # Find the first occurrence of the target weekday
    days_ahead = weekday - first_day.weekday()
    if days_ahead < 0:
        days_ahead += 7
    
    # Get the nth occurrence
    target_date = first_day + timedelta(days=days_ahead + (n - 1) * 7)
    
    return target_date

def _get_last_weekday(year, month, weekday):
    """Get the last weekday of a given month (0=Monday, 6=Sunday)"""
    from datetime import datetime, timedelta
    import calendar
    
    # Get the last day of the month
    last_day = calendar.monthrange(year, month)[1]
    last_date = datetime(year, month, last_day)
    
    # Find the last occurrence of the target weekday
    days_back = (last_date.weekday() - weekday) % 7
    target_date = last_date - timedelta(days=days_back)
    
    return target_date

def _calculate_easter(year):
    """Calculate Easter Sunday for a given year (approximate)"""
    from datetime import datetime, timedelta
    
    # Easter calculation using the algorithm
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    
    return datetime(year, month, day)

@app.route('/api/custom-blackout-lists', methods=['GET'])
@guest_mode_required
def get_custom_blackout_lists():
    """Get all custom blackout date lists for the current user."""
    try:
        user_id = get_current_user_id()
        db_manager = DatabaseManager()
        
        conn = db_manager.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, list_name, description, dates, created_at, updated_at
            FROM custom_blackout_lists
            WHERE user_id = ?
            ORDER BY list_name
        ''', (user_id,))
        
        lists = []
        for row in cursor.fetchall():
            import json
            lists.append({
                'id': row[0],
                'list_name': row[1],
                'description': row[2],
                'dates': json.loads(row[3]) if row[3] else [],
                'created_at': row[4],
                'updated_at': row[5]
            })
        
        conn.close()
        
        return jsonify({
            'success': True,
            'lists': lists
        })
        
    except Exception as e:
        print(f"Error getting custom blackout lists: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/custom-blackout-lists', methods=['POST'])
@guest_mode_required
def create_custom_blackout_list():
    """Create a new custom blackout date list."""
    try:
        user_id = get_current_user_id()
        data = request.get_json()
        
        list_name = data.get('list_name', '').strip()
        description = data.get('description', '').strip()
        dates = data.get('dates', [])
        
        # Validation
        if not list_name:
            return jsonify({'success': False, 'error': 'List name is required'}), 400
        
        if not isinstance(dates, list):
            return jsonify({'success': False, 'error': 'Dates must be a list'}), 400
        
        # Validate date format
        from datetime import datetime
        validated_dates = []
        for date_str in dates:
            try:
                datetime.strptime(date_str, '%Y-%m-%d')
                validated_dates.append(date_str)
            except ValueError:
                return jsonify({'success': False, 'error': f'Invalid date format: {date_str}. Use YYYY-MM-DD'}), 400
        
        db_manager = DatabaseManager()
        conn = db_manager.get_connection()
        cursor = conn.cursor()
        
        # Check if list name already exists for this user
        cursor.execute('''
            SELECT id FROM custom_blackout_lists
            WHERE user_id = ? AND list_name = ?
        ''', (user_id, list_name))
        
        if cursor.fetchone():
            conn.close()
            return jsonify({'success': False, 'error': 'A list with this name already exists'}), 400
        
        # Insert new list
        import json
        cursor.execute('''
            INSERT INTO custom_blackout_lists (user_id, list_name, description, dates)
            VALUES (?, ?, ?, ?)
        ''', (user_id, list_name, description, json.dumps(validated_dates)))
        
        list_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return jsonify({
            'success': True,
            'list_id': list_id,
            'message': 'Custom blackout list created successfully'
        })
        
    except Exception as e:
        print(f"Error creating custom blackout list: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/custom-blackout-lists/<int:list_id>', methods=['PUT'])
@guest_mode_required
def update_custom_blackout_list(list_id):
    """Update an existing custom blackout date list."""
    try:
        user_id = get_current_user_id()
        data = request.get_json()
        
        list_name = data.get('list_name', '').strip()
        description = data.get('description', '').strip()
        dates = data.get('dates', [])
        
        # Validation
        if not list_name:
            return jsonify({'success': False, 'error': 'List name is required'}), 400
        
        if not isinstance(dates, list):
            return jsonify({'success': False, 'error': 'Dates must be a list'}), 400
        
        # Validate date format
        from datetime import datetime
        validated_dates = []
        for date_str in dates:
            try:
                datetime.strptime(date_str, '%Y-%m-%d')
                validated_dates.append(date_str)
            except ValueError:
                return jsonify({'success': False, 'error': f'Invalid date format: {date_str}. Use YYYY-MM-DD'}), 400
        
        db_manager = DatabaseManager()
        conn = db_manager.get_connection()
        cursor = conn.cursor()
        
        # Check if list exists and belongs to user
        cursor.execute('''
            SELECT id FROM custom_blackout_lists
            WHERE id = ? AND user_id = ?
        ''', (list_id, user_id))
        
        if not cursor.fetchone():
            conn.close()
            return jsonify({'success': False, 'error': 'List not found'}), 404
        
        # Check if new name conflicts with existing list
        cursor.execute('''
            SELECT id FROM custom_blackout_lists
            WHERE user_id = ? AND list_name = ? AND id != ?
        ''', (user_id, list_name, list_id))
        
        if cursor.fetchone():
            conn.close()
            return jsonify({'success': False, 'error': 'A list with this name already exists'}), 400
        
        # Update list
        import json
        cursor.execute('''
            UPDATE custom_blackout_lists
            SET list_name = ?, description = ?, dates = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ? AND user_id = ?
        ''', (list_name, description, json.dumps(validated_dates), list_id, user_id))
        
        conn.commit()
        conn.close()
        
        return jsonify({
            'success': True,
            'message': 'Custom blackout list updated successfully'
        })
        
    except Exception as e:
        print(f"Error updating custom blackout list: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/custom-blackout-lists/<int:list_id>', methods=['DELETE'])
@guest_mode_required
def delete_custom_blackout_list(list_id):
    """Delete a custom blackout date list."""
    try:
        user_id = get_current_user_id()
        db_manager = DatabaseManager()
        
        conn = db_manager.get_connection()
        cursor = conn.cursor()
        
        # Check if list exists and belongs to user
        cursor.execute('''
            SELECT id FROM custom_blackout_lists
            WHERE id = ? AND user_id = ?
        ''', (list_id, user_id))
        
        if not cursor.fetchone():
            conn.close()
            return jsonify({'success': False, 'error': 'List not found'}), 404
        
        # Delete list
        cursor.execute('''
            DELETE FROM custom_blackout_lists
            WHERE id = ? AND user_id = ?
        ''', (list_id, user_id))
        
        conn.commit()
        conn.close()
        
        return jsonify({
            'success': True,
            'message': 'Custom blackout list deleted successfully'
        })
        
    except Exception as e:
        print(f"Error deleting custom blackout list: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/blackout-date-lists', methods=['GET'])
@guest_mode_required
def get_blackout_date_lists():
    """Get available blackout date lists and their dates"""
    try:
        # Define built-in blackout date lists
        blackout_date_lists = {
            'FOMC Announcements': _generate_fomc_dates(),
            'End of Month (EOM)': _generate_eom_dates(),
            'End of Quarter (EOQ)': _generate_eoq_dates(),
            'Short Weeks': _generate_short_weeks_dates()
        }
        
        # Add custom blackout date lists for the current user
        try:
            user_id = get_current_user_id()
            db_manager = DatabaseManager()
            conn = db_manager.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT list_name, dates_json 
                FROM custom_blackout_lists 
                WHERE user_id = ?
            ''', (user_id,))
            
            for row in cursor.fetchall():
                list_name, dates_json = row
                import json
                dates = json.loads(dates_json) if dates_json else []
                blackout_date_lists[list_name] = dates
            
            conn.close()
        except Exception as e:
            print(f"Error loading custom blackout lists: {e}")
            # Continue with built-in lists only
        
        return jsonify({
            'success': True,
            'available_lists': list(blackout_date_lists.keys()),
            'date_lists': {name: dates for name, dates in blackout_date_lists.items()}
        })
        
    except Exception as e:
        print(f"Error in get_blackout_date_lists: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/blackout-dates', methods=['POST'])
@guest_mode_required
def get_blackout_dates_analysis():
    """Get blackout dates analysis for strategies"""
    try:
        data = request.get_json()
        date_list_name = data.get('date_list', 'FOMC Announcements')
        whitelist_mode = data.get('whitelist_mode', False)
        max_days = data.get('max_days', 5)
        
        # Check if portfolio has data
        if not portfolio or not portfolio.strategies:
            return jsonify({'success': False, 'error': 'No portfolio data available'})
        
        # Define built-in blackout date lists
        blackout_date_lists = {
            'FOMC Announcements': _generate_fomc_dates(),
            'End of Month (EOM)': _generate_eom_dates(),
            'End of Quarter (EOQ)': _generate_eoq_dates(),
            'Short Weeks': _generate_short_weeks_dates()
        }
        
        # Add custom blackout date lists for the current user
        try:
            user_id = get_current_user_id()
            db_manager = DatabaseManager()
            conn = db_manager.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT list_name, dates FROM custom_blackout_lists
                WHERE user_id = ?
                ORDER BY list_name
            ''', (user_id,))
            
            for row in cursor.fetchall():
                list_name, dates_json = row
                import json
                dates = json.loads(dates_json) if dates_json else []
                blackout_date_lists[list_name] = dates
            
            conn.close()
        except Exception as e:
            print(f"Error loading custom blackout lists: {e}")
            # Continue with built-in lists only
        
        # Get the selected date list
        if date_list_name not in blackout_date_lists:
            return jsonify({'success': False, 'error': f'Date list "{date_list_name}" not found'})
        
        original_blackout_dates = set(blackout_date_lists[date_list_name])
        
        # If whitelist mode is enabled, create a whitelist by removing blackout dates from full date range
        if whitelist_mode:
            # Find the date range from the original blackout dates
            if original_blackout_dates:
                from datetime import datetime, timedelta
                import calendar
                
                # Get min and max years from the blackout dates
                years = set()
                for date_str in original_blackout_dates:
                    try:
                        year = int(date_str.split('-')[0])
                        years.add(year)
                    except (ValueError, IndexError):
                        continue
                
                if years:
                    min_year = min(years)
                    max_year = max(years)
                    
                    # Generate all dates from Jan 1 of min_year to Dec 31 of max_year
                    all_dates = set()
                    for year in range(min_year, max_year + 1):
                        for month in range(1, 13):
                            days_in_month = calendar.monthrange(year, month)[1]
                            for day in range(1, days_in_month + 1):
                                date_str = f"{year:04d}-{month:02d}-{day:02d}"
                                all_dates.add(date_str)
                    
                    # Remove blackout dates to create whitelist
                    blackout_dates = all_dates - original_blackout_dates
                else:
                    blackout_dates = original_blackout_dates
            else:
                blackout_dates = original_blackout_dates
        else:
            blackout_dates = original_blackout_dates
        
        # Analyze each strategy
        strategy_results = []
        
        for strategy in portfolio.strategies.values():
            # Filter trades that were initiated on blackout dates OR had blackout dates during the trade
            blackout_trades = []
            
            for trade in strategy.trades:
                try:
                    # Parse the date opened
                    if hasattr(trade.date_opened, 'strftime'):
                        trade_date_opened = trade.date_opened.strftime('%Y-%m-%d')
                    else:
                        trade_date_opened = str(trade.date_opened).split(' ')[0]
                    
                    # Parse the date closed (if available)
                    trade_date_closed = None
                    if hasattr(trade, 'date_closed') and trade.date_closed:
                        if hasattr(trade.date_closed, 'strftime'):
                            trade_date_closed = trade.date_closed.strftime('%Y-%m-%d')
                        else:
                            trade_date_closed = str(trade.date_closed).split(' ')[0]
                    
                    # Check if trade was initiated on a blackout date
                    if trade_date_opened in blackout_dates:
                        blackout_trades.append(trade)
                        continue
                    
                    # Check if any blackout date falls between opening and closing dates
                    if trade_date_closed:
                        from datetime import datetime, timedelta
                        try:
                            opened_dt = datetime.strptime(trade_date_opened, '%Y-%m-%d')
                            closed_dt = datetime.strptime(trade_date_closed, '%Y-%m-%d')
                            
                            # Check if any blackout date falls within the trade duration
                            # and is within max_days of the close date
                            for blackout_date in blackout_dates:
                                try:
                                    blackout_dt = datetime.strptime(blackout_date, '%Y-%m-%d')
                                    
                                    # Check if blackout date is within trade duration
                                    if opened_dt <= blackout_dt <= closed_dt:
                                        # Check if blackout date is within max_days of close date
                                        days_between = (closed_dt - blackout_dt).days
                                        if days_between <= max_days:
                                            blackout_trades.append(trade)
                                            break  # Found a match, no need to check other blackout dates
                                except ValueError:
                                    continue  # Skip invalid date formats
                        except ValueError:
                            continue  # Skip trades with invalid date formats
                        
                except Exception as e:
                    continue
            
            if blackout_trades:
                # Calculate statistics for blackout trades
                profitable_trades = [t for t in blackout_trades if t.pnl > 0]
                unprofitable_trades = [t for t in blackout_trades if t.pnl <= 0]
                
                total_pnl = sum(t.pnl for t in blackout_trades)
                profitable_count = len(profitable_trades)
                unprofitable_count = len(unprofitable_trades)
                total_count = len(blackout_trades)
                
                # Calculate P&L per lot
                total_contracts = sum(getattr(trade, 'contracts', 1) for trade in blackout_trades)
                pnl_per_lot = round(total_pnl / total_contracts, 2) if total_contracts > 0 else 0
                
                # Convert trade objects to dictionaries for JSON serialization
                trades_data = []
                for trade in blackout_trades:
                    # Handle date_opened conversion safely
                    date_opened_str = 'N/A'
                    try:
                        if hasattr(trade.date_opened, 'strftime'):
                            date_opened_str = trade.date_opened.strftime('%Y-%m-%d')
                        elif trade.date_opened:
                            # If it's a string, try to extract just the date part
                            date_opened_str = str(trade.date_opened).split(' ')[0]
                    except Exception as e:
                        print(f"DEBUG: Error converting date_opened: {e}")
                        date_opened_str = 'N/A'
                    
                    trade_dict = {
                        'pnl': float(trade.pnl),  # Ensure it's a float
                        'contracts': int(getattr(trade, 'contracts', 1)),  # Ensure it's an int
                        'date_opened': date_opened_str
                    }
                    trades_data.append(trade_dict)
                
                strategy_results.append({
                    'strategy': str(strategy.name),  # Ensure string
                    'total_trades': int(total_count),  # Ensure int
                    'profitable_trades': int(profitable_count),  # Ensure int
                    'unprofitable_trades': int(unprofitable_count),  # Ensure int
                    'total_pnl': float(round(total_pnl, 2)),  # Ensure float
                    'pnl_per_lot': float(pnl_per_lot),  # Ensure float
                    'win_rate': float(round((profitable_count / total_count) * 100, 1)) if total_count > 0 else 0.0,  # Ensure float
                    'trades': trades_data  # Include serialized trade data
                })
        
        # Sort by total P&L descending
        strategy_results.sort(key=lambda x: x['total_pnl'], reverse=True)
        
        # Debug: Check for any non-serializable objects
        print(f"DEBUG: About to return strategy_results with {len(strategy_results)} strategies")
        for i, result in enumerate(strategy_results):
            print(f"DEBUG: Strategy {i}: {result['strategy']}, trades count: {len(result.get('trades', []))}")
            if 'trades' in result:
                for j, trade in enumerate(result['trades']):
                    print(f"DEBUG: Trade {j}: type={type(trade)}, keys={list(trade.keys()) if isinstance(trade, dict) else 'not dict'}")
        
        # Prepare chart data for cumulative P&L over time
        chart_data = []
        if strategy_results:
            # Get all trades from all strategies with closing time info
            all_trades = []
            for result in strategy_results:
                if 'trades' in result:
                    for trade in result['trades']:
                        # Get closing time for sorting within each date
                        closing_time = getattr(trade, 'date_closed', None)
                        if closing_time:
                            # Convert to datetime for proper sorting
                            from datetime import datetime
                            if isinstance(closing_time, str):
                                try:
                                    closing_time = datetime.strptime(closing_time, '%Y-%m-%d %H:%M:%S')
                                except:
                                    closing_time = datetime.strptime(closing_time, '%Y-%m-%d')
                            elif hasattr(closing_time, 'strftime'):
                                closing_time = closing_time
                            else:
                                closing_time = None
                        
                        all_trades.append({
                            'date': trade['date_opened'],
                            'pnl': trade['pnl'],
                            'strategy': result['strategy'],
                            'closing_time': closing_time,
                            'trade_data': trade  # Keep original trade data
                        })
            
            # Sort trades by date, then by closing time within each date
            from datetime import datetime
            all_trades.sort(key=lambda x: (x['date'], x['closing_time'] or datetime.min))
            
            # Calculate cumulative P&L for each individual trade
            cumulative_pnl = 0
            for trade in all_trades:
                cumulative_pnl += trade['pnl']
                chart_data.append({
                    'date': trade['date'],
                    'cumulative_pnl': cumulative_pnl,
                    'strategy': trade['strategy'],
                    'trade_pnl': trade['pnl'],
                    'closing_time': trade['closing_time'],
                    'is_latest_of_day': False  # Will be set below
                })
            
            # Mark the latest trade of each day for line connection
            from collections import defaultdict
            daily_trades = defaultdict(list)
            for i, trade in enumerate(chart_data):
                daily_trades[trade['date']].append(i)
            
            # Mark the last trade of each day (by closing time)
            from datetime import datetime
            for date, trade_indices in daily_trades.items():
                if len(trade_indices) > 1:
                    # Find the trade with the latest closing time
                    latest_trade_idx = max(trade_indices, 
                                         key=lambda i: chart_data[i]['closing_time'] or datetime.min)
                    chart_data[latest_trade_idx]['is_latest_of_day'] = True
                else:
                    # Single trade on this date
                    chart_data[trade_indices[0]]['is_latest_of_day'] = True

        return jsonify({
            'success': True,
            'date_list_name': date_list_name,
            'whitelist_mode': whitelist_mode,
            'available_lists': list(blackout_date_lists.keys()),
            'blackout_dates': sorted(list(blackout_dates)),
            'strategy_results': strategy_results,
            'chart_data': chart_data
        })
        
    except Exception as e:
        print(f"Error in blackout dates analysis: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/price-movement/range')
@guest_mode_required
def get_price_movement_range():
    """Get the min and max price movement percentages across all trades."""
    try:
        from analytics import PriceMovementAnalyzer
        
        if not portfolio.strategies:
            return jsonify({
                'success': False,
                'error': 'No portfolio data found'
            })
        
        analyzer = PriceMovementAnalyzer(portfolio)
        result = analyzer.calculate_price_movement_range()
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error getting price movement range: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/price-movement/analysis', methods=['POST'])
@guest_mode_required
def analyze_price_movement():
    """Analyze strategy performance for trades within the specified price movement range."""
    try:
        from analytics import PriceMovementAnalyzer
        
        data = request.get_json()
        min_movement = float(data.get('min_movement', -5.0))
        max_movement = float(data.get('max_movement', 5.0))
        
        if not portfolio.strategies:
            return jsonify({
                'success': False,
                'error': 'No portfolio data found'
            })
        
        analyzer = PriceMovementAnalyzer(portfolio)
        result = analyzer.analyze_price_movement_performance(min_movement, max_movement)
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error analyzing price movement: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/price-movement/trades', methods=['POST'])
@guest_mode_required
def get_price_movement_trades():
    """Get individual trades within the specified price movement range."""
    try:
        from analytics import PriceMovementAnalyzer
        
        data = request.get_json()
        min_movement = float(data.get('min_movement', -5.0))
        max_movement = float(data.get('max_movement', 5.0))
        
        if not portfolio.strategies:
            return jsonify({
                'success': False,
                'error': 'No portfolio data found'
            })
        
        # Get all trades within the price movement range
        all_trades = []
        for strategy in portfolio.strategies.values():
            for trade in strategy.trades:
                if hasattr(trade, 'opening_price') and hasattr(trade, 'closing_price') and trade.opening_price and trade.closing_price:
                    if trade.opening_price > 0:
                        movement = ((trade.closing_price - trade.opening_price) / trade.opening_price) * 100
                        if min_movement <= movement <= max_movement:
                            # Handle date and time formatting
                            date_opened_str = trade.date_opened.strftime('%Y-%m-%d') if hasattr(trade.date_opened, 'strftime') else str(trade.date_opened).split(' ')[0]
                            date_closed_str = trade.date_closed.strftime('%Y-%m-%d') if hasattr(trade.date_closed, 'strftime') else str(trade.date_closed).split(' ')[0]
                            
                            # Handle time formatting - check if time is embedded in date string
                            time_opened_str = None
                            time_closed_str = None
                            
                            # Check if time is in the date_opened string
                            if hasattr(trade, 'date_opened') and trade.date_opened:
                                date_opened_full = str(trade.date_opened)
                                if ' ' in date_opened_full and len(date_opened_full.split(' ')) > 1:
                                    time_opened_str = date_opened_full.split(' ')[1]
                                elif hasattr(trade, 'time_opened') and trade.time_opened:
                                    time_opened_str = str(trade.time_opened)
                            
                            # Check if time is in the date_closed string
                            if hasattr(trade, 'date_closed') and trade.date_closed:
                                date_closed_full = str(trade.date_closed)
                                if ' ' in date_closed_full and len(date_closed_full.split(' ')) > 1:
                                    time_closed_str = date_closed_full.split(' ')[1]
                                elif hasattr(trade, 'time_closed') and trade.time_closed:
                                    time_closed_str = str(trade.time_closed)
                            
                            all_trades.append({
                                'strategy_name': strategy.name,
                                'date_opened': date_opened_str,
                                'time_opened': time_opened_str,
                                'date_closed': date_closed_str,
                                'time_closed': time_closed_str,
                                'opening_price': float(trade.opening_price),
                                'closing_price': float(trade.closing_price),
                                'price_movement': round(movement, 2),
                                'pnl': float(trade.pnl),
                                'contracts': int(getattr(trade, 'contracts', 1)),
                                'pnl_per_lot': float(trade.pnl / getattr(trade, 'contracts', 1))
                            })
        
        return jsonify({
            'success': True,
            'data': {
                'trades': all_trades,
                'min_movement': min_movement,
                'max_movement': max_movement,
                'total_trades': len(all_trades)
            }
        })
        
    except Exception as e:
        print(f"Error getting price movement trades: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/files/overview')
def get_files_overview():
    """Get overview of all uploaded files with their metrics."""
    try:
        # Get list of all saved files
        files = file_manager.get_file_list()
        
        files_overview = []
        
        for file_info in files:
            filename = file_info['filename']
            file_path = file_manager.get_file_path(filename)
            
            if not os.path.exists(file_path):
                continue
                
            try:
                # Read CSV to detect data type and get date range
                df = pd.read_csv(file_path)
                
                # Detect data type
                if 'Initial Premium' in df.columns:
                    data_type = 'Live Data'
                else:
                    data_type = 'Backtest'
                
                # Get date range
                if 'Date Opened' in df.columns and 'Date Closed' in df.columns:
                    # Convert to datetime and find min/max
                    df['Date Opened'] = pd.to_datetime(df['Date Opened'], errors='coerce')
                    df['Date Closed'] = pd.to_datetime(df['Date Closed'], errors='coerce')
                    
                    # Filter out invalid dates
                    valid_dates = df[df['Date Opened'].notna() & df['Date Closed'].notna()]
                    
                    if not valid_dates.empty:
                        start_date = valid_dates['Date Opened'].min().strftime('%a, %d %b %Y')
                        end_date = valid_dates['Date Closed'].max().strftime('%a, %d %b %Y')
                    else:
                        start_date = 'N/A'
                        end_date = 'N/A'
                else:
                    start_date = 'N/A'
                    end_date = 'N/A'
                
                # Calculate basic metrics without loading full portfolio
                total_trades = int(len(df))
                strategy_count = int(df['Strategy'].nunique()) if 'Strategy' in df.columns else 0
                
                # Calculate initial capital and ending capital
                initial_capital = None
                ending_capital = None
                total_pnl = 0
                
                if data_type == 'Live Data':
                    # For live data, extract from filename if available
                    total_pnl = float(df['P/L'].sum()) if 'P/L' in df.columns else 0.0
                    if '__capital=' in filename:
                        try:
                            capital_match = re.search(r'__capital=(\d+)', filename)
                            if capital_match:
                                initial_capital = float(capital_match.group(1))
                        except:
                            pass
                    
                    if initial_capital is not None:
                        ending_capital = initial_capital + total_pnl
                    else:
                        ending_capital = total_pnl  # Fallback
                else:
                    # For backtest files, calculate like in the portfolio loading
                    if 'Funds at Close' in df.columns and not df.empty:
                        # Ensure DataFrame is sorted by Date Closed and Time Closed for chronological calculations
                        # This matches how the main portfolio sorts trades (by date_closed)
                        if 'Date Closed' in df.columns:
                            if 'Time Closed' in df.columns:
                                df_sorted = df.sort_values(by=['Date Closed', 'Time Closed'], ascending=True)
                            else:
                                df_sorted = df.sort_values(by='Date Closed', ascending=True)
                        elif 'Date Opened' in df.columns:
                            if 'Time Opened' in df.columns:
                                df_sorted = df.sort_values(by=['Date Opened', 'Time Opened'], ascending=True)
                            else:
                                df_sorted = df.sort_values(by='Date Opened', ascending=True)
                        else:
                            df_sorted = df  # Cannot sort, rely on original order (less reliable)
                        
                        # Get the first trade's funds_at_close and subtract its P&L to get initial capital
                        first_trade = df_sorted.iloc[0]
                        last_trade = df_sorted.iloc[-1]
                        
                        if pd.notna(first_trade['Funds at Close']) and pd.notna(first_trade['P/L']):
                            initial_capital = float(first_trade['Funds at Close'] - first_trade['P/L'])
                            
                            # For backtest files, calculate total_pnl as the difference between final and initial Funds at Close
                            if pd.notna(last_trade['Funds at Close']):
                                total_pnl = float(last_trade['Funds at Close'] - initial_capital)
                            else:
                                total_pnl = 0.0
                            
                            ending_capital = float(initial_capital + total_pnl)
                        else:
                            initial_capital = 0
                            ending_capital = total_pnl
                    else:
                        initial_capital = 0
                        ending_capital = total_pnl
                
                # Calculate CAGR
                cagr = 0
                if initial_capital and initial_capital > 0 and ending_capital and ending_capital > 0:
                    try:
                        # Get date range for CAGR calculation
                        if 'Date Closed' in df.columns:
                            df['Date Closed'] = pd.to_datetime(df['Date Closed'], errors='coerce')
                            valid_dates = df[df['Date Closed'].notna()]
                            if not valid_dates.empty:
                                start_date = valid_dates['Date Closed'].min()
                                end_date = valid_dates['Date Closed'].max()
                                years = (end_date - start_date).days / 365.25
                                if years > 0:
                                    cagr = float(((ending_capital / initial_capital) ** (1 / years) - 1) * 100)
                    except:
                        pass
                
                # Calculate max drawdown
                max_drawdown = 0
                max_drawdown_pct = 0
                if 'Funds at Close' in df.columns and not df.empty:
                    try:
                        # Sort by Date Closed and Time Closed to ensure chronological order (matches main portfolio logic)
                        if 'Date Closed' in df.columns:
                            if 'Time Closed' in df.columns:
                                df_sorted = df.sort_values(['Date Closed', 'Time Closed'])
                            else:
                                df_sorted = df.sort_values('Date Closed')
                        elif 'Date Opened' in df.columns:
                            if 'Time Opened' in df.columns:
                                df_sorted = df.sort_values(['Date Opened', 'Time Opened'])
                            else:
                                df_sorted = df.sort_values('Date Opened')
                        else:
                            df_sorted = df
                        
                        funds_series = df_sorted['Funds at Close'].dropna()
                        if len(funds_series) > 0:
                            # Calculate running maximum (rolling highest high)
                            running_max = funds_series.expanding().max()
                            # Calculate drawdown
                            drawdown = running_max - funds_series
                            max_drawdown = float(drawdown.max())
                            
                            # Calculate max drawdown percentage using the rolling highest high
                            # Only calculate percentage where running_max > 0 to avoid division by zero
                            valid_indices = running_max > 0
                            if valid_indices.any():
                                drawdown_pct = (drawdown[valid_indices] / running_max[valid_indices]) * 100
                                max_drawdown_pct = float(drawdown_pct.max())
                            else:
                                max_drawdown_pct = 0
                    except Exception as e:
                        print(f"DEBUG {filename}: Error in max drawdown calculation: {e}")
                        import traceback
                        traceback.print_exc()
                        pass
                
                files_overview.append({
                    'filename': filename,
                    'friendly_name': file_info['friendly_name'],
                    'upload_date': file_info['upload_date'],
                    'data_type': data_type,
                    'start_date': start_date,
                    'end_date': end_date,
                    'total_trades': total_trades,
                    'total_pnl': round(total_pnl, 2),
                    'strategy_count': strategy_count,
                    'initial_capital': round(initial_capital, 2) if initial_capital is not None else None,
                    'ending_capital': round(ending_capital, 2) if ending_capital is not None else None,
                    'cagr': round(cagr, 2),
                    'max_drawdown': round(max_drawdown, 2),
                    'max_drawdown_pct': round(max_drawdown_pct, 2),
                    'file_size': file_info.get('file_size', 0)
                })
                
            except Exception as e:
                # If we can't process this file, add it with minimal info
                files_overview.append({
                    'filename': filename,
                    'friendly_name': file_info['friendly_name'],
                    'upload_date': file_info['upload_date'],
                    'data_type': 'Unknown',
                    'start_date': 'N/A',
                    'end_date': 'N/A',
                    'total_trades': 0,
                    'total_pnl': 0,
                    'strategy_count': 0,
                    'initial_capital': None,
                    'ending_capital': 0,
                    'cagr': 0,
                    'max_drawdown': 0,
                    'max_drawdown_pct': 0,
                    'file_size': file_info.get('file_size', 0),
                    'error': str(e)
                })
        
        return jsonify({
            'success': True,
            'data': files_overview
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/meic/stopout-statistics')
@guest_mode_required
def get_meic_stopout_statistics():
    """Get MEIC stopout statistics."""
    try:
        if not portfolio.strategies:
            return jsonify({
                'success': False,
                'error': 'No portfolio data available'
            })
        
        # Get the current filename from session or use None
        current_filename = session.get('current_filename')
        
        from analytics import MEICAnalyzer
        analyzer = MEICAnalyzer(portfolio, current_filename)
        result = analyzer.get_stopout_statistics()
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/meic/time-heatmap')
@guest_mode_required
def get_meic_time_heatmap():
    """Get MEIC time-based heatmap data with optional date filtering."""
    try:
        if not portfolio.strategies:
            return jsonify({
                'success': False,
                'error': 'No portfolio data available'
            })
        
        # Get date filter parameters from query string
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        # Get the current filename from session or use None
        current_filename = session.get('current_filename')
        
        from analytics import MEICAnalyzer
        analyzer = MEICAnalyzer(portfolio, current_filename)
        result = analyzer.get_time_heatmap_data(start_date=start_date, end_date=end_date)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# WSGI entry point for Azure App Service
application = app

if __name__ == '__main__':
    # For development, you can generate a self-signed certificate
    # or run without SSL (recommended for local development)
    
    # Run the app in development mode
    import os
    port = int(os.environ.get('PORT', 8000))
    app.run(debug=False, host='0.0.0.0', port=port)
    
    # If you want to enable HTTPS, uncomment the lines below:
    # from werkzeug.serving import make_ssl_devcert
    # import os
    # 
    # # Generate self-signed certificate (only needed once)
    # if not os.path.exists('ssl_cert.pem'):
    #     make_ssl_devcert('ssl_cert', host='localhost')
    # 
    # app.run(debug=True, host='0.0.0.0', port=5000, 
    #         ssl_context=('ssl_cert.pem', 'ssl_cert.key')) 