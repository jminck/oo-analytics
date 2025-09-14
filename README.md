# Options Trading Analytics Platform

A Flask-based web application for analyzing options trading strategies with comprehensive commission tracking, performance metrics, and interactive visualizations. Designed for traders managing multiple option strategies across different market conditions.

## Core Functionality

### Data Management
- **CSV Upload**: Import trade data from CSV files with automatic format detection
- **Dual Data Types**: Supports both backtest data and live trading data
- **Multi-User Support**: OAuth authentication with user-specific data isolation
- **File Management**: Upload, store, and manage multiple CSV files with metadata tracking
- **Data Transformation**: Automatic conversion of real trade logs to backtest format

### Trade Analysis
- **Strategy Performance**: P&L analysis across multiple trading strategies
- **Commission Analysis**: Detailed breakdown of trading costs including:
  - Opening/closing commissions (actual from CSV or estimated)
  - Exercise fees for expired ITM options
  - Contract count validation from legs data
- **Margin Analysis**: Position sizing and margin requirement tracking
- **Risk Metrics**: Win rates, drawdowns, Sharpe ratios (annualized), efficiency calculations, and performance consistency

### Interactive Charts
- **Cumulative P&L**: Portfolio performance over time
- **Strategy Comparison**: Side-by-side strategy performance analysis
- **Monthly Breakdown**: Time-series analysis with strategy attribution
- **Daily Margin Analysis**: Margin utilization tracking
- **Commission Breakdown**: Visual analysis of trading costs

### Monte Carlo Simulation
- **Portfolio Simulation**: Risk modeling for entire portfolio
- **Strategy-Specific Analysis**: Individual strategy performance projections
- **Confidence Intervals**: Statistical modeling of potential outcomes

## Technical Architecture

### Backend Components
- **Flask Web Framework**: RESTful API with route-based architecture
- **SQLite Database**: Local data storage with user isolation
- **Pandas Data Processing**: Efficient data manipulation and analysis
- **Plotly Visualizations**: Interactive chart generation

### Key Modules
- `app.py` - Main Flask application with 32 API endpoints
- `models.py` - Data models for Trade, Strategy, and Portfolio entities
- `analytics.py` - Strategy analysis algorithms and portfolio metrics
- `charts.py` - Chart generation with ChartFactory pattern
- `commission_config.py` - Commission calculation engine with configurable rates
- `auth.py` - OAuth authentication and user management
- `file_manager.py` - File upload and metadata management

### Data Models

#### Trade Object
```python
- date_opened, date_closed: Trade execution dates
- strategy: Strategy classification
- pnl: Profit/loss amount (absolute)
- pnl_per_lot: Per-contract profit/loss
- contracts: Number of contracts (calculated from legs)
- opening_commissions, closing_commissions: Trading costs
- legs: Options leg details for multi-leg strategies
- reason_for_close: Trade exit reason (expired, closed, stop loss, etc.)
- margin_req: Margin requirements
- funds_at_close: Account balance at trade close
```

#### Strategy Object
```python
- name: Strategy identifier
- trade_type: BULLISH, BEARISH, or NEUTRAL
- trades: Collection of Trade objects
- performance_metrics: Calculated statistics including:
  - win_rate: Percentage of winning trades
  - avg_win, avg_loss: Average winning/losing trade amounts
  - max_win, max_loss: Maximum winning/losing trade amounts
  - expectancy: Expected value per trade
  - sharpe_ratio: Annualized risk-adjusted return
  - efficiency: (Expectancy / Margin) * Win Rate
```

## API Endpoints

### Core Data Endpoints
- `GET /` - Main dashboard interface
- `POST /api/upload` - CSV file upload
- `GET /api/saved-files` - List uploaded files
- `GET /api/load-file/<filename>` - Load specific file
- `DELETE /api/delete-file` - Remove file

### Analytics Endpoints
- `GET /api/portfolio/overview` - Portfolio summary statistics
- `GET /api/portfolio/strategies` - Strategy performance metrics
- `GET /api/strategy-details` - Detailed trade listings with pagination
- `GET /api/commission-analysis` - Commission breakdown analysis
- `GET /api/pnl-by-month` - Monthly P&L analysis
- `GET /api/pnl-by-day-of-week` - Day-of-week performance patterns

### Chart Endpoints
- `GET /api/charts/<chart_type>` - Generate specific chart type
- `GET /api/charts/available` - List available chart types
- `POST /api/monte-carlo/portfolio` - Portfolio Monte Carlo simulation
- `POST /api/monte-carlo/strategy/<name>` - Strategy-specific simulation

### Configuration Endpoints
- `GET|POST /api/commission-config` - Commission rate configuration
- `POST /api/update-friendly-name` - File metadata management

## Commission Analysis Features

### Automated Commission Calculation
- **Live Data**: Estimates commissions using configurable rates (SPX: $1.78/$0.78, QQQ: $1.25/$0.25)
- **Backtest Data**: Uses actual commission amounts from CSV
- **Exercise Cost Tracking**: $9.00 per contract for expired ITM options

### Contract Count Validation
- **Legs Parsing**: Extracts contract quantities from options legs notation
- **Example**: `"44 Sep 22 6370 C STO 110.95 | 22 Sep 22 6300 C BTO 159.35"` = 66 total contracts
- **ITM Detection**: Identifies in-the-money options for exercise fee calculation

### Commission Breakdown Views
- **By Type**: Opening, closing, and exercise costs separately
- **By Strategy**: Commission analysis per trading strategy
- **Contract Tracking**: Separate counts for executed vs. expired contracts

## Data Format Requirements

### Required CSV Columns
```
Date Opened, Date Closed, Strategy, P/L, Funds at Close
```

### Optional Columns (Enhanced Analysis)
```
No. of Contracts, Opening Commissions + Fees, Closing Commissions + Fees,
Legs, Reason For Close, Margin Req., Opening Price, Closing Price
```

### Legs Format (Multi-leg Options)
```
"44 Sep 22 6370 C STO 110.95 | 22 Sep 22 6300 C BTO 159.35"
Format: {quantity} {month} {day} {year} {strike} {C/P} {STO/BTO} {price}
```

### Platform Compatibility

#### OptionOmega.com Integration
This application is fully compatible with CSV exports from **OptionOmega.com** backtesting and trading platform:

- **Live Trading Data**: Direct import of OptionOmega live trading CSV exports
- **Backtest Results**: Import OptionOmega backtest CSV files with strategy performance data
- **Automatic Detection**: Recognizes OptionOmega format based on `Initial Premium` column presence
- **Multi-leg Support**: Parses OptionOmega's legs notation for complex option strategies
- **Commission Analysis**: Calculates accurate commission costs for OptionOmega strategies

**OptionOmega CSV Features Supported:**
- Strategy classification and performance tracking
- Multi-leg options positions (butterflies, iron condors, strangles, etc.)
- Exercise cost calculation for expired ITM options
- Opening/closing price tracking for ITM analysis
- Margin requirement analysis

## Installation and Setup

### Prerequisites
```bash
Python 3.8+
pip install -r requirements.txt
```

### Dependencies
```
Flask==2.3.3, Flask-Login==0.6.3, Flask-SQLAlchemy==3.0.5
Authlib==1.2.1, pandas==2.0.3, numpy==1.24.3, plotly==5.17.0
```

### Local Development
```bash
python app.py
# Access at http://localhost:5000
```

### Environment Configuration
```bash
# Required for OAuth (optional - guest mode available)
GOOGLE_CLIENT_ID=your_client_id
GOOGLE_CLIENT_SECRET=your_client_secret
SECRET_KEY=your_secret_key
```

## File Structure
```
oo-analytics/
├── app.py                    # Main Flask application (2,941 lines)
├── models.py                 # Data models and database management
├── analytics.py              # Strategy analysis algorithms
├── charts.py                 # Chart generation (1,605 lines)
├── commission_config.py      # Commission calculation engine
├── auth.py                   # Authentication system
├── file_manager.py           # File upload management
├── config.py                 # Application configuration
├── templates/
│   └── dashboard.html        # Frontend interface (5,942 lines)
├── data/
│   ├── guest/               # Guest user data
│   └── users/               # Authenticated user data
└── instance/
    └── portfolio_auth.db    # User authentication database
```

## Performance Characteristics
- **Data Capacity**: Handles 25,000+ trades efficiently
- **Response Time**: Sub-second chart generation
- **Storage**: SQLite with file-based CSV storage
- **Scalability**: Single-user focused, multi-user via authentication

## Authentication System
- **OAuth Integration**: Google OAuth for user management
- **Guest Mode**: Anonymous usage without authentication
- **Data Isolation**: User-specific data directories
- **Session Management**: Flask-Login integration

## Chart Types Available
1. **cumulative_pnl** - Portfolio performance timeline
2. **strategy_pnl** - Strategy comparison bar chart
3. **monthly_stacked** - Monthly P&L by strategy
4. **daily_margin** - Margin utilization analysis
5. **commission_breakdown** - Trading cost visualization

## Development Notes
- **Modular Design**: ChartFactory pattern for extensible visualizations
- **Interactive Features**: Custom tooltips with detailed metric explanations
- **Advanced Analytics**: Annualized Sharpe ratios, efficiency calculations, correlation analysis
- **Error Handling**: Comprehensive error logging and user feedback
- **Data Validation**: CSV format validation with helpful error messages
- **Responsive UI**: Bootstrap-based interface with mobile support
- **Performance Optimization**: Efficient data processing for 25,000+ trades
- **Real-time Updates**: Dynamic chart rendering and data refresh capabilities

## Standalone Analysis Scripts

### Trading Analytics Heatmap Generator (`scripts/heatmap.py`)
A comprehensive standalone script for analyzing options trading data:

**Features:**
- **Automated Data Processing**: Finds and combines multiple CSV files matching `*mei*.csv` pattern
- **Time-based Analysis**: Heatmaps showing P&L by time interval and day of week
- **Stop Loss Analysis**: Detailed analysis of stop loss patterns by time, day, and option side
- **Performance Visualizations**: 10+ charts including trade counts, P&L distributions, and reason-for-close analysis
- **Output Management**: Creates timestamped folders in `scripts/logs/` with all charts and data
- **Reproducibility**: Copies original data and script to output folder

**Generated Charts:**
- Average P&L heatmap by time and day of week
- Day summary with trade counts
- Trade count analysis by time interval
- Total P&L by time interval
- Stop loss frequency heatmaps
- Put vs Call stop loss distribution
- Reason for close distribution
- Stop loss frequency reports with detailed statistics

**Usage:**
```bash
cd scripts/
python heatmap.py
# Results saved to: scripts/logs/[filename]_[timestamp]/
```

**Requirements:**
- pandas, matplotlib, seaborn, numpy
- CSV files with columns: Date Opened, Time Opened, P/L, Reason For Close, Legs

## Production Deployment
- **Gunicorn**: WSGI server configuration included (`gunicorn.conf.py`)
- **Azure Deployment**: Configuration documentation provided (`AZURE_DEPLOYMENT.md`)
- **Startup Script**: Automated deployment script (`startup.sh`)
- **File Limits**: 50MB maximum upload size
- **Security**: CSRF protection and secure session management
- **Health Monitoring**: Built-in health check endpoint (`/health`)