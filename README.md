# 📊 Simple Portfolio Strategy Analytics

A streamlined portfolio analytics platform optimized for strategy comparison and diversification analysis. **90% less complex** than enterprise solutions while maintaining powerful features.

## 🎯 What This Does

- **CSV Upload**: Drag & drop trade data (up to 25k trades, 30 strategies)
- **Strategy Analytics**: Compare performance across bullish/bearish/neutral strategies
- **Interactive Charts**: 7 types of interactive Plotly visualizations
- **Portfolio Balance**: Analyze diversification and correlations
- **Position Sizing**: AI-powered allocation recommendations
- **Risk Analysis**: Volatility, drawdown, and Sharpe ratios

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application:**
   ```bash
   python app.py
   ```

3. **Open browser:**
   ```
   http://localhost:5000
   ```

4. **Upload CSV** with these columns:
   - `Date Opened` - Trade opening date
   - `Date Closed` - Trade closing date  
   - `Strategy` - Strategy name
   - `P/L` - Profit/Loss amount
   - `Funds at Close` - Portfolio value at close

## 📋 Features

### ✅ Core Analytics
- **Portfolio Overview**: Total P&L, trade count, max drawdown
- **Strategy Comparison**: Performance ranking and metrics
- **Balance Analysis**: Bullish/bearish/neutral breakdown
- **Diversification Score**: Correlation-based portfolio health
- **Position Sizing**: Performance-based allocation suggestions

### 📊 Interactive Charts
1. **Cumulative P&L** - Portfolio performance over time
2. **Strategy P&L Comparison** - Bar chart of strategy performance
3. **Monthly Stacked** - Monthly P&L breakdown by strategy
4. **Correlation Heatmap** - Strategy correlation matrix
5. **Balance Pie** - Portfolio balance by strategy type
6. **Risk vs Return** - Scatter plot with bubble sizing
7. **Win Rate Comparison** - Win rate analysis by strategy

### 🔧 Technical Features
- **SQLite Database** - Efficient local storage
- **Responsive UI** - Works on desktop, tablet, mobile
- **Drag & Drop Upload** - Easy file handling
- **Real-time Updates** - Instant chart updates
- **Data Validation** - CSV format checking
- **Error Handling** - Graceful error messages

## 📁 Project Structure

```
simple-portfolio/
├── app.py                 # Main Flask application
├── models.py              # Data models (Trade, Strategy, Portfolio)
├── analytics.py           # Strategy analysis functions
├── charts.py              # Interactive chart generation
├── requirements.txt       # Dependencies (4 packages)
├── templates/
│   └── dashboard.html     # Single-page dashboard
└── data/
    └── portfolio.db       # SQLite database (auto-created)
```

## 🔄 API Endpoints

- `GET /` - Dashboard interface
- `GET /api/portfolio/overview` - Portfolio metrics
- `GET /api/portfolio/strategies` - Strategy summary
- `GET /api/portfolio/balance-analysis` - Balance breakdown
- `GET /api/portfolio/position-sizing` - Allocation suggestions
- `GET /api/charts/<type>` - Interactive charts
- `POST /api/upload` - CSV file upload
- `DELETE /api/data/clear` - Clear all data

## 📊 Chart Types

| Chart | Purpose | Key Insights |
|-------|---------|--------------|
| **Cumulative P&L** | Portfolio performance timeline | Overall trajectory, drawdown periods |
| **Strategy P&L** | Strategy comparison | Best/worst performers |
| **Monthly Stacked** | Time-based breakdown | Monthly contribution by strategy |
| **Correlation Heatmap** | Strategy relationships | Diversification opportunities |
| **Balance Pie** | Portfolio composition | Bullish/bearish/neutral allocation |
| **Risk vs Return** | Risk-adjusted performance | Optimal risk/return strategies |
| **Win Rate** | Success rate analysis | Consistency metrics |

## 🎯 Strategy Analysis Features

### Portfolio Balance
- **Bullish Strategies**: Market-up focused
- **Bearish Strategies**: Market-down focused  
- **Neutral Strategies**: Market-direction agnostic
- **Diversification Score**: 0-100 scale based on correlations

### Position Sizing Recommendations
- **Performance Score**: Based on win rate and profit factor
- **Risk Adjustment**: Volatility and drawdown analysis
- **Allocation Suggestions**: Increase/decrease/maintain recommendations

### Risk Metrics
- **Max Drawdown**: Largest peak-to-trough decline
- **Volatility**: Standard deviation of returns
- **Sharpe Ratio**: Risk-adjusted return measure
- **Correlation Analysis**: Strategy interdependencies

## 🚀 Deployment Options

### Local Development
```bash
python app.py
# Access at http://localhost:5000
```

### Cloud Deployment (Heroku)
```bash
# Create Procfile
echo "web: python app.py" > Procfile

# Deploy
git init
heroku create your-app-name
git add .
git commit -m "Deploy portfolio analytics"
git push heroku main
```

### Cloud Deployment (Railway/Render)
1. Connect GitHub repository
2. Set start command: `python app.py`
3. Deploy automatically

## 🔧 Adding New Chart Types

The modular design makes it easy to add new visualizations:

1. **Add method to ChartGenerator** in `charts.py`:
   ```python
   def create_new_chart_type(self) -> Dict:
       # Your chart logic here
       return {'chart': fig.to_json(), 'type': 'new_chart'}
   ```

2. **Register in ChartFactory**:
   ```python
   'new_chart': 'Description of new chart'
   ```

3. **Add to chart methods mapping**:
   ```python
   'new_chart': generator.create_new_chart_type
   ```

The frontend will automatically detect and display new chart types.

## 📈 Performance

- **SQLite**: Handles 25k+ trades efficiently
- **Pandas**: Fast data processing and aggregation
- **Plotly**: Smooth interactive charts
- **Responsive**: Sub-second chart generation

## 🔒 Data Privacy

- **Local Storage**: All data stays on your machine
- **No External Dependencies**: Works offline after initial load
- **SQLite Database**: Encrypted at rest (optional)

## 🆚 Comparison: Before vs After

| Metric | Complex Version | Simple Version |
|--------|-----------------|----------------|
| **Files** | 50+ | 6 |
| **Dependencies** | 40+ packages | 4 packages |
| **Deployment Time** | Hours | Minutes |
| **Infrastructure** | 7 services | 1 service |
| **Maintenance** | High | Minimal |
| **Features** | 357 planned | Core working |
| **Cost** | $100+/month | $0-$10/month |

## 🎯 When to Add Complexity

Only add features back when you have:
- **Proven User Demand** - Users actively requesting specific features
- **Clear Business Value** - ROI justification for complexity
- **Maintenance Resources** - Team capacity to support additional infrastructure

## 🤝 Contributing

1. **Add Analytics**: Extend `analytics.py` with new calculation methods
2. **Create Charts**: Add new chart types to `charts.py`
3. **Improve UI**: Enhance `dashboard.html` template
4. **Optimize Performance**: Database indexing, caching strategies

## 📝 License

MIT License - Use freely for personal and commercial projects.

---

**🚀 Start Simple. Scale Smart.**

This application proves that powerful analytics don't require enterprise complexity. Upload your trade data and start analyzing your portfolio strategies in minutes, not months. 