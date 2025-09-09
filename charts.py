"""
Modular chart generation system for strategy-focused portfolio analytics.
Creates interactive Plotly.js charts optimized for strategy comparison and analysis.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Any
from models import Portfolio
from analytics import StrategyAnalyzer, PortfolioMetrics, MonteCarloSimulator

class ChartGenerator:
    """Generates interactive Plotly charts for strategy analysis."""
    
    def __init__(self, portfolio: Portfolio):
        self.portfolio = portfolio
        self.analyzer = StrategyAnalyzer(portfolio)
        self.metrics = PortfolioMetrics(portfolio)
        
        # Color palette for strategies
        self.colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
            '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5'
        ]
    
    def create_cumulative_pnl_chart(self) -> Dict:
        """Create cumulative P&L chart with strategy breakdown option."""
        overview = self.metrics.get_overview_metrics()
        
        if not overview['cumulative_pnl_series']:
            return self._empty_chart("No data available")
        
        # Get all trades chronologically for the main line
        all_trades = []
        for strategy_name, strategy in self.portfolio.strategies.items():
            for trade in strategy.trades:
                all_trades.append({
                    'date': np.array(pd.to_datetime(trade.date_closed).to_pydatetime()).item(),
                    'pnl': trade.pnl,
                    'strategy': strategy_name,
                    'cumulative': 0  # Will calculate below
                })
        
        if not all_trades:
            return self._empty_chart("No trades found")
        
        # Sort by date and calculate cumulative
        df = pd.DataFrame(all_trades).sort_values('date')
        df['cumulative'] = df['pnl'].cumsum()
        
        fig = go.Figure()
        
        # Main cumulative P&L line
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['cumulative'],
            mode='lines',
            name='Total Portfolio',
            line=dict(color='#1f77b4', width=3),
            hovertemplate='<b>Date:</b> %{x}<br><b>Cumulative P&L:</b> $%{y:,.2f}<extra></extra>'
        ))
        
        # Add strategy-specific lines (initially hidden)
        strategy_colors = {}
        for i, (strategy_name, strategy) in enumerate(self.portfolio.strategies.items()):
            color = self.colors[i % len(self.colors)]
            strategy_colors[strategy_name] = color
            
            strategy_trades = df[df['strategy'] == strategy_name].copy()
            if not strategy_trades.empty:
                strategy_trades['strategy_cumulative'] = strategy_trades['pnl'].cumsum()
                
                fig.add_trace(go.Scatter(
                    x=strategy_trades['date'],
                    y=strategy_trades['strategy_cumulative'],
                    mode='lines',
                    name=strategy_name,
                    line=dict(color=color, width=2),
                    visible='legendonly',  # Hidden by default
                    hovertemplate=f'<b>{strategy_name}</b><br><b>Date:</b> %{{x}}<br><b>Cumulative P&L:</b> $%{{y:,.2f}}<extra></extra>'
                ))
        
        fig.update_layout(
            title='Portfolio Cumulative P&L',
            xaxis_title='Date',
            yaxis_title='Cumulative P&L ($)',
            hovermode='x unified',
            template='plotly_white',
            height=750,
            autosize=True,
            legend=dict(
                orientation='h',
                y=-0.2,
                x=0.5,
                xanchor='center'
            ),
            updatemenus=[{
                'buttons': [
                    {
                        'args': [{'yaxis': {'type': 'linear', 'title': 'Cumulative P&L ($)'}}],
                        'label': 'Linear Scale',
                        'method': 'relayout'
                    },
                    {
                        'args': [{'yaxis': {'type': 'log', 'title': 'Cumulative P&L ($) - Log Scale'}}],
                        'label': 'Log Scale',
                        'method': 'relayout'
                    }
                ],
                'direction': 'down',
                'showactive': True,
                'x': 0.1,
                'xanchor': 'left',
                'y': 1.02,
                'yanchor': 'top'
            }]
        )
        
        return {
            'chart': fig.to_json(),
            'type': 'cumulative_pnl',
            'description': 'Cumulative P&L over time. Click strategy names in legend to show/hide individual strategy performance.'
        }
    
    def create_strategy_pnl_comparison(self) -> Dict:
        """Create bar chart comparing P&L by strategy."""
        strategy_summary = self.portfolio.get_strategy_summary()
        
        if not strategy_summary:
            return self._empty_chart("No strategy data available")
        
        # Sort strategies by total P&L
        strategy_summary.sort(key=lambda x: x['total_pnl'], reverse=True)
        
        # Separate positive and negative P&L for different colors
        strategies = [s['name'] for s in strategy_summary]
        pnls = [s['total_pnl'] for s in strategy_summary]
        colors = ['green' if pnl >= 0 else 'red' for pnl in pnls]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=strategies,
            y=pnls,
            marker_color=colors,
            text=[f'${pnl:,.0f}' for pnl in pnls],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>P&L: $%{y:,.2f}<br>Trades: %{customdata}<extra></extra>',
            customdata=[s['trade_count'] for s in strategy_summary]
        ))
        
        fig.update_layout(
            title='P&L by Strategy',
            xaxis_title='Strategy',
            yaxis_title='Total P&L ($)',
            template='plotly_white',
            height=750,
            autosize=True,
            xaxis={'tickangle': 45},
            margin={'b': 150}  # Add bottom margin for rotated labels
        )
        
        return {
            'chart': fig.to_json(),
            'type': 'strategy_pnl',
            'description': 'Total P&L comparison across all strategies. Green bars are profitable, red bars are losses.'
        }
    
    def create_monthly_pnl_stacked(self) -> Dict:
        """Create stacked bar chart of monthly P&L by strategy."""
        monthly_data = self.portfolio.get_monthly_pnl_by_strategy()
        
        if monthly_data.empty:
            return self._empty_chart("No monthly data available")
        
        # Convert period index to string for plotting
        monthly_data.index = monthly_data.index.astype(str)
        
        fig = go.Figure()
        
        # Calculate monthly totals for hover
        monthly_totals = monthly_data.sum(axis=1)
        
        # Add a bar for each strategy
        for i, strategy in enumerate(monthly_data.columns):
            color = self.colors[i % len(self.colors)]
            
            # Create customdata with monthly totals
            customdata = [monthly_totals[month] for month in monthly_data.index]
            
            fig.add_trace(go.Bar(
                x=monthly_data.index,
                y=monthly_data[strategy],
                name=strategy,
                marker_color=color,
                customdata=customdata,
                hovertemplate=f'<b>{strategy}</b><br>Month: %{{x}}<br>Strategy P&L: $%{{y:,.2f}}<br>Total Month P&L: $%{{customdata:,.2f}}<extra></extra>'
            ))
        
        fig.update_layout(
            title='Monthly P&L by Strategy (Stacked)',
            xaxis_title='Month',
            yaxis_title='P&L ($)',
            barmode='stack',
            template='plotly_white',
            height=750,
            autosize=True,
            xaxis={'tickangle': 45},
            legend=dict(
                orientation='h',
                y=-0.2,
                x=0.5,
                xanchor='center'
            )
        )
        
        return {
            'chart': fig.to_json(),
            'type': 'monthly_stacked',
            'description': 'Monthly P&L breakdown stacked by strategy. Shows contribution of each strategy to monthly performance.'
        }
    
    def create_strategy_correlation_heatmap(self) -> Dict:
        """Create correlation heatmap between strategies."""
        correlations = self.analyzer.calculate_strategy_correlations()
        
        if correlations.empty:
            return self._empty_chart("Insufficient data for correlation analysis")
        
        # Calculate dynamic size based on number of strategies
        num_strategies = len(correlations.columns)
        min_size = 1200  # Increased minimum size
        size_per_strategy = 60  # Much larger size per strategy to accommodate text
        matrix_size = max(min_size, num_strategies * size_per_strategy)
        
        # Add extra width for the colorbar/legend
        total_width = matrix_size + 200  # Add 200px for colorbar and spacing
        
        print(f"DEBUG: Correlation matrix - {num_strategies} strategies, matrix size: {matrix_size}x{matrix_size}px, total width: {total_width}px")
        
        fig = go.Figure(data=go.Heatmap(
            z=correlations.values,
            x=correlations.columns,
            y=correlations.index,
            colorscale='RdBu',
            zmid=0,
            text=correlations.round(2).values,
            texttemplate='%{text}',
            textfont={'size': 10},
            hovertemplate='<b>%{y}</b><br>vs<br><b>%{x}</b><br><br>Correlation: %{z:.3f}<extra></extra>',
            customdata=np.abs(1 - correlations.values),
            colorbar={
                'lenmode': 'pixels',
                'len': 100,
                'thickness': 30,
                'y': 0.5,
                'yanchor': 'middle'
            }
        ))
        
        fig.update_layout(
            title='Strategy Correlation Matrix',
            template='plotly_white',
            height=matrix_size,
            width=total_width,
            hovermode='closest',
            hoverlabel=dict(
                align='left',
                font=dict(size=11)
            ),
            xaxis=dict(
                scaleanchor="y", 
                scaleratio=1,
                tickangle=45,
                tickfont=dict(size=max(10, 14 - num_strategies // 25)),  # Smaller font to fit more strategies
                side='bottom',
                automargin=True  # Enable auto-margin for x-axis labels
            ),
            yaxis=dict(
                scaleanchor="x", 
                scaleratio=1,
                tickfont=dict(size=max(10, 14 - num_strategies // 25)),  # Smaller font to fit more strategies
                automargin=True,
                showticklabels=False  # Hide y-axis tick labels (strategy names)
            ),
            margin=dict(l=300, r=120, t=180, b=300)  # Even larger margins for better label visibility
        )
        
        return {
            'chart': fig.to_json(),
            'type': 'correlation_heatmap',
            'description': 'Correlation matrix between strategy daily returns. Blue = negative correlation, Red = positive correlation.'
        }
    
    def create_strategy_balance_pie(self) -> Dict:
        """Create pie chart showing portfolio balance by strategy type."""
        balance = self.analyzer.get_strategy_balance_analysis()
        
        if not balance:
            return self._empty_chart("No balance data available")
        
        # Prepare data for pie chart
        types = []
        values = []
        colors_pie = []
        
        color_map = {
            'BULLISH': '#2ca02c',  # Green
            'BEARISH': '#d62728',  # Red
            'NEUTRAL': '#ff7f0e'   # Orange
        }
        
        for strategy_type, data in balance.items():
            if data['strategy_count'] > 0:  # Only include types with strategies
                types.append(f"{strategy_type}<br>({data['strategy_count']} strategies)")
                values.append(abs(data['pnl']))  # Use absolute value for pie chart
                colors_pie.append(color_map.get(strategy_type, '#1f77b4'))
        
        if not values:
            return self._empty_chart("No strategy data available")
        
        fig = go.Figure(data=[go.Pie(
            labels=types,
            values=values,
            marker_colors=colors_pie,
            hovertemplate='<b>%{label}</b><br>P&L: $%{value:,.2f}<br>Percentage: %{percent}<extra></extra>'
        )])
        
        fig.update_layout(
            title='Portfolio Balance by Strategy Type',
            template='plotly_white',
            height=650,
            autosize=True
        )
        
        return {
            'chart': fig.to_json(),
            'type': 'balance_pie',
            'description': 'Distribution of portfolio P&L across bullish, bearish, and neutral strategies.'
        }
    
    def create_risk_return_scatter(self) -> Dict:
        """Create risk-return scatter plot for strategies."""
        risk_analysis = self.analyzer.get_risk_analysis()
        strategy_summary = self.portfolio.get_strategy_summary()
        
        if not risk_analysis or not strategy_summary:
            return self._empty_chart("Insufficient data for risk analysis")
        
        # Combine risk and return data
        scatter_data = []
        for strategy_data in strategy_summary:
            strategy_name = strategy_data['name']
            risk_data = risk_analysis.get(strategy_name, {})
            
            if risk_data:
                scatter_data.append({
                    'strategy': strategy_name,
                    'return': strategy_data['total_pnl'],
                    'risk': risk_data['volatility'],
                    'sharpe': risk_data['sharpe_ratio'],
                    'trades': strategy_data['trade_count']
                })
        
        if not scatter_data:
            return self._empty_chart("No risk-return data available")
        
        df_scatter = pd.DataFrame(scatter_data)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df_scatter['risk'],
            y=df_scatter['return'],
            mode='markers',
            marker=dict(
                size=[min(trade_count * 2 + 3, 40) for trade_count in df_scatter['trades']],  # Size based on trade count with min 3, max 40
                color=df_scatter['sharpe'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Sharpe Ratio'),
                line=dict(width=1, color='white')  # Add white border for better visibility
            ),
            hovertemplate='<b>%{customdata[0]}</b><br>Risk (P&L Std Dev): $%{x:,.2f}<br>Return: $%{y:,.2f}<br>Trades: %{customdata[1]}<br>Sharpe Ratio: %{customdata[2]:.3f}<extra></extra>',
            customdata=list(zip(df_scatter['strategy'], df_scatter['trades'], df_scatter['sharpe']))
        ))
        
        fig.update_layout(
            title='Risk-Return Analysis by Strategy',
            xaxis_title='Risk (P&L Standard Deviation)',
            yaxis_title='Total Return ($)',
            template='plotly_white',
            height=750,
            hovermode='closest',
            annotations=[
                dict(
                    x=0.02, y=0.98,
                    xref='paper', yref='paper',
                    text='💡 Hover over points to see strategy details<br>💡 Top-left = Low risk, High return (Ideal)',
                    showarrow=False,
                    bgcolor='rgba(255,255,255,0.8)',
                    bordercolor='rgba(0,0,0,0.3)',
                    borderwidth=1,
                    font=dict(size=12)
                )
            ]
        )
        
        return {
            'chart': fig.to_json(),
            'type': 'risk_return',
            'description': 'Risk vs Return scatter plot. Bubble size = trade count, color = Sharpe ratio. Top-left quadrant is ideal (low risk, high return).'
        }
    
    def create_win_rate_comparison(self) -> Dict:
        """Create win rate comparison chart by strategy."""
        win_rates = self.analyzer.get_strategy_win_rates()
        
        if not win_rates:
            return self._empty_chart("No strategy data available for win rate analysis")
        
        strategies = list(win_rates.keys())
        rates = list(win_rates.values())
        
        fig = go.Figure(data=go.Bar(
            x=strategies,
            y=rates,
            marker_color='lightblue',
            text=[f'{rate:.1f}%' for rate in rates],
            textposition='auto'
        ))
        
        fig.update_layout(
            title='Win Rate by Strategy',
            xaxis_title='Strategy',
            yaxis_title='Win Rate (%)',
            template='plotly_white',
            height=650,
            autosize=True,
            xaxis={'tickangle': 45}
        )
        
        return {
            'chart': fig.to_json(),
            'type': 'win_rate',
            'description': 'Win rate comparison across different strategies.'
        }

    def create_win_rates_table(self) -> Dict:
        """Create a table showing detailed win rate statistics by strategy."""
        strategies = self.portfolio.strategies.values()  # Get Strategy objects, not names
        
        if not strategies:
            return self._empty_chart("No strategy data available for win rate analysis")
        
        # Prepare table data
        table_data = []
        for strategy in strategies:
            stats = strategy.get_summary_stats()
            
            # Format display values - reordered according to user specification
            display_values = [
                strategy.name,
                f"${stats['total_pnl']:,.2f}",
                str(stats['trade_count']),
                str(stats['wins_count']),
                str(stats['losses_count']),
                str(stats['max_win_streak']),
                str(stats['max_loss_streak']),
                f"{stats['win_rate']:.1f}%",
                f"${stats['avg_win_per_lot']:,.2f}",
                f"${stats['avg_loss_per_lot']:,.2f}",
                f"${stats['max_win_per_lot']:,.2f}",
                f"${stats['max_loss_per_lot']:,.2f}",
                f"${stats['expectancy_per_lot']:,.2f}"
            ]
            
            table_data.append(display_values)
        
        # Create table
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=['Strategy', 'Total P&L', 'Trades', 'Wins', 'Losses', 'Max Win Streak', 'Max Lose Streak', 'Win Rate', 'Avg Winner', 'Avg Loser', 'Max Winner', 'Max Loser', 'Expectancy/Lot'],
                fill_color=['#2c3e50', '#2c3e50', '#2c3e50', '#2c3e50', '#2c3e50', '#28a745', '#dc3545', '#2c3e50', '#2c3e50', '#2c3e50', '#2c3e50', '#2c3e50', '#2c3e50'],
                align='center',
                font=dict(size=14, color='white', family='Arial Black'),
                height=40
            ),
            cells=dict(
                values=list(zip(*table_data)),
                fill_color=[['#f8f9fa', '#ffffff'] * (len(strategies) // 2 + 1)][:len(strategies)],
                align='center',
                font=dict(size=12, family='Arial'),
                height=35,
                line=dict(color='#dee2e6', width=1)
            )
        )])
        
        fig.update_layout(
            title=dict(
                text='Strategy Win Rates and Statistics',
                font=dict(size=18, family='Arial Black'),
                x=0.5,
                xanchor='center'
            ),
            template='plotly_white',
            height=500 + (len(strategies) * 35),  # Dynamic height based on number of strategies
            margin=dict(l=20, r=20, t=80, b=20)
        )
        
        return {
            'chart': fig.to_json(),
            'type': 'win_rates_table',
            'description': 'Detailed win rate statistics and performance metrics by strategy.',
            'raw_data': table_data  # Include raw data for sorting
        }

    def create_daily_pnl_chart(self) -> Dict:
        """Create daily P&L distribution chart."""
        all_trades = []
        for strategy in self.portfolio.strategies.values():
            all_trades.extend(strategy.trades)
        
        if not all_trades:
            return self._empty_chart("No trade data available for daily P&L analysis")
        
        # Sort trades by date to calculate account balance progression
        sorted_trades = sorted(all_trades, key=lambda t: pd.to_datetime(t.date_closed))
        
        # Calculate initial balance from first trade
        first_trade = sorted_trades[0]
        initial_balance = first_trade.funds_at_close - first_trade.pnl
        
        # Group trades by date and calculate daily P&L and account balance
        daily_pnl = {}
        daily_balance = {}
        running_pnl = 0
        
        for trade in sorted_trades:
            date = trade.date_closed
            if date:
                running_pnl += trade.pnl
                current_balance = initial_balance + running_pnl
                
                if date not in daily_pnl:
                    daily_pnl[date] = 0
                    daily_balance[date] = current_balance
                
                daily_pnl[date] += trade.pnl
                daily_balance[date] = current_balance  # Update to final balance for the day
        
        if not daily_pnl:
            return self._empty_chart("No valid trade dates for daily P&L analysis")
        
        dates = sorted(daily_pnl.keys())
        pnl_values = [daily_pnl[date] for date in dates]
        balance_values = [daily_balance[date] for date in dates]
        
        # Calculate percentage values
        pnl_percentages = []
        for i, pnl in enumerate(pnl_values):
            balance = balance_values[i]
            if balance > 0:
                percentage = (pnl / balance) * 100
            else:
                percentage = 0
            pnl_percentages.append(percentage)
        
        # Create figure with dual y-axes
        fig = go.Figure()
        
        # Add P&L dollars as primary y-axis (left)
        fig.add_trace(go.Scatter(
            x=dates,
            y=pnl_values,
            mode='lines+markers',
            name='Daily P&L ($)',
            line=dict(color='blue', width=2),
            marker=dict(size=6),
            yaxis='y',
            hovertemplate='<b>%{x}</b><br>Daily P&L: $%{y:,.2f}<br>Percentage: %{customdata:.2f}%<extra></extra>',
            customdata=pnl_percentages
        ))
        
        # Add P&L percentage as secondary y-axis (right)
        fig.add_trace(go.Scatter(
            x=dates,
            y=pnl_percentages,
            mode='lines+markers',
            name='Daily P&L (%)',
            line=dict(color='red', width=2, dash='dash'),
            marker=dict(size=4),
            yaxis='y2',
            hovertemplate='<b>%{x}</b><br>Daily P&L: %{y:.2f}%<br>Dollars: $%{customdata:,.2f}<extra></extra>',
            customdata=pnl_values
        ))
        
        fig.update_layout(
            title='Daily P&L Distribution',
            xaxis_title='Date',
            yaxis=dict(
                title='Daily P&L ($)',
                side='left',
                showgrid=True,
                gridcolor='rgba(0,0,0,0.1)'
            ),
            yaxis2=dict(
                title='Daily P&L (%)',
                side='right',
                overlaying='y',
                showgrid=False
            ),
            template='plotly_white',
            height=650,
            autosize=True,
            xaxis={'tickangle': 45},
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            hovermode='x unified'
        )
        
        return {
            'chart': fig.to_json(),
            'type': 'daily_pnl',
            'description': 'Daily P&L distribution showing portfolio performance over time in both dollars and percentage.'
        }

    def create_drawdown_chart(self) -> Dict:
        """Create portfolio drawdown analysis chart."""
        all_trades = []
        for strategy in self.portfolio.strategies.values():
            all_trades.extend(strategy.trades)
        
        if not all_trades:
            return self._empty_chart("No trade data available for drawdown analysis")
        
        # Sort trades by date (same as Portfolio Overview)
        sorted_trades = sorted(all_trades, key=lambda t: pd.to_datetime(t.date_closed))
        
        if not sorted_trades:
            return self._empty_chart("No valid trade data for drawdown calculation")
        
        # Calculate initial balance from first trade (same as Portfolio Overview)
        first_trade = sorted_trades[0]
        initial_balance = first_trade.funds_at_close - first_trade.pnl
        
        # Calculate account values over time (same as Portfolio Overview)
        cumulative_pnl = []
        running_pnl = 0
        account_values = []
        dates = []
        
        for trade in sorted_trades:
            running_pnl += trade.pnl
            cumulative_pnl.append(running_pnl)
            account_value = initial_balance + running_pnl
            account_values.append(account_value)
            dates.append(trade.date_closed)
        
        # Calculate drawdown using account values (same as Portfolio Overview)
        running_max = np.maximum.accumulate(account_values)
        drawdown_pct = []
        
        for i, account_value in enumerate(account_values):
            peak = running_max[i]
            if peak > 0:
                current_drawdown = (peak - account_value) / peak * 100
            else:
                current_drawdown = 0
            drawdown_pct.append(current_drawdown)
        
        fig = go.Figure()
        
        # Add account values line
        fig.add_trace(go.Scatter(
            x=dates,
            y=account_values,
            mode='lines',
            name='Account Value',
            line=dict(color='blue', width=2),
            yaxis='y'
        ))
        
        # Add drawdown area
        fig.add_trace(go.Scatter(
            x=dates,
            y=drawdown_pct,
            mode='lines',
            name='Drawdown %',
            line=dict(color='red', width=2),
            fill='tonexty',
            fillcolor='rgba(255,0,0,0.1)',
            yaxis='y2'
        ))
        
        fig.update_layout(
            title='Portfolio Drawdown Analysis',
            xaxis_title='Date',
            yaxis=dict(
                title='Account Value ($)',
                side='left'
            ),
            yaxis2=dict(
                title='Drawdown (%)',
                side='right',
                overlaying='y',
                range=[0, max(drawdown_pct) * 1.1]  # Show drawdown from 0 to max
            ),
            template='plotly_white',
            height=650,
            autosize=True,
            xaxis={'tickangle': 45},
            legend=dict(x=0.02, y=0.98)
        )
        
        return {
            'chart': fig.to_json(),
            'type': 'drawdown',
            'description': 'Portfolio drawdown analysis showing peak-to-trough declines.'
        }
    
    def create_monte_carlo_distribution_chart(self, simulation_data: Dict) -> Dict:
        """Create Monte Carlo outcome distribution histogram."""
        if not simulation_data or 'final_balance_stats' not in simulation_data:
            return self._empty_chart("No Monte Carlo simulation data available")
        
        try:
            # Get simulation results
            simulation_results = simulation_data.get('simulation_results', [])
            if not simulation_results:
                return self._empty_chart("No simulation results available")
            
            # Extract final balances
            final_balances = [result['final_balance'] for result in simulation_results]
            
            # Get statistics
            stats = simulation_data['final_balance_stats']
            historical_balance = simulation_data['simulation_summary']['historical_final_balance']
            
            # Create histogram
            fig = go.Figure()
            
            fig.add_trace(go.Histogram(
                x=final_balances,
                nbinsx=50,
                name='Simulated Outcomes',
                marker_color='lightblue',
                opacity=0.7
            ))
            
            # Add vertical lines for key statistics
            fig.add_vline(
                x=stats['mean'],
                line=dict(color='blue', width=2, dash='dash'),
                annotation_text=f"Mean: ${stats['mean']:,.0f}",
                annotation_position="top right"
            )
            
            fig.add_vline(
                x=stats['median'],
                line=dict(color='green', width=2, dash='dash'),
                annotation_text=f"Median: ${stats['median']:,.0f}",
                annotation_position="top left"
            )
            
            fig.add_vline(
                x=historical_balance,
                line=dict(color='red', width=3),
                annotation_text=f"Historical: ${historical_balance:,.0f}",
                annotation_position="bottom right"
            )
            
            # Update layout
            fig.update_layout(
                title={
                    'text': f'Monte Carlo Final Balance Distribution<br><sub>{simulation_data["simulation_summary"]["num_simulations"]:,} simulations</sub>',
                    'x': 0.5,
                    'xanchor': 'center'
                },
                xaxis_title='Final Balance ($)',
                yaxis_title='Frequency',
                height=650,
                autosize=True,
                showlegend=False,
                template='plotly_white'
            )
            
            return {
                'chart': fig.to_json(),
                'type': 'monte_carlo_distribution'
            }
            
        except Exception as e:
            return self._empty_chart(f"Error creating Monte Carlo distribution chart: {str(e)}")
    
    def create_monte_carlo_confidence_intervals_chart(self, simulation_data: Dict) -> Dict:
        """Create Monte Carlo confidence intervals chart."""
        if not simulation_data or 'simulation_results' not in simulation_data:
            return self._empty_chart("No Monte Carlo simulation data available")
        
        try:
            # Get simulation results (first 20 for visualization)
            simulation_results = simulation_data.get('simulation_results', [])[:20]
            if not simulation_results:
                return self._empty_chart("No simulation results available")
            
            # Create figure
            fig = go.Figure()
            
            # Add individual simulation paths (first 20)
            for i, result in enumerate(simulation_results):
                if i < 20:  # Limit to first 20 for clarity
                    fig.add_trace(go.Scatter(
                        x=list(range(1, len(result['account_balance']) + 1)),
                        y=result['account_balance'],
                        mode='lines',
                        line=dict(color='lightgray', width=1),
                        opacity=0.3,
                        showlegend=False,
                        hovertemplate='Simulation %{fullData.name}<br>Trade: %{x}<br>Balance: $%{y:,.0f}<extra></extra>',
                        name=f'Sim {i+1}'
                    ))
            
            # Calculate percentiles for confidence bands
            if len(simulation_results) > 0:
                trade_count = len(simulation_results[0]['account_balance'])
                trades = list(range(1, trade_count + 1))
                
                # Calculate percentiles for each trade
                p5_values = []
                p25_values = []
                p75_values = []
                p95_values = []
                median_values = []
                
                for trade_idx in range(trade_count):
                    # Get a larger sample for better percentile calculation
                    all_results = simulation_data.get('simulation_results', [])
                    if len(all_results) > 100:
                        balances_at_trade = [result['account_balance'][trade_idx] for result in all_results]
                    else:
                        balances_at_trade = [result['account_balance'][trade_idx] for result in simulation_results]
                    
                    p5_values.append(np.percentile(balances_at_trade, 5))
                    p25_values.append(np.percentile(balances_at_trade, 25))
                    p75_values.append(np.percentile(balances_at_trade, 75))
                    p95_values.append(np.percentile(balances_at_trade, 95))
                    median_values.append(np.percentile(balances_at_trade, 50))
                
                # Add confidence bands
                fig.add_trace(go.Scatter(
                    x=trades + trades[::-1],
                    y=p95_values + p5_values[::-1],
                    fill='toself',
                    fillcolor='rgba(135, 206, 235, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='90% Confidence',
                    hoverinfo='skip'
                ))
                
                fig.add_trace(go.Scatter(
                    x=trades + trades[::-1],
                    y=p75_values + p25_values[::-1],
                    fill='toself',
                    fillcolor='rgba(135, 206, 235, 0.4)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='50% Confidence',
                    hoverinfo='skip'
                ))
                
                # Add median line
                fig.add_trace(go.Scatter(
                    x=trades,
                    y=median_values,
                    mode='lines',
                    line=dict(color='blue', width=3),
                    name='Median Path'
                ))
            
            # Update layout
            fig.update_layout(
                title={
                    'text': f'Monte Carlo Simulation Paths<br><sub>Showing {len(simulation_results)} of {simulation_data["simulation_summary"]["num_simulations"]:,} simulations</sub>',
                    'x': 0.5,
                    'xanchor': 'center'
                },
                xaxis_title='Trade Number',
                yaxis_title='Account Balance ($)',
                height=750,
                template='plotly_white'
            )
            
            return {
                'chart': fig.to_json(),
                'type': 'monte_carlo_confidence'
            }
            
        except Exception as e:
            return self._empty_chart(f"Error creating Monte Carlo confidence chart: {str(e)}")
    
    def create_monte_carlo_risk_metrics_chart(self, simulation_data: Dict) -> Dict:
        """Create Monte Carlo risk metrics comparison chart."""
        if not simulation_data:
            return self._empty_chart("No Monte Carlo simulation data available")
        
        try:
            # Get statistics
            balance_stats = simulation_data.get('final_balance_stats', {})
            drawdown_stats = simulation_data.get('drawdown_stats', {})
            probabilities = simulation_data.get('probabilities', {})
            
            if not balance_stats or not drawdown_stats:
                return self._empty_chart("Insufficient simulation data for risk metrics")
            
            # Prepare data for comparison
            metrics = [
                'Win Probability (%)',
                'Loss Probability (%)',
                'Avg Max Drawdown ($)',
                'Worst Case Drawdown ($)',
                'Best Case Balance ($)',
                'Worst Case Balance ($)'
            ]
            
            values = [
                probabilities.get('win_probability', 0),
                probabilities.get('loss_probability', 0),
                abs(drawdown_stats.get('mean', 0)),
                abs(drawdown_stats.get('max', 0)),
                balance_stats.get('max', 0),
                balance_stats.get('min', 0)
            ]
            
            # Create bar chart
            colors = ['green', 'red', 'orange', 'darkred', 'darkgreen', 'darkred']
            
            fig = go.Figure()
            
            for i, (metric, value, color) in enumerate(zip(metrics, values, colors)):
                # Format values appropriately
                if 'Probability' in metric:
                    display_value = f"{value:.1f}%"
                    bar_value = value
                else:
                    display_value = f"${value:,.0f}"
                    bar_value = value / 1000  # Convert to thousands for better scale
                
                fig.add_trace(go.Bar(
                    x=[metric],
                    y=[bar_value],
                    text=[display_value],
                    textposition='auto',
                    marker_color=color,
                    opacity=0.7,
                    showlegend=False
                ))
            
            # Update layout
            fig.update_layout(
                title={
                    'text': 'Monte Carlo Risk Metrics Summary',
                    'x': 0.5,
                    'xanchor': 'center'
                },
                xaxis_title='Risk Metrics',
                yaxis_title='Value ($ in thousands, % as shown)',
                height=650,
                template='plotly_white'
            )
            
            return {
                'chart': fig.to_json(),
                'type': 'monte_carlo_risk_metrics'
            }
            
        except Exception as e:
            return self._empty_chart(f"Error creating Monte Carlo risk metrics chart: {str(e)}")
    
    def create_daily_margin_analysis_chart(self):
        """Create a daily margin analysis showing margin usage each day in dollars and as percentage of equity."""
        try:
            from datetime import datetime, timedelta
            import pandas as pd
            
            # Get all trades chronologically - optimized to avoid repeated date parsing
            all_trades = []
            for strategy in self.portfolio.strategies.values():
                for trade in strategy.trades:
                    if trade.date_closed and trade.funds_at_close is not None:
                        # Pre-parse dates to avoid repeated parsing
                        trade.close_date = pd.to_datetime(trade.date_closed).date()
                        trade.close_datetime = np.array(pd.to_datetime(trade.date_closed).to_pydatetime()).item()
                        all_trades.append(trade)
            
            if not all_trades:
                return self._empty_chart("No trade data available for daily margin analysis.")
            
            # Sort trades by closing date to process them chronologically
            all_trades.sort(key=lambda x: x.close_datetime)
            
            # Create daily data structure - optimized approach
            daily_data = {}
            
            # Pre-calculate margin amounts to avoid repeated calculations
            for trade in all_trades:
                # Calculate margin for this trade once
                margin_amount = 0
                if hasattr(trade, 'margin_req') and trade.margin_req > 0:
                    margin_amount = trade.margin_req
                elif trade.legs and trade.contracts > 0:
                    try:
                        from app import calculate_margin_from_legs
                        margin_info = calculate_margin_from_legs(trade.legs, context={
                            'strategy': trade.strategy,
                            'date_opened': trade.date_opened,
                            'contracts': trade.contracts
                        })
                        margin_amount = margin_info.get('overall_margin', 0)
                    except Exception:
                        margin_amount = 0
                
                trade.margin_amount = margin_amount
            
            # Group trades by closing date for efficient processing
            trades_by_date = {}
            for trade in all_trades:
                close_date = trade.close_date
                if close_date not in trades_by_date:
                    trades_by_date[close_date] = []
                trades_by_date[close_date].append(trade)
            
            # Process each date efficiently
            sorted_dates = sorted(trades_by_date.keys())
            current_balance = None
            
            for date in sorted_dates:
                trades_on_date = trades_by_date[date]
                
                # Find the last trade on this date (already sorted by datetime)
                last_trade = trades_on_date[-1]
                current_balance = last_trade.funds_at_close
                
                # Calculate totals for this date
                total_margin = sum(trade.margin_amount for trade in trades_on_date)
                trades_count = len(trades_on_date)
                strategies = set(trade.strategy for trade in trades_on_date)
                
                daily_data[date] = {
                    'total_margin': total_margin,
                    'account_balance': current_balance,
                    'trades_count': trades_count,
                    'strategies_count': len(strategies)
                }
            
            # Convert to list and sort by date descending - optimized
            daily_list = []
            for date in sorted(daily_data.keys(), reverse=True):  # Sort by date descending
                data = daily_data[date]
                margin_pct = (data['total_margin'] / data['account_balance'] * 100) if data['account_balance'] > 0 else 0
                daily_list.append({
                    'date': date,
                    'margin_dollars': data['total_margin'],
                    'account_balance': data['account_balance'],
                    'margin_percentage': margin_pct,
                    'trades_count': data['trades_count'],
                    'strategies_count': data['strategies_count']
                })
            
            if not daily_list:
                return self._empty_chart("No margin data available for daily analysis.")
            
            # Create the chart data (reverse order for chronological display)
            chart_data = list(reversed(daily_list))  # Reverse for chronological chart display
            dates = [str(item['date']) for item in chart_data]
            margin_dollars = [item['margin_dollars'] for item in chart_data]
            margin_percentages = [item['margin_percentage'] for item in chart_data]
            account_balances = [item['account_balance'] for item in chart_data]
            trades_counts = [item['trades_count'] for item in chart_data]
            strategies_counts = [item['strategies_count'] for item in chart_data]
            
            # Create a figure with subplots
            fig = go.Figure()
            
            # Add margin dollars as bars
            fig.add_trace(go.Bar(
                x=dates,
                y=margin_dollars,
                name='Margin ($)',
                yaxis='y',
                marker_color='rgba(255, 99, 132, 0.7)',
                hovertemplate='<b>%{x}</b><br>' +
                            'Margin: $%{y:,.2f}<br>' +
                            '<extra></extra>'
            ))
            
            # Add margin percentage as line on secondary y-axis
            fig.add_trace(go.Scatter(
                x=dates,
                y=margin_percentages,
                name='Margin (%)',
                yaxis='y2',
                line=dict(color='rgba(54, 162, 235, 1)', width=2),
                hovertemplate='<b>%{x}</b><br>' +
                            'Margin: %{y:.2f}% ($%{customdata:,.2f})<br>' +
                            '<extra></extra>',
                customdata=margin_dollars
            ))
            
            # Update layout
            fig.update_layout(
                title='Daily Margin Usage Analysis',
                xaxis=dict(
                    title='Date',
                    tickangle=45
                ),
                yaxis=dict(
                    title='Margin ($)',
                    side='left',
                    showgrid=True,
                    gridcolor='rgba(0,0,0,0.1)'
                ),
                yaxis2=dict(
                    title='Margin (%)',
                    side='right',
                    overlaying='y',
                    range=[0, max(margin_percentages) * 1.1 if margin_percentages else 100],
                    showgrid=False
                ),
                barmode='group',
                hovermode='x unified',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                height=600
            )
            
            # Create table data for detailed view with pagination support
            table_data = {
                'data': daily_list,  # Full dataset for client-side pagination and sorting
                'total_records': len(daily_list),
                'page_size': 50,  # Default page size
                'columns': [
                    {'key': 'date', 'label': 'Date', 'sortable': True, 'type': 'date'},
                    {'key': 'margin_dollars', 'label': 'Margin ($)', 'sortable': True, 'type': 'currency'},
                    {'key': 'margin_percentage', 'label': 'Margin (%)', 'sortable': True, 'type': 'percentage'},
                    {'key': 'account_balance', 'label': 'Account Balance ($)', 'sortable': True, 'type': 'currency'},
                    {'key': 'trades_count', 'label': 'Trades', 'sortable': True, 'type': 'number'},
                    {'key': 'strategies_count', 'label': 'Strategies', 'sortable': True, 'type': 'number'}
                ]
            }
            
            # Create a simple table figure for now (will be replaced by custom HTML table)
            table_fig = go.Figure(data=[go.Table(
                header=dict(
                    values=['Date', 'Margin ($)', 'Margin (%)', 'Account Balance ($)', 'Trades', 'Strategies'],
                    fill_color='#2E86AB',
                    font=dict(color='white', size=14),
                    align='center'
                ),
                cells=dict(
                    values=[
                        dates,
                        [f"${val:,.2f}" for val in margin_dollars],
                        [f"{val:.2f}%" for val in margin_percentages],
                        [f"${val:,.2f}" for val in account_balances],
                        trades_counts,
                        strategies_counts
                    ],
                    fill_color='#F7F7F7',
                    font=dict(size=12),
                    align='center',
                    height=35
                ),
                columnwidth=[120, 120, 120, 150, 80, 100]
            )])
            
            table_fig.update_layout(
                title='Daily Margin Usage Table',
                height=600
            )
            
            return {
                'chart': fig.to_json(),
                'table': table_fig.to_json(),
                'table_data': table_data,  # Raw data for custom sorting
                'summary': {
                    'total_days': len(daily_list),
                    'avg_margin_dollars': sum(margin_dollars) / len(margin_dollars),
                    'avg_margin_percentage': sum(margin_percentages) / len(margin_percentages),
                    'max_margin_dollars': max(margin_dollars),
                    'max_margin_percentage': max(margin_percentages),
                    'min_margin_dollars': min(margin_dollars),
                    'min_margin_percentage': min(margin_percentages)
                }
            }
            
        except Exception as e:
            return self._empty_chart(f"Error creating daily margin analysis chart: {str(e)}")

    def create_margin_analysis_chart(self):
        """Create a margin analysis table showing lowest, average, and highest margin per contract per strategy."""
        try:
            strategy_data = []
            
            # Import the margin calculation function and reset debug log
            from app import calculate_margin_from_legs, reset_margin_debug_log
            from datetime import datetime, timedelta
            
            # Reset debug log to start a new session
            reset_margin_debug_log()
            
            # Find the most recent trade date across all strategies to calculate 90-day window
            all_trade_dates = []
            for strategy in self.portfolio.strategies.values():
                for trade in strategy.trades:
                    if trade.date_opened:
                        all_trade_dates.append(trade.date_opened)
            
            if not all_trade_dates:
                return self._empty_chart("No trade dates found for margin analysis.")
            
            # Calculate 90-day cutoff date from the most recent trade
            most_recent_date = max(all_trade_dates)
            cutoff_date = most_recent_date - timedelta(days=90)
            
            # Collect margin per contract values for each strategy
            for strategy_name, strategy in self.portfolio.strategies.items():
                strategy_margins = []
                
                recent_margins = []  # For 90-day calculation
                
                for trade in strategy.trades:
                    margin_per_contract = 0
                    used_source = 'none'
                    
                    # First, try to use CSV margin data if available (for backtest data)
                    if hasattr(trade, 'margin_req') and hasattr(trade, 'contracts') and trade.margin_req > 0 and trade.contracts > 0:
                        margin_per_contract = trade.margin_req / trade.contracts
                        used_source = 'csv'
                    # Fallback to calculating from legs if CSV data not available (for live trade data)
                    elif trade.legs and trade.contracts > 0:
                        try:
                            # Import the margin calculation function
                            from app import calculate_margin_from_legs
                            
                            # Format date_opened as string if it's a datetime object
                            date_opened = getattr(trade, 'date_opened', '')
                            if hasattr(date_opened, 'strftime'):
                                date_opened = date_opened.strftime('%Y-%m-%d')
                            elif date_opened:
                                date_opened = str(date_opened)
                            
                            margin_info = calculate_margin_from_legs(trade.legs, context={
                                'strategy': strategy_name,
                                'date_opened': date_opened,
                                'contracts': getattr(trade, 'contracts', '')
                            })
                            margin_per_contract = margin_info.get('margin_per_contract', 0)
                            used_source = 'legs'
                        except Exception as e:
                            print(f"Error calculating margin from legs for trade: {e}")
                            margin_per_contract = 0
                    
                    # Only add if we have valid margin data
                    if margin_per_contract > 0:
                        strategy_margins.append((margin_per_contract, trade.date_opened))
                        
                        # Check if this trade is within the last 90 days
                        if trade.date_opened and trade.date_opened >= cutoff_date:
                            recent_margins.append(margin_per_contract)
                
                if strategy_margins:
                    # Extract margin values and dates
                    margin_values = [item[0] for item in strategy_margins]
                    margin_dates = [item[1] for item in strategy_margins]
                    
                    # Find min and max with their corresponding dates
                    min_idx = margin_values.index(min(margin_values))
                    max_idx = margin_values.index(max(margin_values))
                    
                    lowest_margin = margin_values[min_idx]
                    lowest_date = margin_dates[min_idx]
                    highest_margin = margin_values[max_idx]
                    highest_date = margin_dates[max_idx]
                    average_margin = sum(margin_values) / len(margin_values)
                    
                    # Calculate 90-day average
                    avg_90_days = sum(recent_margins) / len(recent_margins) if recent_margins else 0
                    
                    # Calculate expectancy using the strategy's property
                    expectancy = strategy.expectancy_per_lot
                    
                    # Calculate efficiency (expectancy / avg margin)
                    efficiency = expectancy / average_margin if average_margin > 0 else 0
                    
                    strategy_data.append({
                        'strategy': strategy_name,
                        'lowest': lowest_margin,
                        'lowest_date': lowest_date,
                        'average': average_margin,
                        'highest': highest_margin,
                        'highest_date': highest_date,
                        'avg_90_days': avg_90_days,
                        'recent_trades': len(recent_margins),
                        'trades': len(strategy_margins),
                        'avg_win': strategy.avg_win_per_lot,
                        'avg_loss': strategy.avg_loss_per_lot,
                        'max_win': strategy.max_win_per_lot,
                        'max_loss': strategy.max_loss_per_lot,
                        'win_rate': strategy.win_rate,
                        'expectancy': expectancy,
                        'efficiency': efficiency
                    })
            
            if not strategy_data:
                return self._empty_chart("No margin data available. Margin data requires either 'Margin Req.' and 'No. of Contracts' columns (for backtest data) or 'Legs' column with option leg details (for live trade data).")
            
            # Sort by average margin descending
            strategy_data.sort(key=lambda x: x['average'], reverse=True)
            
            # Prepare table data
            strategies = [data['strategy'] for data in strategy_data]
            
            # Display formatted values
            lowest_margins = [f"${data['lowest']:,.2f} ({data['lowest_date']})" for data in strategy_data]
            average_margins = [f"${data['average']:,.2f}" for data in strategy_data]
            highest_margins = [f"${data['highest']:,.2f} ({data['highest_date']})" for data in strategy_data]
            avg_90_days_margins = [f"${data['avg_90_days']:,.2f} ({data['recent_trades']} trades)" for data in strategy_data]
            trade_counts = [str(data['trades']) for data in strategy_data]
            # Format winner/loser values per contract and win rate (without HTML tags)
            avg_wins = [f"${data['avg_win']:,.2f}" for data in strategy_data]
            avg_losses = [f"${data['avg_loss']:,.2f}" for data in strategy_data]
            max_wins = [f"${data['max_win']:,.2f}" for data in strategy_data]
            max_losses = [f"${data['max_loss']:,.2f}" for data in strategy_data]
            win_rates = [f"{data['win_rate']:.1f}%" for data in strategy_data]
            expectancies = [f"${data['expectancy']:,.2f}" for data in strategy_data]
            efficiencies = [f"{data['efficiency']:.4f}" for data in strategy_data]
            
            # Create simple working table with all columns
            fig = go.Figure(data=[go.Table(
                header=dict(
                    values=['Strategy', 'Trades', 'Avg Margin/Contract', 'Avg Margin/Contract (Last 90 Days)', 'Min Margin/Contract (Date)', 'Max Margin/Contract (Date)', 'Win Rate', 'Avg Winner/Contract', 'Avg Loser/Contract', 'Max Winner/Contract', 'Max Loser/Contract', 'Expectancy/Contract', 'Efficiency'],
                    fill_color='#2E86AB',
                    font=dict(color='white', size=14),
                    align='center'
                ),
                cells=dict(
                    values=[
                        strategies,
                        trade_counts,
                        average_margins,
                        avg_90_days_margins,
                        lowest_margins,
                        highest_margins,
                        win_rates,
                        avg_wins,
                        avg_losses,
                        max_wins,
                        max_losses,
                        expectancies,
                        efficiencies
                    ],
                    fill_color='#F7F7F7',
                    font=dict(size=12),
                    align='center',
                    height=35,
                    # Add conditional formatting for winner/loser columns
                    font_color=[
                        ['black'] * len(strategies),  # Strategy
                        ['black'] * len(strategies),  # Trades
                        ['black'] * len(strategies),  # Avg Margin/Contract
                        ['black'] * len(strategies),  # Avg Margin/Contract (Last 90 Days)
                        ['black'] * len(strategies),  # Min Margin/Contract (Date)
                        ['black'] * len(strategies),  # Max Margin/Contract (Date)
                        ['black'] * len(strategies),  # Win Rate
                        ['green'] * len(strategies),  # Avg Winner
                        ['red'] * len(strategies),    # Avg Loser
                        ['green'] * len(strategies),  # Max Winner
                        ['red'] * len(strategies),    # Max Loser
                        ['black'] * len(strategies),  # Expectancy
                        ['black'] * len(strategies)   # Efficiency
                    ]
                )
            )])
            
            fig.update_layout(
                title='Margin Analysis: Margin per Contract by Strategy',
                height=min(600, 150 + len(strategy_data) * 45),
                margin=dict(l=20, r=20, t=60, b=20)
            )
            
            return {
                'chart': fig.to_json(),
                'type': 'margin_analysis',
                'description': 'Margin per contract statistics by strategy with P&L performance metrics. Uses CSV margin data when available, otherwise calculated from option legs.'
            }
            
        except Exception as e:
            print(f"Error creating margin analysis chart: {e}")
            return self._empty_chart(f"Error creating margin analysis chart: {str(e)}")

    def create_pnl_by_day_of_week_chart(self):
        """Create P&L by day of week table for all strategies."""
        try:
            if not self.portfolio.strategies:
                return self._empty_chart("No strategies found")
            
            # Initialize days of week (excluding weekends)
            days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
            
            # Collect P&L data for all strategies
            pnl_data = []
            
            for strategy_name, strategy in self.portfolio.strategies.items():
                # Initialize P&L for each day of week
                day_pnl = {day: 0.0 for day in days_of_week}
                
                for trade in strategy.trades:
                    if trade.date_closed:
                        # Get day of week from date_closed
                        day_of_week = trade.date_closed.strftime('%A')
                        day_pnl[day_of_week] += trade.pnl
                
                # Create row data for this strategy
                row_data = {
                    'Strategy': strategy_name,
                    'Trades': len(strategy.trades)
                }
                
                # Add P&L for each day
                for day in days_of_week:
                    row_data[day] = day_pnl[day]
                
                pnl_data.append(row_data)
            
            # Sort by total P&L descending
            pnl_data.sort(key=lambda x: sum(x[day] for day in days_of_week), reverse=True)
            
            if not pnl_data:
                return self._empty_chart("No P&L data available")
            
            # Prepare table data
            header_values = ['Strategy', 'Trades'] + days_of_week
            cell_values = [
                [row['Strategy'] for row in pnl_data],
                [row['Trades'] for row in pnl_data]
            ]
            
            # Add P&L values for each day with color formatting
            for day in days_of_week:
                day_values = []
                for row in pnl_data:
                    pnl_value = row[day]
                    if pnl_value > 0:
                        day_values.append(f"<span style='color: green; font-weight: bold;'>${pnl_value:,.2f}</span>")
                    elif pnl_value < 0:
                        day_values.append(f"<span style='color: red; font-weight: bold;'>${pnl_value:,.2f}</span>")
                    else:
                        day_values.append(f"${pnl_value:,.2f}")
                cell_values.append(day_values)
            
            # Create table
            fig = go.Figure(data=[go.Table(
                header=dict(
                    values=header_values,
                    fill_color='#2E86AB',
                    font=dict(color='white', size=14),
                    align='center'
                ),
                cells=dict(
                    values=cell_values,
                    fill_color=[['#F7F7F7', '#E8F4F8', '#F7F7F7', '#E8F4F8', '#F7F7F7', '#E8F4F8']],
                    font=dict(size=12),
                    align='center',
                    height=35
                )
            )])
            
            fig.update_layout(
                title='P&L by Day of Week',
                height=min(600, 150 + len(pnl_data) * 45),
                margin=dict(l=20, r=20, t=60, b=20)
            )
            
            return {
                'chart': fig.to_json(),
                'type': 'pnl_by_day_of_week',
                'description': 'P&L breakdown by day of week for each strategy.'
            }
            
        except Exception as e:
            print(f"Error creating P&L by day of week chart: {e}")
            return self._empty_chart(f"Error creating P&L by day of week chart: {str(e)}")
    
    def _empty_chart(self, message: str) -> Dict:
        """Return empty chart with message."""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            x=0.5, y=0.5,
            xref='paper', yref='paper',
            showarrow=False,
            font_size=16
        )
        fig.update_layout(
            template='plotly_white',
            height=650,
            autosize=True,
            xaxis={'visible': False},
            yaxis={'visible': False}
        )
        
        return {
            'chart': fig.to_json(),
            'type': 'empty',
            'description': message
        }

class ChartFactory:
    """Factory for creating specific chart types easily."""
    
    def __init__(self, portfolio: Portfolio):
        self.generator = ChartGenerator(portfolio)
    
    def get_available_charts(self) -> Dict[str, str]:
        """Get list of available chart types with descriptions."""
        return {
            'overview': 'Strategy performance overview and portfolio analysis',
            'cumulative_pnl': 'Cumulative profit and loss over time',
            'daily_pnl': 'Daily profit and loss analysis',
            'monthly_pnl_stacked': 'Monthly P&L breakdown by strategy',
            'pnl_by_day_of_week': 'P&L by day of week table',
            'pnl_by_month': 'P&L by month table',
            'commission_analysis': 'Commission analysis',
            'drawdown': 'Portfolio drawdown analysis',
            'portfolio_balance': 'Portfolio allocation by strategy',
            'strategy_detail': 'Detailed trade analysis by strategy',
            'win_rates_table': 'Win rate statistics table',
            'risk_return': 'Risk vs return scatter plot',
            'strategy_correlation': 'Strategy correlation heatmap',
            'margin_analysis': 'Margin requirements analysis',
            'daily_margin_analysis': 'Daily margin usage analysis',
            'monte_carlo': 'Monte Carlo simulation analysis'
        }
    
    def get_monte_carlo_charts() -> Dict[str, str]:
        """Get list of available Monte Carlo chart types with descriptions."""
        return {
            'monte_carlo_distribution': 'Final balance distribution histogram',
            'monte_carlo_confidence': 'Simulation paths with confidence intervals',
            'monte_carlo_risk_metrics': 'Risk metrics summary from simulation'
        }
    
    def create_chart(self, chart_type: str) -> Dict:
        """Create a chart based on the chart type."""
        chart_methods = {
            'overview': self.generator._empty_chart,  # Will be handled specially in loadChart
            'cumulative_pnl': self.generator.create_cumulative_pnl_chart,
            'daily_pnl': self.generator.create_daily_pnl_chart,
            'monthly_pnl_stacked': self.generator.create_monthly_pnl_stacked,
            'drawdown': self.generator.create_drawdown_chart,
            'portfolio_balance': self.generator.create_strategy_balance_pie,
            'strategy_detail': self.generator._empty_chart,  # Will be handled specially in loadChart
            'win_rates_table': self.generator.create_win_rates_table,
            'risk_return': self.generator.create_risk_return_scatter,
            'strategy_correlation': self.generator.create_strategy_correlation_heatmap,
            'margin_analysis': self.generator.create_margin_analysis_chart,
            'daily_margin_analysis': self.generator.create_daily_margin_analysis_chart,
            'pnl_by_day_of_week': self.generator.create_pnl_by_day_of_week_chart
        }
        
        if chart_type in chart_methods:
            # Call the method without passing portfolio (it's already available in the generator)
            return chart_methods[chart_type]()
        else:
            return self.generator._empty_chart(f"Unknown chart type: {chart_type}")
    
    @staticmethod
    def create_monte_carlo_chart(chart_type: str, simulation_data: Dict, portfolio: Portfolio) -> Dict:
        """Create a Monte Carlo chart with simulation data."""
        generator = ChartGenerator(portfolio)
        
        monte_carlo_methods = {
            'monte_carlo_distribution': generator.create_monte_carlo_distribution_chart,
            'monte_carlo_confidence': generator.create_monte_carlo_confidence_intervals_chart,
            'monte_carlo_risk_metrics': generator.create_monte_carlo_risk_metrics_chart
        }
        
        if chart_type not in monte_carlo_methods:
            return generator._empty_chart(f"Unknown Monte Carlo chart type: {chart_type}")
        
        return monte_carlo_methods[chart_type](simulation_data) 