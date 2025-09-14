"""
Strategy-focused analytics functions for portfolio analysis.
Handles calculations for strategy comparison, balance analysis, and diversification.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from models import Portfolio, Strategy, Trade

class StrategyAnalyzer:
    """Handles strategy-focused analytics for portfolio balance and diversification."""
    
    def __init__(self, portfolio: Portfolio):
        self.portfolio = portfolio
    
    def get_strategy_balance_analysis(self) -> Dict:
        """Analyze portfolio balance across bullish/bearish/neutral strategies."""
        balance = self.portfolio.get_strategy_types_breakdown()
        
        # Calculate allocation percentages
        total_pnl = abs(self.portfolio.total_pnl) if self.portfolio.total_pnl != 0 else 1
        
        balance_summary = {}
        for strategy_type, strategies in balance.items():
            type_pnl = sum(s['pnl'] for s in strategies)
            type_trades = sum(s['trade_count'] for s in strategies)
            
            balance_summary[strategy_type] = {
                'pnl': round(type_pnl, 2),
                'pnl_percentage': round((type_pnl / total_pnl) * 100, 2) if total_pnl != 0 else 0,
                'trade_count': type_trades,
                'strategy_count': len(strategies),
                'strategies': strategies
            }
        
        return balance_summary
    
    def calculate_strategy_correlations(self) -> pd.DataFrame:
        """Calculate correlations between strategy daily returns."""
        # Get daily P&L by strategy
        daily_returns = self._get_daily_returns_by_strategy()
        
        if daily_returns.empty:
            # Fallback: create correlation matrix with all strategies
            return self._create_fallback_correlation_matrix()
        
        # Calculate correlation matrix
        correlation_matrix = daily_returns.corr()
        
        # If we have fewer strategies than expected, add missing ones
        if len(correlation_matrix.columns) < len(self.portfolio.strategies):
            correlation_matrix = self._add_missing_strategies_to_correlation(correlation_matrix)
        
        return correlation_matrix
    
    def _create_fallback_correlation_matrix(self) -> pd.DataFrame:
        """Create a fallback correlation matrix when no daily data is available."""
        strategy_names = list(self.portfolio.strategies.keys())
        n_strategies = len(strategy_names)
        
        # Create identity matrix (perfect correlation with self, no correlation with others)
        correlation_matrix = pd.DataFrame(
            np.eye(n_strategies),  # Identity matrix
            index=strategy_names,
            columns=strategy_names
        )
        
        print(f"DEBUG: Created fallback correlation matrix for {n_strategies} strategies")
        return correlation_matrix
    
    def _add_missing_strategies_to_correlation(self, correlation_matrix: pd.DataFrame) -> pd.DataFrame:
        """Add missing strategies to correlation matrix with zero correlation."""
        all_strategies = list(self.portfolio.strategies.keys())
        existing_strategies = list(correlation_matrix.columns)
        missing_strategies = [s for s in all_strategies if s not in existing_strategies]
        
        if not missing_strategies:
            return correlation_matrix
        
        # Add missing strategies as columns and rows
        for strategy in missing_strategies:
            correlation_matrix[strategy] = 0.0  # Add as column
            correlation_matrix.loc[strategy] = 0.0  # Add as row
        
        # Set diagonal to 1.0 for missing strategies
        for strategy in missing_strategies:
            correlation_matrix.loc[strategy, strategy] = 1.0
        
        print(f"DEBUG: Added {len(missing_strategies)} missing strategies to correlation matrix")
        return correlation_matrix
    
    def get_diversification_score(self) -> Dict:
        """Calculate portfolio diversification metrics."""
        correlations = self.calculate_strategy_correlations()
        
        if correlations.empty:
            return {'score': 0, 'status': 'No data'}
        
        # Average absolute correlation (lower is better for diversification)
        avg_correlation = correlations.abs().mean().mean()
        
        # Diversification score (0-100, higher is better)
        diversification_score = max(0, 100 - (avg_correlation * 100))
        
        # Count of low-correlation pairs (< 0.3)
        low_corr_pairs = 0
        total_pairs = 0
        
        for i in range(len(correlations.columns)):
            for j in range(i+1, len(correlations.columns)):
                if not pd.isna(correlations.iloc[i, j]):
                    total_pairs += 1
                    if abs(correlations.iloc[i, j]) < 0.3:
                        low_corr_pairs += 1
        
        return {
            'score': round(diversification_score, 2),
            'avg_correlation': round(avg_correlation, 3),
            'low_correlation_pairs': low_corr_pairs,
            'total_pairs': total_pairs,
            'status': self._get_diversification_status(diversification_score)
        }
    
    def get_risk_analysis(self) -> Dict:
        """Analyze risk across strategies."""
        risk_metrics = {}
        
        for strategy_name, strategy in self.portfolio.strategies.items():
            if not strategy.trades:
                continue
            
            # Calculate drawdown for this strategy using absolute P&L
            pnl_series = pd.Series([trade.pnl for trade in strategy.trades])
            cumulative_pnl = pnl_series.cumsum()
            
            # Calculate maximum drawdown
            running_max = cumulative_pnl.expanding().max()
            drawdown = cumulative_pnl - running_max
            max_drawdown = drawdown.min()
            
            # Calculate volatility using per-lot P&L for fair comparison
            pnl_per_lot_series = pd.Series([trade.pnl_per_lot for trade in strategy.trades])
            volatility_per_lot = pnl_per_lot_series.std()
            
            # Trade-based efficiency ratio (simple ratio without time normalization)
            trade_efficiency_ratio = pnl_per_lot_series.mean() / volatility_per_lot if volatility_per_lot != 0 else 0
            
            # Calculate proper Sharpe ratio with time normalization
            sharpe_ratio = self._calculate_proper_sharpe_ratio(strategy.trades)
            
            risk_metrics[strategy_name] = {
                'max_drawdown': round(max_drawdown, 2),
                'volatility': round(volatility_per_lot, 2),
                'sharpe_ratio': round(sharpe_ratio, 3),
                'trade_efficiency_ratio': round(trade_efficiency_ratio, 3),
                'risk_level': self._classify_risk_level(volatility_per_lot, max_drawdown)
            }
        
        return risk_metrics
    
    def _calculate_proper_sharpe_ratio(self, trades: List) -> float:
        """Calculate proper Sharpe ratio with time normalization and risk-free rate."""
        if len(trades) < 2:
            return 0.0
        
        # Calculate time span and trade frequency
        start_date = pd.to_datetime(trades[0].date_closed)
        end_date = pd.to_datetime(trades[-1].date_closed)
        days_elapsed = (end_date - start_date).days
        
        if days_elapsed <= 0:
            return 0.0
        
        # Calculate trades per year
        trades_per_year = len(trades) * 365.25 / days_elapsed
        
        # Calculate returns as percentage (using per-lot P&L relative to a standard margin base)
        # Use a standard margin base of $1000 for normalization
        standard_margin_base = 1000.0
        returns = []
        
        for trade in trades:
            # Calculate return as percentage of standard margin base
            return_pct = trade.pnl_per_lot / standard_margin_base
            returns.append(return_pct)
        
        if not returns:
            return 0.0
        
        # Calculate mean and standard deviation of returns
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        # Annualize the metrics
        annual_return = mean_return * trades_per_year
        annual_volatility = std_return * np.sqrt(trades_per_year)
        
        # Risk-free rate (assume 2% annual)
        risk_free_rate = 0.02
        
        # Calculate Sharpe ratio
        if annual_volatility > 0:
            sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility
        else:
            sharpe_ratio = 0.0
        
        return sharpe_ratio
    
    def suggest_position_sizing(self) -> Dict:
        """Suggest position sizing based on strategy performance and risk."""
        suggestions = {}
        risk_analysis = self.get_risk_analysis()
        
        total_strategies = len(self.portfolio.strategies)
        base_allocation = 100 / total_strategies if total_strategies > 0 else 0
        
        for strategy_name, strategy in self.portfolio.strategies.items():
            risk_data = risk_analysis.get(strategy_name, {})
            
            # Adjust allocation based on performance and risk
            performance_score = self._calculate_performance_score(strategy)
            risk_adjustment = self._calculate_risk_adjustment(risk_data)
            
            suggested_allocation = base_allocation * performance_score * risk_adjustment
            
            suggestions[strategy_name] = {
                'current_trades': strategy.trade_count,
                'performance_score': round(performance_score, 2),
                'risk_adjustment': round(risk_adjustment, 2),
                'suggested_allocation_pct': round(suggested_allocation, 2),
                'recommendation': self._get_sizing_recommendation(suggested_allocation, base_allocation)
            }
        
        return suggestions
    
    def _get_daily_returns_by_strategy(self) -> pd.DataFrame:
        """Get daily returns for each strategy."""
        daily_data = []
        
        for strategy_name, strategy in self.portfolio.strategies.items():
            for trade in strategy.trades:
                # Only include trades with valid close dates
                if trade.date_closed:
                    try:
                        date = pd.to_datetime(trade.date_closed)
                        daily_data.append({
                            'date': date,
                            'strategy': strategy_name,
                            'pnl': trade.pnl_per_lot
                        })
                    except (ValueError, TypeError):
                        # Skip trades with invalid dates
                        continue
        
        if not daily_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(daily_data)
        
        # Group by date and strategy, sum P&L for each day
        daily_returns = df.groupby(['date', 'strategy'])['pnl'].sum().unstack(fill_value=0)
        
        # Debug: Print strategy count
        print(f"DEBUG: Correlation calculation found {len(daily_returns.columns)} strategies: {list(daily_returns.columns)}")
        
        return daily_returns
    
    def _get_diversification_status(self, score: float) -> str:
        """Get diversification status based on score."""
        if score >= 70:
            return "Well Diversified"
        elif score >= 50:
            return "Moderately Diversified"
        elif score >= 30:
            return "Poorly Diversified"
        else:
            return "Highly Correlated"
    
    def _classify_risk_level(self, volatility: float, max_drawdown: float) -> str:
        """Classify strategy risk level."""
        # Simple risk classification based on volatility and drawdown
        risk_score = (volatility * 0.6) + (abs(max_drawdown) * 0.4)
        
        if risk_score < 100:
            return "LOW"
        elif risk_score < 300:
            return "MEDIUM"
        else:
            return "HIGH"
    
    def _calculate_performance_score(self, strategy: Strategy) -> float:
        """Calculate performance score for position sizing (0.5 to 1.5)."""
        if not strategy.trades:
            return 1.0
        
        # Base score on win rate and profit factor
        win_rate_score = strategy.win_rate / 100  # 0-1
        
        profit_factor = abs(strategy.avg_win / strategy.avg_loss) if strategy.avg_loss != 0 else 1
        profit_factor_score = min(profit_factor / 2, 1.0)  # Cap at 1.0
        
        # Combined score (0.5 to 1.5 range)
        performance_score = 0.5 + (win_rate_score * 0.5) + (profit_factor_score * 0.5)
        
        return max(0.5, min(1.5, performance_score))
    
    def _calculate_risk_adjustment(self, risk_data: Dict) -> float:
        """Calculate risk adjustment for position sizing (0.5 to 1.2)."""
        if not risk_data:
            return 1.0
        
        risk_level = risk_data.get('risk_level', 'MEDIUM')
        
        # Adjust based on risk level
        if risk_level == 'LOW':
            return 1.2  # Increase allocation for low-risk strategies
        elif risk_level == 'MEDIUM':
            return 1.0  # Keep normal allocation
        else:  # HIGH
            return 0.5  # Reduce allocation for high-risk strategies
    
    def _get_sizing_recommendation(self, suggested: float, base: float) -> str:
        """Get position sizing recommendation."""
        ratio = suggested / base if base != 0 else 1
        
        if ratio > 1.2:
            return "INCREASE - Strong performer, low risk"
        elif ratio < 0.8:
            return "DECREASE - High risk or poor performance"
        else:
            return "MAINTAIN - Balanced allocation"

class PortfolioMetrics:
    """Calculate comprehensive portfolio-level metrics."""
    
    def __init__(self, portfolio: Portfolio):
        self.portfolio = portfolio
    
    def get_overview_metrics(self) -> Dict:
        """Get high-level portfolio metrics."""
        if not self.portfolio.strategies:
            return self._empty_metrics()
        
        # Calculate cumulative P&L over time
        all_trades = self._get_all_trades_chronologically()
        
        if not all_trades:
            return self._empty_metrics()
        
        # Calculate initial account balance from first trade
        first_trade = all_trades[0]
        initial_balance = first_trade.funds_at_close - first_trade.pnl
        
        cumulative_pnl = np.cumsum([trade.pnl for trade in all_trades])
        account_values = initial_balance + cumulative_pnl
        
        # Calculate key metrics
        total_return = cumulative_pnl[-1]
        max_drawdown = self._calculate_max_drawdown(account_values)
        max_drawdown_pct = self._calculate_max_drawdown_percentage(account_values)
        
        # Calculate CAGR
        final_balance = account_values[-1]
        start_date = pd.to_datetime(first_trade.date_closed)
        end_date = pd.to_datetime(all_trades[-1].date_closed)
        years = (end_date - start_date).days / 365.25
        
        if years > 0 and initial_balance > 0 and final_balance > 0:
            cagr = ((final_balance / initial_balance) ** (1 / years) - 1) * 100
        else:
            cagr = 0
        
        # Calculate proper Sharpe ratio using time normalization
        sharpe_ratio = self._calculate_portfolio_sharpe_ratio(all_trades)
        
        # Calculate average trade P&L (for portfolio metrics)
        returns = [trade.pnl for trade in all_trades]
        
        # Calculate average and longest days to new P&L high
        # Create a daily account value series to properly calculate calendar days
        start_date = pd.to_datetime(first_trade.date_closed)
        end_date = pd.to_datetime(all_trades[-1].date_closed)
        
        # Create a daily date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Create a DataFrame with daily account values
        daily_data = []
        current_balance = initial_balance
        trade_idx = 0
        
        for date in date_range:
            # Apply any trades that occurred on this date
            while (trade_idx < len(all_trades) and 
                   pd.to_datetime(all_trades[trade_idx].date_closed).date() == date.date()):
                current_balance += all_trades[trade_idx].pnl
                trade_idx += 1
            
            daily_data.append({
                'date': date,
                'balance': current_balance
            })
        
        # Find new highs in daily data
        daily_balances = [d['balance'] for d in daily_data]
        daily_dates = [d['date'] for d in daily_data]
        
        high_indices = []
        running_max = float('-inf')
        for i, val in enumerate(daily_balances):
            if val > running_max:
                high_indices.append(i)
                running_max = val
        
        avg_days_to_new_high = None
        longest_days_since_to_new_high = None
        if len(high_indices) >= 2:
            high_dates = [daily_dates[i] for i in high_indices]
            day_diffs = [(high_dates[i] - high_dates[i-1]).days for i in range(1, len(high_dates))]
            # Calculate days since last high
            last_high_date = high_dates[-1]
            most_recent_date = daily_dates[-1]
            days_since_last_high = (most_recent_date - last_high_date).days
            max_interval = max(day_diffs) if day_diffs else 0
            # If current interval is longer, use it
            if days_since_last_high > max_interval:
                longest_days_since_to_new_high = days_since_last_high
            else:
                longest_days_since_to_new_high = max_interval
            if day_diffs:
                avg_days_to_new_high = round(np.mean(day_diffs), 2)
        
        # Calculate MAR (CAGR / Max Drawdown %)
        mar = cagr / abs(max_drawdown_pct) if abs(max_drawdown_pct) > 0 else 0
        
        return {
            'total_pnl': round(total_return, 2),
            'total_trades': len(all_trades),
            'initial_balance': round(initial_balance, 2),
            'final_balance': round(final_balance, 2),
            'cagr': round(cagr, 2),
            'max_drawdown': round(max_drawdown, 2),
            'max_drawdown_pct': round(max_drawdown_pct, 2),
            'mar': round(mar, 2),
            'sharpe_ratio': round(sharpe_ratio, 3),
            'strategy_count': len(self.portfolio.strategies),
            'avg_trade_pnl': round(np.mean(returns), 2),
            'cumulative_pnl_series': cumulative_pnl.tolist(),
            'avg_days_to_new_high': avg_days_to_new_high,
            'longest_days_since_to_new_high': longest_days_since_to_new_high,
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d')
        }
    
    def _get_all_trades_chronologically(self) -> List[Trade]:
        """Get all trades sorted by date."""
        all_trades = []
        
        for strategy in self.portfolio.strategies.values():
            all_trades.extend(strategy.trades)
        
        # Sort by date_closed
        all_trades.sort(key=lambda t: pd.to_datetime(t.date_closed))
        
        return all_trades
    
    def _calculate_portfolio_sharpe_ratio(self, all_trades: List) -> float:
        """Calculate proper portfolio-level Sharpe ratio with time normalization."""
        if len(all_trades) < 2:
            return 0.0
        
        # Calculate time span
        start_date = pd.to_datetime(all_trades[0].date_closed)
        end_date = pd.to_datetime(all_trades[-1].date_closed)
        days_elapsed = (end_date - start_date).days
        
        if days_elapsed <= 0:
            return 0.0
        
        # Calculate trades per year for the portfolio
        trades_per_year = len(all_trades) * 365.25 / days_elapsed
        
        # Calculate returns as percentage of a standard base
        # Use average margin per lot across all trades as base
        total_margin = sum(getattr(trade, 'margin_req', 0) for trade in all_trades)
        total_contracts = sum(getattr(trade, 'contracts', 1) for trade in all_trades)
        avg_margin_per_lot = total_margin / total_contracts if total_contracts > 0 else 1000.0
        
        # Use standard base if no margin data
        if avg_margin_per_lot == 0:
            avg_margin_per_lot = 1000.0
        
        returns = []
        for trade in all_trades:
            # Calculate return as percentage of margin base
            return_pct = trade.pnl_per_lot / avg_margin_per_lot
            returns.append(return_pct)
        
        if not returns:
            return 0.0
        
        # Calculate mean and standard deviation
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        # Annualize the metrics
        annual_return = mean_return * trades_per_year
        annual_volatility = std_return * np.sqrt(trades_per_year)
        
        # Risk-free rate (2% annual)
        risk_free_rate = 0.02
        
        # Calculate Sharpe ratio
        if annual_volatility > 0:
            sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility
        else:
            sharpe_ratio = 0.0
        
        return sharpe_ratio
    
    def _calculate_max_drawdown(self, account_values: np.ndarray) -> float:
        """Calculate maximum drawdown from account values."""
        running_max = np.maximum.accumulate(account_values)
        drawdown = account_values - running_max
        return np.min(drawdown)
    
    def _calculate_max_drawdown_percentage(self, account_values: np.ndarray) -> float:
        """Calculate maximum drawdown percentage from peak."""
        if len(account_values) == 0:
            return 0.0
        
        # Calculate running maximum (peaks)
        running_max = np.maximum.accumulate(account_values)
        
        # Calculate drawdown percentage at each point: (peak - current) / peak * 100
        # Avoid division by zero by setting minimum peak value
        running_max = np.maximum(running_max, 1)  # Avoid division by zero
        drawdown_pct = (running_max - account_values) / running_max * 100
        
        # Return the maximum drawdown percentage
        return np.max(drawdown_pct)
    
    def _empty_metrics(self) -> Dict:
        """Return empty metrics when no data available."""
        return {
            'total_pnl': 0,
            'total_trades': 0,
            'initial_balance': 0,
            'final_balance': 0,
            'cagr': 0,
            'max_drawdown': 0,
            'max_drawdown_pct': 0,
            'sharpe_ratio': 0,
            'strategy_count': 0,
            'avg_trade_pnl': 0,
            'cumulative_pnl_series': [],
            'avg_days_to_new_high': None,
            'longest_days_since_to_new_high': None
        }


class MonteCarloSimulator:
    """Monte Carlo simulation for portfolio strategy analysis."""
    
    def __init__(self, portfolio: Portfolio):
        self.portfolio = portfolio
    
    def run_simulation(self, num_simulations: int = 1000, num_trades: int = None, 
                      confidence_levels: List[float] = [0.05, 0.25, 0.5, 0.75, 0.95]) -> Dict:
        """
        Run Monte Carlo simulation by randomly sampling from historical trade outcomes.
        
        Args:
            num_simulations: Number of simulation runs
            num_trades: Number of trades per simulation (default: same as historical)
            confidence_levels: Percentiles to calculate for results
        
        Returns:
            Dictionary with simulation results and statistics
        """
        if not self.portfolio.strategies:
            return self._empty_simulation_results()
        
        # Get all historical trades
        all_trades = []
        for strategy in self.portfolio.strategies.values():
            all_trades.extend(strategy.trades)
        
        if not all_trades:
            return self._empty_simulation_results()
        
        # Use historical trade count if not specified
        if num_trades is None:
            num_trades = len(all_trades)
        
        # Extract trade returns for sampling
        trade_returns = [trade.pnl for trade in all_trades]
        
        # Run simulations
        simulation_results = []
        final_balances = []
        max_drawdowns = []
        
        # Calculate initial balance from historical data
        metrics = PortfolioMetrics(self.portfolio)
        historical_metrics = metrics.get_overview_metrics()
        initial_balance = historical_metrics['initial_balance']
        
        for sim_num in range(num_simulations):
            # Randomly sample trades with replacement
            simulated_pnl = np.random.choice(trade_returns, size=num_trades, replace=True)
            
            # Calculate cumulative P&L and account balance over time
            cumulative_pnl = np.cumsum(simulated_pnl)
            account_balance = initial_balance + cumulative_pnl
            
            # Calculate final balance and max drawdown
            final_balance = account_balance[-1]
            max_drawdown = self._calculate_simulation_max_drawdown(account_balance)
            
            simulation_results.append({
                'simulation_id': sim_num + 1,
                'final_balance': final_balance,
                'total_pnl': cumulative_pnl[-1],
                'max_drawdown': max_drawdown,
                'cumulative_pnl': cumulative_pnl.tolist(),
                'account_balance': account_balance.tolist(),
                'simulated_trades': simulated_pnl.tolist()  # Store the actual trade P&L values
            })
            
            final_balances.append(final_balance)
            max_drawdowns.append(max_drawdown)
        
        # Calculate statistics
        final_balances = np.array(final_balances)
        max_drawdowns = np.array(max_drawdowns)
        total_pnls = final_balances - initial_balance
        
        # Calculate percentiles
        balance_percentiles = {}
        pnl_percentiles = {}
        drawdown_percentiles = {}
        
        for level in confidence_levels:
            percentile = level * 100
            balance_percentiles[f'p{int(percentile)}'] = np.percentile(final_balances, percentile)
            pnl_percentiles[f'p{int(percentile)}'] = np.percentile(total_pnls, percentile)
            drawdown_percentiles[f'p{int(percentile)}'] = np.percentile(max_drawdowns, percentile)
        
        # Calculate additional statistics
        win_probability = (total_pnls > 0).mean() * 100
        loss_probability = (total_pnls < 0).mean() * 100
        
        return {
            'simulation_summary': {
                'num_simulations': num_simulations,
                'num_trades_per_sim': num_trades,
                'initial_balance': initial_balance,
                'historical_final_balance': historical_metrics['final_balance']
            },
            'final_balance_stats': {
                'mean': np.mean(final_balances),
                'median': np.median(final_balances),
                'std': np.std(final_balances),
                'min': np.min(final_balances),
                'max': np.max(final_balances),
                'percentiles': balance_percentiles
            },
            'pnl_stats': {
                'mean': np.mean(total_pnls),
                'median': np.median(total_pnls),
                'std': np.std(total_pnls),
                'min': np.min(total_pnls),
                'max': np.max(total_pnls),
                'percentiles': pnl_percentiles
            },
            'drawdown_stats': {
                'mean': np.mean(max_drawdowns),
                'median': np.median(max_drawdowns),
                'std': np.std(max_drawdowns),
                'min': np.min(max_drawdowns),
                'max': np.max(max_drawdowns),
                'percentiles': drawdown_percentiles
            },
            'probabilities': {
                'win_probability': win_probability,
                'loss_probability': loss_probability,
                'break_even_probability': 100 - win_probability - loss_probability
            },
            'simulation_results': simulation_results[:100],  # Return first 100 for charts
            'confidence_levels': confidence_levels
        }
    
    def run_strategy_specific_simulation(self, strategy_name: str, num_simulations: int = 1000, 
                                       num_trades: int = None) -> Dict:
        """
        Run Monte Carlo simulation for a specific strategy.
        
        Args:
            strategy_name: Name of the strategy to simulate
            num_simulations: Number of simulation runs
            num_trades: Number of trades per simulation
        
        Returns:
            Dictionary with strategy-specific simulation results
        """
        if strategy_name not in self.portfolio.strategies:
            return self._empty_simulation_results()
        
        strategy = self.portfolio.strategies[strategy_name]
        
        if not strategy.trades:
            return self._empty_simulation_results()
        
        # Use strategy's historical trade count if not specified
        if num_trades is None:
            num_trades = len(strategy.trades)
        
        # Extract trade returns for this strategy
        trade_returns = [trade.pnl for trade in strategy.trades]
        
        # Calculate initial balance (for single strategy, use average account balance)
        account_balances = [trade.funds_at_close for trade in strategy.trades]
        initial_balance = np.mean(account_balances) - strategy.total_pnl
        
        # Run simulations
        simulation_results = []
        final_pnls = []
        
        for sim_num in range(num_simulations):
            # Randomly sample trades with replacement
            simulated_pnl = np.random.choice(trade_returns, size=num_trades, replace=True)
            
            # Calculate cumulative P&L
            cumulative_pnl = np.cumsum(simulated_pnl)
            final_pnl = cumulative_pnl[-1]
            
            simulation_results.append({
                'simulation_id': sim_num + 1,
                'final_pnl': final_pnl,
                'cumulative_pnl': cumulative_pnl.tolist()
            })
            
            final_pnls.append(final_pnl)
        
        # Calculate statistics
        final_pnls = np.array(final_pnls)
        
        return {
            'strategy_name': strategy_name,
            'simulation_summary': {
                'num_simulations': num_simulations,
                'num_trades_per_sim': num_trades,
                'historical_pnl': strategy.total_pnl,
                'historical_trade_count': len(strategy.trades)
            },
            'pnl_stats': {
                'mean': np.mean(final_pnls),
                'median': np.median(final_pnls),
                'std': np.std(final_pnls),
                'min': np.min(final_pnls),
                'max': np.max(final_pnls)
            },
            'probabilities': {
                'win_probability': (final_pnls > 0).mean() * 100,
                'loss_probability': (final_pnls < 0).mean() * 100
            },
            'simulation_results': simulation_results[:100]  # Return first 100 for charts
        }
    
    def _calculate_simulation_max_drawdown(self, account_balance: np.ndarray) -> float:
        """Calculate maximum drawdown for a simulation run."""
        if len(account_balance) == 0:
            return 0.0
        
        # Calculate running maximum (peaks)
        running_max = np.maximum.accumulate(account_balance)
        
        # Calculate drawdown at each point
        drawdown = running_max - account_balance
        
        # Return maximum drawdown
        return np.max(drawdown)
    
    def _empty_simulation_results(self) -> Dict:
        """Return empty results when no data available."""
        return {
            'simulation_summary': {
                'num_simulations': 0,
                'num_trades_per_sim': 0,
                'initial_balance': 0,
                'historical_final_balance': 0
            },
            'final_balance_stats': {},
            'pnl_stats': {},
            'drawdown_stats': {},
            'probabilities': {},
            'simulation_results': [],
            'confidence_levels': []
        } 