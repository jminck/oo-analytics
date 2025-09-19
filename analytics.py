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
        
        # Risk-free rate (2% annual Treasury bill rate)
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
        
        # Risk-free rate (2% annual Treasury bill rate)
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
                      trade_size_percent: float = 1.0,
                      confidence_levels: List[float] = [0.05, 0.25, 0.5, 0.75, 0.95]) -> Dict:
        """
        Run Monte Carlo simulation by randomly sampling from historical trade outcomes.
        
        Args:
            num_simulations: Number of simulation runs
            num_trades: Number of trades per simulation (default: same as historical)
            trade_size_percent: Percentage of account balance to risk per trade (default: 1.0%)
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
        
        # Extract trade returns for sampling (use aggregate P&L for portfolio-level analysis)
        # Note: trade_size_percent affects position sizing, not individual trade returns
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
            simulated_trade_pnl = np.random.choice(trade_returns, size=num_trades, replace=True)
            
            # Calculate account balance over time
            # Note: Historical trades already represent the actual performance
            # The trade size percentage is for reference only, not for scaling
            cumulative_pnl = np.cumsum(simulated_trade_pnl)
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
                'simulated_trades': simulated_trade_pnl.tolist()  # Store the actual trade P&L values
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
        
        # Calculate risk-adjusted ratios
        risk_ratios = self._calculate_risk_ratios(total_pnls, final_balances, None, trade_size_percent)
        
        # Calculate t-test statistics
        t_test_results = self._calculate_t_tests(total_pnls, final_balances, initial_balance)
        
        # Calculate p-values for statistical significance
        p_values = self._calculate_p_values(total_pnls, final_balances, max_drawdowns, initial_balance)
        
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
            'risk_ratios': risk_ratios,
            't_test_results': t_test_results,
            'p_values': p_values,
            'simulation_results': simulation_results[:100],  # Return first 100 for charts
            'confidence_levels': confidence_levels
        }
    
    def run_strategy_specific_simulation(self, strategy_name: str, num_simulations: int = 1000, 
                                       num_trades: int = None, trade_size_percent: float = 1.0) -> Dict:
        """
        Run Monte Carlo simulation for a specific strategy.
        
        Args:
            strategy_name: Name of the strategy to simulate
            num_simulations: Number of simulation runs
            num_trades: Number of trades per simulation
            trade_size_percent: Percentage of account balance to risk per trade (default: 1.0%)
        
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
        
        # Extract trade returns for this strategy (use per-lot P&L for consistency)
        # For per-lot statistics, use raw per-lot P&L (not scaled by trade size)
        trade_returns = [trade.pnl_per_lot for trade in strategy.trades]
        
        # Debug: Check the actual values
        print(f"DEBUG: Strategy {strategy_name} - Sample trade_returns:")
        for i, trade in enumerate(strategy.trades[:5]):  # First 5 trades
            print(f"DEBUG: Trade {i+1}: P&L=${trade.pnl:,.2f}, Contracts={trade.contracts}, P&L/Lot=${trade.pnl_per_lot:,.2f}")
        print(f"DEBUG: Trade returns range: ${min(trade_returns):,.2f} to ${max(trade_returns):,.2f}")
        print(f"DEBUG: Mean trade return: ${np.mean(trade_returns):,.2f}")
        
        # Calculate initial balance (for single strategy, use average account balance)
        # Since we're using per-lot P&L, we need to normalize the initial balance calculation
        account_balances = [trade.funds_at_close for trade in strategy.trades]
        avg_contracts = np.mean([getattr(trade, 'contracts', 1) for trade in strategy.trades])
        # For per-lot analysis, use a standard initial balance per lot
        initial_balance = 1000.0  # Standard $1000 per lot initial balance
        
        # Run simulations
        simulation_results = []
        final_pnls = []
        
        for sim_num in range(num_simulations):
            # Randomly sample trades with replacement
            simulated_pnl = np.random.choice(trade_returns, size=num_trades, replace=True)
            
            # Calculate cumulative P&L
            cumulative_pnl = np.cumsum(simulated_pnl)
            final_pnl = cumulative_pnl[-1]
            
            # Calculate account balance for chart compatibility
            account_balance = initial_balance + cumulative_pnl
            
            simulation_results.append({
                'simulation_id': sim_num + 1,
                'final_pnl': final_pnl,
                'final_balance': account_balance[-1],  # For compatibility with portfolio simulation
                'total_pnl': final_pnl,  # For compatibility with portfolio simulation
                'cumulative_pnl': cumulative_pnl.tolist(),
                'account_balance': account_balance.tolist(),  # For chart compatibility
                'simulated_trades': simulated_pnl.tolist()  # Store individual trade returns
            })
            
            final_pnls.append(final_pnl)
        
        # Calculate statistics
        final_pnls = np.array(final_pnls)
        # Since we're using per-lot P&L, the final_balances should also be per-lot
        final_balances = initial_balance + final_pnls
        
        # Debug: Check simulation results
        print(f"DEBUG: Simulation results for {strategy_name}:")
        print(f"DEBUG: Final P&Ls range: ${np.min(final_pnls):,.2f} to ${np.max(final_pnls):,.2f}")
        print(f"DEBUG: Mean final P&L: ${np.mean(final_pnls):,.2f}")
        print(f"DEBUG: Initial balance: ${initial_balance:,.2f}")
        print(f"DEBUG: Final balances range: ${np.min(final_balances):,.2f} to ${np.max(final_balances):,.2f}")
        
        # Calculate max drawdowns for each simulation
        max_drawdowns = []
        for result in simulation_results:
            cumulative_pnl = np.array(result['cumulative_pnl'])
            account_balance = initial_balance + cumulative_pnl
            max_drawdown = self._calculate_simulation_max_drawdown(account_balance)
            max_drawdowns.append(max_drawdown)
        
        max_drawdowns = np.array(max_drawdowns)
        
        # Calculate percentiles
        pnl_percentiles = {}
        balance_percentiles = {}
        drawdown_percentiles = {}
        confidence_levels = [0.05, 0.25, 0.5, 0.75, 0.95]
        
        for level in confidence_levels:
            percentile = level * 100
            pnl_percentiles[f'p{int(percentile)}'] = np.percentile(final_pnls, percentile)
            balance_percentiles[f'p{int(percentile)}'] = np.percentile(final_balances, percentile)
            drawdown_percentiles[f'p{int(percentile)}'] = np.percentile(max_drawdowns, percentile)
        
        # Calculate additional statistics
        win_probability = (final_pnls > 0).mean() * 100
        loss_probability = (final_pnls < 0).mean() * 100
        
        # Calculate risk-adjusted ratios
        risk_ratios = self._calculate_risk_ratios(final_pnls, final_balances, strategy.trades, trade_size_percent)
        
        # Calculate t-test statistics
        t_test_results = self._calculate_t_tests(final_pnls, final_balances, initial_balance)
        
        # Calculate p-values for statistical significance
        p_values = self._calculate_p_values(final_pnls, final_balances, max_drawdowns, initial_balance)
        
        return {
            'strategy_name': strategy_name,
            'simulation_summary': {
                'num_simulations': num_simulations,
                'num_trades_per_sim': num_trades,
                'initial_balance': initial_balance,
                'historical_pnl': strategy.total_pnl,
                'historical_trade_count': len(strategy.trades)
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
                'mean': np.mean(final_pnls) / num_trades,  # Average per-lot P&L per trade
                'median': np.median(final_pnls) / num_trades,  # Median per-lot P&L per trade
                'std': np.std(final_pnls) / num_trades,  # Standard deviation per-lot P&L per trade
                'min': np.min(final_pnls) / num_trades,  # Min per-lot P&L per trade
                'max': np.max(final_pnls) / num_trades,  # Max per-lot P&L per trade
                'percentiles': {k: v / num_trades for k, v in pnl_percentiles.items()}  # Percentiles of per-lot P&L per trade
            },
            'drawdown_stats': {
                'mean': np.mean(max_drawdowns),  # Average per-lot max drawdown across simulations
                'median': np.median(max_drawdowns),  # Median per-lot max drawdown across simulations
                'std': np.std(max_drawdowns),  # Standard deviation of per-lot max drawdown across simulations
                'min': np.min(max_drawdowns),  # Min per-lot max drawdown across simulations
                'max': np.max(max_drawdowns),  # Max per-lot max drawdown across simulations
                'percentiles': drawdown_percentiles  # Percentiles of per-lot max drawdown across simulations
            },
            'probabilities': {
                'win_probability': win_probability,
                'loss_probability': loss_probability,
                'break_even_probability': 100 - win_probability - loss_probability
            },
            'risk_ratios': risk_ratios,
            't_test_results': t_test_results,
            'p_values': p_values,
            'simulation_results': simulation_results[:100],  # Return first 100 for charts
            'confidence_levels': confidence_levels
        }
    
    def run_all_strategies_simulation(self, num_simulations: int = 1000, num_trades: int = None, 
                                    trade_size_percent: float = 1.0) -> Dict:
        """
        Run Monte Carlo simulation for each strategy individually to identify fragility.
        
        Args:
            num_simulations: Number of simulation runs per strategy
            num_trades: Number of trades per simulation (default: strategy's historical count)
            trade_size_percent: Percentage of account balance to risk per trade (default: 1.0%)
        
        Returns:
            Dictionary with results for each strategy and comparative analysis
        """
        if not self.portfolio.strategies:
            return {'error': 'No strategies available'}
        
        strategy_results = {}
        fragility_analysis = {}
        
        # Run simulation for each strategy
        for strategy_name in self.portfolio.strategies.keys():
            strategy_results[strategy_name] = self.run_strategy_specific_simulation(
                strategy_name, num_simulations, num_trades, trade_size_percent
            )
        
        # Analyze fragility across strategies
        fragility_metrics = self._analyze_strategy_fragility(strategy_results)
        
        return {
            'strategy_results': strategy_results,
            'fragility_analysis': fragility_metrics,
            'summary': {
                'total_strategies': len(self.portfolio.strategies),
                'num_simulations_per_strategy': num_simulations,
                'analysis_timestamp': pd.Timestamp.now().isoformat()
            }
        }
    
    def _analyze_strategy_fragility(self, strategy_results: Dict) -> Dict:
        """
        Analyze strategy fragility by comparing individual strategy performance.
        
        Args:
            strategy_results: Results from individual strategy simulations
            
        Returns:
            Dictionary with fragility analysis metrics
        """
        fragility_metrics = {}
        
        # Collect key metrics from each strategy
        win_probabilities = []
        sharpe_ratios = []
        max_drawdowns = []
        pnl_volatilities = []
        
        for strategy_name, results in strategy_results.items():
            if 'probabilities' in results:
                win_probabilities.append(results['probabilities']['win_probability'])
            if 'risk_ratios' in results and results['risk_ratios'].get('sharpe_ratio'):
                sharpe_ratios.append(results['risk_ratios']['sharpe_ratio'])
            if 'drawdown_stats' in results:
                max_drawdowns.append(results['drawdown_stats']['max'])
            if 'pnl_stats' in results:
                pnl_volatilities.append(results['pnl_stats']['std'])
        
        # Calculate fragility indicators
        if win_probabilities:
            fragility_metrics['win_probability_range'] = {
                'min': min(win_probabilities),
                'max': max(win_probabilities),
                'std': np.std(win_probabilities),
                'coefficient_of_variation': np.std(win_probabilities) / np.mean(win_probabilities) if np.mean(win_probabilities) > 0 else 0
            }
        
        if sharpe_ratios:
            sharpe_ratios = [r for r in sharpe_ratios if r is not None]
            if sharpe_ratios:
                fragility_metrics['sharpe_ratio_range'] = {
                    'min': min(sharpe_ratios),
                    'max': max(sharpe_ratios),
                    'std': np.std(sharpe_ratios),
                    'coefficient_of_variation': np.std(sharpe_ratios) / np.mean(sharpe_ratios) if np.mean(sharpe_ratios) > 0 else 0
                }
        
        if max_drawdowns:
            fragility_metrics['drawdown_range'] = {
                'min': min(max_drawdowns),
                'max': max(max_drawdowns),
                'std': np.std(max_drawdowns),
                'coefficient_of_variation': np.std(max_drawdowns) / np.mean(max_drawdowns) if np.mean(max_drawdowns) > 0 else 0
            }
        
        if pnl_volatilities:
            fragility_metrics['volatility_range'] = {
                'min': min(pnl_volatilities),
                'max': max(pnl_volatilities),
                'std': np.std(pnl_volatilities),
                'coefficient_of_variation': np.std(pnl_volatilities) / np.mean(pnl_volatilities) if np.mean(pnl_volatilities) > 0 else 0
            }
        
        # Create rankings based on established risk-adjusted metrics
        sharpe_ratios = {}
        sortino_ratios = {}
        
        for strategy_name, results in strategy_results.items():
            # Get Sharpe ratio
            if 'risk_ratios' in results and results['risk_ratios'].get('sharpe_ratio') is not None:
                sharpe_ratios[strategy_name] = results['risk_ratios']['sharpe_ratio']
            
            # Get Sortino ratio (include None values as they represent infinite Sortino ratios)
            if 'risk_ratios' in results and 'sortino_ratio' in results['risk_ratios']:
                sortino_ratios[strategy_name] = results['risk_ratios']['sortino_ratio']
        
        # Sort by Sharpe ratio (descending - higher is better)
        sharpe_ranking = sorted(sharpe_ratios.items(), key=lambda x: x[1], reverse=True)
        
        # Sort by Sortino ratio (descending - higher is better, None values = infinite = best)
        def sortino_sort_key(item):
            value = item[1]
            if value is None:  # None represents infinite Sortino ratio
                return float('inf')
            return value
        
        sortino_ranking = sorted(sortino_ratios.items(), key=sortino_sort_key, reverse=True)
        
        fragility_metrics['sharpe_ranking'] = sharpe_ranking
        fragility_metrics['sortino_ranking'] = sortino_ranking
        fragility_metrics['best_sharpe'] = sharpe_ranking[0][0] if sharpe_ranking else None
        fragility_metrics['worst_sharpe'] = sharpe_ranking[-1][0] if sharpe_ranking else None
        fragility_metrics['best_sortino'] = sortino_ranking[0][0] if sortino_ranking else None
        fragility_metrics['worst_sortino'] = sortino_ranking[-1][0] if sortino_ranking else None
        
        return fragility_metrics
    
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
    
    def _calculate_risk_ratios(self, total_pnls: np.ndarray, final_balances: np.ndarray, strategy_trades: List = None, trade_size_percent: float = 1.0) -> Dict:
        """
        Calculate risk-adjusted performance ratios for Monte Carlo results.
        
        Args:
            total_pnls: Array of total P&L values from simulations
            final_balances: Array of final balance values from simulations
            strategy_trades: List of actual strategy trades for risk calculation
            trade_size_percent: Percentage of account balance to risk per trade
            
        Returns:
            Dictionary with Sharpe ratio, Sortino ratio, and other risk metrics
        """
        print(f"DEBUG: _calculate_risk_ratios called with strategy_trades: {strategy_trades is not None}")
        if strategy_trades:
            print(f"DEBUG: Number of strategy trades: {len(strategy_trades)}")
        else:
            print("DEBUG: No strategy trades provided, using portfolio-level calculation")
        
        # Use standard 2% annual risk-free rate (Treasury bill rate)
        risk_free_rate = 0.02
        
        print(f"DEBUG: Input parameters:")
        print(f"DEBUG: Total P&Ls shape: {total_pnls.shape if hasattr(total_pnls, 'shape') else len(total_pnls)}")
        print(f"DEBUG: Final balances shape: {final_balances.shape if hasattr(final_balances, 'shape') else len(final_balances)}")
        print(f"DEBUG: Trade size percent: {trade_size_percent}%")
        print(f"DEBUG: Risk-free rate: {risk_free_rate*100:.4f}%")
        
        try:
            
            # For strategy-specific analysis, use time-normalized annualized returns
            # This gives us proper Sharpe/Sortino ratios that account for compounding and time
            if strategy_trades:
                # Sort trades by date to ensure chronological order
                sorted_trades = sorted(strategy_trades, key=lambda t: pd.to_datetime(t.date_closed))
                
                # Calculate time span and trade frequency for proper annualization
                start_date = pd.to_datetime(sorted_trades[0].date_closed)
                end_date = pd.to_datetime(sorted_trades[-1].date_closed)
                days_elapsed = (end_date - start_date).days
                
                print(f"DEBUG: Start date: {start_date}, End date: {end_date}")
                print(f"DEBUG: Days elapsed: {days_elapsed}")
                
                if days_elapsed > 0:
                    print("DEBUG: Using time-normalized annualized returns calculation")
                    # Calculate trades per year
                    trades_per_year = len(sorted_trades) * 365.25 / days_elapsed
                    
                    # Calculate actual position sizing and compounding for each trade
                    # Scale the position sizing by the trade size percentage
                    trade_returns = []
                    current_balance = sorted_trades[0].funds_at_close - sorted_trades[0].pnl  # Initial balance
                    
                    print(f"DEBUG: Strategy {sorted_trades[0].strategy if hasattr(sorted_trades[0], 'strategy') else 'Unknown'}")
                    print(f"DEBUG: Initial balance: ${current_balance:,.2f}")
                    print(f"DEBUG: Number of trades: {len(sorted_trades)}")
                    
                    for i, trade in enumerate(sorted_trades[:5]):  # Debug first 5 trades
                        # Calculate margin requirement for this trade
                        margin_req = getattr(trade, 'margin_req', 0)
                        if margin_req == 0:
                            # Fallback: estimate margin from contracts and margin_per_contract
                            margin_per_contract = getattr(trade, 'margin_per_contract', 1000.0)
                            margin_req = trade.contracts * margin_per_contract
                        
                        # Calculate position size as percentage of current balance
                        # Scale by the trade size percentage parameter
                        if current_balance > 0 and margin_req > 0:
                            # Use the specified trade size percentage instead of actual margin requirement
                            position_size_pct = trade_size_percent / 100.0
                            # Calculate return based on actual margin requirement and position sizing
                            # This gives us the true return on the capital at risk
                            trade_return = (trade.pnl / margin_req) * position_size_pct
                        else:
                            # Fallback to per-lot calculation scaled by trade size
                            trade_return = (trade.pnl_per_lot / 1000.0) * (trade_size_percent / 100.0)
                        
                        print(f"DEBUG: Trade {i+1}: Balance=${current_balance:,.2f}, Margin=${margin_req:,.2f}, P&L=${trade.pnl:,.2f}, Position%={position_size_pct*100:.2f}%, Return={trade_return*100:.4f}%")
                        
                        trade_returns.append(trade_return)
                        
                        # Update balance for next trade (compounding effect)
                        current_balance = trade.funds_at_close
                    
                    # Calculate middle trades without debug
                    for trade in sorted_trades[5:-5]:
                        margin_req = getattr(trade, 'margin_req', 0)
                        if margin_req == 0:
                            margin_per_contract = getattr(trade, 'margin_per_contract', 1000.0)
                            margin_req = trade.contracts * margin_per_contract
                        
                        # Use the specified trade size percentage
                        position_size_pct = trade_size_percent / 100.0
                        # Calculate return based on actual margin requirement and position sizing
                        trade_return = (trade.pnl / margin_req) * position_size_pct
                        
                        trade_returns.append(trade_return)
                        current_balance = trade.funds_at_close
                    
                    # Debug last 5 trades
                    print("DEBUG: Last 5 trades:")
                    for i, trade in enumerate(sorted_trades[-5:]):
                        margin_req = getattr(trade, 'margin_req', 0)
                        if margin_req == 0:
                            margin_per_contract = getattr(trade, 'margin_per_contract', 1000.0)
                            margin_req = trade.contracts * margin_per_contract
                        
                        # Use the specified trade size percentage
                        position_size_pct = trade_size_percent / 100.0
                        # Calculate return based on actual margin requirement and position sizing
                        trade_return = (trade.pnl / margin_req) * position_size_pct
                        
                        print(f"DEBUG: Trade {len(sorted_trades)-4+i}: Balance=${current_balance:,.2f}, Margin=${margin_req:,.2f}, P&L=${trade.pnl:,.2f}, Position%={position_size_pct*100:.2f}%, Return={trade_return*100:.4f}%")
                        
                        trade_returns.append(trade_return)
                        current_balance = trade.funds_at_close
                    
                    # Use the actual individual trade returns for risk calculation
                    # This gives us the true risk profile of the strategy
                    returns = np.array(trade_returns)
                    
                    # Scale risk-free rate to match trade frequency
                    # If we have 50 trades per year, each trade represents 1/50th of a year
                    trade_period_risk_free_rate = risk_free_rate / trades_per_year
                    
                    print(f"DEBUG: Mean return per trade: {np.mean(trade_returns)*100:.4f}%")
                    print(f"DEBUG: Trades per year: {trades_per_year:.2f}")
                    print(f"DEBUG: Annual return: {np.mean(trade_returns)*trades_per_year*100:.2f}%")
                    print(f"DEBUG: Annual volatility: {np.std(trade_returns)*np.sqrt(trades_per_year)*100:.2f}%")
                    print(f"DEBUG: Trade period risk-free rate: {trade_period_risk_free_rate*100:.4f}%")
                else:
                    print("DEBUG: Using fallback per-lot returns calculation (days_elapsed <= 0)")
                    # Fallback to per-lot returns if no time data
                    initial_balance = 1000.0
                    returns = np.array([trade.pnl_per_lot / initial_balance for trade in strategy_trades])
            else:
                # For portfolio-level analysis, use the final P&L from each simulation
                initial_balance = 1000.0  # Standard per-lot initial balance
                returns = total_pnls / initial_balance  # Convert to percentage returns
            
            # Calculate mean return and standard deviation
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            print(f"DEBUG: Basic return statistics:")
            print(f"DEBUG: Mean return: {mean_return*100:.6f}%")
            print(f"DEBUG: Standard deviation: {std_return*100:.6f}%")
            print(f"DEBUG: Number of returns: {len(returns)}")
            print(f"DEBUG: Min return: {np.min(returns)*100:.6f}%")
            print(f"DEBUG: Max return: {np.max(returns)*100:.6f}%")
            
            # Sharpe Ratio: (Annual Return - Risk Free Rate) / Annual Volatility
            print(f"DEBUG: === SHARPE RATIO CALCULATION ===")
            if std_return > 0:
                if strategy_trades and days_elapsed > 0:
                    # Calculate annualized metrics for proper Sharpe ratio
                    trades_per_year = len(strategy_trades) * 365.25 / days_elapsed
                    annual_return = mean_return * trades_per_year
                    annual_volatility = std_return * np.sqrt(trades_per_year)
                    excess_return = annual_return - risk_free_rate
                    sharpe_ratio = excess_return / annual_volatility
                    
                    print(f"DEBUG: Strategy-specific Sharpe calculation:")
                    print(f"DEBUG: Trades per year: {trades_per_year:.2f}")
                    print(f"DEBUG: Annual return: {annual_return*100:.4f}%")
                    print(f"DEBUG: Annual volatility: {annual_volatility*100:.4f}%")
                    print(f"DEBUG: Risk-free rate: {risk_free_rate*100:.4f}%")
                    print(f"DEBUG: Excess return: {excess_return*100:.4f}%")
                    print(f"DEBUG: Sharpe ratio: {sharpe_ratio:.4f}")
                else:
                    # For portfolio-level, use the returns as-is (already annualized)
                    excess_return = mean_return - risk_free_rate
                    sharpe_ratio = excess_return / std_return
                    
                    print(f"DEBUG: Portfolio-level Sharpe calculation:")
                    print(f"DEBUG: Mean return: {mean_return*100:.4f}%")
                    print(f"DEBUG: Standard deviation: {std_return*100:.4f}%")
                    print(f"DEBUG: Risk-free rate: {risk_free_rate*100:.4f}%")
                    print(f"DEBUG: Excess return: {excess_return*100:.4f}%")
                    print(f"DEBUG: Sharpe ratio: {sharpe_ratio:.4f}")
            else:
                sharpe_ratio = 0
                print(f"DEBUG: Sharpe ratio = 0 (standard deviation is 0)")
            
            # Sortino Ratio: (Annual Return - Risk Free Rate) / Annual Downside Deviation
            # Downside deviation only considers negative returns
            negative_returns = returns[returns < 0]
            
            print(f"DEBUG: === SORTINO RATIO CALCULATION ===")
            print(f"DEBUG: Total returns: {len(returns)}")
            print(f"DEBUG: Negative returns: {len(negative_returns)}")
            if len(negative_returns) > 0:
                print(f"DEBUG: Negative returns range: {np.min(negative_returns)*100:.6f}% to {np.max(negative_returns)*100:.6f}%")
                print(f"DEBUG: Mean negative return: {np.mean(negative_returns)*100:.6f}%")
                print(f"DEBUG: Std dev of negative returns: {np.std(negative_returns)*100:.6f}%")
            else:
                print(f"DEBUG: No negative returns found - perfect downside protection")
            
            if strategy_trades and days_elapsed > 0:
                # Calculate annualized metrics for proper Sortino ratio
                trades_per_year = len(strategy_trades) * 365.25 / days_elapsed
                annual_return = mean_return * trades_per_year
                
                print(f"DEBUG: Strategy-specific Sortino calculation:")
                print(f"DEBUG: Trades per year: {trades_per_year:.2f}")
                print(f"DEBUG: Annual return: {annual_return*100:.4f}%")
                print(f"DEBUG: Risk-free rate: {risk_free_rate*100:.4f}%")
                
                if len(negative_returns) > 1:  # Need at least 2 negative returns for meaningful std dev
                    downside_deviation = np.std(negative_returns) * np.sqrt(trades_per_year)  # Annualize downside deviation
                    if downside_deviation > 0:
                        sortino_ratio = (annual_return - risk_free_rate) / downside_deviation
                        print(f"DEBUG: Multiple negative returns case:")
                        print(f"DEBUG: Downside deviation (annualized): {downside_deviation*100:.4f}%")
                        print(f"DEBUG: Sortino ratio: {sortino_ratio:.4f}")
                    else:
                        # All negative returns are identical - use a small value to avoid division by zero
                        sortino_ratio = (annual_return - risk_free_rate) / 0.001
                        print(f"DEBUG: Identical negative returns case (using 0.001 divisor):")
                        print(f"DEBUG: Sortino ratio: {sortino_ratio:.4f}")
                elif len(negative_returns) == 1:
                    # Only one negative return - use its absolute value as downside deviation
                    downside_deviation = abs(negative_returns[0]) * np.sqrt(trades_per_year)  # Annualize
                    sortino_ratio = (annual_return - risk_free_rate) / downside_deviation
                    print(f"DEBUG: Single negative return case:")
                    print(f"DEBUG: Single negative return: {negative_returns[0]*100:.6f}%")
                    print(f"DEBUG: Downside deviation (annualized): {downside_deviation*100:.4f}%")
                    print(f"DEBUG: Sortino ratio: {sortino_ratio:.4f}")
                else:
                    # If no negative returns, Sortino ratio is infinite (perfect downside protection)
                    sortino_ratio = float('inf') if annual_return > risk_free_rate else 0
                    print(f"DEBUG: No negative returns case:")
                    print(f"DEBUG: Sortino ratio: {'∞' if sortino_ratio == float('inf') else sortino_ratio}")
            else:
                # For portfolio-level, use the returns as-is (already annualized)
                print(f"DEBUG: Portfolio-level Sortino calculation:")
                print(f"DEBUG: Mean return: {mean_return*100:.4f}%")
                print(f"DEBUG: Risk-free rate: {risk_free_rate*100:.4f}%")
                
                if len(negative_returns) > 1:
                    downside_deviation = np.std(negative_returns)
                    if downside_deviation > 0:
                        sortino_ratio = (mean_return - risk_free_rate) / downside_deviation
                        print(f"DEBUG: Multiple negative returns case:")
                        print(f"DEBUG: Downside deviation: {downside_deviation*100:.4f}%")
                        print(f"DEBUG: Sortino ratio: {sortino_ratio:.4f}")
                    else:
                        sortino_ratio = (mean_return - risk_free_rate) / 0.001
                        print(f"DEBUG: Identical negative returns case (using 0.001 divisor):")
                        print(f"DEBUG: Sortino ratio: {sortino_ratio:.4f}")
                elif len(negative_returns) == 1:
                    downside_deviation = abs(negative_returns[0])
                    sortino_ratio = (mean_return - risk_free_rate) / downside_deviation
                    print(f"DEBUG: Single negative return case:")
                    print(f"DEBUG: Single negative return: {negative_returns[0]*100:.6f}%")
                    print(f"DEBUG: Downside deviation: {downside_deviation*100:.4f}%")
                    print(f"DEBUG: Sortino ratio: {sortino_ratio:.4f}")
                else:
                    sortino_ratio = float('inf') if mean_return > risk_free_rate else 0
                    print(f"DEBUG: No negative returns case:")
                    print(f"DEBUG: Sortino ratio: {'∞' if sortino_ratio == float('inf') else sortino_ratio}")
            
            # Calmar Ratio: Annual Return / Maximum Drawdown
            # For Monte Carlo, we'll use the mean return and mean max drawdown
            max_drawdowns = []
            for balance in final_balances:
                # Calculate drawdown for this simulation
                initial = balance - total_pnls[final_balances == balance][0]
                running_max = np.maximum.accumulate([initial] + [balance])
                drawdown = (running_max[-1] - balance) / running_max[-1] * 100
                max_drawdowns.append(drawdown)
            
            mean_max_drawdown = np.mean(max_drawdowns)
            if mean_max_drawdown > 0:
                calmar_ratio = mean_return / mean_max_drawdown
            else:
                calmar_ratio = float('inf') if mean_return > 0 else 0
            
            # Information Ratio: (Portfolio Return - Benchmark Return) / Tracking Error
            # For this calculation, we'll use 0 as benchmark (cash)
            benchmark_return = 0
            excess_returns = returns - benchmark_return
            tracking_error = np.std(excess_returns)
            
            if tracking_error > 0:
                information_ratio = np.mean(excess_returns) / tracking_error
            else:
                information_ratio = 0
            
            # Return to Risk Ratio: Mean Return / Standard Deviation
            return_to_risk = mean_return / std_return if std_return > 0 else 0
            
            # Handle infinity values for JSON serialization
            def safe_float(value):
                if value == float('inf') or value == float('-inf'):
                    return None
                return value
            
            return {
                'sharpe_ratio': safe_float(sharpe_ratio),
                'sortino_ratio': safe_float(sortino_ratio),
                'calmar_ratio': safe_float(calmar_ratio),
                'information_ratio': safe_float(information_ratio),
                'return_to_risk': safe_float(return_to_risk),
                'mean_return_pct': mean_return,
                'std_return_pct': std_return,
                'downside_deviation_pct': np.std(negative_returns) if len(negative_returns) > 0 else 0,
                'mean_max_drawdown_pct': mean_max_drawdown
            }
            
        except Exception as e:
            # Return default values if calculation fails
            return {
                'sharpe_ratio': 0,
                'sortino_ratio': 0,
                'calmar_ratio': 0,
                'information_ratio': 0,
                'return_to_risk': 0,
                'mean_return_pct': 0,
                'std_return_pct': 0,
                'downside_deviation_pct': 0,
                'mean_max_drawdown_pct': 0
            }
    
    def _calculate_t_tests(self, total_pnls: np.ndarray, final_balances: np.ndarray, 
                          initial_balance: float) -> Dict:
        """
        Calculate t-test statistics for Monte Carlo results.
        
        Args:
            total_pnls: Array of total P&L values from simulations
            final_balances: Array of final balance values from simulations
            initial_balance: Initial account balance
            
        Returns:
            Dictionary with t-test statistics and interpretations
        """
        try:
            from scipy import stats
            
            t_test_results = {}
            
            # 1. One-sample t-test for profitability (H0: mean P&L = 0)
            if len(total_pnls) > 1 and np.std(total_pnls) > 0:
                t_stat_profit, p_val_profit = stats.ttest_1samp(total_pnls, 0)
                t_test_results['profitability'] = {
                    't_statistic': t_stat_profit,
                    'p_value': p_val_profit,
                    'degrees_of_freedom': len(total_pnls) - 1,
                    'hypothesis': 'H0: mean P&L = 0, H1: mean P&L ≠ 0',
                    'interpretation': self._interpret_t_test(t_stat_profit, p_val_profit, 'profitability')
                }
            else:
                t_test_results['profitability'] = None
            
            # 2. One-sample t-test for outperforming initial balance (H0: mean final balance = initial balance)
            if len(final_balances) > 1 and np.std(final_balances) > 0:
                t_stat_balance, p_val_balance = stats.ttest_1samp(final_balances, initial_balance)
                t_test_results['outperform_initial'] = {
                    't_statistic': t_stat_balance,
                    'p_value': p_val_balance,
                    'degrees_of_freedom': len(final_balances) - 1,
                    'hypothesis': 'H0: mean final balance = initial balance, H1: mean final balance ≠ initial balance',
                    'interpretation': self._interpret_t_test(t_stat_balance, p_val_balance, 'outperform_initial')
                }
            else:
                t_test_results['outperform_initial'] = None
            
            # 3. One-sample t-test for positive returns (H0: mean return = 0)
            returns = (total_pnls / initial_balance) * 100  # Convert to percentage returns
            if len(returns) > 1 and np.std(returns) > 0:
                t_stat_return, p_val_return = stats.ttest_1samp(returns, 0)
                t_test_results['positive_returns'] = {
                    't_statistic': t_stat_return,
                    'p_value': p_val_return,
                    'degrees_of_freedom': len(returns) - 1,
                    'hypothesis': 'H0: mean return = 0%, H1: mean return ≠ 0%',
                    'interpretation': self._interpret_t_test(t_stat_return, p_val_return, 'positive_returns')
                }
            else:
                t_test_results['positive_returns'] = None
            
            # 4. Two-sample t-test comparing to risk-free rate (H0: mean return = risk_free_rate)
            risk_free_rate = 2.0  # 2% annual risk-free rate
            if len(returns) > 1 and np.std(returns) > 0:
                t_stat_risk_free, p_val_risk_free = stats.ttest_1samp(returns, risk_free_rate)
                t_test_results['vs_risk_free'] = {
                    't_statistic': t_stat_risk_free,
                    'p_value': p_val_risk_free,
                    'degrees_of_freedom': len(returns) - 1,
                    'hypothesis': f'H0: mean return = {risk_free_rate}%, H1: mean return ≠ {risk_free_rate}%',
                    'interpretation': self._interpret_t_test(t_stat_risk_free, p_val_risk_free, 'vs_risk_free')
                }
            else:
                t_test_results['vs_risk_free'] = None
            
            # 5. Paired t-test for consistency (comparing first half vs second half of simulations)
            if len(total_pnls) > 10:  # Need enough data for meaningful comparison
                mid_point = len(total_pnls) // 2
                first_half = total_pnls[:mid_point]
                second_half = total_pnls[mid_point:]
                
                if len(first_half) > 1 and len(second_half) > 1:
                    t_stat_consistency, p_val_consistency = stats.ttest_rel(first_half, second_half)
                    t_test_results['consistency'] = {
                        't_statistic': t_stat_consistency,
                        'p_value': p_val_consistency,
                        'degrees_of_freedom': len(first_half) - 1,
                        'hypothesis': 'H0: first half = second half, H1: first half ≠ second half',
                        'interpretation': self._interpret_t_test(t_stat_consistency, p_val_consistency, 'consistency')
                    }
                else:
                    t_test_results['consistency'] = None
            else:
                t_test_results['consistency'] = None
            
            return t_test_results
            
        except ImportError:
            # If scipy is not available, return None for all t-tests
            return {
                'profitability': None,
                'outperform_initial': None,
                'positive_returns': None,
                'vs_risk_free': None,
                'consistency': None
            }
        except Exception as e:
            # Return None for all t-tests if calculation fails
            return {
                'profitability': None,
                'outperform_initial': None,
                'positive_returns': None,
                'vs_risk_free': None,
                'consistency': None
            }
    
    def _interpret_t_test(self, t_statistic: float, p_value: float, test_type: str) -> str:
        """
        Interpret t-test results based on test type and significance.
        
        Args:
            t_statistic: The calculated t-statistic
            p_value: The p-value from the test
            test_type: Type of test being performed
            
        Returns:
            String interpretation of the test results
        """
        if p_value < 0.001:
            significance = "highly significant (p < 0.001)"
        elif p_value < 0.01:
            significance = "very significant (p < 0.01)"
        elif p_value < 0.05:
            significance = "significant (p < 0.05)"
        elif p_value < 0.1:
            significance = "marginally significant (p < 0.1)"
        else:
            significance = "not significant (p ≥ 0.1)"
        
        if test_type == 'profitability':
            if t_statistic > 0 and p_value < 0.05:
                return f"Strategy is significantly profitable ({significance})"
            elif t_statistic < 0 and p_value < 0.05:
                return f"Strategy is significantly unprofitable ({significance})"
            else:
                return f"Strategy profitability is not statistically significant ({significance})"
        
        elif test_type == 'outperform_initial':
            if t_statistic > 0 and p_value < 0.05:
                return f"Strategy significantly outperforms initial balance ({significance})"
            elif t_statistic < 0 and p_value < 0.05:
                return f"Strategy significantly underperforms initial balance ({significance})"
            else:
                return f"Strategy performance vs initial balance is not significant ({significance})"
        
        elif test_type == 'positive_returns':
            if t_statistic > 0 and p_value < 0.05:
                return f"Strategy has significantly positive returns ({significance})"
            elif t_statistic < 0 and p_value < 0.05:
                return f"Strategy has significantly negative returns ({significance})"
            else:
                return f"Strategy returns are not significantly different from zero ({significance})"
        
        elif test_type == 'vs_risk_free':
            if t_statistic > 0 and p_value < 0.05:
                return f"Strategy significantly outperforms risk-free rate ({significance})"
            elif t_statistic < 0 and p_value < 0.05:
                return f"Strategy significantly underperforms risk-free rate ({significance})"
            else:
                return f"Strategy performance vs risk-free rate is not significant ({significance})"
        
        elif test_type == 'consistency':
            if p_value < 0.05:
                return f"Strategy shows significant inconsistency between periods ({significance})"
            else:
                return f"Strategy shows consistent performance across periods ({significance})"
        
        else:
            return f"Test result: {significance}"
    
    def _calculate_p_values(self, total_pnls: np.ndarray, final_balances: np.ndarray, 
                           max_drawdowns: np.ndarray, initial_balance: float) -> Dict:
        """
        Calculate p-values for various statistical tests on Monte Carlo results.
        
        Args:
            total_pnls: Array of total P&L values from simulations
            final_balances: Array of final balance values from simulations
            max_drawdowns: Array of maximum drawdown values from simulations
            initial_balance: Initial account balance
            
        Returns:
            Dictionary with p-values for different statistical tests
        """
        try:
            from scipy import stats
            
            p_values = {}
            
            # 1. P-value for profitability (one-sample t-test: H0: mean P&L <= 0)
            if len(total_pnls) > 1 and np.std(total_pnls) > 0:
                t_stat_profit, p_values['profitability'] = stats.ttest_1samp(total_pnls, 0)
                # For one-tailed test (H0: mean <= 0, H1: mean > 0)
                if t_stat_profit > 0:
                    p_values['profitability'] = p_values['profitability'] / 2
                else:
                    p_values['profitability'] = 1 - (p_values['profitability'] / 2)
            else:
                p_values['profitability'] = None
            
            # 2. P-value for outperforming initial balance (one-sample t-test: H0: mean final balance <= initial balance)
            if len(final_balances) > 1 and np.std(final_balances) > 0:
                t_stat_balance, p_values['outperform_initial'] = stats.ttest_1samp(final_balances, initial_balance)
                # For one-tailed test (H0: mean <= initial, H1: mean > initial)
                if t_stat_balance > 0:
                    p_values['outperform_initial'] = p_values['outperform_initial'] / 2
                else:
                    p_values['outperform_initial'] = 1 - (p_values['outperform_initial'] / 2)
            else:
                p_values['outperform_initial'] = None
            
            # 3. P-value for Sharpe ratio significance (if we can calculate it)
            if len(total_pnls) > 1 and np.std(total_pnls) > 0 and np.mean(total_pnls) != 0:
                # Calculate Sharpe ratio (assuming risk-free rate = 0 for simplicity)
                sharpe_ratio = np.mean(total_pnls) / np.std(total_pnls)
                # Test if Sharpe ratio is significantly different from 0
                t_stat_sharpe, p_values['sharpe_ratio'] = stats.ttest_1samp(total_pnls, 0)
                p_values['sharpe_ratio'] = p_values['sharpe_ratio'] / 2  # One-tailed test
            else:
                p_values['sharpe_ratio'] = None
            
            # 4. P-value for drawdown risk (test if mean drawdown is significantly different from 0)
            if len(max_drawdowns) > 1 and np.std(max_drawdowns) > 0:
                t_stat_drawdown, p_values['drawdown_risk'] = stats.ttest_1samp(max_drawdowns, 0)
                # For one-tailed test (H0: mean drawdown <= 0, H1: mean drawdown > 0)
                if t_stat_drawdown > 0:
                    p_values['drawdown_risk'] = p_values['drawdown_risk'] / 2
                else:
                    p_values['drawdown_risk'] = 1 - (p_values['drawdown_risk'] / 2)
            else:
                p_values['drawdown_risk'] = None
            
            # 5. P-value for consistency (test if variance is significantly different from a benchmark)
            # This tests if the strategy has consistent returns (lower variance = more consistent)
            if len(total_pnls) > 1:
                # Chi-square test for variance (H0: variance = benchmark_variance)
                # Using a reasonable benchmark variance (e.g., 10% of mean absolute P&L)
                benchmark_variance = (np.mean(np.abs(total_pnls)) * 0.1) ** 2
                if benchmark_variance > 0:
                    chi2_stat = (len(total_pnls) - 1) * np.var(total_pnls) / benchmark_variance
                    p_values['consistency'] = 1 - stats.chi2.cdf(chi2_stat, len(total_pnls) - 1)
                else:
                    p_values['consistency'] = None
            else:
                p_values['consistency'] = None
            
            return p_values
            
        except ImportError:
            # If scipy is not available, return None for all p-values
            return {
                'profitability': None,
                'outperform_initial': None,
                'sharpe_ratio': None,
                'drawdown_risk': None,
                'consistency': None
            }
        except Exception as e:
            # Return None for all p-values if calculation fails
            return {
                'profitability': None,
                'outperform_initial': None,
                'sharpe_ratio': None,
                'drawdown_risk': None,
                'consistency': None
            }
    
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
            'risk_ratios': {},
            't_test_results': {},
            'p_values': {},
            'simulation_results': [],
            'confidence_levels': []
        }


class CrossFileAnalyzer:
    """Analyzes correlations between strategies across multiple uploaded files."""
    
    def __init__(self, file_manager):
        self.file_manager = file_manager
    
    def get_cross_file_correlations(self, max_pairs: int = 30, selected_files: List[str] = None) -> Dict:
        """
        Calculate correlations between strategies across selected uploaded files.
        
        Args:
            max_pairs: Maximum number of uncorrelated pairs to return
            selected_files: List of filenames to analyze (if None, analyzes all files)
            
        Returns:
            Dictionary with correlation analysis results
        """
        try:
            # Get uploaded files (filtered by selection if provided)
            all_files = self.file_manager.get_file_list()
            
            if selected_files:
                # Filter to only selected files
                file_list = [f for f in all_files if f['filename'] in selected_files]
                print(f"DEBUG: Filtering to {len(file_list)} selected files out of {len(all_files)} total files")
            else:
                # Use all files (backward compatibility)
                file_list = all_files
                print(f"DEBUG: Using all {len(file_list)} files for analysis")
            
            if len(file_list) < 2:
                return {
                    'success': False,
                    'error': 'Need at least 2 files for cross-file correlation analysis',
                    'file_count': len(file_list)
                }
            
            # Load all portfolios and their strategies
            portfolios_data = []
            
            print(f"DEBUG: Found {len(file_list)} files to analyze:")
            for file_info in file_list:
                print(f"  - {file_info['friendly_name']} ({file_info['filename']})")
            
            for file_info in file_list:
                try:
                    file_path = self.file_manager.get_file_path(file_info['filename'])
                    print(f"DEBUG: Loading file {file_info['friendly_name']} from {file_path}")
                    
                    # Create temporary portfolio for this file
                    from models import Portfolio
                    temp_portfolio = Portfolio()
                    temp_portfolio.load_from_csv(file_path)
                    
                    print(f"DEBUG: Loaded {len(temp_portfolio.strategies)} strategies from {file_info['friendly_name']}")
                    
                    # Get date range for this file
                    all_trades = []
                    for strategy in temp_portfolio.strategies.values():
                        all_trades.extend(strategy.trades)
                    
                    print(f"DEBUG: Found {len(all_trades)} total trades in {file_info['friendly_name']}")
                    
                    if all_trades:
                        # Debug: Show sample dates before parsing
                        sample_dates = [trade.date_closed for trade in all_trades[:3]]
                        print(f"DEBUG: Sample date_closed values from {file_info['friendly_name']}: {sample_dates}")
                        
                        dates = []
                        for trade in all_trades:
                            try:
                                parsed_date = pd.to_datetime(trade.date_closed)
                                dates.append(parsed_date)
                            except Exception as e:
                                print(f"DEBUG: Failed to parse date '{trade.date_closed}' from {file_info['friendly_name']}: {e}")
                                continue
                        
                        if dates:
                            start_date = min(dates)
                            end_date = max(dates)
                            
                            print(f"DEBUG: Date range for {file_info['friendly_name']}: {start_date.date()} to {end_date.date()}")
                            
                            portfolios_data.append({
                                'filename': file_info['filename'],
                                'friendly_name': file_info['friendly_name'],
                                'portfolio': temp_portfolio,
                                'start_date': start_date,
                                'end_date': end_date,
                                'trade_count': len(all_trades)
                            })
                        else:
                            print(f"DEBUG: No valid dates found in {file_info['friendly_name']}")
                    else:
                        print(f"DEBUG: No trades found in {file_info['friendly_name']}")
                except Exception as e:
                    print(f"Warning: Could not load file {file_info['filename']}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            if len(portfolios_data) < 2:
                return {
                    'success': False,
                    'error': 'Could not load enough valid files for analysis',
                    'loaded_files': len(portfolios_data)
                }
            
            # Find common date range
            common_start = max(p['start_date'] for p in portfolios_data)
            common_end = min(p['end_date'] for p in portfolios_data)
            
            # Debug: Print date ranges for troubleshooting
            print(f"DEBUG: Cross-file correlation date analysis:")
            for p in portfolios_data:
                print(f"  {p['friendly_name']}: {p['start_date'].date()} to {p['end_date'].date()}")
                print(f"    Raw dates: {p['start_date']} to {p['end_date']}")
                print(f"    Date types: {type(p['start_date'])} to {type(p['end_date'])}")
            
            print(f"  Common range: {common_start.date()} to {common_end.date()}")
            print(f"  Raw common dates: {common_start} to {common_end}")
            print(f"  Overlap check: {common_start.date()} > {common_end.date()} = {common_start > common_end}")
            print(f"  Raw comparison: {common_start} > {common_end} = {common_start > common_end}")
            
            # Additional debugging: show the actual values being compared
            print(f"  common_start value: {repr(common_start)}")
            print(f"  common_end value: {repr(common_end)}")
            print(f"  common_start > common_end: {common_start > common_end}")
            print(f"  common_start >= common_end: {common_start >= common_end}")
            print(f"  common_start == common_end: {common_start == common_end}")
            
            if common_start > common_end:
                return {
                    'success': False,
                    'error': 'No overlapping date range found between files',
                    'date_ranges': {p['friendly_name']: f"{p['start_date'].date()} to {p['end_date'].date()}" for p in portfolios_data},
                    'common_start': common_start.strftime('%Y-%m-%d'),
                    'common_end': common_end.strftime('%Y-%m-%d'),
                    'debug_info': 'common_start > common_end means no overlap'
                }
            
            # Calculate daily returns for each strategy within common date range
            strategy_returns = {}
            
            for portfolio_data in portfolios_data:
                portfolio = portfolio_data['portfolio']
                file_prefix = portfolio_data['friendly_name']
                
                for strategy_name, strategy in portfolio.strategies.items():
                    # Filter trades to common date range
                    filtered_trades = []
                    for trade in strategy.trades:
                        trade_date = pd.to_datetime(trade.date_closed)
                        if common_start <= trade_date <= common_end:
                            filtered_trades.append(trade)
                    
                    if len(filtered_trades) < 5:  # Need minimum trades for correlation
                        continue
                    
                    # Create unique strategy identifier
                    unique_strategy_name = f"{file_prefix}::{strategy_name}"
                    
                    # Calculate daily returns using per-lot P&L
                    daily_data = []
                    for trade in filtered_trades:
                        daily_data.append({
                            'date': pd.to_datetime(trade.date_closed).date(),
                            'pnl': trade.pnl_per_lot
                        })
                    
                    if daily_data:
                        df = pd.DataFrame(daily_data)
                        # Group by date and sum P&L (in case multiple trades per day)
                        daily_returns = df.groupby('date')['pnl'].sum()
                        strategy_returns[unique_strategy_name] = daily_returns
            
            if len(strategy_returns) < 2:
                return {
                    'success': False,
                    'error': 'Not enough strategies with sufficient data in common date range',
                    'common_date_range': f"{common_start.date()} to {common_end.date()}",
                    'strategies_found': len(strategy_returns)
                }
            
            # Create correlation matrix
            returns_df = pd.DataFrame(strategy_returns).fillna(0)
            correlation_matrix = returns_df.corr()
            
            # Limit strategies if max_pairs is specified and we have more strategies than the limit
            if max_pairs and len(correlation_matrix.columns) > max_pairs:
                print(f"DEBUG: Limiting correlation matrix to top {max_pairs} most uncorrelated strategies (from {len(correlation_matrix.columns)} total)")
                
                # Find the most uncorrelated strategies by calculating average absolute correlation for each strategy
                strategy_correlations = {}
                for strategy in correlation_matrix.columns:
                    # Get correlations with all other strategies (excluding self-correlation of 1.0)
                    other_correlations = correlation_matrix[strategy].drop(strategy)
                    # Calculate average absolute correlation (lower = more uncorrelated)
                    avg_abs_correlation = other_correlations.abs().mean()
                    strategy_correlations[strategy] = avg_abs_correlation
                
                # Select strategies with lowest average absolute correlation (most uncorrelated)
                most_uncorrelated_strategies = sorted(strategy_correlations.items(), key=lambda x: x[1])[:max_pairs]
                top_strategies = [strategy for strategy, _ in most_uncorrelated_strategies]
                
                print(f"DEBUG: Selected {len(top_strategies)} most uncorrelated strategies:")
                for strategy, avg_corr in most_uncorrelated_strategies:
                    print(f"  {strategy.split('::')[1]}: avg correlation = {avg_corr:.3f}")
                
                # Filter correlation matrix to only include most uncorrelated strategies
                correlation_matrix = correlation_matrix.loc[top_strategies, top_strategies]
            
            # Find uncorrelated pairs (lowest correlations)
            uncorrelated_pairs = []
            
            for i in range(len(correlation_matrix.columns)):
                for j in range(i + 1, len(correlation_matrix.columns)):
                    strategy1 = correlation_matrix.columns[i]
                    strategy2 = correlation_matrix.columns[j]
                    correlation = correlation_matrix.iloc[i, j]
                    
                    # Only include pairs from different files
                    file1 = strategy1.split('::')[0]
                    file2 = strategy2.split('::')[0]
                    
                    if file1 != file2:  # Different files
                        uncorrelated_pairs.append({
                            'strategy1': strategy1,
                            'strategy2': strategy2,
                            'strategy1_display': strategy1.split('::')[1],
                            'strategy2_display': strategy2.split('::')[1],
                            'file1': file1,
                            'file2': file2,
                            'correlation': round(correlation, 4),
                            'abs_correlation': abs(correlation)
                        })
            
            # Sort by absolute correlation (lowest first = most uncorrelated)
            uncorrelated_pairs.sort(key=lambda x: x['abs_correlation'])
            
            # Limit results
            top_uncorrelated = uncorrelated_pairs[:max_pairs]
            
            return {
                'success': True,
                'data': {
                    'common_date_range': {
                        'start': common_start.strftime('%Y-%m-%d'),
                        'end': common_end.strftime('%Y-%m-%d'),
                        'days': (common_end - common_start).days
                    },
                    'files_analyzed': [
                        {
                            'filename': p['friendly_name'],
                            'strategies_count': len([s for s in strategy_returns.keys() if s.startswith(p['friendly_name'])]),
                            'date_range': f"{p['start_date'].date()} to {p['end_date'].date()}",
                            'trades_in_range': sum(1 for s in strategy_returns.keys() if s.startswith(p['friendly_name']))
                        } for p in portfolios_data
                    ],
                    'total_strategy_pairs': len(uncorrelated_pairs),
                    'uncorrelated_pairs': top_uncorrelated,
                    'correlation_matrix': correlation_matrix.round(4).to_dict(),
                    'summary_stats': {
                        'avg_correlation': round(np.mean([pair['correlation'] for pair in uncorrelated_pairs]), 4),
                        'min_correlation': round(min([pair['correlation'] for pair in uncorrelated_pairs]), 4),
                        'max_correlation': round(max([pair['correlation'] for pair in uncorrelated_pairs]), 4)
                    }
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Cross-file correlation analysis failed: {str(e)}'
            }


class PriceMovementAnalyzer:
    """Handles price movement analysis for filtering trades by underlying price changes."""
    
    def __init__(self, portfolio: Portfolio):
        self.portfolio = portfolio
    
    def calculate_price_movement_range(self) -> Dict:
        """Calculate the min and max price movement percentages across all trades."""
        try:
            all_trades = []
            for strategy in self.portfolio.strategies.values():
                all_trades.extend(strategy.trades)
            
            if not all_trades:
                return {
                    'success': False,
                    'error': 'No trades found in portfolio'
                }
            
            movements = []
            for trade in all_trades:
                if hasattr(trade, 'opening_price') and hasattr(trade, 'closing_price') and trade.opening_price and trade.closing_price:
                    if trade.opening_price > 0:
                        movement = ((trade.closing_price - trade.opening_price) / trade.opening_price) * 100
                        movements.append(movement)
            
            if not movements:
                return {
                    'success': False,
                    'error': 'No trades with valid price data found'
                }
            
            return {
                'success': True,
                'data': {
                    'min_movement': round(min(movements), 2),
                    'max_movement': round(max(movements), 2),
                    'trade_count': len(movements)
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Failed to calculate price movement range: {str(e)}'
            }
    
    def analyze_price_movement_performance(self, min_movement: float, max_movement: float) -> Dict:
        """Analyze strategy performance for trades within the specified price movement range."""
        try:
            all_trades = []
            for strategy in self.portfolio.strategies.values():
                for trade in strategy.trades:
                    if hasattr(trade, 'opening_price') and hasattr(trade, 'closing_price') and trade.opening_price and trade.closing_price:
                        if trade.opening_price > 0:
                            movement = ((trade.closing_price - trade.opening_price) / trade.opening_price) * 100
                            if min_movement <= movement <= max_movement:
                                all_trades.append({
                                    'trade': trade,
                                    'strategy_name': strategy.name,
                                    'movement': movement
                                })
            
            if not all_trades:
                return {
                    'success': True,
                    'data': {
                        'strategies': [],
                        'min_movement': min_movement,
                        'max_movement': max_movement,
                        'total_trades': 0
                    }
                }
            
            # Group trades by strategy
            strategy_data = {}
            for trade_data in all_trades:
                strategy_name = trade_data['strategy_name']
                trade = trade_data['trade']
                
                if strategy_name not in strategy_data:
                    strategy_data[strategy_name] = {
                        'name': strategy_name,
                        'trades': [],
                        'total_pnl': 0,
                        'total_contracts': 0,
                        'wins': 0,
                        'losses': 0,
                        'win_amounts': [],
                        'loss_amounts': []
                    }
                
                strategy_data[strategy_name]['trades'].append(trade)
                strategy_data[strategy_name]['total_pnl'] += trade.pnl
                strategy_data[strategy_name]['total_contracts'] += getattr(trade, 'contracts', 1)
                
                if trade.pnl > 0:
                    strategy_data[strategy_name]['wins'] += 1
                    strategy_data[strategy_name]['win_amounts'].append(trade.pnl)
                else:
                    strategy_data[strategy_name]['losses'] += 1
                    strategy_data[strategy_name]['loss_amounts'].append(trade.pnl)
            
            # Calculate summary statistics for each strategy
            strategies = []
            for strategy_name, data in strategy_data.items():
                trade_count = len(data['trades'])
                avg_pnl = data['total_pnl'] / trade_count if trade_count > 0 else 0
                avg_pnl_per_lot = data['total_pnl'] / data['total_contracts'] if data['total_contracts'] > 0 else 0
                avg_win = np.mean(data['win_amounts']) if data['win_amounts'] else 0
                avg_loss = np.mean(data['loss_amounts']) if data['loss_amounts'] else 0
                
                strategies.append({
                    'name': strategy_name,
                    'trade_count': trade_count,
                    'total_pnl': round(data['total_pnl'], 2),
                    'avg_pnl': round(avg_pnl, 2),
                    'avg_pnl_per_lot': round(avg_pnl_per_lot, 2),
                    'total_contracts': data['total_contracts'],
                    'wins': data['wins'],
                    'losses': data['losses'],
                    'avg_win': round(avg_win, 2),
                    'avg_loss': round(avg_loss, 2)
                })
            
            # Sort by total P&L descending
            strategies.sort(key=lambda x: x['total_pnl'], reverse=True)
            
            return {
                'success': True,
                'data': {
                    'strategies': strategies,
                    'min_movement': min_movement,
                    'max_movement': max_movement,
                    'total_trades': len(all_trades)
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Failed to analyze price movement performance: {str(e)}'
            } 