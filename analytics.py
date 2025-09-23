"""
Strategy-focused analytics functions for portfolio analysis.
Handles calculations for strategy comparison, balance analysis, and diversification.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from models import Portfolio, Strategy, Trade
from config import Config

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
        
        # Debug: Created fallback correlation matrix for {n_strategies} strategies
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
        
        # Debug: Added {len(missing_strategies)} missing strategies to correlation matrix
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
        
        # Risk-free rate (configurable, default 4%)
        risk_free_rate = Config.RISK_FREE_RATE
        
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
        """Get daily returns for each strategy (optimized)."""
        # Pre-allocate data structures for better performance
        all_dates = []
        all_strategies = []
        all_pnls = []
        
        # Collect all trade data in one pass
        for strategy_name, strategy in self.portfolio.strategies.items():
            for trade in strategy.trades:
                if trade.date_closed:
                    try:
                        # Use vectorized date parsing
                        date = pd.to_datetime(trade.date_closed)
                        all_dates.append(date)
                        all_strategies.append(strategy_name)
                        all_pnls.append(trade.pnl_per_lot)
                    except (ValueError, TypeError):
                        continue
        
        if not all_dates:
            return pd.DataFrame()
        
        # Create DataFrame from pre-allocated lists (faster than append)
        df = pd.DataFrame({
            'date': all_dates,
            'strategy': all_strategies,
            'pnl': all_pnls
        })
        
        # Use more efficient grouping with categorical data
        df['strategy'] = df['strategy'].astype('category')
        
        # Group by date and strategy, sum P&L for each day
        daily_returns = df.groupby(['date', 'strategy'])['pnl'].sum().unstack(fill_value=0)
        
        # Debug: Print strategy count
        # Debug: Correlation calculation found {len(daily_returns.columns)} strategies: {list(daily_returns.columns)}
        
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
        
        # Calculate cumulative P&L over time (optimized)
        all_trades = self._get_all_trades_chronologically()
        
        if not all_trades:
            return self._empty_metrics()
        
        # Calculate initial account balance from first trade
        first_trade = all_trades[0]
        initial_balance = first_trade.funds_at_close - first_trade.pnl
        
        # Use numpy for vectorized operations
        pnl_values = np.array([trade.pnl for trade in all_trades])
        cumulative_pnl = np.cumsum(pnl_values)
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
        
        # Use numpy array for returns (already calculated)
        returns = pnl_values
        
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
        
        # Calculate actual contract count from legs, fallback to CSV field
        from commission_config import CommissionCalculator
        calc = CommissionCalculator()
        total_contracts = 0
        for trade in all_trades:
            legs = getattr(trade, 'legs', '')
            if legs and legs.strip():
                contracts = calc.calculate_actual_contracts_from_legs(legs)
            else:
                contracts = getattr(trade, 'contracts', 1)
            total_contracts += contracts
            
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
        
        # Risk-free rate (configurable, default 4%)
        risk_free_rate = Config.RISK_FREE_RATE
        
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
        # OPTIMIZATION 7: Add caching for simulation results
        self._simulation_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
    
    def clear_cache(self):
        """Clear the simulation cache to free memory."""
        self._simulation_cache.clear()
        print(f"🧹 Monte Carlo cache cleared. Previous stats: {self._cache_hits} hits, {self._cache_misses} misses")
    
    def _calculate_daily_portfolio_returns(self, trades: List) -> List[float]:
        """
        Calculate daily portfolio returns from historical trades.
        
        Args:
            trades: List of trades sorted by date
            
        Returns:
            List of daily returns as percentages
        """
        if not trades:
            return []
        
        # Group trades by date and calculate daily P&L
        daily_pnl = {}
        for trade in trades:
            trade_date = pd.to_datetime(trade.date_closed).date()
            if trade_date not in daily_pnl:
                daily_pnl[trade_date] = 0.0
            daily_pnl[trade_date] += trade.pnl
        
        # Sort dates and calculate daily returns
        sorted_dates = sorted(daily_pnl.keys())
        daily_returns = []
        
        # Calculate running balance and daily returns
        running_balance = trades[0].funds_at_close - trades[0].pnl  # Initial balance
        
        for date in sorted_dates:
            daily_pnl_value = daily_pnl[date]
            if running_balance > 0:
                daily_return = daily_pnl_value / running_balance
                daily_returns.append(daily_return)
            else:
                daily_returns.append(0.0)
            
            # Update running balance
            running_balance += daily_pnl_value
        
        return daily_returns
    
    def _calculate_historical_margin_percentage(self, all_trades: List, initial_balance: float) -> float:
        """Calculate historical margin percentage using the same logic as the overview page."""
        if not all_trades or initial_balance <= 0:
            return 1.0  # Default fallback
        
        # Use the chronologically first trade to calculate margin percentage (same as Strategy.get_summary_stats)
        sorted_trades_for_calc = sorted(all_trades, key=lambda t: pd.to_datetime(t.date_closed))
        first_trade = sorted_trades_for_calc[0]
        first_trade_margin_req = getattr(first_trade, 'margin_req', 0)
        
        # Debug: Check the first few trades to see what margin data looks like
        # Sort trades by date to get the chronologically first trades
        sorted_trades_for_debug = sorted(all_trades, key=lambda t: pd.to_datetime(t.date_closed))
        print(f"DEBUG: First 5 trades margin data (chronologically):")
        for i, trade in enumerate(sorted_trades_for_debug[:5]):
            margin_req = getattr(trade, 'margin_req', 0)
            contracts = getattr(trade, 'contracts', 1)
            pnl = getattr(trade, 'pnl', 0)
            funds_at_close = getattr(trade, 'funds_at_close', 0)
            date_closed = getattr(trade, 'date_closed', 'Unknown')
            print(f"DEBUG: Trade {i+1} ({date_closed}): margin_req=${margin_req:,.2f}, contracts={contracts}, pnl=${pnl:,.2f}, funds_at_close=${funds_at_close:,.2f}")
        
        if first_trade_margin_req > 0:
            # Use the same calculation as Strategy Performance section (models.py line 474)
            funds_at_close = getattr(first_trade, 'funds_at_close', 0)
            first_trade_pnl = first_trade.pnl
            
            # Initial balance = funds at close - P&L of first trade (same as Strategy Performance)
            calculated_initial_balance = funds_at_close - first_trade_pnl
            
            print(f"DEBUG: Historical margin calculation (using Strategy Performance method):")
            print(f"DEBUG: First trade margin req: ${first_trade_margin_req:,.2f}")
            print(f"DEBUG: First trade funds at close: ${funds_at_close:,.2f}")
            print(f"DEBUG: First trade P&L: ${first_trade_pnl:,.2f}")
            print(f"DEBUG: Calculated initial balance: ${calculated_initial_balance:,.2f}")
            print(f"DEBUG: Passed initial balance: ${initial_balance:,.2f}")
            
            if calculated_initial_balance > 0:
                margin_percentage = (first_trade_margin_req / calculated_initial_balance) * 100
                print(f"DEBUG: Calculated margin percentage: {margin_percentage:.2f}%")
                return margin_percentage
            else:
                # Fallback to using the passed initial_balance
                margin_percentage = (first_trade_margin_req / initial_balance) * 100
                print(f"DEBUG: Fallback margin percentage: {margin_percentage:.2f}%")
                return margin_percentage
        else:
            # Fallback: estimate from average margin per contract
            total_margin = 0
            total_contracts = 0
            
            for trade in all_trades[:100]:  # Sample first 100 trades for efficiency
                margin_req = getattr(trade, 'margin_req', 0)
                contracts = getattr(trade, 'contracts', 1)
                
                if margin_req > 0 and contracts > 0:
                    total_margin += margin_req
                    total_contracts += contracts
            
            if total_contracts > 0:
                avg_margin_per_contract = total_margin / total_contracts
                # Estimate margin percentage based on average margin per contract
                estimated_margin_percentage = (avg_margin_per_contract / initial_balance) * 100
                print(f"DEBUG: Fallback margin calculation:")
                print(f"DEBUG: Average margin per contract: ${avg_margin_per_contract:,.2f}")
                print(f"DEBUG: Estimated margin percentage: {estimated_margin_percentage:.2f}%")
                return estimated_margin_percentage
            else:
                # Final fallback
                print(f"DEBUG: Using default margin percentage: 1.0%")
                return 1.0
    
    def run_simulation(self, num_simulations: int = 1000, num_trades: int = None, 
                      risk_free_rate: float = None,
                      confidence_levels: List[float] = [0.05, 0.25, 0.5, 0.75, 0.95]) -> Dict:
        
        print(f"DEBUG: run_simulation called with:")
        print(f"DEBUG: num_simulations: {num_simulations}")
        print(f"DEBUG: num_trades: {num_trades}")
        print(f"DEBUG: risk_free_rate: {risk_free_rate}")
        """
        Run Monte Carlo simulation by randomly sampling from historical trade outcomes.
        Uses each strategy's historical margin percentage for position sizing.
        
        Args:
            num_simulations: Number of simulation runs
            num_trades: Number of trades per simulation (default: same as historical)
            risk_free_rate: Risk-free rate for risk ratio calculations
            confidence_levels: Percentiles to calculate for results
        
        Returns:
            Dictionary with simulation results and statistics
        """
        if not self.portfolio.strategies:
            return self._empty_simulation_results()
        
        # OPTIMIZATION 7: Check cache first
        cache_key = f"portfolio_{num_simulations}_{num_trades}_{hash(str(confidence_levels))}"
        if cache_key in self._simulation_cache:
            self._cache_hits += 1
            print(f"🚀 Monte Carlo cache hit! ({self._cache_hits} hits, {self._cache_misses} misses)")
            return self._simulation_cache[cache_key]
        
        self._cache_misses += 1
        
        # Get all historical trades
        all_trades = []
        for strategy in self.portfolio.strategies.values():
            all_trades.extend(strategy.trades)
        
        if not all_trades:
            return self._empty_simulation_results()
        
        # Use historical trade count if not specified
        if num_trades is None:
            num_trades = len(all_trades)
        
        # For portfolio-level Monte Carlo, we need to handle multiple strategies
        # Each strategy has its own position size based on its margin requirements
        
        # Calculate initial balance from historical data first
        metrics = PortfolioMetrics(self.portfolio)
        historical_metrics = metrics.get_overview_metrics()
        initial_balance = historical_metrics['initial_balance']
        
        # Calculate position size for each strategy
        strategy_data = {}
        total_trades = 0
        
        for strategy_name, strategy in self.portfolio.strategies.items():
            if strategy.trades:
                # Calculate this strategy's margin percentage
                strategy_margin_pct = self._calculate_historical_margin_percentage(strategy.trades, initial_balance)
                
                # Use historical margin percentage to maintain same position sizing as original data
                # This ensures Monte Carlo simulation uses the same position sizing as historical data
                strategy_position_size = strategy_margin_pct
                
                print(f"DEBUG: Portfolio simulation - Strategy {strategy_name}:")
                print(f"DEBUG: Historical margin: {strategy_margin_pct:.2f}%")
                print(f"DEBUG: Using historical margin for position sizing")
                
                strategy_data[strategy_name] = {
                    'strategy': strategy,
                    'trades': strategy.trades,
                    'margin_pct': strategy_margin_pct,
                    'position_size': strategy_position_size,
                    'trade_count': len(strategy.trades)
                }
                total_trades += len(strategy.trades)
                
                print(f"DEBUG: Strategy '{strategy_name}':")
                print(f"DEBUG: - Historical margin: {strategy_margin_pct:.2f}%")
                print(f"DEBUG: - Position size: {strategy_position_size:.2f}%")
                print(f"DEBUG: - Trade count: {len(strategy.trades)}")
        
        print(f"DEBUG: Total strategies: {len(strategy_data)}")
        print(f"DEBUG: Total trades across all strategies: {total_trades}")
        
        print(f"DEBUG: Monte Carlo simulation parameters:")
        print(f"DEBUG: Total historical trades: {total_trades}")
        print(f"DEBUG: Trades per simulation: {num_trades}")
        print(f"DEBUG: Number of simulations: {num_simulations}")
        print(f"DEBUG: Using historical position sizing for each strategy")
        print(f"DEBUG: Initial balance: ${initial_balance:,.2f}")
        
        # Multi-strategy Monte Carlo simulation with proper position sizing per strategy
        print(f"DEBUG: Using multi-strategy bootstrap sampling for realistic Monte Carlo simulation")
        
        # Calculate total win rate across all strategies
        total_positive_trades = sum(sum(1 for trade in data['trades'] if getattr(trade, 'pnl', 0) > 0) for data in strategy_data.values())
        total_negative_trades = sum(sum(1 for trade in data['trades'] if getattr(trade, 'pnl', 0) < 0) for data in strategy_data.values())
        overall_win_rate = total_positive_trades / (total_positive_trades + total_negative_trades) * 100 if (total_positive_trades + total_negative_trades) > 0 else 0
        
        print(f"DEBUG: Overall portfolio statistics:")
        print(f"DEBUG: - Total trades: {total_trades}")
        print(f"DEBUG: - Positive trades: {total_positive_trades}")
        print(f"DEBUG: - Negative trades: {total_negative_trades}")
        print(f"DEBUG: - Win rate: {overall_win_rate:.1f}%")
        
        # Run simulations using multi-strategy bootstrap sampling
        simulation_results = []
        
        try:
            print(f"DEBUG: Starting Monte Carlo simulation with {num_simulations} simulations")
            for sim_num in range(num_simulations):
                # Sample trades from each strategy proportionally
                simulated_trade_dollars = []
                current_balance = initial_balance
                
                # Calculate how many trades to sample from each strategy
                trades_per_strategy = {}
                for strategy_name, data in strategy_data.items():
                    # Sample proportionally based on the strategy's trade count
                    strategy_trade_count = int(num_trades * data['trade_count'] / total_trades)
                    trades_per_strategy[strategy_name] = max(1, strategy_trade_count)  # At least 1 trade per strategy
                
                # Adjust if we don't have exactly num_trades
                total_sampled = sum(trades_per_strategy.values())
                if total_sampled != num_trades:
                    # Adjust the largest strategy to match
                    largest_strategy = max(strategy_data.keys(), key=lambda k: trades_per_strategy[k])
                    trades_per_strategy[largest_strategy] += (num_trades - total_sampled)
                
                # Sample trades from each strategy
                for strategy_name, data in strategy_data.items():
                    strategy_trades = data['trades']
                    num_strategy_trades = trades_per_strategy[strategy_name]
                    
                    # Sample trades randomly with replacement from this strategy
                    sampled_trades = np.random.choice(strategy_trades, size=num_strategy_trades, replace=True)
                    
                    # Apply position sizing for this strategy
                    strategy_position_size = data['position_size'] / 100.0  # Convert to decimal
                    strategy_historical_margin = data['margin_pct'] / 100.0  # Convert to decimal
                    position_scaling_factor = strategy_position_size / strategy_historical_margin
                    
                    print(f"DEBUG: Strategy {strategy_name} position scaling:")
                    print(f"DEBUG: Strategy position size: {strategy_position_size*100:.2f}%")
                    print(f"DEBUG: Historical margin: {strategy_historical_margin*100:.2f}%")
                    print(f"DEBUG: Position scaling factor: {position_scaling_factor:.4f}")
                    
                    for trade in sampled_trades:
                        # Get the actual P&L from the trade
                        trade_pnl = getattr(trade, 'pnl', 0)
                        if trade_pnl == 0:
                            trade_pnl = trade.pnl_per_lot * trade.contracts
                        
                        # Scale the P&L based on this strategy's position size
                        scaled_trade_pnl = trade_pnl * position_scaling_factor
                        simulated_trade_dollars.append(scaled_trade_pnl)
                        
                        # Update balance for next trade (compounding effect)
                        current_balance += scaled_trade_pnl
                
                # Calculate cumulative P&L for this simulation
                cumulative_pnl = np.cumsum(simulated_trade_dollars)
                account_balance = initial_balance + cumulative_pnl
                
                # Calculate final balance and max drawdown
                final_balance = account_balance[-1]
                max_drawdown = self._calculate_simulation_max_drawdown(account_balance)
                
                simulation_results.append({
                    'final_balance': final_balance,
                    'total_pnl': cumulative_pnl[-1],
                    'max_drawdown': max_drawdown,
                    'cumulative_pnl': cumulative_pnl.tolist(),
                    'account_balance': account_balance.tolist(),
                    'simulated_trades': simulated_trade_dollars
                })
            
                # Debug first few simulations
                if sim_num < 3:
                    print(f"DEBUG: Simulation {sim_num + 1}:")
                    print(f"DEBUG: - Final balance: ${final_balance:,.2f}")
                    print(f"DEBUG: - Total P&L: ${cumulative_pnl[-1]:,.2f}")
                    print(f"DEBUG: - Max drawdown: ${max_drawdown:,.2f}")
                    print(f"DEBUG: - Winning trades: {sum(1 for x in simulated_trade_dollars if x > 0)}")
                    print(f"DEBUG: - Losing trades: {sum(1 for x in simulated_trade_dollars if x < 0)}")
                    print(f"DEBUG: - Win rate: {sum(1 for x in simulated_trade_dollars if x > 0) / len(simulated_trade_dollars) * 100:.1f}%")
                    print(f"DEBUG: - Trades per strategy: {trades_per_strategy}")
        
        except Exception as e:
            error_msg = f"Monte Carlo simulation failed: {str(e)}"
            print(f"DEBUG: {error_msg}")
            import traceback
            traceback.print_exc()
            # Log to file for debugging
            import logging
            logging.basicConfig(level=logging.DEBUG, filename='monte_carlo_errors.log')
            logging.error(f"Monte Carlo simulation error: {str(e)}")
            logging.error(f"Traceback: {traceback.format_exc()}")
            raise Exception(error_msg)
        
        # Extract results for analysis
        final_balances = np.array([sim['final_balance'] for sim in simulation_results])
        total_pnls = np.array([sim['total_pnl'] for sim in simulation_results])
        max_drawdowns = np.array([sim['max_drawdown'] for sim in simulation_results])
        
        # Store simulation results with proper IDs
        formatted_simulation_results = []
        for i, sim in enumerate(simulation_results):
            formatted_simulation_results.append({
                'simulation_id': i + 1,
                'final_balance': float(sim['final_balance']),
                'total_pnl': float(sim['total_pnl']),
                'max_drawdown': float(sim['max_drawdown']),
                'cumulative_pnl': sim['cumulative_pnl'],
                'account_balance': sim['account_balance'],
                'simulated_trades': sim['simulated_trades']
            })
        
        # Calculate statistics (arrays already created above)
        total_pnls = final_balances - initial_balance
        
        # OPTIMIZATION 6: Vectorized percentile calculation
        # Calculate all percentiles at once instead of in a loop
        percentiles = np.array([level * 100 for level in confidence_levels])
        balance_percentiles = dict(zip([f'p{int(p)}' for p in percentiles], np.percentile(final_balances, percentiles)))
        pnl_percentiles = dict(zip([f'p{int(p)}' for p in percentiles], np.percentile(total_pnls, percentiles)))
        drawdown_percentiles = dict(zip([f'p{int(p)}' for p in percentiles], np.percentile(max_drawdowns, percentiles)))
        
        # Calculate additional statistics
        win_probability = (total_pnls > 0).mean() * 100
        loss_probability = (total_pnls < 0).mean() * 100
        
        # Calculate risk-adjusted ratios
        risk_ratios = self._calculate_risk_ratios(total_pnls, final_balances, None, risk_free_rate, max_drawdowns)
        
        # Calculate t-test statistics
        t_test_results = self._calculate_t_tests(total_pnls, final_balances, initial_balance)
        
        # Calculate p-values for statistical significance
        p_values = self._calculate_p_values(total_pnls, final_balances, max_drawdowns, initial_balance)
        
        # OPTIMIZATION 7: Store result in cache
        result = {
            'simulation_summary': {
                'num_simulations': num_simulations,
                'num_trades_per_sim': num_trades,
                'trade_size_percent': 'Historical position sizing',
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
            'simulation_results': formatted_simulation_results[:50],  # Return first 50 for charts (reduced for performance)
            'confidence_levels': confidence_levels
        }
        
        # Store in cache and return
        self._simulation_cache[cache_key] = result
        return result
    
    def run_strategy_specific_simulation(self, strategy_name: str, num_simulations: int = 1000, 
                                       num_trades: int = None, risk_free_rate: float = None) -> Dict:
        """
        Run Monte Carlo simulation for a specific strategy.
        Uses the strategy's historical margin percentage for position sizing.
        
        Args:
            strategy_name: Name of the strategy to simulate
            num_simulations: Number of simulation runs
            num_trades: Number of trades per simulation
            risk_free_rate: Risk-free rate for risk ratio calculations
        
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
        
        # Extract actual trade P&L values for this strategy (use actual dollar P&L for consistency with portfolio simulation)
        trade_pnls = []
        for trade in strategy.trades:
            # Use the actual P&L from the trade
            trade_pnl = getattr(trade, 'pnl', 0)
            if trade_pnl == 0:
                # Fallback: calculate from per-lot P&L
                trade_pnl = trade.pnl_per_lot * trade.contracts
            trade_pnls.append(trade_pnl)
        
        # Debug: Check the actual values
        print(f"DEBUG: Strategy {strategy_name} - Sample trade P&Ls:")
        for i, trade in enumerate(strategy.trades[:5]):  # First 5 trades
            print(f"DEBUG: Trade {i+1}: P&L=${trade.pnl:,.2f}, Contracts={trade.contracts}, P&L/Lot=${trade.pnl_per_lot:,.2f}")
        print(f"DEBUG: Trade P&L range: ${min(trade_pnls):,.2f} to ${max(trade_pnls):,.2f}")
        print(f"DEBUG: Mean trade P&L: ${np.mean(trade_pnls):,.2f}")
        
        # Calculate initial balance (for single strategy, use average account balance)
        # Since we're using per-lot P&L, we need to normalize the initial balance calculation
        account_balances = [trade.funds_at_close for trade in strategy.trades]
        
        # Calculate actual contract count from legs, fallback to CSV field
        from commission_config import CommissionCalculator
        calc = CommissionCalculator()
        contract_counts = []
        for trade in strategy.trades:
            legs = getattr(trade, 'legs', '')
            if legs and legs.strip():
                contracts = calc.calculate_actual_contracts_from_legs(legs)
            else:
                contracts = getattr(trade, 'contracts', 1)
            contract_counts.append(contracts)
        avg_contracts = np.mean(contract_counts)
        # Calculate actual initial balance from chronologically first trade of this strategy
        if strategy.trades:
            # Sort trades by date to get the chronologically first trade
            sorted_trades = sorted(strategy.trades, key=lambda t: pd.to_datetime(t.date_closed))
            first_trade = sorted_trades[0]
            # Starting balance = funds at close - P&L (this gives us the balance before the trade)
            initial_balance = first_trade.funds_at_close - first_trade.pnl
            print(f"DEBUG: Strategy '{strategy_name}' first trade date: {first_trade.date_closed}")
            print(f"DEBUG: First trade funds_at_close: ${first_trade.funds_at_close:,.2f}")
            print(f"DEBUG: First trade P&L: ${first_trade.pnl:,.2f}")
            print(f"DEBUG: Calculated initial balance: ${initial_balance:,.2f}")
        else:
            initial_balance = 1000.0  # Fallback if no trades
        
        # Calculate strategy's margin percentage for position sizing
        strategy_margin_pct = self._calculate_historical_margin_percentage(strategy.trades, initial_balance)
        print(f"DEBUG: Strategy '{strategy_name}' margin percentage: {strategy_margin_pct:.2f}%")
        
        # Use the user-specified trade size if provided, otherwise use strategy's historical margin
        # Use historical margin percentage to maintain same position sizing as original data
        # This ensures Monte Carlo simulation uses the same position sizing as historical data
        strategy_position_size = strategy_margin_pct
        
        print(f"DEBUG: Using historical margin percentage: {strategy_margin_pct:.2f}%")
        
        print(f"DEBUG: Strategy '{strategy_name}' position size: {strategy_position_size:.2f}%")
        
        # Run simulations
        simulation_results = []
        final_pnls = []
        
        try:
            print(f"DEBUG: Starting strategy-specific simulation for '{strategy_name}' with {num_simulations} simulations")
            for sim_num in range(num_simulations):
                # Randomly sample trades with replacement
                simulated_trades = np.random.choice(trade_pnls, size=num_trades, replace=True)
                
                # Apply position sizing for this strategy
                strategy_position_size_pct = strategy_position_size / 100.0  # Convert to decimal
                strategy_historical_margin_pct = strategy_margin_pct / 100.0  # Convert to decimal
                position_scaling_factor = strategy_position_size_pct / strategy_historical_margin_pct
                
                # Scale the P&L based on this strategy's position size
                scaled_trades = simulated_trades * position_scaling_factor
                
                # Calculate cumulative P&L
                cumulative_pnl = np.cumsum(scaled_trades)
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
                    'simulated_trades': scaled_trades.tolist()  # Store individual trade returns
                })
                
                final_pnls.append(final_pnl)
                
                # Debug first few simulations
                if sim_num < 3:
                    print(f"DEBUG: Simulation {sim_num + 1}:")
                    print(f"DEBUG: - Final balance: ${account_balance[-1]:,.2f}")
                    print(f"DEBUG: - Total P&L: ${final_pnl:,.2f}")
                    print(f"DEBUG: - Position size: {strategy_position_size:.2f}%")
                    print(f"DEBUG: - Position scaling factor: {position_scaling_factor:.4f}")
                    print(f"DEBUG: - Winning trades: {sum(1 for x in scaled_trades if x > 0)}")
                    print(f"DEBUG: - Losing trades: {sum(1 for x in scaled_trades if x < 0)}")
                    print(f"DEBUG: - Win rate: {sum(1 for x in scaled_trades if x > 0) / len(scaled_trades) * 100:.1f}%")
        
        except Exception as e:
            error_msg = f"Strategy-specific Monte Carlo simulation failed for '{strategy_name}': {str(e)}"
            print(f"DEBUG: {error_msg}")
            import traceback
            traceback.print_exc()
            # Log to file for debugging
            import logging
            logging.basicConfig(level=logging.DEBUG, filename='monte_carlo_errors.log')
            logging.error(f"Strategy-specific simulation error for '{strategy_name}': {str(e)}")
            logging.error(f"Traceback: {traceback.format_exc()}")
            raise Exception(error_msg)
        
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
        risk_ratios = self._calculate_risk_ratios(final_pnls, final_balances, strategy.trades, risk_free_rate, max_drawdowns)
        
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
            'simulation_results': simulation_results[:50],  # Return first 50 for charts (reduced for performance)
            'confidence_levels': confidence_levels
        }
    
    def run_all_strategies_simulation(self, num_simulations: int = 1000, num_trades: int = None, 
                                    risk_free_rate: float = None) -> Dict:
        """
        Run Monte Carlo simulation for each strategy individually to identify fragility.
        
        Args:
            num_simulations: Number of simulation runs per strategy
            num_trades: Number of trades per simulation (default: strategy's historical count)
            risk_free_rate: Risk-free rate for risk ratio calculations
        
        Returns:
            Dictionary with results for each strategy and comparative analysis
        """
        if not self.portfolio.strategies:
            return {'error': 'No strategies available'}
        
        strategy_results = {}
        fragility_analysis = {}
        
        # Run simulation for each strategy
        for strategy_name in self.portfolio.strategies.keys():
            # Calculate actual margin percentage for this strategy
            strategy = self.portfolio.strategies[strategy_name]
            if strategy.trades:
                # Calculate historical margin percentage for this strategy
                strategy_margin_pct = self._calculate_historical_margin_percentage(strategy.trades)
                print(f"DEBUG: Strategy '{strategy_name}' calculated margin percentage: {strategy_margin_pct:.4f}%")
            else:
                strategy_margin_pct = 1.0  # Fallback value
                print(f"DEBUG: Strategy '{strategy_name}' has no trades, using fallback: {strategy_margin_pct:.4f}%")
            
            strategy_results[strategy_name] = self.run_strategy_specific_simulation(
                strategy_name, num_simulations, num_trades, strategy_margin_pct, risk_free_rate
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
    
    def _calculate_vectorized_max_drawdown(self, account_balances: np.ndarray) -> np.ndarray:
        """Calculate maximum drawdown for all simulations at once (vectorized)."""
        if account_balances.size == 0:
            return np.array([])
        
        # Calculate running maximum for each simulation (along axis=1)
        running_max = np.maximum.accumulate(account_balances, axis=1)
        
        # Calculate drawdown at each point
        drawdowns = running_max - account_balances
        
        # Return maximum drawdown for each simulation
        return np.max(drawdowns, axis=1)
    
    def _calculate_risk_ratios(self, total_pnls: np.ndarray, final_balances: np.ndarray, strategy_trades: List = None, risk_free_rate: float = None, max_drawdowns: np.ndarray = None) -> Dict:
        """
        Calculate risk-adjusted performance ratios for Monte Carlo results.
        
        Args:
            total_pnls: Array of total P&L values from simulations
            final_balances: Array of final balance values from simulations
            strategy_trades: List of actual strategy trades for risk calculation
            risk_free_rate: Risk-free rate for risk ratio calculations
            max_drawdowns: Array of maximum drawdown values from simulations
            
        Returns:
            Dictionary with Sharpe ratio, Sortino ratio, and other risk metrics
        """
        # Handle infinity values for JSON serialization
        def safe_float(value):
            if value == float('inf') or value == float('-inf'):
                print(f"DEBUG: Converting infinity to None: {value}")
                return None
            if abs(value) > 1000:  # Cap extremely large values
                print(f"DEBUG: Capping large value: {value} -> 1000")
                return 1000.0
            return value
        
        print(f"DEBUG: _calculate_risk_ratios called with strategy_trades: {strategy_trades is not None}")
        if strategy_trades:
            print(f"DEBUG: Number of strategy trades: {len(strategy_trades)}")
        else:
            print("DEBUG: No strategy trades provided, using portfolio-level calculation")
        
        # For portfolio simulations (strategy_trades=None), calculate risk ratios directly from simulation results
        if strategy_trades is None:
            print("DEBUG: Portfolio simulation - calculating risk ratios from simulation results")
            
            # Calculate time period from historical data for annualization
            all_trades = []
            for strategy in self.portfolio.strategies.values():
                all_trades.extend(strategy.trades)
            
            if len(all_trades) > 1:
                sorted_trades = sorted(all_trades, key=lambda t: pd.to_datetime(t.date_closed))
                start_date = pd.to_datetime(sorted_trades[0].date_closed)
                end_date = pd.to_datetime(sorted_trades[-1].date_closed)
                days_elapsed = (end_date - start_date).days
                years = days_elapsed / 365.25
                
                if years > 0:
                    # Calculate annual return from simulation results
                    # Use the mean of the simulation results to get the expected annual return
                    mean_final_balance = np.mean(final_balances)
                    initial_balance = sorted_trades[0].funds_at_close - sorted_trades[0].pnl
                    total_return = (mean_final_balance - initial_balance) / initial_balance
                    annual_return = (1 + total_return) ** (1/years) - 1
                    
                    print(f"DEBUG: Annual return calculation details:")
                    print(f"DEBUG: Mean final balance: ${mean_final_balance:,.2f}")
                    print(f"DEBUG: Initial balance: ${initial_balance:,.2f}")
                    print(f"DEBUG: Total return: {total_return*100:.2f}%")
                    print(f"DEBUG: Years: {years:.2f}")
                    print(f"DEBUG: Annual return: {annual_return*100:.2f}%")
                    
                    # Calculate volatility from actual historical data, not simulation results
                    # The simulation results show uncertainty in final outcomes, not strategy volatility
                    # We need to calculate the actual volatility of the strategy over time
                    
                    # Get all historical trades and calculate daily returns
                    all_trades = []
                    for strategy in self.portfolio.strategies.values():
                        all_trades.extend(strategy.trades)
                    
                    if len(all_trades) > 1:
                        # Sort trades by date
                        sorted_trades = sorted(all_trades, key=lambda t: pd.to_datetime(t.date_closed))
                        
                        # Calculate daily portfolio returns from historical data
                        daily_returns = self._calculate_daily_portfolio_returns(sorted_trades)
                        
                        if len(daily_returns) > 1:
                            # Convert to numpy array for calculations
                            daily_returns = np.array(daily_returns)
                            # Calculate annualized volatility from daily returns
                            daily_volatility = np.std(daily_returns)
                            annual_volatility = daily_volatility * np.sqrt(252)  # Annualize daily volatility
                        else:
                            # Fallback: use a reasonable estimate
                            annual_volatility = 0.15  # 15% annual volatility as reasonable estimate
                    else:
                        # Fallback: use a reasonable estimate
                        annual_volatility = 0.15  # 15% annual volatility as reasonable estimate
                    
                    print(f"DEBUG: Volatility calculation details:")
                    print(f"DEBUG: Number of simulations: {len(total_pnls)}")
                    print(f"DEBUG: Using historical daily returns for volatility calculation")
                    print(f"DEBUG: Annual volatility: {annual_volatility*100:.2f}%")
                    print(f"DEBUG: Years: {years:.2f}")
                    
                    # Calculate downside deviation from historical daily returns
                    if 'daily_returns' in locals() and len(daily_returns) > 1:
                        negative_daily_returns = daily_returns[daily_returns < 0]
                        if len(negative_daily_returns) > 0:
                            # Calculate annualized downside deviation from daily returns
                            daily_downside_deviation = np.std(negative_daily_returns)
                            downside_deviation = daily_downside_deviation * np.sqrt(252)  # Annualize
                        else:
                            # If no negative daily returns, use a reasonable estimate
                            downside_deviation = annual_volatility * 0.5  # Assume downside is 50% of total volatility
                    else:
                        # Fallback: use a reasonable estimate based on volatility
                        downside_deviation = annual_volatility * 0.5  # Assume downside is 50% of total volatility
                    
                    # Calculate risk ratios - convert everything to decimal for calculation
                    annual_return_decimal = annual_return  # Already in decimal form
                    annual_volatility_decimal = annual_volatility  # Already in decimal form
                    downside_deviation_decimal = downside_deviation  # Already in decimal form
                    risk_free_rate_decimal = risk_free_rate / 100.0 if risk_free_rate > 1 else risk_free_rate
                    
                    excess_return_decimal = annual_return_decimal - risk_free_rate_decimal
                    sharpe_ratio = excess_return_decimal / annual_volatility_decimal if annual_volatility_decimal > 0 else 0
                    sortino_ratio = excess_return_decimal / downside_deviation_decimal if downside_deviation_decimal > 0 else 0
                    
                    # Cap extreme ratios to prevent unrealistic values
                    if sortino_ratio > 10.0:
                        sortino_ratio = 10.0
                    
                    print(f"DEBUG: Risk ratio calculations (all in decimal form):")
                    print(f"DEBUG: Annual return: {annual_return_decimal:.6f} ({annual_return_decimal*100:.2f}%)")
                    print(f"DEBUG: Risk-free rate: {risk_free_rate_decimal:.6f} ({risk_free_rate_decimal*100:.2f}%)")
                    print(f"DEBUG: Excess return: {excess_return_decimal:.6f} ({excess_return_decimal*100:.2f}%)")
                    print(f"DEBUG: Annual volatility: {annual_volatility_decimal:.6f} ({annual_volatility_decimal*100:.2f}%)")
                    print(f"DEBUG: Downside deviation: {downside_deviation_decimal:.6f} ({downside_deviation_decimal*100:.2f}%)")
                    print(f"DEBUG: Sharpe ratio: {sharpe_ratio:.4f}")
                    print(f"DEBUG: Sortino ratio: {sortino_ratio:.4f}")
                    
                    # Calculate MAR ratio using actual historical maximum drawdown
                    # Get the actual maximum drawdown from portfolio overview for consistency
                    metrics = PortfolioMetrics(self.portfolio)
                    portfolio_metrics = metrics.get_overview_metrics()
                    actual_max_drawdown_pct = abs(portfolio_metrics['max_drawdown_pct']) / 100.0
                    print(f"DEBUG: Portfolio metrics max_drawdown_pct: {portfolio_metrics['max_drawdown_pct']:.2f}%")
                    print(f"DEBUG: Actual max drawdown (decimal): {actual_max_drawdown_pct:.4f}")
                    mar_ratio = annual_return / actual_max_drawdown_pct if actual_max_drawdown_pct > 0 else 0
                    
                    # Information ratio (using same calculation as Sharpe for now)
                    information_ratio = sharpe_ratio
                    
                    print(f"DEBUG: Portfolio simulation risk ratios:")
                    print(f"DEBUG: Annual return: {annual_return*100:.4f}%")
                    print(f"DEBUG: Annual volatility: {annual_volatility*100:.4f}%")
                    print(f"DEBUG: Downside deviation: {downside_deviation*100:.4f}%")
                    print(f"DEBUG: Sharpe ratio: {sharpe_ratio:.4f}")
                    print(f"DEBUG: Sortino ratio: {sortino_ratio:.4f}")
                    print(f"DEBUG: MAR ratio: {mar_ratio:.4f}")
                    
                    return {
                        'sharpe_ratio': safe_float(sharpe_ratio),
                        'sortino_ratio': safe_float(sortino_ratio),
                        'mar_ratio': safe_float(mar_ratio),
                        'information_ratio': safe_float(information_ratio),
                        'return_to_risk': safe_float(sharpe_ratio),
                        'mean_return_pct': annual_return_decimal * 100,  # Convert to percentage
                        'std_return_pct': annual_volatility_decimal * 100,  # Convert to percentage
                        'downside_deviation_pct': downside_deviation_decimal * 100,  # Convert to percentage
                        'mean_max_drawdown_pct': actual_max_drawdown_pct * 100  # Convert to percentage
                    }
        
        # Use provided risk-free rate or fall back to config default
        if risk_free_rate is None:
            risk_free_rate = Config.RISK_FREE_RATE
        else:
            # Convert percentage to decimal if needed
            if risk_free_rate > 1:
                risk_free_rate = risk_free_rate / 100.0
        
        print(f"DEBUG: Input parameters:")
        print(f"DEBUG: Total P&Ls shape: {total_pnls.shape if hasattr(total_pnls, 'shape') else len(total_pnls)}")
        print(f"DEBUG: Final balances shape: {final_balances.shape if hasattr(final_balances, 'shape') else len(final_balances)}")
        print(f"DEBUG: Using historical position sizing")
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
                    print(f"DEBUG: Days elapsed: {days_elapsed}, Valid for annualization")
                    # Calculate trades per year
                    trades_per_year = len(sorted_trades) * 365.25 / days_elapsed
                    print(f"DEBUG: Trades per year: {trades_per_year:.2f}")
                    
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
                            position_size_pct = 1.0  # Use historical position sizing
                            # Calculate return based on actual margin requirement and position sizing
                            # This gives us the true return on the capital at risk
                            trade_return = (trade.pnl / margin_req) * position_size_pct
                        else:
                            # Fallback to per-lot calculation scaled by trade size
                            trade_return = (trade.pnl_per_lot / 1000.0) * 1.0  # Use historical position sizing
                        
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
                        position_size_pct = 1.0  # Use historical position sizing
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
                        position_size_pct = 1.0  # Use historical position sizing
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
                # For portfolio-level analysis, calculate daily portfolio returns from historical data
                # This gives us a more accurate representation of portfolio volatility
                print(f"DEBUG: Portfolio-level analysis - calculating daily portfolio returns")
                
                # Get all trades from all strategies, sorted by date
                all_trades = []
                for strategy in self.portfolio.strategies.values():
                    all_trades.extend(strategy.trades)
                
                if len(all_trades) > 1:
                    # Sort trades by date
                    sorted_trades = sorted(all_trades, key=lambda t: pd.to_datetime(t.date_closed))
                    
                    # Calculate daily portfolio returns
                    daily_returns = self._calculate_daily_portfolio_returns(sorted_trades)
                    
                    if len(daily_returns) > 1:
                        returns = np.array(daily_returns)
                        print(f"DEBUG: Portfolio-level daily returns calculation:")
                        print(f"DEBUG: - Number of trading days: {len(returns)}")
                        print(f"DEBUG: - Mean daily return: {np.mean(returns)*100:.6f}%")
                        print(f"DEBUG: - Std daily return: {np.std(returns)*100:.6f}%")
                        print(f"DEBUG: - Min daily return: {np.min(returns)*100:.6f}%")
                        print(f"DEBUG: - Max daily return: {np.max(returns)*100:.6f}%")
                    else:
                        print(f"DEBUG: Portfolio-level analysis: Insufficient daily return data")
                        returns = np.array([0.0])
                else:
                    print(f"DEBUG: Portfolio-level analysis: Insufficient trade data")
                    returns = np.array([0.0])
            
            # Calculate mean return and standard deviation
            try:
                mean_return = np.mean(returns)
                std_return = np.std(returns)
                
                print(f"DEBUG: Basic return statistics:")
                print(f"DEBUG: Mean return: {mean_return*100:.6f}%")
                print(f"DEBUG: Standard deviation: {std_return*100:.6f}%")
                print(f"DEBUG: Number of returns: {len(returns)}")
                print(f"DEBUG: Min return: {np.min(returns)*100:.6f}%")
                print(f"DEBUG: Max return: {np.max(returns)*100:.6f}%")
            except Exception as e:
                print(f"DEBUG: Error calculating basic return statistics: {e}")
                mean_return = 0
                std_return = 0
            
            # Check if returns are reasonable (not too large)
            if mean_return > 10.0:  # More than 1000% return
                print(f"DEBUG: WARNING: Mean return is very large ({mean_return*100:.2f}%), this might cause issues")
            if std_return > 5.0:  # More than 500% volatility
                print(f"DEBUG: WARNING: Standard deviation is very large ({std_return*100:.2f}%), this might cause issues")
            
            # Sharpe Ratio: (Annual Return - Risk Free Rate) / Annual Volatility
            print(f"DEBUG: === SHARPE RATIO CALCULATION ===")
            print(f"DEBUG: About to calculate Sharpe ratio with mean_return={mean_return:.6f}, std_return={std_return:.6f}")
            try:
                if std_return > 0:
                    if strategy_trades and days_elapsed > 0:
                        # Calculate annualized metrics for proper Sharpe ratio
                        years = days_elapsed / 365.25
                        trades_per_year = len(strategy_trades) / years
                        
                        # For strategy-specific, calculate the actual total return over the period
                        # Use the same approach as portfolio-level calculation
                        sorted_trades = sorted(strategy_trades, key=lambda t: pd.to_datetime(t.date_closed))
                        initial_balance = sorted_trades[0].funds_at_close - sorted_trades[0].pnl
                        final_balance = sorted_trades[-1].funds_at_close
                        total_return = (final_balance - initial_balance) / initial_balance
                        annual_return = (1 + total_return) ** (1/years) - 1
                        # Calculate volatility from actual historical daily returns (same as portfolio simulation)
                        daily_returns = self._calculate_daily_portfolio_returns(sorted_trades)
                        if len(daily_returns) > 1:
                            daily_returns = np.array(daily_returns)
                            daily_volatility = np.std(daily_returns)
                            annual_volatility = daily_volatility * np.sqrt(252)  # Annualize daily volatility
                        else:
                            # Fallback: use a reasonable estimate
                            annual_volatility = 0.15  # 15% annual volatility as reasonable estimate
                        excess_return = annual_return - risk_free_rate
                        sharpe_ratio = excess_return / annual_volatility
                        
                        print(f"DEBUG: Strategy-specific Sharpe calculation:")
                        print(f"DEBUG: Time period: {years:.2f} years")
                        print(f"DEBUG: Trades per year: {trades_per_year:.2f}")
                        print(f"DEBUG: Mean return per trade: {mean_return*100:.4f}%")
                        print(f"DEBUG: Total return over period: {total_return*100:.4f}%")
                        print(f"DEBUG: Annual return: {annual_return*100:.4f}%")
                        print(f"DEBUG: Annual volatility: {annual_volatility*100:.4f}%")
                        print(f"DEBUG: Risk-free rate: {risk_free_rate*100:.4f}%")
                        print(f"DEBUG: Excess return: {excess_return*100:.4f}%")
                        print(f"DEBUG: Sharpe ratio: {sharpe_ratio:.4f}")
                    else:
                        # For portfolio-level, we need to annualize the returns
                        # The returns are total returns over the entire simulation period
                        # We need to calculate the time period and annualize
                        
                        # Calculate time period from historical data
                        all_trades = []
                        for strategy in self.portfolio.strategies.values():
                            all_trades.extend(strategy.trades)
                        
                        if len(all_trades) > 1:
                            sorted_trades = sorted(all_trades, key=lambda t: pd.to_datetime(t.date_closed))
                            start_date = pd.to_datetime(sorted_trades[0].date_closed)
                            end_date = pd.to_datetime(sorted_trades[-1].date_closed)
                            days_elapsed = (end_date - start_date).days
                            years = days_elapsed / 365.25
                        
                            if years > 0:
                                # For portfolio-level, calculate annual return from simulation results
                                # Use the actual portfolio performance from the simulation
                                initial_balance = sorted_trades[0].funds_at_close - sorted_trades[0].pnl
                                final_balance = sorted_trades[-1].funds_at_close
                                total_return = (final_balance - initial_balance) / initial_balance
                                annual_return = (1 + total_return) ** (1/years) - 1
                                annual_volatility = std_return * np.sqrt(252)  # Annualize daily volatility
                                excess_return = annual_return - risk_free_rate
                                sharpe_ratio = excess_return / annual_volatility
                                
                                print(f"DEBUG: Portfolio-level Sharpe calculation (annualized):")
                                print(f"DEBUG: Time period: {years:.2f} years")
                                print(f"DEBUG: Total return: {total_return*100:.4f}%")
                                print(f"DEBUG: Annual return: {annual_return*100:.4f}%")
                                print(f"DEBUG: Annual volatility: {annual_volatility*100:.4f}%")
                                print(f"DEBUG: Risk-free rate: {risk_free_rate*100:.4f}%")
                                print(f"DEBUG: Excess return: {excess_return*100:.4f}%")
                                print(f"DEBUG: Sharpe ratio: {sharpe_ratio:.4f}")
                            else:
                                sharpe_ratio = 0
                                print(f"DEBUG: Portfolio-level Sharpe calculation: Time period too short ({years:.2f} years)")
                        else:
                            sharpe_ratio = 0
                            print(f"DEBUG: Portfolio-level Sharpe calculation: Insufficient trade data")
                else:
                    sharpe_ratio = 0
                    print(f"DEBUG: Sharpe ratio = 0 (standard deviation is 0)")
            except Exception as e:
                print(f"DEBUG: Error in Sharpe ratio calculation: {e}")
                sharpe_ratio = 0
            
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
                years = days_elapsed / 365.25
                trades_per_year = len(strategy_trades) / years
                
                # Use the same annual return calculation as Sharpe ratio
                sorted_trades = sorted(strategy_trades, key=lambda t: pd.to_datetime(t.date_closed))
                initial_balance = sorted_trades[0].funds_at_close - sorted_trades[0].pnl
                final_balance = sorted_trades[-1].funds_at_close
                total_return = (final_balance - initial_balance) / initial_balance
                annual_return = (1 + total_return) ** (1/years) - 1
                
                print(f"DEBUG: Strategy-specific Sortino calculation:")
                print(f"DEBUG: Time period: {years:.2f} years")
                print(f"DEBUG: Trades per year: {trades_per_year:.2f}")
                print(f"DEBUG: Annual return: {annual_return*100:.4f}%")
                print(f"DEBUG: Risk-free rate: {risk_free_rate*100:.4f}%")
                
                # Calculate downside deviation from historical daily returns (same as portfolio simulation)
                daily_returns = self._calculate_daily_portfolio_returns(sorted_trades)
                if len(daily_returns) > 1:
                    daily_returns = np.array(daily_returns)
                    negative_daily_returns = daily_returns[daily_returns < 0]
                    if len(negative_daily_returns) > 0:
                        # Calculate annualized downside deviation from daily returns
                        daily_downside_deviation = np.std(negative_daily_returns)
                        downside_deviation = daily_downside_deviation * np.sqrt(252)  # Annualize
                        sortino_ratio = (annual_return - risk_free_rate) / downside_deviation
                        print(f"DEBUG: Multiple negative daily returns case:")
                        print(f"DEBUG: Downside deviation (annualized): {downside_deviation*100:.4f}%")
                        print(f"DEBUG: Sortino ratio: {sortino_ratio:.4f}")
                    else:
                        # If no negative daily returns, use a reasonable estimate
                        downside_deviation = annual_volatility * 0.5  # Assume downside is 50% of total volatility
                        sortino_ratio = (annual_return - risk_free_rate) / downside_deviation
                        print(f"DEBUG: No negative daily returns case (using 50% of volatility):")
                        print(f"DEBUG: Downside deviation: {downside_deviation*100:.4f}%")
                        print(f"DEBUG: Sortino ratio: {sortino_ratio:.4f}")
                else:
                    # Fallback: use a reasonable estimate
                    downside_deviation = annual_volatility * 0.5  # Assume downside is 50% of total volatility
                    sortino_ratio = (annual_return - risk_free_rate) / downside_deviation
                    print(f"DEBUG: Fallback downside deviation case:")
                    print(f"DEBUG: Downside deviation: {downside_deviation*100:.4f}%")
                    print(f"DEBUG: Sortino ratio: {sortino_ratio:.4f}")
            else:
                # For portfolio-level, we need to annualize the returns for Sortino calculation
                print(f"DEBUG: Portfolio-level Sortino calculation:")
                print(f"DEBUG: Mean return: {mean_return*100:.4f}%")
                print(f"DEBUG: Risk-free rate: {risk_free_rate*100:.4f}%")
                
                # Calculate time period from historical data (same as Sharpe calculation)
                all_trades = []
                for strategy in self.portfolio.strategies.values():
                    all_trades.extend(strategy.trades)
                
                if len(all_trades) > 1:
                    sorted_trades = sorted(all_trades, key=lambda t: pd.to_datetime(t.date_closed))
                    start_date = pd.to_datetime(sorted_trades[0].date_closed)
                    end_date = pd.to_datetime(sorted_trades[-1].date_closed)
                    days_elapsed = (end_date - start_date).days
                    years = days_elapsed / 365.25
                    
                    if years > 0:
                        # For portfolio-level, use the same annual return calculation as Sharpe ratio
                        initial_balance = sorted_trades[0].funds_at_close - sorted_trades[0].pnl
                        final_balance = sorted_trades[-1].funds_at_close
                        total_return = (final_balance - initial_balance) / initial_balance
                        annual_return = (1 + total_return) ** (1/years) - 1
                        
                        if len(negative_returns) > 1:
                            # Annualize downside deviation for daily returns
                            downside_deviation = np.std(negative_returns) * np.sqrt(252)
                            if downside_deviation > 0:
                                sortino_ratio = (annual_return - risk_free_rate) / downside_deviation
                                print(f"DEBUG: Multiple negative returns case (annualized):")
                                print(f"DEBUG: Time period: {years:.2f} years")
                                print(f"DEBUG: Annual return: {annual_return*100:.4f}%")
                                print(f"DEBUG: Downside deviation (annualized): {downside_deviation*100:.4f}%")
                                print(f"DEBUG: Sortino ratio: {sortino_ratio:.4f}")
                            else:
                                sortino_ratio = (annual_return - risk_free_rate) / 0.001
                                print(f"DEBUG: Identical negative returns case (using 0.001 divisor):")
                                print(f"DEBUG: Sortino ratio: {sortino_ratio:.4f}")
                        elif len(negative_returns) == 1:
                            # Annualize single negative return for daily returns
                            downside_deviation = abs(negative_returns[0]) * np.sqrt(252)
                            sortino_ratio = (annual_return - risk_free_rate) / downside_deviation
                            print(f"DEBUG: Single negative return case (annualized):")
                            print(f"DEBUG: Time period: {years:.2f} years")
                            print(f"DEBUG: Annual return: {annual_return*100:.4f}%")
                            print(f"DEBUG: Single negative return: {negative_returns[0]*100:.6f}%")
                            print(f"DEBUG: Downside deviation (annualized): {downside_deviation*100:.4f}%")
                            print(f"DEBUG: Sortino ratio: {sortino_ratio:.4f}")
                        else:
                            # When no negative returns, use a more realistic approach
                            if annual_return > risk_free_rate:
                                # Use the minimum return as a proxy for downside risk
                                min_return = np.min(returns)
                                if min_return < 0:
                                    # Use the minimum return as downside deviation (annualized)
                                    downside_deviation = abs(min_return) / np.sqrt(years)
                                    sortino_ratio = (annual_return - risk_free_rate) / downside_deviation
                                else:
                                    # All returns are positive, use a conservative estimate
                                    # Cap at a reasonable maximum (e.g., 10.0)
                                    sortino_ratio = min(10.0, (annual_return - risk_free_rate) / 0.01)
                            else:
                                sortino_ratio = 0
                                
                            print(f"DEBUG: No negative returns case (using conservative estimate, annualized):")
                            print(f"DEBUG: Time period: {years:.2f} years")
                            print(f"DEBUG: Annual return: {annual_return*100:.4f}%")
                            print(f"DEBUG: Sortino ratio: {sortino_ratio:.4f}")
                    else:
                        sortino_ratio = 0
                        print(f"DEBUG: Portfolio-level Sortino calculation: Time period too short ({years:.2f} years)")
                else:
                    sortino_ratio = 0
                    print(f"DEBUG: Portfolio-level Sortino calculation: Insufficient trade data")
            
            # MAR (Maximum Adverse Return): Annual Return / Maximum Drawdown
            # Use the max drawdowns from the simulation results
            print(f"DEBUG: === MAR (MAXIMUM ADVERSE RETURN) CALCULATION ===")
            if max_drawdowns is not None:
                print(f"DEBUG: Max drawdowns array length: {len(max_drawdowns)}")
                if len(max_drawdowns) > 0:
                    print(f"DEBUG: Sample max drawdowns (first 5): {max_drawdowns[:5]}")
                    print(f"DEBUG: Mean max drawdown: ${np.mean(max_drawdowns):,.2f}")
                else:
                    print(f"DEBUG: No max drawdowns available")
            else:
                print(f"DEBUG: Max drawdowns is None")
            
            if max_drawdowns is not None and len(max_drawdowns) > 0:
                mean_max_drawdown = np.mean(max_drawdowns)
                if mean_max_drawdown > 0:
                    # Calculate annual return for MAR (Maximum Adverse Return)
                    if strategy_trades and days_elapsed > 0:
                        # Strategy-specific: already have annual_return
                        # Calculate initial balance for this strategy
                        strategy_initial_balance = sorted_trades[0].funds_at_close - sorted_trades[0].pnl
                        print(f"DEBUG: Strategy-specific MAR calculation:")
                        print(f"DEBUG: Annual return: {annual_return*100:.4f}%")
                        print(f"DEBUG: Mean max drawdown: ${mean_max_drawdown:,.2f}")
                        print(f"DEBUG: Strategy initial balance: ${strategy_initial_balance:,.2f}")
                        print(f"DEBUG: Max drawdown as % of balance: {(mean_max_drawdown/strategy_initial_balance)*100:.4f}%")
                        # Use the actual maximum drawdown from portfolio overview for consistency
                        # Get the actual max drawdown from portfolio metrics
                        metrics = PortfolioMetrics(self.portfolio)
                        portfolio_metrics = metrics.get_overview_metrics()
                        actual_max_drawdown_pct = abs(portfolio_metrics['max_drawdown_pct']) / 100.0
                        print(f"DEBUG: Portfolio metrics max_drawdown_pct: {portfolio_metrics['max_drawdown_pct']:.2f}%")
                        print(f"DEBUG: Actual max drawdown (decimal): {actual_max_drawdown_pct:.4f}")
                        mar_ratio = annual_return / actual_max_drawdown_pct
                        print(f"DEBUG: MAR (Maximum Adverse Return): {mar_ratio:.4f}")
                    else:
                        # Portfolio-level: need to calculate annual return
                        all_trades = []
                        for strategy in self.portfolio.strategies.values():
                            all_trades.extend(strategy.trades)
                        
                        if len(all_trades) > 1:
                            sorted_trades = sorted(all_trades, key=lambda t: pd.to_datetime(t.date_closed))
                            start_date = pd.to_datetime(sorted_trades[0].date_closed)
                            end_date = pd.to_datetime(sorted_trades[-1].date_closed)
                            days_elapsed = (end_date - start_date).days
                            years = days_elapsed / 365.25
                            
                            if years > 0:
                                # Use the same annual return calculation as Sharpe and Sortino ratios
                                initial_balance = sorted_trades[0].funds_at_close - sorted_trades[0].pnl
                                final_balance = sorted_trades[-1].funds_at_close
                                total_return = (final_balance - initial_balance) / initial_balance
                                annual_return = (1 + total_return) ** (1/years) - 1
                                
                                # Use the actual maximum drawdown from portfolio overview for consistency
                                metrics = PortfolioMetrics(self.portfolio)
                                portfolio_metrics = metrics.get_overview_metrics()
                                actual_max_drawdown_pct = abs(portfolio_metrics['max_drawdown_pct']) / 100.0
                                mar_ratio = annual_return / actual_max_drawdown_pct
                                
                                print(f"DEBUG: Portfolio-level MAR calculation:")
                                print(f"DEBUG: Time period: {years:.2f} years")
                                print(f"DEBUG: Annual return: {annual_return*100:.4f}%")
                                print(f"DEBUG: Portfolio metrics max_drawdown_pct: {portfolio_metrics['max_drawdown_pct']:.2f}%")
                                print(f"DEBUG: Actual max drawdown (decimal): {actual_max_drawdown_pct:.4f}")
                                print(f"DEBUG: MAR (Maximum Adverse Return): {mar_ratio:.4f}")
                            else:
                                mar_ratio = 0
                                print(f"DEBUG: Portfolio-level MAR calculation: Time period too short ({years:.2f} years)")
                        else:
                            mar_ratio = 0
                            print(f"DEBUG: Portfolio-level MAR calculation: Insufficient trade data")
                else:
                    # When no drawdown, use a conservative estimate
                    if strategy_trades and days_elapsed > 0:
                        mar_ratio = min(10.0, annual_return / 0.01)
                    else:
                        mar_ratio = min(10.0, mean_return / 0.01)
            else:
                # Fallback
                if strategy_trades and days_elapsed > 0:
                    mar_ratio = min(10.0, annual_return / 0.01)
                else:
                    mar_ratio = min(10.0, mean_return / 0.01)
            
            print(f"DEBUG: MAR (Maximum Adverse Return) calculation completed: {mar_ratio}")
            
            # Ensure actual_max_drawdown_pct is defined for return values
            if 'actual_max_drawdown_pct' not in locals():
                # Fallback: get from portfolio metrics
                metrics = PortfolioMetrics(self.portfolio)
                portfolio_metrics = metrics.get_overview_metrics()
                actual_max_drawdown_pct = abs(portfolio_metrics['max_drawdown_pct']) / 100.0
            
            # Information Ratio: (Portfolio Return - Benchmark Return) / Tracking Error
            # Use annualized returns for consistency with other ratios
            if strategy_trades and days_elapsed > 0:
                # Strategy-specific: use the annualized return we already calculated
                benchmark_return = risk_free_rate  # Use risk-free rate as benchmark
                excess_return = annual_return - benchmark_return
                # Use annualized volatility for tracking error
                tracking_error = annual_volatility
            else:
                # Portfolio-level: use annualized return
                benchmark_return = risk_free_rate
                excess_return = annual_return - benchmark_return
                # Use annualized volatility for tracking error
                tracking_error = annual_volatility
            
            if tracking_error > 0:
                information_ratio = excess_return / tracking_error
            else:
                information_ratio = 0
            
            # Return to Risk Ratio: Annual Return / Annual Volatility
            # Use annualized values for consistency with other ratios
            return_to_risk = annual_return / annual_volatility if annual_volatility > 0 else 0
            print(f"DEBUG: Return to Risk calculation:")
            print(f"DEBUG: Mean return: {mean_return:.6f}")
            print(f"DEBUG: Std return: {std_return:.6f}")
            print(f"DEBUG: Return to Risk: {return_to_risk:.6f}")
            
            # Debug the raw risk ratio values
            print(f"DEBUG: Raw risk ratio values:")
            print(f"DEBUG: Sharpe ratio: {sharpe_ratio} (type: {type(sharpe_ratio)})")
            print(f"DEBUG: Sortino ratio: {sortino_ratio} (type: {type(sortino_ratio)})")
            print(f"DEBUG: MAR (Maximum Adverse Return): {mar_ratio} (type: {type(mar_ratio)})")
            print(f"DEBUG: Information ratio: {information_ratio} (type: {type(information_ratio)})")
            print(f"DEBUG: Return to risk: {return_to_risk} (type: {type(return_to_risk)})")
            
            # Debug the safe_float conversions
            print(f"DEBUG: After safe_float conversion:")
            print(f"DEBUG: Sharpe ratio: {safe_float(sharpe_ratio)}")
            print(f"DEBUG: Sortino ratio: {safe_float(sortino_ratio)}")
            print(f"DEBUG: MAR (Maximum Adverse Return): {safe_float(mar_ratio)}")
            print(f"DEBUG: Information ratio: {safe_float(information_ratio)}")
            print(f"DEBUG: Return to risk: {safe_float(return_to_risk)}")
            
            print(f"DEBUG: === FINAL RISK RATIO RETURN ===")
            print(f"DEBUG: About to return risk ratios")
            
            return {
                'sharpe_ratio': safe_float(sharpe_ratio),
                'sortino_ratio': safe_float(sortino_ratio),
                'mar_ratio': safe_float(mar_ratio),
                'information_ratio': safe_float(information_ratio),
                'return_to_risk': safe_float(return_to_risk),
                'mean_return_pct': annual_return * 100,  # Convert to percentage
                'std_return_pct': annual_volatility * 100,  # Convert to percentage
                'downside_deviation_pct': (np.std(negative_returns) * np.sqrt(trades_per_year) if len(negative_returns) > 0 and 'trades_per_year' in locals() else 0) * 100,  # Convert to percentage
                'mean_max_drawdown_pct': actual_max_drawdown_pct
            }
            
        except Exception as e:
            # Return default values if calculation fails
            print(f"DEBUG: Exception in _calculate_risk_ratios: {e}")
            import traceback
            print(f"DEBUG: Traceback: {traceback.format_exc()}")
            return {
                'sharpe_ratio': 0,
                'sortino_ratio': 0,
                'mar_ratio': 0,
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
            risk_free_rate = Config.RISK_FREE_RATE_PCT  # Use configurable risk-free rate
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
                
                # Calculate actual contract count from legs, fallback to CSV field
                legs = getattr(trade, 'legs', '')
                if legs and legs.strip():
                    from commission_config import CommissionCalculator
                    calc = CommissionCalculator()
                    contracts = calc.calculate_actual_contracts_from_legs(legs)
                else:
                    contracts = getattr(trade, 'contracts', 1)
                strategy_data[strategy_name]['total_contracts'] += contracts
                
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

class MEICAnalyzer:
    """Analyzes Multiple Entry Iron Condor (MEIC) trades."""
    
    def __init__(self, portfolio: Portfolio, filename: str = None):
        self.portfolio = portfolio
        self.filename = filename
        self._meic_trades_cache = None
        self._leg_groups_cache = None
        self.meic_trades = self._get_meic_trades()
        self.leg_groups = self._group_trades_by_entry_time()
    
    def _get_meic_trades(self) -> List[Trade]:
        """Get all MEIC trades from the portfolio (optimized with caching)."""
        # Return cached result if available
        if self._meic_trades_cache is not None:
            return self._meic_trades_cache
        
        meic_trades = []
        
        # Pre-check filename to avoid repeated string operations
        filename_has_meic = self.filename and 'meic' in self.filename.lower()
        
        for strategy_name, strategy in self.portfolio.strategies.items():
            # Check strategy name once per strategy
            strategy_has_meic = 'meic' in strategy_name.lower()
            
            # If strategy name contains 'meic', add all trades
            if strategy_has_meic:
                meic_trades.extend(strategy.trades)
            elif filename_has_meic:
                # Only check individual trades if filename has 'meic' but strategy doesn't
                for trade in strategy.trades:
                    if trade.is_meic_trade(self.filename):
                        meic_trades.append(trade)
        
        # Cache the result
        self._meic_trades_cache = meic_trades
        return meic_trades
    
    def _group_trades_by_entry_time(self) -> Dict[str, Dict]:
        """Group MEIC trades by entry time to identify put/call spread pairs (optimized with caching)."""
        # Return cached result if available
        if self._leg_groups_cache is not None:
            return self._leg_groups_cache
        
        leg_groups = {}
        
        # Pre-allocate the groups structure for better performance
        for trade in self.meic_trades:
            # Create a key based on entry date and time (optimized string formatting)
            entry_key = f"{trade.date_opened}_{trade.time_opened}"
            
            # Use setdefault for cleaner code and better performance
            if entry_key not in leg_groups:
                leg_groups[entry_key] = {
                    'put_spread': None,
                    'call_spread': None,
                    'entry_date': trade.date_opened,
                    'entry_time': trade.time_opened
                }
            
            # Classify the trade as put or call spread (optimized)
            if trade.is_put_spread():
                leg_groups[entry_key]['put_spread'] = trade
            elif trade.is_call_spread():
                leg_groups[entry_key]['call_spread'] = trade
        
        # Cache the result
        self._leg_groups_cache = leg_groups
        return leg_groups
    
    def get_stopout_statistics(self) -> Dict:
        """Calculate stopout statistics for MEIC leg groups."""
        # Filter leg groups to only include valid MEIC pairs (both put and call spreads)
        valid_leg_groups = {}
        for entry_key, group in self.leg_groups.items():
            if group['put_spread'] and group['call_spread']:
                valid_leg_groups[entry_key] = group
        
        if not valid_leg_groups:
            return {
                'success': False,
                'error': 'No valid MEIC trades found',
                'data': {
                    'total_leg_groups': 0,
                    'total_contracts': 0,
                    'total_commissions': 0,
                    'no_stopouts': {'count': 0, 'percentage': 0},
                    'puts_stopped': {'count': 0, 'percentage': 0},
                    'calls_stopped': {'count': 0, 'percentage': 0},
                    'both_stopped': {'count': 0, 'percentage': 0}
                }
            }
        
        total_groups = len(valid_leg_groups)
        no_stopouts = 0
        puts_stopped = 0
        calls_stopped = 0
        both_stopped = 0
        
        for group in valid_leg_groups.values():
            put_stopped = group['put_spread'] and group['put_spread'].was_stopped_out()
            call_stopped = group['call_spread'] and group['call_spread'].was_stopped_out()
            
            if not put_stopped and not call_stopped:
                no_stopouts += 1
            elif put_stopped and not call_stopped:
                puts_stopped += 1
            elif not put_stopped and call_stopped:
                calls_stopped += 1
            elif put_stopped and call_stopped:
                both_stopped += 1
        
        # Calculate additional statistics
        from commission_config import CommissionCalculator
        calc = CommissionCalculator()
        
        total_contracts = 0
        for trade in self.meic_trades:
            legs = getattr(trade, 'legs', '')
            if legs and legs.strip():
                contracts = calc.calculate_actual_contracts_from_legs(legs)
            else:
                contracts = getattr(trade, 'contracts', 1)
            total_contracts += contracts
            
        total_commissions = sum(trade.total_commissions for trade in self.meic_trades)
        
        return {
            'success': True,
            'data': {
                'total_leg_groups': total_groups,
                'total_contracts': total_contracts,
                'total_commissions': round(total_commissions, 2),
                'no_stopouts': {
                    'count': no_stopouts,
                    'percentage': round((no_stopouts / total_groups) * 100, 1) if total_groups > 0 else 0
                },
                'puts_stopped': {
                    'count': puts_stopped,
                    'percentage': round((puts_stopped / total_groups) * 100, 1) if total_groups > 0 else 0
                },
                'calls_stopped': {
                    'count': calls_stopped,
                    'percentage': round((calls_stopped / total_groups) * 100, 1) if total_groups > 0 else 0
                },
                'both_stopped': {
                    'count': both_stopped,
                    'percentage': round((both_stopped / total_groups) * 100, 1) if total_groups > 0 else 0
                }
            }
        }
    
    def get_time_heatmap_data(self, start_date: str = None, end_date: str = None) -> Dict:
        """Generate heatmap data showing P&L by day of week and entry time with optional date filtering."""
        # Filter leg groups to only include valid MEIC pairs (both put and call spreads)
        valid_leg_groups = {}
        for entry_key, group in self.leg_groups.items():
            if group['put_spread'] and group['call_spread']:
                valid_leg_groups[entry_key] = group
        
        if not valid_leg_groups:
            return {
                'success': False,
                'error': 'No valid MEIC trades found',
                'data': {
                    'pnl_heatmap': {},
                    'summary_stats': {
                        'total_leg_groups': 0,
                        'total_pnl': 0,
                        'avg_pnl_per_group': 0
                    },
                    'date_range': {
                        'start_date': None,
                        'end_date': None
                    }
                }
            }
        
        # Parse date filters if provided
        start_date_parsed = None
        end_date_parsed = None
        
        if start_date:
            try:
                start_date_parsed = pd.to_datetime(start_date)
            except:
                pass
        
        if end_date:
            try:
                end_date_parsed = pd.to_datetime(end_date)
            except:
                pass
        
        # Create a DataFrame for easier manipulation
        heatmap_data = []
        
        for group in valid_leg_groups.values():
            entry_date = pd.to_datetime(group['entry_date'])
            
            # Apply date filtering
            if start_date_parsed and entry_date < start_date_parsed:
                continue
            if end_date_parsed and entry_date > end_date_parsed:
                continue
            
            day_of_week = entry_date.strftime('%A')
            entry_time = group['entry_time']
            
            # Calculate total P&L for this leg group
            total_pnl = 0
            if group['put_spread']:
                total_pnl += group['put_spread'].pnl
            if group['call_spread']:
                total_pnl += group['call_spread'].pnl
            
            heatmap_data.append({
                'day_of_week': day_of_week,
                'entry_time': entry_time,
                'total_pnl': total_pnl,
                'entry_date': entry_date
            })
        
        df = pd.DataFrame(heatmap_data)
        
        # Group by day of week and entry time
        heatmap_summary = df.groupby(['day_of_week', 'entry_time']).agg({
            'total_pnl': ['sum', 'count', 'mean']
        }).round(2)
        
        # Flatten column names
        heatmap_summary.columns = ['total_pnl', 'trade_count', 'avg_pnl']
        heatmap_summary = heatmap_summary.reset_index()
        
        # Create pivot table for heatmap
        pivot_pnl = heatmap_summary.pivot(index='day_of_week', columns='entry_time', values='total_pnl').fillna(0)
        pivot_count = heatmap_summary.pivot(index='day_of_week', columns='entry_time', values='trade_count').fillna(0)
        
        # Order days of week
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        pivot_pnl = pivot_pnl.reindex([day for day in day_order if day in pivot_pnl.index])
        pivot_count = pivot_count.reindex([day for day in day_order if day in pivot_pnl.index])
        
        # Convert to ordered dictionary to preserve day order
        ordered_pnl_heatmap = {}
        for day in pivot_pnl.index:
            ordered_pnl_heatmap[day] = pivot_pnl.loc[day].to_dict()
        
        ordered_count_heatmap = {}
        for day in pivot_count.index:
            ordered_count_heatmap[day] = pivot_count.loc[day].to_dict()
        
        # Calculate P&L totals per day
        daily_pnl_totals = {}
        for day in pivot_pnl.index:
            daily_pnl_totals[day] = round(pivot_pnl.loc[day].sum(), 2)
        
        # Calculate date range for sliders
        date_range = None
        if not df.empty:
            min_date = df['entry_date'].min()
            max_date = df['entry_date'].max()
            date_range = {
                'min_date': min_date.strftime('%Y-%m-%d'),
                'max_date': max_date.strftime('%Y-%m-%d'),
                'total_trades': len(df)
            }

        # Calculate total P&L from ALL MEIC trades (gross P&L, no commission subtraction)
        total_meic_pnl = sum(trade.pnl for trade in self.meic_trades)
        
        return {
            'success': True,
            'data': {
                'pnl_heatmap': ordered_pnl_heatmap,
                'count_heatmap': ordered_count_heatmap,
                'daily_pnl_totals': daily_pnl_totals,
                'summary_stats': {
                    'total_leg_groups': len(self.leg_groups),
                    'total_pnl': round(total_meic_pnl, 2),  # Gross P&L from all MEIC trades
                    'avg_pnl_per_group': round(df['total_pnl'].mean(), 2) if not df.empty else 0,
                    'best_day': df.groupby('day_of_week')['total_pnl'].sum().idxmax() if not df.empty else None,
                    'worst_day': df.groupby('day_of_week')['total_pnl'].sum().idxmin() if not df.empty else None
                },
                'date_range': date_range
            }
        } 