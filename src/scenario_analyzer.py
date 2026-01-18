"""
Scenario Analysis Module

Purpose: Calculate portfolio P&L under different market scenarios using greeks approximation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from datetime import datetime

# Handle imports for both package and direct script execution
try:
    from .portfolio_aggregator import PortfolioAggregator
except ImportError:
    from portfolio_aggregator import PortfolioAggregator


class ScenarioAnalyzer:
    """
    Analyzes portfolio performance under different market scenarios using greeks approximation.
    
    Uses the Taylor expansion formula:
    P&L ≈ Δ*ΔS + 0.5*Γ*(ΔS)² + ν*Δσ + θ*Δt + ρ*Δr
    
    Where:
    - Δ = delta (price sensitivity)
    - Γ = gamma (second-order price sensitivity)
    - ν = vega (volatility sensitivity)
    - θ = theta (time decay)
    - ρ = rho (interest rate sensitivity)
    - ΔS = price change
    - Δσ = volatility change
    - Δt = time change
    - Δr = interest rate change
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize ScenarioAnalyzer.
        
        Args:
            data_dir: Directory containing data files
        """
        self.data_dir = data_dir
        self.portfolio_aggregator = PortfolioAggregator(data_dir=data_dir)
    
    def calculate_scenario_pnl(
        self,
        price_change_pct: float,
        vol_change_pct: float,
        rate_change_bps: int,
        time_decay_days: int,
        portfolio_summary: Optional[Dict] = None,
        positions: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        Calculate portfolio P&L under a given scenario.
        
        Args:
            price_change_pct: Price change percentage (e.g., 5.0 for +5%)
            vol_change_pct: Volatility change percentage (e.g., 10.0 for +10%)
            rate_change_bps: Interest rate change in basis points (e.g., 25 for +25 bps)
            time_decay_days: Time decay in days (e.g., 7 for 7 days)
            portfolio_summary: Optional portfolio summary dict. If None, will load from file.
            positions: Optional positions DataFrame. If None, will load from file.
        
        Returns:
            Dictionary containing:
            - total_pnl: Total P&L in dollars
            - delta_pnl: P&L from delta (price movement)
            - gamma_pnl: P&L from gamma (second-order price effect)
            - vega_pnl: P&L from vega (volatility change)
            - theta_pnl: P&L from theta (time decay)
            - rho_pnl: P&L from rho (interest rate change)
            - scenario_params: Dictionary of input parameters
            - breakdown: DataFrame with detailed breakdown
        """
        # Load portfolio summary if not provided
        if portfolio_summary is None:
            portfolio_summary = self.portfolio_aggregator.load_and_aggregate_portfolio_greeks()
        
        # Load positions if needed for average spot price
        if positions is None:
            try:
                positions = self.portfolio_aggregator.load_positions_with_greeks()
            except FileNotFoundError:
                positions = pd.DataFrame()
        
        # Get average spot price for scaling
        if 'spot_price' in positions.columns and not positions.empty:
            avg_spot = float(positions['spot_price'].mean())
        else:
            avg_spot = 100.0  # Default
        
        # Calculate changes
        delta_S = avg_spot * (price_change_pct / 100.0)  # Price change in dollars
        delta_sigma = (vol_change_pct / 100.0)  # Volatility change (absolute)
        delta_t = time_decay_days / 365.0  # Time decay in years
        delta_r = rate_change_bps / 10000.0  # Rate change in decimal (bps to decimal)
        
        # Calculate P&L components using greeks approximation
        # P&L ≈ Δ*ΔS + 0.5*Γ*(ΔS)² + ν*Δσ + θ*Δt + ρ*Δr
        delta_pnl = portfolio_summary['total_delta'] * delta_S
        gamma_pnl = 0.5 * portfolio_summary['total_gamma'] * (delta_S ** 2)
        vega_pnl = portfolio_summary['total_vega'] * delta_sigma
        theta_pnl = portfolio_summary['total_theta'] * delta_t
        rho_pnl = portfolio_summary['total_rho'] * delta_r
        
        total_pnl = delta_pnl + gamma_pnl + vega_pnl + theta_pnl + rho_pnl
        
        # Create breakdown DataFrame
        breakdown = pd.DataFrame({
            'Component': ['Delta', 'Gamma', 'Vega', 'Theta', 'Rho', 'Total'],
            'P&L ($)': [delta_pnl, gamma_pnl, vega_pnl, theta_pnl, rho_pnl, total_pnl],
            'Input Change': [
                f"{price_change_pct:+.1f}%",
                f"{price_change_pct:+.1f}%",
                f"{vol_change_pct:+.1f}%",
                f"{time_decay_days} days",
                f"{rate_change_bps:+d} bps",
                "Combined"
            ],
            'Greek Exposure': [
                f"{portfolio_summary['total_delta']:,.2f}",
                f"{portfolio_summary['total_gamma']:,.4f}",
                f"{portfolio_summary['total_vega']:,.2f}",
                f"{portfolio_summary['total_theta']:,.2f}",
                f"{portfolio_summary['total_rho']:,.2f}",
                "-"
            ]
        })
        
        return {
            'total_pnl': total_pnl,
            'delta_pnl': delta_pnl,
            'gamma_pnl': gamma_pnl,
            'vega_pnl': vega_pnl,
            'theta_pnl': theta_pnl,
            'rho_pnl': rho_pnl,
            'scenario_params': {
                'price_change_pct': price_change_pct,
                'vol_change_pct': vol_change_pct,
                'rate_change_bps': rate_change_bps,
                'time_decay_days': time_decay_days,
                'avg_spot_price': avg_spot
            },
            'breakdown': breakdown,
            'portfolio_summary': portfolio_summary
        }
    
    def calculate_pnl_percentage(self, total_pnl: float, total_notional: float) -> Optional[float]:
        """
        Calculate P&L as percentage of notional.
        
        Args:
            total_pnl: Total P&L in dollars
            total_notional: Total notional value
        
        Returns:
            P&L percentage, or None if notional is zero
        """
        if total_notional > 0:
            return (total_pnl / total_notional) * 100.0
        return None
