import pandas as pd
import numpy as np
from typing import Dict, List
import os


class PortfolioAggregator:
    """
    Runs all aggregations and exports a summary report.
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
    
    def _calculate_notional(self, positions: pd.DataFrame) -> pd.Series:
        """Calculate notional value for positions."""
        if 'quantity' in positions.columns and 'spot_price' in positions.columns:
            return positions['quantity'].abs() * positions['spot_price']
        return pd.Series(0.0, index=positions.index)
    
    def load_positions_with_greeks(self) -> pd.DataFrame:
        """
        Load positions with greeks from CSV.
        
        Returns:
            DataFrame with positions and their greeks
        
        Raises:
            FileNotFoundError: If positions_with_greeks.csv does not exist
        """
        filepath = os.path.join(self.data_dir, "positions_with_greeks.csv")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Positions file not found: {filepath}")
        return pd.read_csv(filepath)
    
    def aggregate_portfolio_greeks(self, positions: pd.DataFrame) -> Dict:
        """
        Sum all position greeks and calculate total notional.
        
        Args:
            positions: DataFrame with positions and their greeks (from positions_with_greeks.csv)
        
        Returns:
            Dictionary with total_delta, total_gamma, total_vega, total_theta, total_rho, total_notional, num_positions
        """
        greeks = ['delta', 'gamma', 'vega', 'theta', 'rho']
        
        exposures = {}
        for greek in greeks:
            col = f'position_{greek}'
            exposures[f'total_{greek}'] = float(positions[col].sum()) if col in positions.columns else 0.0
        
        # Calculate total notional (absolute value of quantity * spot_price)
        if 'quantity' in positions.columns and 'spot_price' in positions.columns:
            exposures['total_notional'] = float((positions['quantity'].abs() * positions['spot_price']).sum())
        else:
            exposures['total_notional'] = 0.0
        
        exposures['num_positions'] = len(positions)
        
        return exposures
    
    def load_and_aggregate_portfolio_greeks(self) -> Dict:
        """
        Load positions with greeks and aggregate to portfolio-level totals.
        
        This is a convenience method that combines load_positions_with_greeks() 
        and aggregate_portfolio_greeks().
        
        Returns:
            Dictionary with total_delta, total_gamma, total_vega, total_theta, total_rho, total_notional, num_positions
        
        Raises:
            FileNotFoundError: If positions_with_greeks.csv does not exist
        """
        positions = self.load_positions_with_greeks()
        return self.aggregate_portfolio_greeks(positions)
    
    def get_unique_symbols(self) -> List[str]:
        """
        Get unique symbols from positions with greeks.
        
        Returns:
            List of unique symbol strings
        
        Raises:
            FileNotFoundError: If positions_with_greeks.csv does not exist
        """
        positions = self.load_positions_with_greeks()
        if 'symbol' in positions.columns:
            return positions['symbol'].unique().tolist()
        return []
    
    def aggregate_by_symbol(self, positions: pd.DataFrame) -> pd.DataFrame:
        """
        Group exposures per stock symbol.
        
        Args:
            positions: DataFrame with positions and their greeks
        
        Returns:
            DataFrame with columns: symbol, delta, gamma, vega, theta, rho, notional, num_positions
            Sorted by absolute delta (largest risk first)
        """
        if positions.empty or 'symbol' not in positions.columns:
            return pd.DataFrame({
                'symbol': [], 'delta': [], 'gamma': [], 'vega': [], 
                'theta': [], 'rho': [], 'notional': [], 'num_positions': []
            })
        
        # Calculate notional for each position first
        positions_copy = positions.copy()
        if 'quantity' in positions_copy.columns and 'spot_price' in positions_copy.columns:
            positions_copy['notional'] = positions_copy['quantity'].abs() * positions_copy['spot_price']
        else:
            positions_copy['notional'] = 0.0
        
        # Group by symbol and aggregate
        grouped = positions_copy.groupby('symbol').agg({
            'position_delta': 'sum',
            'position_gamma': 'sum',
            'position_vega': 'sum',
            'position_theta': 'sum',
            'position_rho': 'sum',
            'notional': 'sum',
            'position_id': 'count'
        }).reset_index()
        
        # Rename columns
        result = pd.DataFrame({
            'symbol': grouped['symbol'],
            'delta': grouped['position_delta'],
            'gamma': grouped['position_gamma'],
            'vega': grouped['position_vega'],
            'theta': grouped['position_theta'],
            'rho': grouped['position_rho'],
            'notional': grouped['notional'],
            'num_positions': grouped['position_id']
        })
        
        # Sort by absolute delta (largest risk first)
        abs_deltas = np.abs(np.asarray(result['delta']))
        sorted_indices = np.argsort(abs_deltas)[::-1]
        result = result.iloc[sorted_indices].reset_index(drop=True)
        
        return result
    
    def aggregate_by_instrument_type(self, positions: pd.DataFrame) -> pd.DataFrame:
        """
        Split exposures by equities vs. options.
        
        Args:
            positions: DataFrame with positions and their greeks
        
        Returns:
            DataFrame with columns: instrument_type, delta, gamma, vega, theta, rho, notional, num_positions
        """
        if positions.empty or 'instrument_type' not in positions.columns:
            return pd.DataFrame({
                'instrument_type': [], 'delta': [], 'gamma': [], 'vega': [], 
                'theta': [], 'rho': [], 'notional': [], 'num_positions': []
            })
        
        # Calculate notional for each position first
        positions_copy = positions.copy()
        positions_copy['notional'] = self._calculate_notional(positions_copy)
        
        # Group by instrument_type and aggregate
        grouped = positions_copy.groupby('instrument_type').agg({
            'position_delta': 'sum',
            'position_gamma': 'sum',
            'position_vega': 'sum',
            'position_theta': 'sum',
            'position_rho': 'sum',
            'notional': 'sum',
            'position_id': 'count'
        }).reset_index()
        
        # Rename columns
        result = pd.DataFrame({
            'instrument_type': grouped['instrument_type'],
            'delta': grouped['position_delta'],
            'gamma': grouped['position_gamma'],
            'vega': grouped['position_vega'],
            'theta': grouped['position_theta'],
            'rho': grouped['position_rho'],
            'notional': grouped['notional'],
            'num_positions': grouped['position_id']
        })
        
        return result
    
    def identify_top_risks(self, positions: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        """
        List largest individual positions by absolute delta.
        
        Args:
            positions: DataFrame with positions and their greeks
            top_n: Number of top risky positions to return (default: 10)
        
        Returns:
            DataFrame with top N risky positions, including position_id, symbol, 
            instrument_type, position_delta, and other key fields
        """
        if positions.empty or 'position_delta' not in positions.columns:
            return pd.DataFrame()
        
        # Select relevant columns for top risks
        risk_cols = ['position_id', 'symbol', 'instrument_type', 'quantity', 
                     'position_delta', 'position_gamma', 'position_vega', 
                     'position_theta', 'position_rho']
        
        # Filter to available columns
        available_cols = [col for col in risk_cols if col in positions.columns]
        risk_df = positions[available_cols].copy()
        
        # Sort by absolute position_delta (largest risk first) and return top N
        abs_deltas = np.abs(np.asarray(risk_df['position_delta']))
        sorted_indices = np.argsort(abs_deltas)[::-1][:top_n]
        top_risks_df = risk_df.iloc[sorted_indices].copy()
        
        return top_risks_df.reset_index(drop=True)
    
    def generate_summary_report(self, positions: pd.DataFrame) -> Dict:
        """
        Generate complete portfolio summary report with all aggregations.
        
        Args:
            positions: DataFrame with positions and their greeks
        
        Returns:
            Dictionary containing:
            - portfolio_summary: Portfolio-level totals
            - symbol_breakdown: Aggregation by symbol
            - instrument_type_breakdown: Aggregation by instrument type
            - top_risks: Top N risky positions
        """
        report = {}
        
        # Portfolio-level summary
        report['portfolio_summary'] = self.aggregate_portfolio_greeks(positions)
        
        # Symbol breakdown
        report['symbol_breakdown'] = self.aggregate_by_symbol(positions).to_dict('records')
        
        # Instrument type breakdown
        report['instrument_type_breakdown'] = self.aggregate_by_instrument_type(positions).to_dict('records')
        
        # Top risks
        report['top_risks'] = self.identify_top_risks(positions, top_n=10).to_dict('records')
        
        return report
    
    def load_and_aggregate_by_symbol(self) -> pd.DataFrame:
        """
        Convenience method: Load positions and aggregate by symbol.
        
        Returns:
            DataFrame with columns: symbol, delta, gamma, vega, theta, rho, notional, num_positions
        """
        positions = self.load_positions_with_greeks()
        return self.aggregate_by_symbol(positions)
    
    def load_and_aggregate_by_instrument_type(self) -> pd.DataFrame:
        """
        Convenience method: Load positions and aggregate by instrument type.
        
        Returns:
            DataFrame with columns: instrument_type, delta, gamma, vega, theta, rho, notional, num_positions
        """
        positions = self.load_positions_with_greeks()
        return self.aggregate_by_instrument_type(positions)
    
    def load_and_identify_top_risks(self, top_n: int = 10) -> pd.DataFrame:
        """
        Convenience method: Load positions and identify top risks.
        
        Args:
            top_n: Number of top risky positions to return (default: 10)
        
        Returns:
            DataFrame with top N risky positions
        """
        positions = self.load_positions_with_greeks()
        return self.identify_top_risks(positions, top_n=top_n)
    
    def load_and_generate_summary_report(self) -> Dict:
        """
        Convenience method: Load positions and generate complete summary report.
        
        Returns:
            Dictionary containing portfolio_summary, symbol_breakdown, 
            instrument_type_breakdown, and top_risks
        """
        positions = self.load_positions_with_greeks()
        return self.generate_summary_report(positions)
