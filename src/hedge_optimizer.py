import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import os
from datetime import datetime
from scipy.optimize import minimize
from greeks_calculator import GreeksCalculator


class HedgeOptimizer:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.greeks_calculator = GreeksCalculator(data_dir)
    
    def _parse_config(self, config: Optional[Dict]) -> Dict:
        """Parse and return configuration with defaults."""
        config = config or {}
        return {
            'include_etfs': config.get('include_etfs', True),
            'etf_symbols': config.get('etf_symbols', ['SPY']),
            'default_transaction_cost_bps': config.get('default_transaction_cost_bps', 5.0),
            'default_max_quantity': config.get('default_max_quantity', 100000.0),
            'use_market_data': config.get('use_market_data', True)
        }
    
    def _build_hedge_symbols_list(self, symbols: List[str], include_etfs: bool, 
                                  etf_symbols: List[str]) -> List[str]:
        """Build list of hedge symbols including ETFs and removing duplicates."""
        return list(dict.fromkeys(symbols + (etf_symbols if include_etfs else [])))
    
    def _load_market_data_dict(self, use_market_data: bool) -> Dict[str, Tuple[float, float, Optional[float]]]:
        """
        Load market data from CSV and create a lookup dictionary.
        
        Returns:
            Dictionary mapping symbol to (spot_price, borrow_cost_bps, transaction_cost_bps)
        """
        if not use_market_data:
            return {}
        
        try:
            market_data_path = os.path.join(self.data_dir, "market_data.csv")
            if not os.path.exists(market_data_path):
                return {}
            
            market_data = pd.read_csv(market_data_path)
            market_data_dict: Dict[str, Tuple[float, float, Optional[float]]] = {}
            
            # Handle case where transaction_cost_bps might not exist in older CSV files
            has_transaction_cost = 'transaction_cost_bps' in market_data.columns
            for _, row in market_data.iterrows():
                transaction_cost = None
                if has_transaction_cost:
                    val = row['transaction_cost_bps']
                    # Check if value is valid (not NaN)
                    if isinstance(val, (int, float)) and not np.isnan(val):
                        transaction_cost = float(val)
                
                market_data_dict[str(row['symbol'])] = (
                    float(row['spot_price']),
                    float(row['borrow_cost_bps']),
                    transaction_cost
                )
            
            return market_data_dict
        except Exception:
            return {}
    
    def _extract_market_data_for_symbol(self, symbol: str, 
                                       market_data_dict: Dict[str, Tuple[float, float, Optional[float]]],
                                       default_transaction_cost_bps: float) -> Dict[str, float]:
        """Extract market data for a symbol, using defaults if not found."""
        if symbol in market_data_dict:
            spot_price, borrow_cost_bps, transaction_cost = market_data_dict[symbol]
            return {
                'spot_price': float(spot_price),
                'borrow_cost_bps': float(borrow_cost_bps),
                'transaction_cost_bps': float(transaction_cost) if transaction_cost is not None else default_transaction_cost_bps
            }
        return {'spot_price': 100.0, 'borrow_cost_bps': 20.0, 'transaction_cost_bps': default_transaction_cost_bps}
    
    def _calculate_rho_per_unit(self, instrument_type: str, spot_price: float,
                                 duration_years: Optional[float] = None,
                                 yield_to_maturity: float = 0.0) -> float:
        """Calculate rho per unit for an instrument using greeks_calculator."""
        if instrument_type in ['equity', 'etf']:
            return 0.0  # Equities/ETFs have no rho
        elif instrument_type == 'bond':
            if duration_years is None:
                duration_years = 5.0  # Default 5-year bond
            return self.greeks_calculator.compute_bond_rho(spot_price, duration_years, yield_to_maturity)
        elif instrument_type == 'fx_forward':
            # FX forwards have rho through interest rate differential
            # Use duration approximation
            duration = duration_years if duration_years else 0.25
            return self.greeks_calculator.compute_bond_rho(spot_price, duration, yield_to_maturity)
        return 0.0
    
    def _create_universe_record(self, symbol: str, instrument_type: str,
                               market_data: Dict[str, float], 
                               max_quantity: float,
                               duration_years: Optional[float] = None,
                               yield_to_maturity: float = 0.0) -> Dict:
        """Create a universe record for a hedge instrument."""
        spot_price = float(market_data['spot_price'])
        return {
            'symbol': symbol,
            'instrument_type': instrument_type,
            'spot_price': spot_price,
            'borrow_cost_bps': float(market_data['borrow_cost_bps']),
            'transaction_cost_bps': float(market_data['transaction_cost_bps']),
            'max_long_quantity': float(max_quantity),
            'max_short_quantity': float(max_quantity),
            'delta_per_unit': 1.0 if instrument_type in ['equity', 'etf'] else 0.0,
            'rho_per_unit': self._calculate_rho_per_unit(instrument_type, spot_price, duration_years, yield_to_maturity),
            'duration_years': duration_years if duration_years else 0.0
        }
    
    def _add_interest_rate_instruments(self, config: Dict) -> List[Dict]:
        """Add interest rate instruments (bonds/Treasuries) to hedge rho."""
        from data_loader import DataLoader
        
        ir_instruments = []
        treasury_symbols = config.get('treasury_symbols', [])
        if not treasury_symbols:
            treasury_symbols = ['TLT', 'IEF', 'SHY']
        
        # Fetch Treasury ETF data using data_loader
        data_loader = DataLoader(self.data_dir)
        try:
            # Try loading from permanent CSV first, then fetch if needed
            treasury_data = data_loader.load_treasury_etf_data()
            if treasury_data is None or treasury_data.empty:
                treasury_data = data_loader.fetch_treasury_etf_data(treasury_symbols, use_cache=config.get('use_market_data', True))
            else:
                # Filter to only requested symbols
                treasury_data = treasury_data[treasury_data['symbol'].isin(treasury_symbols)]
                if treasury_data.empty:
                    treasury_data = data_loader.fetch_treasury_etf_data(treasury_symbols, use_cache=config.get('use_market_data', True))
        except Exception as e:
            print(f"Warning: Could not fetch Treasury ETF data: {str(e)}")
            return []
        
        default_max_quantity = config['default_max_quantity']
        
        for _, row in treasury_data.iterrows():
            symbol = str(row['symbol'])
            market_data = {
                'spot_price': float(row['spot_price']),
                'borrow_cost_bps': float(row['borrow_cost_bps']),
                'transaction_cost_bps': float(row['transaction_cost_bps'])
            }
            duration_years = float(row['duration_years'])
            ytm_val = row.get('yield_to_maturity', 0.0) if 'yield_to_maturity' in row else 0.0
            if ytm_val is None or (isinstance(ytm_val, float) and np.isnan(ytm_val)):
                yield_to_maturity = 0.0
            else:
                yield_to_maturity = float(ytm_val)
            
            ir_instruments.append(self._create_universe_record(
                symbol, 'bond', market_data, default_max_quantity, duration_years, yield_to_maturity
            ))
        
        return ir_instruments
    
    def build_hedge_universe(self, symbols: List[str], config: Optional[Dict] = None) -> pd.DataFrame:
        """
        Build the list of available hedge instruments with limits and costs.
        
        Args:
            symbols: List of stock symbols to include in hedge universe (e.g., ['AAPL', 'MSFT'])
            config: Optional configuration dictionary with keys:
                - include_etfs: bool, whether to include common ETFs (default: True)
                - etf_symbols: List[str], additional ETF symbols to include (default: ['SPY'])
                - include_ir_instruments: bool, whether to include interest rate instruments for rho hedging (default: True)
                - treasury_symbols: List[str], Treasury bond/ETF symbols (default: ['TLT', 'IEF', 'SHY'])
                - default_transaction_cost_bps: float, default transaction cost in basis points (default: 5.0)
                - default_max_quantity: float, default maximum position size (default: 100000)
                - use_market_data: bool, whether to load market data for prices/costs (default: True)
        
        Returns:
            DataFrame with columns:
                - symbol: Hedge instrument symbol
                - instrument_type: 'equity', 'etf', or 'bond'
                - spot_price: Current market price
                - borrow_cost_bps: Cost to short in basis points
                - transaction_cost_bps: Transaction cost in basis points
                - max_long_quantity: Maximum long position allowed
                - max_short_quantity: Maximum short position allowed
                - delta_per_unit: Delta contribution per unit (1.0 for equities/ETFs, 0.0 for bonds)
                - rho_per_unit: Rho contribution per unit (0.0 for equities/ETFs, negative for bonds)
                - maturity_years: Maturity in years (for bonds)
        """
        parsed_config = self._parse_config(config)
        parsed_config['include_ir_instruments'] = config.get('include_ir_instruments', True) if config else True
        parsed_config['treasury_symbols'] = config.get('treasury_symbols', []) if config else []
        
        hedge_symbols = self._build_hedge_symbols_list(
            symbols, parsed_config['include_etfs'], parsed_config['etf_symbols']
        )
        
        market_data_dict = self._load_market_data_dict(parsed_config['use_market_data'])
        universe_records = []
        
        # Add equity/ETF instruments
        for symbol in hedge_symbols:
            instrument_type = 'etf' if symbol in parsed_config['etf_symbols'] else 'equity'
            market_data = self._extract_market_data_for_symbol(
                symbol, market_data_dict, parsed_config['default_transaction_cost_bps']
            )
            universe_records.append(self._create_universe_record(
                symbol, instrument_type, market_data, parsed_config['default_max_quantity']
            ))
        
        # Add interest rate instruments for rho hedging
        if parsed_config['include_ir_instruments']:
            ir_instruments = self._add_interest_rate_instruments(parsed_config)
            universe_records.extend(ir_instruments)
        
        return pd.DataFrame(universe_records)
    
    def _extract_optimization_parameters(self, portfolio_exposures: Dict, targets: Dict) -> Dict:
        """Extract optimization parameters with defaults."""
        return {
            'portfolio_delta': float(portfolio_exposures.get('total_delta', 0.0)),
            'portfolio_rho': float(portfolio_exposures.get('total_rho', 0.0)),
            'delta_target': float(targets.get('delta_target', 0.0)),
            'delta_tolerance': float(targets.get('delta_tolerance', 0.01)),
            'rho_target': float(targets.get('rho_target', 0.0)),
            'rho_tolerance': float(targets.get('rho_tolerance', 10000.0))
        }
    
    def _extract_hedge_instrument_parameters(self, hedge_universe: pd.DataFrame) -> Dict:
        """Extract hedge instrument parameters as numpy arrays."""
        n = len(hedge_universe)
        # Use rho_per_unit from hedge_universe if available, otherwise default to zeros
        if 'rho_per_unit' in hedge_universe.columns:
            rho_per_unit = hedge_universe['rho_per_unit'].values.astype(float)
        else:
            rho_per_unit = np.zeros(n)
        
        return {
            'spot_prices': hedge_universe['spot_price'].values.astype(float),
            'transaction_costs_bps': hedge_universe['transaction_cost_bps'].values.astype(float),
            'borrow_costs_bps': hedge_universe['borrow_cost_bps'].values.astype(float),
            'max_long': hedge_universe['max_long_quantity'].values.astype(float),
            'max_short': hedge_universe['max_short_quantity'].values.astype(float),
            'delta_per_unit': hedge_universe['delta_per_unit'].values.astype(float),
            'rho_per_unit': rho_per_unit,
            'n_instruments': n
        }
    
    def _build_objective_function(self, spot_prices: np.ndarray, transaction_costs_bps: np.ndarray,
                                  borrow_costs_bps: np.ndarray, portfolio_delta: float, portfolio_rho: float,
                                  delta_target: float, rho_target: float,
                                  delta_per_unit: np.ndarray, rho_per_unit: np.ndarray):
        """Build objective function that minimizes total hedging cost and penalizes deviation from targets."""
        def objective(x):
            # Cost component
            notional = np.abs(x) * spot_prices
            transaction_cost = np.sum(transaction_costs_bps * notional / 10000)
            short_mask = x < 0
            borrow_cost = np.sum(borrow_costs_bps[short_mask] * notional[short_mask] / 10000)
            total_cost = transaction_cost + borrow_cost
            
            # Penalty component for deviation from targets
            hedge_delta = np.sum(x * delta_per_unit)
            hedge_rho = np.sum(x * rho_per_unit)
            residual_delta = portfolio_delta + hedge_delta
            residual_rho = portfolio_rho + hedge_rho
            
            # Penalty: square of deviation from target (scaled to be comparable to cost)
            # Use larger penalty weights to prioritize meeting targets over minimizing cost
            delta_penalty = (residual_delta - delta_target) ** 2 * 0.1  # Scale factor
            rho_penalty = (residual_rho - rho_target) ** 2 * 0.001  # Scale factor (rho is typically larger, but we want to hedge it)
            
            return total_cost + delta_penalty + rho_penalty
        return objective
    
    def _build_constraints(self, portfolio_delta: float, portfolio_rho: float,
                          delta_target: float, delta_tolerance: float,
                          rho_target: float, rho_tolerance: float,
                          delta_per_unit: np.ndarray, rho_per_unit: np.ndarray) -> List[Dict]:
        """
        Build optimization constraints for delta and rho targets.
        
        Uses smooth two-sided inequality constraints instead of absolute values
        for better optimization convergence.
        """
        def make_lower_bound(portfolio_val, target, tolerance, per_unit):
            """Lower bound: portfolio_val + hedge >= target - tolerance"""
            def constraint(x):
                hedge_val = np.sum(x * per_unit)
                return (portfolio_val + hedge_val) - (target - tolerance)
            return constraint
        
        def make_upper_bound(portfolio_val, target, tolerance, per_unit):
            """Upper bound: portfolio_val + hedge <= target + tolerance"""
            def constraint(x):
                hedge_val = np.sum(x * per_unit)
                return (target + tolerance) - (portfolio_val + hedge_val)
            return constraint
        
        # Create two constraints per target (lower and upper bounds)
        constraints = [
            # Delta constraints
            {'type': 'ineq', 'fun': make_lower_bound(portfolio_delta, delta_target, delta_tolerance, delta_per_unit)},
            {'type': 'ineq', 'fun': make_upper_bound(portfolio_delta, delta_target, delta_tolerance, delta_per_unit)},
            # Rho constraints
            {'type': 'ineq', 'fun': make_lower_bound(portfolio_rho, rho_target, rho_tolerance, rho_per_unit)},
            {'type': 'ineq', 'fun': make_upper_bound(portfolio_rho, rho_target, rho_tolerance, rho_per_unit)}
        ]
        
        return constraints
    
    def _solve_optimization(self, objective, bounds: List[Tuple[float, float]],
                           constraints: List[Dict], n_instruments: int) -> Tuple[np.ndarray, float, str]:
        """Solve the optimization problem."""
        try:
            result = minimize(objective, np.zeros(n_instruments), method='SLSQP',
                            bounds=bounds, constraints=constraints,
                            options={'maxiter': 1000, 'ftol': 1e-6})
            if result.success:
                return result.x, result.fun, 'optimal'
            return np.zeros(n_instruments), 0.0, f'failed: {result.message}'
        except Exception as e:
            return np.zeros(n_instruments), 0.0, f'error: {str(e)}'
    
    def _calculate_residual_exposures(self, hedge_quantities: np.ndarray,
                                     delta_per_unit: np.ndarray, rho_per_unit: np.ndarray,
                                     portfolio_delta: float, portfolio_rho: float) -> Dict[str, float]:
        """Calculate residual exposures after hedging."""
        hedge_delta = float(np.sum(hedge_quantities * delta_per_unit))
        hedge_rho = float(np.sum(hedge_quantities * rho_per_unit))
        return {
            'hedge_delta': hedge_delta,
            'hedge_rho': hedge_rho,
            'residual_delta': portfolio_delta + hedge_delta,
            'residual_rho': portfolio_rho + hedge_rho
        }
    
    def _create_hedge_recommendations(self, hedge_universe: pd.DataFrame,
                                     hedge_quantities: np.ndarray,
                                     rho_per_unit: np.ndarray) -> pd.DataFrame:
        """Create hedge recommendations DataFrame from optimization results."""
        hedge_recommendations = []
        timestamp = datetime.now().isoformat()
        
        for idx, (_, row) in enumerate(hedge_universe.iterrows()):
            quantity = hedge_quantities[idx]
            if abs(quantity) <= 1e-6:
                continue
            
            spot_price = float(row['spot_price'])
            notional = abs(quantity) * spot_price
            cost_bps = float(row['transaction_cost_bps']) + (float(row['borrow_cost_bps']) if quantity < 0 else 0)
            
            hedge_recommendations.append({
                'symbol': row['symbol'],
                'instrument_type': row['instrument_type'],
                'hedge_quantity': float(quantity),
                'side': 'buy' if quantity > 0 else 'sell',
                'estimated_cost': cost_bps * notional / 10000,
                'delta_contribution': float(quantity * row['delta_per_unit']),
                'rho_contribution': float(quantity * rho_per_unit[idx]),
                'timestamp': timestamp
            })
        
        return pd.DataFrame(hedge_recommendations)
    
    def _create_optimization_summary(self, solver_status: str, total_cost: float,
                                    residual_exposures: Dict[str, float],
                                    hedge_recommendations_df: pd.DataFrame,
                                    portfolio_exposures: Dict) -> Dict:
        """Create optimization summary dictionary."""
        residual = {'total_delta': residual_exposures['residual_delta'], 'total_rho': residual_exposures['residual_rho']}
        return {
            'solver_status': solver_status,
            'total_hedge_cost': float(total_cost),
            'residual_delta': float(residual_exposures['residual_delta']),
            'residual_rho': float(residual_exposures['residual_rho']),
            'num_hedge_trades': len(hedge_recommendations_df),
            'hedge_effectiveness_pct': self.compute_hedge_effectiveness(portfolio_exposures, residual)
        }
    
    def optimize_hedge_portfolio(self, portfolio_exposures: Dict, hedge_universe: pd.DataFrame,
                                 market_data: pd.DataFrame, targets: Dict, 
                                 constraints: Optional[Dict] = None) -> Tuple[pd.DataFrame, Dict]:
        """
        Solve constrained optimization problem to meet targets at minimal cost.
        
        Args:
            portfolio_exposures: Current portfolio greeks summary with keys:
                - total_delta: Current portfolio delta exposure
                - total_rho: Current portfolio rho exposure (optional, defaults to 0)
            hedge_universe: Available hedge instruments DataFrame with columns:
                - symbol, instrument_type, spot_price, borrow_cost_bps, transaction_cost_bps,
                  max_long_quantity, max_short_quantity, delta_per_unit
            market_data: Current market prices (not directly used, but kept for API consistency)
            targets: User-defined targets with keys:
                - delta_target: Target delta (default: 0)
                - delta_tolerance: Tolerance for delta constraint (default: 0.01)
                - rho_target: Target rho (default: 0)
                - rho_tolerance: Tolerance for rho constraint (default: 10000.0)
            constraints: Additional constraints (currently unused, reserved for future use)
        
        Returns:
            Tuple of (hedge_recommendations DataFrame, optimization_summary Dict)
        """
        # Extract and validate parameters
        params = self._extract_optimization_parameters(portfolio_exposures, targets)
        
        # Validate hedge universe
        if hedge_universe.empty:
            return self._create_empty_hedge_results(
                params['portfolio_delta'], params['portfolio_rho'],
                params['delta_target'], params['rho_target']
            )
        
        # Extract hedge instrument parameters
        instrument_params = self._extract_hedge_instrument_parameters(hedge_universe)
        
        # Build objective function
        objective = self._build_objective_function(
            instrument_params['spot_prices'],
            instrument_params['transaction_costs_bps'],
            instrument_params['borrow_costs_bps'],
            params['portfolio_delta'],
            params['portfolio_rho'],
            params['delta_target'],
            params['rho_target'],
            instrument_params['delta_per_unit'],
            instrument_params['rho_per_unit']
        )
        
        # Build constraints
        constraints_list = self._build_constraints(
            params['portfolio_delta'],
            params['portfolio_rho'],
            params['delta_target'],
            params['delta_tolerance'],
            params['rho_target'],
            params['rho_tolerance'],
            instrument_params['delta_per_unit'],
            instrument_params['rho_per_unit']
        )
        
        # Build bounds
        bounds = list(zip(-instrument_params['max_short'], instrument_params['max_long']))
        
        # Solve optimization
        hedge_quantities, total_cost, solver_status = self._solve_optimization(
            objective, bounds, constraints_list, instrument_params['n_instruments']
        )
        
        # Calculate residual exposures
        residual_exposures = self._calculate_residual_exposures(
            hedge_quantities,
            instrument_params['delta_per_unit'],
            instrument_params['rho_per_unit'],
            params['portfolio_delta'],
            params['portfolio_rho']
        )
        
        # Create hedge recommendations
        hedge_recommendations_df = self._create_hedge_recommendations(
            hedge_universe,
            hedge_quantities,
            instrument_params['rho_per_unit']
        )
        
        # Create optimization summary
        optimization_summary = self._create_optimization_summary(
            solver_status,
            total_cost,
            residual_exposures,
            hedge_recommendations_df,
            portfolio_exposures
        )
        
        return hedge_recommendations_df, optimization_summary
    
    def _create_empty_hedge_results(self, portfolio_delta: float, portfolio_rho: float,
                                   delta_target: float, rho_target: float) -> Tuple[pd.DataFrame, Dict]:
        """Create empty hedge results when optimization cannot be performed."""
        cols = ['symbol', 'instrument_type', 'hedge_quantity', 'side', 'estimated_cost',
                'delta_contribution', 'rho_contribution', 'timestamp']
        return (
            pd.DataFrame({col: [] for col in cols}),
            {
                'solver_status': 'no_instruments',
                'total_hedge_cost': 0.0,
                'residual_delta': float(portfolio_delta),
                'residual_rho': float(portfolio_rho),
                'num_hedge_trades': 0,
                'hedge_effectiveness_pct': 0.0
            }
        )
    
    def compute_hedge_effectiveness(self, original_exposures: Dict, 
                                   residual_exposures: Dict) -> float:
        """Calculate risk reduction percentage (0-100)."""
        orig_delta, orig_rho = abs(original_exposures.get('total_delta', 0.0)), abs(original_exposures.get('total_rho', 0.0))
        res_delta, res_rho = abs(residual_exposures.get('total_delta', 0.0)), abs(residual_exposures.get('total_rho', 0.0))
        
        if orig_delta == 0 and orig_rho == 0:
            return 100.0
        
        norm_orig_rho, norm_res_rho = orig_rho / 10000.0, res_rho / 10000.0
        delta_reduction = (1.0 - res_delta / orig_delta) if orig_delta > 0 else (1.0 if res_delta == 0 else 0.0)
        rho_reduction = (1.0 - norm_res_rho / norm_orig_rho) if norm_orig_rho > 0 else (1.0 if norm_res_rho == 0 else 0.0)
        
        total_mag = orig_delta + norm_orig_rho
        if total_mag > 0:
            delta_weight, rho_weight = orig_delta / total_mag, norm_orig_rho / total_mag
        else:
            delta_weight, rho_weight = 0.8, 0.2
        
        return max(0.0, min(100.0, (delta_weight * delta_reduction + rho_weight * rho_reduction) * 100.0))
    
    def save_hedge_tickets(self, hedge_recommendations: pd.DataFrame):
        """
        Save hedge recommendations to CSV.
        
        Args:
            hedge_recommendations: DataFrame with columns:
                - symbol: Hedge instrument symbol
                - instrument_type: Type of instrument (equity/etf)
                - hedge_quantity: Quantity to hedge (can be positive or negative)
                - side: 'buy' or 'sell'
                - estimated_cost: Estimated cost of the hedge
                - delta_contribution: Delta contribution of this hedge
                - rho_contribution: Rho contribution of this hedge
                - timestamp: Timestamp of when the recommendation was created
        
        Raises:
            ValueError: If required columns are missing
            IOError: If file cannot be written
        """
        # Ensure data directory exists
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Required columns for hedge tickets
        required_columns = [
            'symbol', 'instrument_type', 'hedge_quantity', 'side',
            'estimated_cost', 'delta_contribution', 'rho_contribution', 'timestamp'
        ]
        
        # Validate DataFrame
        if hedge_recommendations.empty:
            print("Warning: No hedge recommendations to save (empty DataFrame)")
            # Create empty file with correct columns for consistency
            empty_dict = {col: [] for col in required_columns}
            hedge_recommendations = pd.DataFrame(empty_dict)
        else:
            # Validate required columns are present
            missing_columns = [col for col in required_columns if col not in hedge_recommendations.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Sort by symbol for better readability
        if not hedge_recommendations.empty:
            hedge_recommendations = hedge_recommendations.sort_values('symbol').copy()
        
        # Save to CSV
        filepath = os.path.join(self.data_dir, "hedge_tickets.csv")
        try:
            hedge_recommendations.to_csv(filepath, index=False)
            num_trades = len(hedge_recommendations)
            total_cost = float(hedge_recommendations['estimated_cost'].sum()) if not hedge_recommendations.empty else 0.0
            print(f"Saved hedge_tickets.csv ({num_trades} trades, total cost: ${total_cost:,.2f})")
        except Exception as e:
            raise IOError(f"Failed to save hedge tickets to {filepath}: {str(e)}") from e
    
    def save_optimization_summary(self, summary: Dict):
        """
        Save optimization summary to JSON.
        
        Fields: solver_status, total_hedge_cost, residual_delta, residual_rho,
                num_hedge_trades, hedge_effectiveness_pct
        """
        import json
        os.makedirs(self.data_dir, exist_ok=True)
        filepath = os.path.join(self.data_dir, "optimization_summary.json")
        try:
            with open(filepath, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"Saved optimization_summary.json")
        except Exception as e:
            raise IOError(f"Failed to save optimization summary to {filepath}: {str(e)}") from e
    
    def load_portfolio_exposures(self) -> Dict:
        """Load and aggregate portfolio exposures from positions_with_greeks.csv."""
        positions_path = os.path.join(self.data_dir, "positions_with_greeks.csv")
        if not os.path.exists(positions_path):
            raise FileNotFoundError(f"Positions file not found: {positions_path}")
        
        positions = pd.read_csv(positions_path)
        greeks = ['delta', 'rho', 'gamma', 'vega', 'theta']
        
        exposures = {}
        for greek in greeks:
            col = f'position_{greek}'
            exposures[f'total_{greek}'] = float(positions[col].sum()) if col in positions.columns else 0.0
        
        if 'quantity' in positions.columns and 'spot_price' in positions.columns:
            exposures['total_notional'] = float((positions['quantity'] * positions['spot_price']).sum())
        else:
            exposures['total_notional'] = 0.0
        exposures['num_positions'] = len(positions)
        return exposures
    
    def load_market_data(self) -> pd.DataFrame:
        """
        Load market data from CSV.
        
        Returns:
            DataFrame with market data
        """
        market_data_path = os.path.join(self.data_dir, "market_data.csv")
        if not os.path.exists(market_data_path):
            raise FileNotFoundError(f"Market data file not found: {market_data_path}")
        return pd.read_csv(market_data_path)
    
    def run_pipeline(self, targets: Optional[Dict] = None) -> Tuple[pd.DataFrame, Dict]:
        """Run the complete hedge optimization pipeline."""
        positions = pd.read_csv(os.path.join(self.data_dir, "positions_with_greeks.csv"))
        symbols = positions['symbol'].unique().tolist()
        targets = targets or {'delta_target': 0.0, 'delta_tolerance': 0.01, 'rho_target': 0.0, 'rho_tolerance': 10.0}
        portfolio_exposures = self.load_portfolio_exposures()
        hedge_universe = self.build_hedge_universe(symbols)
        market_data = self.load_market_data() if os.path.exists(os.path.join(self.data_dir, "market_data.csv")) else pd.DataFrame()
        hedge_recommendations, optimization_summary = self.optimize_hedge_portfolio(portfolio_exposures, hedge_universe, market_data, targets)
        self.save_hedge_tickets(hedge_recommendations)
        self.save_optimization_summary(optimization_summary)
        return hedge_recommendations, optimization_summary
    
    def run_end_to_end(self, symbols: List[str], targets: Dict,
                      hedge_config: Optional[Dict] = None,
                      save_results: bool = True) -> Tuple[pd.DataFrame, Dict]:
        """
        Run the complete hedge optimization pipeline end-to-end.
        
        This method orchestrates the entire hedge optimization process:
        1. Load portfolio exposures from positions_with_greeks.csv
        2. Build hedge universe for the given symbols
        3. Load market data
        4. Optimize hedge portfolio to meet targets
        5. Save hedge tickets and optimization summary (if save_results=True)
        
        Args:
            symbols: List of stock symbols to include in hedge universe
            targets: User-defined targets dictionary with keys:
                - delta_target: Target delta (default: 0)
                - delta_tolerance: Tolerance for delta constraint (default: 0.01)
                - rho_target: Target rho (default: 0)
                - rho_tolerance: Tolerance for rho constraint (default: 10000.0)
            hedge_config: Optional configuration for hedge universe with keys:
                - include_etfs: bool, whether to include common ETFs (default: True)
                - etf_symbols: List[str], additional ETF symbols to include (default: ['SPY'])
                - default_transaction_cost_bps: float, default transaction cost (default: 5.0)
                - default_max_quantity: float, default maximum position size (default: 100000)
                - use_market_data: bool, whether to load market data (default: True)
            save_results: If True, save hedge tickets and optimization summary to files
        
        Returns:
            Tuple of (hedge_recommendations DataFrame, optimization_summary Dict)
        
        Raises:
            FileNotFoundError: If required input files are missing
            ValueError: If portfolio exposures cannot be loaded
        """
        print("Hedge Optimization Pipeline")
        
        print("\nLoading portfolio exposures...")
        portfolio_exposures = self.load_portfolio_exposures()
        print(f"  Delta: {portfolio_exposures['total_delta']:,.2f}, Rho: {portfolio_exposures['total_rho']:,.2f}, Positions: {portfolio_exposures['num_positions']}")
        
        print(f"\nBuilding hedge universe...")
        hedge_universe = self.build_hedge_universe(symbols, hedge_config)
        print(f"  Instruments: {len(hedge_universe)}")
        
        print("\nLoading market data...")
        try:
            market_data = self.load_market_data()
            print(f"  Loaded: {len(market_data)} symbols")
        except Exception as e:
            print(f"  Warning: {str(e)}")
            market_data = pd.DataFrame()
        
        print("\n[Optimizing...")
        hedge_recommendations, optimization_summary = self.optimize_hedge_portfolio(
            portfolio_exposures, hedge_universe, market_data, targets
        )
        s = optimization_summary
        print(f"  Status: {s['solver_status']}, Trades: {s['num_hedge_trades']}, Cost: ${s['total_hedge_cost']:,.2f}")
        print(f"  Residual Delta: {s['residual_delta']:,.2f}, Rho: {s['residual_rho']:,.2f}, Effectiveness: {s['hedge_effectiveness_pct']:.1f}%")
        
        if save_results:
            print("\nSaving results...")
            self.save_hedge_tickets(hedge_recommendations)
            self.save_optimization_summary(optimization_summary)
        
        print("\nComplete")
        
        return hedge_recommendations, optimization_summary
