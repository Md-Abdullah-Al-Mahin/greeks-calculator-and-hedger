import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional
from scipy.stats import norm
import os


class GreeksCalculator:
    """
    Computes Black-Scholes greeks for positions.
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
    
    def load_data(self) -> tuple:
        """
        Load all required data files.
        
        Returns:
            Tuple of (positions, market_data, rates, vol_surface) DataFrames
        """
        positions = pd.read_csv(os.path.join(self.data_dir, "positions.csv"))
        market_data = pd.read_csv(os.path.join(self.data_dir, "market_data.csv"))
        rates = pd.read_csv(os.path.join(self.data_dir, "rates.csv"))
        vol_surface = pd.read_csv(os.path.join(self.data_dir, "vol_surface.csv"))
        return positions, market_data, rates, vol_surface
    
    def compute_time_to_expiry(self, expiry_date: pd.Series, current_date: Optional[datetime] = None) -> pd.Series:
        """
        Convert expiry dates to year fractions.
        """
        if current_date is None:
            current_date = datetime.now()
        return (pd.to_datetime(expiry_date) - current_date).dt.days / 365.0
    
    def interpolate_interest_rate(self, time_to_expiry: float, rates: pd.DataFrame) -> float:
        """
        Interpolate interest rate for given time horizon.
        
        Args:
            time_to_expiry: Time to expiry in years (float)
            rates: DataFrame with columns 'tenor_days' and 'rate' (decimal)
        
        Returns:
            Interpolated interest rate (decimal)
        
        Notes:
            - Uses linear interpolation between tenor points
            - If time_to_expiry is before the first tenor, returns the first rate
            - If time_to_expiry is after the last tenor, returns the last rate
        """
        if rates.empty:
            raise ValueError("Rates DataFrame is empty")
        
        # Convert time_to_expiry from years to days
        time_to_expiry_days = time_to_expiry * 365.0
        
        # Ensure rates are sorted by tenor_days
        rates_sorted = rates.sort_values('tenor_days').copy()
        
        # Extract tenor_days and rates as numpy arrays
        tenor_days = np.array(rates_sorted['tenor_days'].values, dtype=float)
        rate_values = np.array(rates_sorted['rate'].values, dtype=float)
        
        # Handle edge cases
        if time_to_expiry_days <= tenor_days[0]:
            # Before first tenor: return first rate
            return float(rate_values[0])
        elif time_to_expiry_days >= tenor_days[-1]:
            # After last tenor: return last rate
            return float(rate_values[-1])
        else:
            # Linear interpolation using numpy
            interpolated_rate = np.interp(time_to_expiry_days, tenor_days, rate_values)
            return float(interpolated_rate)
    
    def interpolate_volatility(self, symbol: str, strike: float, expiry: datetime, 
                              spot_price: float, vol_surface: pd.DataFrame) -> float:
        """
        Interpolate implied volatility from surface.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            strike: Strike price of the option
            expiry: Expiry date as datetime
            spot_price: Current spot price of the underlying
            vol_surface: DataFrame with columns 'symbol', 'expiry', 'strike', 'moneyness', 'implied_vol'
        
        Returns:
            Interpolated implied volatility (decimal, e.g., 0.25 for 25%)
        
        Notes:
            - First filters by symbol
            - Finds the closest expiry date in the surface
            - Interpolates based on strike for that expiry
            - Falls back to default volatility (0.25) if no data found
        """
        DEFAULT_VOL = 0.25
        MIN_VOL = 0.01  # Minimum reasonable volatility (1%) to avoid numerical issues
        
        if vol_surface.empty:
            return DEFAULT_VOL
        
        # Filter by symbol and convert expiry to datetime
        symbol_data = vol_surface[vol_surface['symbol'] == symbol].copy()
        if symbol_data.empty:
            return DEFAULT_VOL
        
        symbol_data['expiry'] = pd.to_datetime(symbol_data['expiry'])
        
        # Find closest expiry
        expiry_ts = pd.Timestamp(expiry)
        unique_expiries = pd.Series(symbol_data['expiry']).drop_duplicates().tolist()
        if not unique_expiries:
            return DEFAULT_VOL
        
        time_diffs = [abs((pd.Timestamp(exp) - expiry_ts).total_seconds()) for exp in unique_expiries]
        closest_expiry = pd.Timestamp(unique_expiries[np.argmin(time_diffs)])
        
        # Filter by closest expiry, valid vols (prefer reasonable vols >= MIN_VOL), and sort by strike
        expiry_data = pd.DataFrame(
            symbol_data[(symbol_data['expiry'] == closest_expiry) & (symbol_data['implied_vol'] >= MIN_VOL)]
        ).sort_values('strike')
        
        # If no data with reasonable vols, try without the MIN_VOL filter
        if expiry_data.empty:
            expiry_data = pd.DataFrame(
                symbol_data[(symbol_data['expiry'] == closest_expiry) & (symbol_data['implied_vol'] > 0)]
            ).sort_values('strike')
            if expiry_data.empty:
                return DEFAULT_VOL
        
        # Extract arrays and interpolate
        strikes = expiry_data['strike'].values.astype(float)
        vols = expiry_data['implied_vol'].values.astype(float)
        
        # Interpolate
        if strike <= strikes[0]:
            interpolated_vol = float(vols[0])
        elif strike >= strikes[-1]:
            interpolated_vol = float(vols[-1])
        else:
            interpolated_vol = float(np.interp(strike, strikes, vols))
        
        # Enforce minimum volatility to avoid numerical issues in Black-Scholes
        return max(interpolated_vol, MIN_VOL)        
    
    def compute_black_scholes_greeks(self, spot: float, strike: float, time_to_expiry: float,
                                    rate: float, volatility: float, option_type: str,
                                    dividend_yield: float = 0.0) -> dict:
        """
        Compute Black-Scholes greeks for a single option with dividend adjustment.
        
        Args:
            spot: Current spot price of the underlying
            strike: Strike price of the option
            time_to_expiry: Time to expiry in years
            rate: Risk-free interest rate (decimal)
            volatility: Implied volatility (decimal)
            option_type: 'call' or 'put'
            dividend_yield: Dividend yield (decimal, default: 0.0)
        
        Returns:
            Dictionary with delta, gamma, vega, theta, rho
        """
        # Handle edge cases
        if time_to_expiry <= 0:
            # Option expired
            return {'delta': 0.0, 'gamma': 0.0, 'vega': 0.0, 'theta': 0.0, 'rho': 0.0}
        
        # Enforce minimum volatility to avoid numerical issues
        MIN_VOL = 0.01  # 1% minimum
        if volatility < MIN_VOL:
            volatility = MIN_VOL
        
        if volatility <= 0 or spot <= 0 or strike <= 0:
            return {'delta': 0.0, 'gamma': 0.0, 'vega': 0.0, 'theta': 0.0, 'rho': 0.0}
        
        # Calculate d1 and d2 with dividend adjustment (Merton model)
        # For dividend-paying stocks, use (r - q) instead of r
        sqrt_T = np.sqrt(time_to_expiry)
        d1 = (np.log(spot / strike) + (rate - dividend_yield + 0.5 * volatility ** 2) * time_to_expiry) / (volatility * sqrt_T)
        d2 = d1 - volatility * sqrt_T
        
        # Standard normal CDF and PDF
        N_d1 = norm.cdf(d1)
        N_d2 = norm.cdf(d2)
        N_neg_d1 = norm.cdf(-d1)
        N_neg_d2 = norm.cdf(-d2)
        n_d1 = norm.pdf(d1)  # PDF at d1
        
        # Discount factors
        discount_rate = np.exp(-rate * time_to_expiry)
        discount_dividend = np.exp(-dividend_yield * time_to_expiry)
        
        # Common terms
        vega_term = spot * discount_dividend * n_d1 * sqrt_T
        gamma_term = discount_dividend * n_d1 / (spot * volatility * sqrt_T)
        theta_vol_term = -spot * discount_dividend * n_d1 * volatility / (2 * sqrt_T)
        
        if option_type.lower() == 'call':
            delta = discount_dividend * N_d1
            theta = (theta_vol_term 
                    - rate * strike * discount_rate * N_d2
                    + dividend_yield * spot * discount_dividend * N_d1)
            rho = strike * time_to_expiry * discount_rate * N_d2
        else:  # put
            delta = discount_dividend * (N_d1 - 1.0)  # or -discount_dividend * N_neg_d1
            theta = (theta_vol_term 
                    + rate * strike * discount_rate * N_neg_d2
                    - dividend_yield * spot * discount_dividend * N_neg_d1)
            rho = -strike * time_to_expiry * discount_rate * N_neg_d2
        
        # Gamma and Vega are the same for calls and puts (with dividend adjustment)
        gamma = gamma_term
        vega = vega_term
        
        return {
            'delta': float(delta),
            'gamma': float(gamma),
            'vega': float(vega),
            'theta': float(theta),
            'rho': float(rho)
        }
    
    def compute_bond_rho(self, spot_price: float, duration_years: float, 
                        yield_to_maturity: float = 0.0) -> float:
        """
        Compute rho (interest rate sensitivity) for a bond or Treasury ETF.
        
        Rho represents the change in bond price for a 1% (0.01) change in interest rates.
        Formula: rho = -modified_duration * price * 0.01
        
        For bonds, modified_duration ≈ duration_years / (1 + yield)
        
        Args:
            spot_price: Current price of the bond/ETF
            duration_years: Duration of the bond in years (e.g., 7.5 for IEF)
            yield_to_maturity: Current yield to maturity (decimal, e.g., 0.04 for 4%)
        
        Returns:
            Rho value (negative, as bond prices fall when rates rise)
        """
        if duration_years <= 0 or spot_price <= 0:
            return 0.0
        
        # Modified duration = Macaulay duration / (1 + yield)
        # For small yields, this is approximately equal to duration
        modified_duration = duration_years / (1.0 + yield_to_maturity) if yield_to_maturity > -0.99 else duration_years
        
        # Rho = -modified_duration * price * 0.01 (for 1% rate change)
        rho = -modified_duration * spot_price * 0.01
        
        return float(rho)
    
    def enrich_positions(self, positions: pd.DataFrame, market_data: pd.DataFrame,
                        rates: pd.DataFrame, vol_surface: pd.DataFrame) -> pd.DataFrame:
        """
        Join positions with market inputs and compute derived fields.
        
        Args:
            positions: DataFrame with position_id, symbol, quantity, instrument_type, strike, expiry, option_type
            market_data: DataFrame with symbol, spot_price, dividend_yield, borrow_cost_bps, last_updated
            rates: DataFrame with tenor_days, rate
            vol_surface: DataFrame with symbol, expiry, strike, moneyness, implied_vol
        
        Returns:
            DataFrame with all original fields plus: spot_price, dividend_yield, borrow_cost_bps,
            time_to_expiry, interpolated_rate, interpolated_vol
        """
        # Start with a copy of positions
        enriched = positions.copy()
        
        # Join with market_data on symbol
        enriched = enriched.merge(
            market_data[['symbol', 'spot_price', 'dividend_yield', 'borrow_cost_bps']],
            on='symbol',
            how='left'
        )
        
        # Initialize derived fields
        enriched['time_to_expiry'] = 0.0
        enriched['interpolated_rate'] = 0.0
        enriched['interpolated_vol'] = 0.0
        
        # Get default short-term rate for equities
        default_rate = 0.0
        if not rates.empty:
            short_rates = pd.DataFrame(rates[rates['tenor_days'] <= 30])
            if not short_rates.empty:
                default_rate = float(short_rates.iloc[0]['rate'])
            else:
                default_rate = float(pd.DataFrame(rates).iloc[0]['rate'])
        
        # Process options
        option_mask = enriched['instrument_type'] == 'option'
        if option_mask.any():
            # Compute time to expiry for options
            expiry_series = pd.Series(pd.to_datetime(enriched.loc[option_mask, 'expiry']))
            time_to_expiry_series = self.compute_time_to_expiry(expiry_series)
            enriched.loc[option_mask, 'time_to_expiry'] = time_to_expiry_series.values
            
            # Interpolate rates and vols for options
            for idx in enriched[option_mask].index:
                time_to_exp = enriched.at[idx, 'time_to_expiry']
                if time_to_exp > 0:
                    # Interpolate rate
                    rate = self.interpolate_interest_rate(time_to_exp, rates)
                    enriched.at[idx, 'interpolated_rate'] = rate
                    
                    # Interpolate volatility
                    strike = float(enriched.at[idx, 'strike'])
                    spot_price = float(enriched.at[idx, 'spot_price'])
                    expiry_date = pd.to_datetime(enriched.at[idx, 'expiry'])
                    vol = self.interpolate_volatility(
                        enriched.at[idx, 'symbol'],
                        strike,
                        expiry_date,
                        spot_price,
                        vol_surface
                    )
                    enriched.at[idx, 'interpolated_vol'] = vol
        
        # Set default rate for equities
        equity_mask = enriched['instrument_type'] == 'equity'
        enriched.loc[equity_mask, 'interpolated_rate'] = default_rate
        
        return enriched
    
    def compute_position_greeks(self, enriched_positions: pd.DataFrame) -> pd.DataFrame:
        """
        Compute greeks for all positions (equities get delta=1, others computed).
        
        Args:
            enriched_positions: DataFrame with enriched position data including spot_price,
                              time_to_expiry, interpolated_rate, interpolated_vol
        
        Returns:
            DataFrame with all original fields plus unit greeks (delta, gamma, vega, theta, rho)
            and position greeks (position_delta, position_gamma, position_vega, position_theta, position_rho)
        """
        positions = enriched_positions.copy()
        
        # Initialize unit greeks columns
        positions['delta'] = 0.0
        positions['gamma'] = 0.0
        positions['vega'] = 0.0
        positions['theta'] = 0.0
        positions['rho'] = 0.0
        
        # Process equities: delta=1, other greeks=0
        equity_mask = positions['instrument_type'] == 'equity'
        positions.loc[equity_mask, 'delta'] = 1.0
        
        # Process options: compute Black-Scholes greeks
        option_mask = positions['instrument_type'] == 'option'
        for idx in positions[option_mask].index:
            row = positions.loc[idx]
            # Get dividend yield, defaulting to 0.0 if not available
            dividend_yield = float(row.get('dividend_yield', 0.0)) if 'dividend_yield' in row else 0.0
            greeks = self.compute_black_scholes_greeks(
                spot=float(row['spot_price']),
                strike=float(row['strike']),
                time_to_expiry=float(row['time_to_expiry']),
                rate=float(row['interpolated_rate']),
                volatility=float(row['interpolated_vol']),
                option_type=str(row['option_type']),
                dividend_yield=dividend_yield
            )
            positions.loc[idx, ['delta', 'gamma', 'vega', 'theta', 'rho']] = [
                greeks['delta'], greeks['gamma'], greeks['vega'], greeks['theta'], greeks['rho']
            ]
        
        # Compute position-level greeks (unit greeks * quantity)
        positions['position_delta'] = positions['delta'] * positions['quantity']
        positions['position_gamma'] = positions['gamma'] * positions['quantity']
        positions['position_vega'] = positions['vega'] * positions['quantity']
        positions['position_theta'] = positions['theta'] * positions['quantity']
        positions['position_rho'] = positions['rho'] * positions['quantity']
        
        return positions
    
    def run_pipeline(self, validate: bool = True) -> pd.DataFrame:
        """
        Run the full greeks calculation pipeline.
        
        Args:
            validate: If True, run validation checks on computed greeks
        
        Returns:
            DataFrame with positions and their greeks
        """
        positions, market_data, rates, vol_surface = self.load_data()
        enriched_positions = self.enrich_positions(positions, market_data, rates, vol_surface)
        positions_with_greeks = self.compute_position_greeks(enriched_positions)
        
        # Run validation checks
        if validate:
            self.validate_greeks(positions_with_greeks, verbose=True)
        
        self.save_results(positions_with_greeks)
        return positions_with_greeks
    
    def validate_greeks(self, positions_with_greeks: pd.DataFrame, 
                        tolerance: float = 0.1, verbose: bool = True) -> Dict[str, bool]:
        """
        Validate greeks for reasonableness.
        
        Args:
            positions_with_greeks: DataFrame with computed greeks
            tolerance: Tolerance for ATM call delta check (default: 0.1)
            verbose: If True, print validation results
        
        Returns:
            Dictionary with validation results (key: check_name, value: passed)
        """
        results = {}
        warnings = []
        
        # Check 1: All gamma values should be >= 0
        if 'gamma' in positions_with_greeks.columns:
            negative_gamma = positions_with_greeks[positions_with_greeks['gamma'] < 0]
            gamma_check = len(negative_gamma) == 0
            results['gamma_non_negative'] = gamma_check
            if not gamma_check and verbose:
                warnings.append(f"Warning: Found {len(negative_gamma)} positions with negative gamma")
        else:
            results['gamma_non_negative'] = True
        
        # Check 2: ATM call options should have delta ~0.5
        if 'instrument_type' in positions_with_greeks.columns:
            option_mask = positions_with_greeks['instrument_type'] == 'option'
            if option_mask.any():
                options = positions_with_greeks[option_mask].copy()
                if 'option_type' in options.columns and 'delta' in options.columns:
                    call_options = options[options['option_type'] == 'call']
                    if len(call_options) > 0 and 'strike' in call_options.columns and 'spot_price' in call_options.columns:
                        # Find ATM options (strike within 2% of spot)
                        call_options['moneyness'] = call_options['strike'] / call_options['spot_price']
                        atm_calls = call_options[(call_options['moneyness'] >= 0.98) & (call_options['moneyness'] <= 1.02)]
                        if len(atm_calls) > 0:
                            avg_delta = atm_calls['delta'].mean()
                            delta_check = abs(avg_delta - 0.5) <= tolerance
                            results['atm_call_delta'] = delta_check
                            if not delta_check and verbose:
                                warnings.append(f"Warning: ATM call delta is {avg_delta:.3f}, expected ~0.5")
                        else:
                            results['atm_call_delta'] = True  # No ATM calls to check
                    else:
                        results['atm_call_delta'] = True
                else:
                    results['atm_call_delta'] = True
            else:
                results['atm_call_delta'] = True
        
        # Check 3: Vega should be positive for all options
        if 'vega' in positions_with_greeks.columns:
            option_mask = positions_with_greeks['instrument_type'] == 'option'
            if option_mask.any():
                options = positions_with_greeks[option_mask]
                negative_vega = options[options['vega'] < 0]
                vega_check = len(negative_vega) == 0
                results['vega_non_negative'] = vega_check
                if not vega_check and verbose:
                    warnings.append(f"Warning: Found {len(negative_vega)} options with negative vega")
            else:
                results['vega_non_negative'] = True
        else:
            results['vega_non_negative'] = True
        
        # Check 4: Theta should be negative for long positions (time decay)
        if 'theta' in positions_with_greeks.columns and 'quantity' in positions_with_greeks.columns:
            long_positions = positions_with_greeks[positions_with_greeks['quantity'] > 0]
            if len(long_positions) > 0:
                # For long positions, theta should be negative (value decreases over time)
                option_mask = long_positions['instrument_type'] == 'option'
                if option_mask.any():
                    long_options = long_positions[option_mask]
                    positive_theta = long_options[long_options['theta'] > 0]
                    theta_check = len(positive_theta) == 0
                    results['long_theta_negative'] = theta_check
                    if not theta_check and verbose:
                        warnings.append(f"Warning: Found {len(positive_theta)} long options with positive theta")
                else:
                    results['long_theta_negative'] = True
            else:
                results['long_theta_negative'] = True
        else:
            results['long_theta_negative'] = True
        
        # Check 5: Equity delta should be exactly 1.0
        if 'delta' in positions_with_greeks.columns and 'instrument_type' in positions_with_greeks.columns:
            equity_mask = positions_with_greeks['instrument_type'] == 'equity'
            if equity_mask.any():
                equities = positions_with_greeks[equity_mask]
                non_unit_delta = equities[abs(equities['delta'] - 1.0) > 1e-6]
                equity_delta_check = len(non_unit_delta) == 0
                results['equity_delta_one'] = equity_delta_check
                if not equity_delta_check and verbose:
                    warnings.append(f"Warning: Found {len(non_unit_delta)} equities with delta != 1.0")
            else:
                results['equity_delta_one'] = True
        else:
            results['equity_delta_one'] = True
        
        if verbose:
            if warnings:
                print("\nGreeks Validation Warnings:")
                for warning in warnings:
                    print(f"  {warning}")
            else:
                print("\nGreeks Validation: All checks passed ✓")
        
        return results
    
    def save_results(self, positions_with_greeks: pd.DataFrame):
        """
        Save enriched positions with greeks to CSV.
        """
        filepath = os.path.join(self.data_dir, "positions_with_greeks.csv")
        positions_with_greeks.to_csv(filepath, index=False)
