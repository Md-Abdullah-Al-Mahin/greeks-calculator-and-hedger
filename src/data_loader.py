"""
Data Loader Module

Purpose: Fetch real market data and generate synthetic positions.
- Fetch Stock Data: Gets prices and dividends for selected symbols from Yahoo Finance
- Fetch Risk-Free Rates: Gets latest Treasury rates from the Federal Reserve
- Fetch Options Chain: Gets calls and puts with strikes, prices, volumes
- Build Volatility Surface: Assembles implied vols across strikes and expiries
- Generate Synthetic Positions: Creates a realistic portfolio (mix of stocks and options)
- Generate Synthetic Borrow Costs: Produces plausible borrow costs when unavailable publicly
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import json
import os
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

try:
    from pandas_datareader import data as web
    HAS_DATAREADER = True
except ImportError:
    HAS_DATAREADER = False
    web = None


class DataLoader:
    """
    Orchestrates data fetching, caching, and saving CSVs with timestamps.
    """
    
    def __init__(self, data_dir: str = "data", cache_expiry_hours: int = 1):
        self.data_dir = data_dir
        self.cache_expiry_hours = cache_expiry_hours
        os.makedirs(data_dir, exist_ok=True)
    
    def _get_cache_path(self, cache_key: str, metadata: bool = False) -> str:
        """Get cache file path for a given key."""
        suffix = "_metadata.json" if metadata else ".csv"
        return os.path.join(self.data_dir, f"cache_{cache_key}{suffix}")
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache exists and is still valid."""
        cache_path = self._get_cache_path(cache_key)
        metadata_path = self._get_cache_path(cache_key, metadata=True)
        
        if not (os.path.exists(cache_path) and os.path.exists(metadata_path)):
            return False
        
        try:
            with open(metadata_path) as f:
                metadata = json.load(f)
            cache_time = datetime.fromisoformat(metadata.get('timestamp', ''))
            return (datetime.now() - cache_time).total_seconds() < (self.cache_expiry_hours * 3600)
        except Exception:
            return False
    
    def _load_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Load data from cache if valid."""
        if not self._is_cache_valid(cache_key):
            return None
        try:
            return pd.read_csv(self._get_cache_path(cache_key))
        except Exception as e:
            print(f"Warning: Could not load cache for {cache_key}: {str(e)}")
            return None
    
    def _save_to_cache(self, data: pd.DataFrame, cache_key: str):
        """Save data to cache with metadata."""
        try:
            data.to_csv(self._get_cache_path(cache_key), index=False)
            metadata = {'timestamp': datetime.now().isoformat(), 'cache_key': cache_key, 'rows': len(data)}
            with open(self._get_cache_path(cache_key, metadata=True), 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save cache for {cache_key}: {str(e)}")
    
    def _get_price(self, ticker: yf.Ticker, info: Dict) -> Optional[float]:
        """Extract price from ticker info or history."""
        price = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose')
        if price is None:
            try:
                hist = ticker.history(period='1d')
                if not hist.empty:
                    price = hist['Close'].iloc[-1]
            except Exception:
                pass
        return price
    
    def _estimate_borrow_cost_from_liquidity(self, info: Dict, spot_price: float) -> float:
        """
        Estimate borrow cost (in basis points) from liquidity metrics.
        
        Uses bid-ask spread, volume, market cap, and short interest.
        """
        # Bid-ask spread factor
        bid, ask = info.get('bid'), info.get('ask')
        spread_pct = ((ask - bid) / bid * 100) if (bid and ask and bid > 0) else 0.1
        spread_factor = min(spread_pct * 100, 200)
        
        # Volume factor (inverse: lower volume = higher cost)
        avg_volume = info.get('averageVolume10days') or info.get('averageVolume', 0)
        if avg_volume > 0:
            volume_factor = min(max(0, 50 - np.log10(avg_volume) * 10), 100)
        else:
            volume_factor = 100
        
        # Market cap factor
        market_cap = info.get('marketCap', 0)
        cap_factor = 0 if market_cap >= 10e9 else (20 if market_cap >= 1e9 else 50)
        
        # Short interest factor
        shares_short = info.get('sharesShort', 0)
        shares_outstanding = info.get('sharesOutstanding', 0)
        short_pct = (shares_short / shares_outstanding * 100) if shares_outstanding > 0 else 0
        short_factor = 50 if short_pct > 20 else (20 if short_pct > 10 else 0)
        
        # Combine factors (base 10 bps, capped at 5-500 bps)
        total = 10 + spread_factor + volume_factor + cap_factor + short_factor
        return round(max(5, min(total, 500)), 2)
    
    def fetch_stock_data(self, symbols: List[str], use_cache: bool = True) -> pd.DataFrame:
        """
        Fetch stock prices and dividends from Yahoo Finance.
        
        Args:
            symbols: List of stock symbols to fetch (e.g., ['AAPL', 'MSFT'])
            use_cache: If True, load from cache if available and valid
        
        Returns:
            DataFrame with columns: symbol, spot_price, dividend_yield, borrow_cost_bps, last_updated
        """
        cache_key = f"stock_data_{'_'.join(sorted(symbols))}"
        if use_cache and (cached := self._load_from_cache(cache_key)) is not None:
            print(f"Loaded stock data from cache for {len(symbols)} symbols")
            return cached
        
        results = []
        current_time = datetime.now()
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                spot_price = self._get_price(ticker, info)
                if spot_price is None:
                    print(f"Warning: Could not fetch price for {symbol}, skipping...")
                    continue
                
                dividend_yield = info.get('dividendYield') or 0.0
                borrow_cost_bps = self._estimate_borrow_cost_from_liquidity(info, spot_price)
                
                results.append({
                    'symbol': symbol,
                    'spot_price': float(spot_price),
                    'dividend_yield': float(dividend_yield),
                    'borrow_cost_bps': float(borrow_cost_bps),
                    'last_updated': current_time.isoformat()
                })
            except Exception as e:
                print(f"Error fetching data for {symbol}: {str(e)}")
                continue
        
        if not results:
            raise ValueError("No stock data could be fetched. Please check your symbols and internet connection.")
        
        df = pd.DataFrame(results)
        if use_cache:
            self._save_to_cache(df, cache_key)
        return df
    
    def fetch_risk_free_rates(self, use_cache: bool = True) -> pd.DataFrame:
        """
        Fetch latest Treasury rates from Federal Reserve (FRED).
        
        Args:
            use_cache: If True, load from cache if available and valid
        
        Returns:
            DataFrame with columns: tenor_days, rate (decimal)
        """
        cache_key = "risk_free_rates"
        if use_cache and (cached := self._load_from_cache(cache_key)) is not None:
            print("Loaded risk-free rates from cache")
            return cached
        
        if not HAS_DATAREADER or web is None:
            raise ImportError("pandas-datareader is required. Install with: pip install pandas-datareader")
        
        # FRED series IDs for Treasury rates
        fred_series = {
            30: 'DGS1MO',      # 1 month
            90: 'DGS3MO',      # 3 months
            180: 'DGS6MO',     # 6 months
            365: 'DGS1',       # 1 year
            730: 'DGS2',       # 2 years
            1095: 'DGS3',      # 3 years
            1825: 'DGS5',      # 5 years
            2555: 'DGS7',      # 7 years
            3650: 'DGS10',     # 10 years
        }
        
        results = []
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5)
        
        for tenor_days, series_id in fred_series.items():
            try:
                data = web.DataReader(series_id, 'fred', start_date, end_date)
                if not data.empty:
                    rate = data[series_id].dropna().iloc[-1] / 100  # Convert to decimal
                    results.append({
                        'tenor_days': tenor_days,
                        'rate': float(rate)
                    })
            except Exception as e:
                print(f"Warning: Could not fetch {series_id} ({tenor_days} days): {str(e)}")
                continue
        
        if not results:
            raise ValueError("No Treasury rates could be fetched. Please check your internet connection.")
        
        df = pd.DataFrame(results).sort_values('tenor_days')
        if use_cache:
            self._save_to_cache(df, cache_key)
        return df
    
    def fetch_options_chain(self, symbol: str, use_cache: bool = True) -> pd.DataFrame:
        """
        Fetch options chain with strikes, prices, volumes, and implied volatility.
        
        Args:
            symbol: Stock symbol to fetch options for
            use_cache: If True, load from cache if available and valid
        
        Returns:
            DataFrame with columns: symbol, expiry, strike, option_type, lastPrice, 
            volume, openInterest, impliedVolatility, bid, ask
        """
        cache_key = f"options_chain_{symbol}"
        if use_cache and (cached := self._load_from_cache(cache_key)) is not None:
            print(f"Loaded options chain from cache for {symbol}")
            return cached
        
        try:
            ticker = yf.Ticker(symbol)
            expirations = ticker.options
            
            if not expirations:
                print(f"Warning: No options data available for {symbol}")
                return pd.DataFrame()
            
            all_options = []
            
            # Fetch options for all available expirations
            for expiry in expirations:
                try:
                    chain = ticker.option_chain(expiry)
                    # Process calls and puts
                    for option_type, chain_df in [('call', chain.calls), ('put', chain.puts)]:
                        if not chain_df.empty:
                            df_copy = chain_df.copy()
                            df_copy['option_type'] = option_type
                            df_copy['expiry'] = expiry
                            df_copy['symbol'] = symbol
                            all_options.append(df_copy)
                except Exception as e:
                    print(f"Warning: Could not fetch options for {symbol} expiry {expiry}: {str(e)}")
                    continue
            
            if not all_options:
                return pd.DataFrame()
            
            # Combine all options
            df = pd.concat(all_options, ignore_index=True)
            
            # Select and ensure required columns exist
            required_cols = ['symbol', 'expiry', 'strike', 'option_type', 'lastPrice', 
                           'volume', 'openInterest', 'impliedVolatility', 'bid', 'ask']
            for col in required_cols:
                if col not in df.columns:
                    df[col] = np.nan
            
            df_result = df[required_cols].copy()
            
            # Filter out illiquid options
            if 'volume' in df_result.columns and 'openInterest' in df_result.columns:
                volume_filled = pd.Series(df_result['volume']).fillna(0)
                oi_filled = pd.Series(df_result['openInterest']).fillna(0)
                mask = (volume_filled > 0) | (oi_filled > 0)
                df_result = df_result.loc[mask].copy()
            
            if use_cache and isinstance(df_result, pd.DataFrame) and not df_result.empty:
                self._save_to_cache(df_result, cache_key)
            return df_result if isinstance(df_result, pd.DataFrame) else pd.DataFrame()
            
        except Exception as e:
            print(f"Error fetching options chain for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def build_volatility_surface(self, symbols: List[str], use_cache: bool = True) -> pd.DataFrame:
        """
        Build volatility surface from options chains.
        
        Args:
            symbols: List of stock symbols to build volatility surface for
            use_cache: If True, load from cache if available and valid
        
        Returns:
            DataFrame with columns: symbol, expiry, strike, moneyness, implied_vol
        """
        cache_key = f"volatility_surface_{'_'.join(sorted(symbols))}"
        if use_cache and (cached := self._load_from_cache(cache_key)) is not None:
            print(f"Loaded volatility surface from cache for {len(symbols)} symbols")
            return cached
        
        all_surfaces = []
        
        # Get spot prices for all symbols
        try:
            market_data = self.fetch_stock_data(symbols)
            spot_prices = dict(zip(market_data['symbol'], market_data['spot_price']))
        except Exception as e:
            print(f"Warning: Could not fetch spot prices: {str(e)}")
            spot_prices = {}
        
        for symbol in symbols:
            try:
                # Fetch options chain
                options_df = self.fetch_options_chain(symbol)
                
                if options_df.empty:
                    print(f"Warning: No options data for {symbol}, skipping...")
                    continue
                
                # Get spot price
                spot_price = spot_prices.get(symbol)                
                if spot_price is None or spot_price <= 0:
                    continue
                
                # Calculate moneyness (strike / spot)
                options_df = options_df.copy()
                options_df['moneyness'] = options_df['strike'] / spot_price
                
                # Extract implied volatility
                if 'impliedVolatility' in options_df.columns:
                    # Filter out invalid implied vols
                    valid_iv = options_df['impliedVolatility'].notna() & (options_df['impliedVolatility'] > 0)
                    options_df = options_df[valid_iv].copy()
                    
                    # Select required columns and create new DataFrame with renamed column
                    surface_df = pd.DataFrame({
                        'symbol': options_df['symbol'],
                        'expiry': options_df['expiry'],
                        'strike': options_df['strike'],
                        'moneyness': options_df['moneyness'],
                        'implied_vol': options_df['impliedVolatility']
                    })
                    
                    # Filter reasonable moneyness range (0.5x to 2x spot)
                    surface_df = surface_df[
                        (surface_df['moneyness'] >= 0.5) & 
                        (surface_df['moneyness'] <= 2.0)
                    ]
                    
                    all_surfaces.append(surface_df)
                else:
                    print(f"Warning: No implied volatility data for {symbol}")
                    
            except Exception as e:
                print(f"Error building volatility surface for {symbol}: {str(e)}")
                continue
        
        if not all_surfaces:
            return pd.DataFrame({'symbol': [], 'expiry': [], 'strike': [], 'moneyness': [], 'implied_vol': []})
        
        result = pd.concat(all_surfaces, ignore_index=True).sort_values(['symbol', 'expiry', 'strike'])
        if use_cache and not result.empty:
            self._save_to_cache(result, cache_key)
        return result
    
    def generate_synthetic_positions(self, symbols: List[str], num_positions: int = 20, seed: Optional[int] = None) -> pd.DataFrame:
        """
        Generate synthetic portfolio positions.
        
        Args:
            symbols: List of stock symbols to generate positions for
            num_positions: Number of positions to generate
            seed: Random seed for reproducibility
        
        Returns:
            DataFrame with columns: position_id, symbol, quantity, instrument_type, strike, expiry, option_type
        """
        if seed is not None:
            np.random.seed(seed)
        
        if not symbols:
            return pd.DataFrame({'position_id': [], 'symbol': [], 'quantity': [], 'instrument_type': [], 'strike': [], 'expiry': [], 'option_type': []})
        
        # Get spot prices for realistic strike generation
        try:
            market_data = self.fetch_stock_data(symbols)
            spot_prices = dict(zip(market_data['symbol'], market_data['spot_price']))
        except Exception:
            # Fallback: use default prices if fetch fails
            spot_prices = {symbol: 100.0 for symbol in symbols}
        
        positions = []
        
        for i in range(num_positions):
            # Randomly select symbol
            symbol = np.random.choice(symbols)
            spot_price = spot_prices.get(symbol, 100.0)
            
            # 60% equities, 40% options
            is_option = np.random.random() < 0.4
            
            if is_option:
                # Generate option position
                option_type = np.random.choice(['call', 'put'])
                
                # Strike around the money (0.8x to 1.2x spot)
                moneyness = np.random.uniform(0.8, 1.2)
                strike = round(spot_price * moneyness, 2)
                
                # Expiry: 30 to 365 days from now
                days_to_expiry = np.random.randint(30, 365)
                expiry = (datetime.now() + timedelta(days=days_to_expiry)).strftime('%Y-%m-%d')
                
                # Quantity: typically smaller for options (1-100 contracts)
                quantity = np.random.choice([-1, 1]) * np.random.randint(1, 101)
                
                positions.append({
                    'position_id': f'POS_{i+1:04d}',
                    'symbol': symbol,
                    'quantity': quantity,
                    'instrument_type': 'option',
                    'strike': strike,
                    'expiry': expiry,
                    'option_type': option_type
                })
            else:
                # Generate equity position
                # Quantity: can be long or short, typically larger (10-1000 shares)
                quantity = np.random.choice([-1, 1]) * np.random.randint(10, 1001)
                
                positions.append({
                    'position_id': f'POS_{i+1:04d}',
                    'symbol': symbol,
                    'quantity': quantity,
                    'instrument_type': 'equity',
                    'strike': None,
                    'expiry': None,
                    'option_type': None
                })
        
        df = pd.DataFrame(positions)
        return df
    
    def generate_synthetic_borrow_costs(self, symbols: List[str]) -> pd.DataFrame:
        """
        Generate synthetic borrow costs for symbols (without fetching market data).
        
        This is useful for testing or when market data is unavailable.
        Uses deterministic generation based on symbol hash for reproducibility.
        
        Args:
            symbols: List of stock symbols to generate borrow costs for
        
        Returns:
            DataFrame with columns: symbol, borrow_cost_bps
        """
        results = []
        
        for symbol in symbols:
            # Use symbol hash for deterministic but varied costs
            symbol_hash = hash(symbol) % 1000
            
            # Base cost: 10-50 bps for most stocks
            base_cost = 10 + (symbol_hash % 40)
            
            # Add variation based on symbol characteristics
            # Longer symbols or certain patterns might indicate different liquidity
            if len(symbol) > 4:
                base_cost += 10  # Longer symbols might be less liquid
            
            # Add some randomness but keep it bounded
            variation = (symbol_hash % 20) - 10  # -10 to +10
            borrow_cost_bps = base_cost + variation
            
            # Cap at reasonable range (5-200 bps for synthetic)
            borrow_cost_bps = max(5, min(borrow_cost_bps, 200))
            
            results.append({
                'symbol': symbol,
                'borrow_cost_bps': round(borrow_cost_bps, 2)
            })
        
        return pd.DataFrame(results)
    
    def save_data(self, data: pd.DataFrame, filename: str, metadata: Optional[Dict] = None):
        """
        Save DataFrame to CSV with timestamp metadata.
        """
        filepath = os.path.join(self.data_dir, filename)
        data.to_csv(filepath, index=False)
        
        if metadata:
            metadata_file = os.path.join(self.data_dir, "metadata.json")
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    existing_metadata = json.load(f)
            else:
                existing_metadata = {}
            
            existing_metadata[filename] = {
                **metadata,
                "timestamp": datetime.now().isoformat()
            }
            
            with open(metadata_file, 'w') as f:
                json.dump(existing_metadata, f, indent=2)
    
    def load_data(self, filename: str) -> pd.DataFrame:
        """
        Load DataFrame from CSV.
        """
        filepath = os.path.join(self.data_dir, filename)
        return pd.read_csv(filepath)
