import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple
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
    
    def __init__(self, data_dir: str = "data", cache_expiry_hours: int = 1, borrow_cap_bps: float = 5000.0):
        self.data_dir = data_dir
        self.cache_expiry_hours = cache_expiry_hours
        self.borrow_cap_bps = borrow_cap_bps  # cap for borrow cost (e.g. 5000 = 50%); HTB can exceed
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
        except (FileNotFoundError, json.JSONDecodeError, ValueError, KeyError):
            return False
    
    def _delete_stale_cache(self, cache_key: str) -> None:
        """Remove cache .csv and _metadata.json for a key if they exist."""
        for path in [self._get_cache_path(cache_key), self._get_cache_path(cache_key, metadata=True)]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
    
    def _load_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Load data from cache if valid. If invalid, deletes the stale cache files."""
        if not self._is_cache_valid(cache_key):
            self._delete_stale_cache(cache_key)
            return None
        try:
            return pd.read_csv(self._get_cache_path(cache_key))
        except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
            print(f"Warning: Could not load cache for {cache_key}: {str(e)}")
            return None
    
    def load_treasury_etf_data(self) -> Optional[pd.DataFrame]:
        """Load Treasury ETF data from permanent CSV file if it exists."""
        filepath = os.path.join(self.data_dir, "treasury_etf_data.csv")
        if os.path.exists(filepath):
            try:
                return pd.read_csv(filepath)
            except Exception as e:
                print(f"Warning: Could not load treasury_etf_data.csv: {str(e)}")
                return None
        return None
    
    def _save_to_cache(self, data: pd.DataFrame, cache_key: str):
        """Save data to cache with metadata."""
        try:
            data.to_csv(self._get_cache_path(cache_key), index=False)
            metadata = {'timestamp': datetime.now().isoformat(), 'cache_key': cache_key, 'rows': len(data)}
            with open(self._get_cache_path(cache_key, metadata=True), 'w') as f:
                json.dump(metadata, f, indent=2)
        except (IOError, OSError, PermissionError):
            # Silently fail on cache write errors - not critical
            pass
    
    def _get_price(self, ticker: yf.Ticker, info: Dict) -> Optional[float]:
        """Extract price from ticker info or history."""
        price = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose')
        if price is None:
            try:
                hist = ticker.history(period='1d')
                if not hist.empty:
                    price = hist['Close'].iloc[-1]
            except (KeyError, IndexError, AttributeError):
                pass
        return price
    
    def _get_spot_and_dividend(self, symbol: str) -> Tuple[Any, float]:
        """Get underlying spot price and dividend yield for a symbol. Reuses _get_price."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            spot = self._get_price(ticker, info)
            dy = info.get('dividendYield') or info.get('trailingAnnualDividendYield') or 0.0
            dy = float(dy) if dy is not None else 0.0
            if dy > 1.0:
                dy = dy / 100.0
            return (float(spot) if spot is not None else np.nan, dy)
        except (TypeError, AttributeError, ValueError, KeyError):
            return (np.nan, 0.0)
    
    def _estimate_borrow_cost_from_liquidity(
        self,
        info: Dict,
        spot_price: float,
        *,
        borrow_cap_bps: float = 5000.0,
    ) -> Dict[str, Any]:
        """
        Estimate borrow cost (in basis points) from liquidity metrics.
        
        Uses: bid-ask spread, volume, market cap, short interest, dividend yield,
        and an HTB (hard-to-borrow) premium when short interest is very high.
        Returns total_bps and a breakdown dict for logging and UI.
        
        - Cap: default 5000 bps (50%) so HTB names can reach 50%+; override via
          borrow_cap_bps if needed.
        - Dividend: shorts pay dividends to the lender; we add dividend_yield * 10^4
          (capped at 1000 bps) as a proxy for this carry.
        - HTB premium: when short_pct > 25%, we add 500–2500 bps to approximate
          the jump when a name goes special (beyond what short_factor captures).
        """
        bid = info.get('bid')
        ask = info.get('ask')
        spread_pct = ((ask - bid) / bid * 100) if (bid and ask and bid > 0) else 0.1
        spread_factor = min(spread_pct * 100, 200)
        
        avg_volume = float(info.get('averageVolume10days') or info.get('averageVolume') or 0)
        if avg_volume > 0:
            volume_factor = min(max(0, 50 - np.log10(avg_volume) * 10), 100)
        else:
            volume_factor = 100.0
        
        market_cap = float(info.get('marketCap') or 0)
        cap_factor = 0 if market_cap >= 10e9 else (20 if market_cap >= 1e9 else 50)
        
        shares_short = float(info.get('sharesShort') or 0)
        shares_outstanding = float(info.get('sharesOutstanding') or 0)
        short_pct = (shares_short / shares_outstanding * 100) if shares_outstanding > 0 else 0.0
        short_factor = 50 if short_pct > 20 else (20 if short_pct > 10 else 0)
        
        dy = float(info.get('dividendYield') or info.get('trailingAnnualDividendYield') or 0)
        if dy > 1.0:
            dy = dy / 100.0
        dividend_bps = min(dy * 10000, 1000.0)
        
        # HTB premium: when short_pct is very high, borrow can spike to 20–50%+; add overlay
        if short_pct > 45:
            htb_premium = 2500.0
        elif short_pct > 35:
            htb_premium = 1500.0
        elif short_pct > 25:
            htb_premium = 500.0
        else:
            htb_premium = 0.0
        
        base = 10.0
        total = base + spread_factor + volume_factor + cap_factor + short_factor + dividend_bps + htb_premium
        total = round(max(5, min(total, borrow_cap_bps)), 2)
        
        return {
            "total_bps": total,
            "borrow_base_bps": base,
            "borrow_spread_bps": round(spread_factor, 2),
            "borrow_volume_bps": round(volume_factor, 2),
            "borrow_cap_bps": cap_factor,
            "borrow_short_bps": short_factor,
            "borrow_dividend_bps": round(dividend_bps, 2),
            "borrow_htb_premium_bps": round(htb_premium, 2),
            "dividend_yield_borrow": round(dy, 4),
            "bid": float(bid) if bid is not None else None,
            "ask": float(ask) if ask is not None else None,
            "avg_volume": avg_volume,
            "market_cap": market_cap if market_cap > 0 else None,
            "short_pct": round(short_pct, 2),
            "spread_pct": round(spread_pct, 4),
        }
    
    def _estimate_transaction_cost_from_liquidity(self, info: Dict, spot_price: float) -> Dict[str, Any]:
        """
        Estimate transaction cost (in basis points) from liquidity metrics.
        
        Uses bid-ask spread, volume, market cap, and volatility.
        Transaction cost represents the cost of executing a trade (one-way).
        Returns total_bps and a breakdown dict for logging and UI.
        
        Components:
        - Bid-ask spread (primary): Half the spread as one-way cost
        - Volume factor: Lower volume = higher market impact
        - Market cap factor: Smaller cap = higher cost
        - Volatility adjustment: Higher volatility increases execution risk and
          dealer/inventory risk; we use 52-week range (realized vol proxy) and
          beta (market-relative vol). Adds 0–20 bps.
        - Base commission: Assumed institutional commission
        """
        bid = info.get('bid')
        ask = info.get('ask')
        if bid and ask and bid > 0:
            mid_price = (bid + ask) / 2
            spread_bps_raw = ((ask - bid) / mid_price) * 10000
            spread_cost = min(spread_bps_raw / 2, 50)  # One-way cost, cap at 50 bps
        else:
            spread_bps_raw = 5.0  # default assumption
            spread_cost = 2.5  # Default spread cost (half of typical 5 bps spread)
        
        avg_volume = float(info.get('averageVolume10days') or info.get('averageVolume') or 0)
        if avg_volume > 0:
            volume_factor = max(0, 10 - np.log10(max(avg_volume, 1)) * 2)
        else:
            volume_factor = 10.0
        
        market_cap = float(info.get('marketCap') or 0)
        if market_cap >= 10e9:
            cap_factor = 0
        elif market_cap >= 1e9:
            cap_factor = 3
        else:
            cap_factor = 8
        
        # --- Volatility adjustment ---
        # 52-week range: (high - low) / spot as a volatility proxy; scale to bps and cap
        high = info.get('fiftyTwoWeekHigh') or info.get('52WeekHigh')
        low = info.get('fiftyTwoWeekLow') or info.get('52WeekLow')
        if high is not None and low is not None and spot_price > 0 and high > low:
            range_pct = (float(high) - float(low)) / spot_price
            range_bps = range_pct * 10000
            vol_52w_bps = min(range_bps * 0.04, 12.0)  # 4% of 52w range in bps, cap 12
        else:
            vol_52w_bps = 0.0
        
        # Beta: beta > 1 implies higher systematic volatility; add (beta - 1) * 4 bps, cap 10
        raw_beta = info.get('beta') or info.get('beta3Year')
        if raw_beta is not None:
            beta = float(raw_beta)
            vol_beta_bps = min(max(0, (beta - 1.0) * 4.0), 10.0) if beta > 1.0 else 0.0
        else:
            vol_beta_bps = 0.0
        
        tx_volatility_bps = round(min(vol_52w_bps + vol_beta_bps, 20.0), 2)  # cap combined at 20 bps
        
        base_commission = 2.0
        total = base_commission + spread_cost + volume_factor + cap_factor + tx_volatility_bps
        total = round(max(1, min(total, 100)), 2)
        
        return {
            "total_bps": total,
            "tx_base_bps": base_commission,
            "tx_spread_bps": round(spread_cost, 2),
            "tx_volume_bps": round(volume_factor, 2),
            "tx_cap_bps": cap_factor,
            "tx_volatility_bps": tx_volatility_bps,
            "tx_vol_52w_bps": round(vol_52w_bps, 2),
            "tx_vol_beta_bps": round(vol_beta_bps, 2),
            "bid": float(bid) if bid is not None else None,
            "ask": float(ask) if ask is not None else None,
            "avg_volume": avg_volume,
            "market_cap": market_cap if market_cap > 0 else None,
            "spread_bps_raw": round(spread_bps_raw, 2) if (bid and ask and bid > 0) else None,
        }
    
    def fetch_stock_data(self, symbols: List[str], use_cache: bool = True) -> pd.DataFrame:
        """
        Fetch stock prices and dividends from Yahoo Finance.
        
        Args:
            symbols: List of stock symbols to fetch (e.g., ['AAPL', 'MSFT'])
            use_cache: If True, load from cache if available and valid
        
        Returns:
            DataFrame with columns: symbol, spot_price, dividend_yield, borrow_cost_bps, 
                                  transaction_cost_bps, last_updated
        """
        cache_key = f"stock_data_{'_'.join(sorted(symbols))}"
        if use_cache and (cached := self._load_from_cache(cache_key)) is not None:
            print(f"Using cached stock data for {len(symbols)} symbols")
            return cached
        
        print(f"Fetching stock data for {len(symbols)} symbols")
        results = []
        failed_symbols = []
        current_time = datetime.now()
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                spot_price = self._get_price(ticker, info)
                if spot_price is None:
                    failed_symbols.append(f"{symbol} (no price data)")
                    continue
                
                dividend_yield = info.get('dividendYield') or 0.0
                borrow = self._estimate_borrow_cost_from_liquidity(info, spot_price, borrow_cap_bps=self.borrow_cap_bps)
                tx = self._estimate_transaction_cost_from_liquidity(info, spot_price)
                borrow_cost_bps = borrow["total_bps"]
                transaction_cost_bps = tx["total_bps"]
                
                # Log costs per ticker
                print(f"  {symbol}: transaction_cost={transaction_cost_bps} bps (base={tx['tx_base_bps']}, spread={tx['tx_spread_bps']}, vol={tx['tx_volume_bps']}, cap={tx['tx_cap_bps']}, volatility={tx.get('tx_volatility_bps', 0)}); borrow_cost={borrow_cost_bps} bps (base={borrow['borrow_base_bps']}, spread={borrow['borrow_spread_bps']}, vol={borrow['borrow_volume_bps']}, cap={borrow['borrow_cap_bps']}, short={borrow['borrow_short_bps']}, div={borrow.get('borrow_dividend_bps', 0)}, htb={borrow.get('borrow_htb_premium_bps', 0)})")
                
                row: Dict[str, Any] = {
                    'symbol': symbol,
                    'spot_price': float(spot_price),
                    'dividend_yield': float(dividend_yield),
                    'borrow_cost_bps': float(borrow_cost_bps),
                    'transaction_cost_bps': float(transaction_cost_bps),
                    'last_updated': current_time.isoformat(),
                    # Transaction cost breakdown
                    'tx_base_bps': tx['tx_base_bps'],
                    'tx_spread_bps': tx['tx_spread_bps'],
                    'tx_volume_bps': tx['tx_volume_bps'],
                    'tx_cap_bps': tx['tx_cap_bps'],
                    'tx_volatility_bps': tx.get('tx_volatility_bps', 0),
                    'tx_vol_52w_bps': tx.get('tx_vol_52w_bps', 0),
                    'tx_vol_beta_bps': tx.get('tx_vol_beta_bps', 0),
                    'spread_bps_raw': tx.get('spread_bps_raw'),
                    # Borrow cost breakdown
                    'borrow_base_bps': borrow['borrow_base_bps'],
                    'borrow_spread_bps': borrow['borrow_spread_bps'],
                    'borrow_volume_bps': borrow['borrow_volume_bps'],
                    'borrow_cap_bps': borrow['borrow_cap_bps'],
                    'borrow_short_bps': borrow['borrow_short_bps'],
                    'borrow_dividend_bps': borrow.get('borrow_dividend_bps', 0),
                    'borrow_htb_premium_bps': borrow.get('borrow_htb_premium_bps', 0),
                    'dividend_yield_borrow': borrow.get('dividend_yield_borrow'),
                    'short_pct': borrow['short_pct'],
                    'spread_pct': borrow['spread_pct'],
                    # Inputs (shared)
                    'bid': borrow.get('bid') or tx.get('bid'),
                    'ask': borrow.get('ask') or tx.get('ask'),
                    'avg_volume': borrow.get('avg_volume') if borrow.get('avg_volume', 0) > 0 else tx.get('avg_volume'),
                    'market_cap': borrow.get('market_cap') or tx.get('market_cap'),
                }
                results.append(row)
            except (AttributeError, KeyError, ValueError, TypeError) as e:
                failed_symbols.append(f"{symbol} ({str(e)})")
                continue
            except Exception as e:
                # Catch any other unexpected errors
                failed_symbols.append(f"{symbol} (unexpected error: {str(e)})")
                continue
        
        # Report failed symbols if any
        if failed_symbols:
            print(f"Warning: Failed to fetch data for {len(failed_symbols)} symbol(s): {', '.join(failed_symbols)}")
        
        if not results:
            raise ValueError(f"No stock data could be fetched for any symbols. Failed symbols: {', '.join(failed_symbols) if failed_symbols else 'all symbols'}. Please check your symbols and internet connection.")
        
        df = pd.DataFrame(results)
        if use_cache:
            self._save_to_cache(df, cache_key)
        print(f"Fetched stock data for {len(df)} symbols")
        return df
    
    def fetch_risk_free_rates(self, use_cache: bool = True) -> pd.DataFrame:
        """
        Fetch latest Treasury rates from Federal Reserve (FRED) or yfinance fallback.
        
        Args:
            use_cache: If True, load from cache if available and valid
        
        Returns:
            DataFrame with columns: tenor_days, rate (decimal)
        """
        cache_key = "risk_free_rates"
        if use_cache and (cached := self._load_from_cache(cache_key)) is not None:
            print("Using cached risk-free rates")
            return cached
        
        print("Fetching risk-free rates")
        
        results = []
        
        # Try FRED via pandas-datareader first
        if HAS_DATAREADER and web is not None:
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
                except (KeyError, IndexError, ValueError, AttributeError):
                    continue
        
        # Fallback to yfinance if FRED failed or pandas-datareader unavailable
        if not results:
            print("Using yfinance fallback for Treasury rates")
            # yfinance tickers for Treasury rates
            yf_tickers = {
                30: '^IRX',    # 13-week (closest to 1 month)
                90: '^IRX',    # 13-week (closest to 3 months)
                180: '^IRX',   # 13-week (closest to 6 months)
                365: '^IRX',   # 13-week (closest to 1 year)
                730: '^FVX',  # 5-year (closest to 2 years)
                1095: '^FVX', # 5-year (closest to 3 years)
                1825: '^FVX', # 5-year
                2555: '^FVX', # 5-year (closest to 7 years)
                3650: '^TNX', # 10-year
            }
            
            for tenor_days, ticker_symbol in yf_tickers.items():
                try:
                    ticker = yf.Ticker(ticker_symbol)
                    hist = ticker.history(period='5d')
                    if not hist.empty:
                        rate = hist['Close'].iloc[-1] / 100  # Convert to decimal
                        results.append({
                            'tenor_days': tenor_days,
                            'rate': float(rate)
                        })
                except (KeyError, IndexError, ValueError, AttributeError):
                    continue
        
        # If still no results, use reasonable defaults
        if not results:
            print("Warning: Could not fetch Treasury rates. Using default rates.")
            defaults = {
                30: 0.045,   # 4.5% for short-term
                90: 0.045,
                180: 0.045,
                365: 0.047,  # 4.7% for 1 year
                730: 0.048,  # 4.8% for 2 years
                1095: 0.049, # 4.9% for 3 years
                1825: 0.050, # 5.0% for 5 years
                2555: 0.051, # 5.1% for 7 years
                3650: 0.052, # 5.2% for 10 years
            }
            results = [{'tenor_days': k, 'rate': v} for k, v in defaults.items()]
        
        df = pd.DataFrame(results).sort_values('tenor_days')
        if use_cache:
            self._save_to_cache(df, cache_key)
        print(f"Fetched {len(df)} risk-free rate tenors")
        return df
    
    def fetch_options_chain(
        self,
        symbol: str,
        use_cache: bool = True,
        min_volume: int = 0,
        min_open_interest: int = 0,
    ) -> pd.DataFrame:
        """
        Fetch options chain with strikes, prices, volumes, and implied volatility.
        
        Args:
            symbol: Stock symbol to fetch options for
            use_cache: If True, load from cache if available and valid
            min_volume: Optional minimum volume to keep (0 = no extra filter)
            min_open_interest: Optional minimum open interest to keep (0 = no extra filter)
        
        Returns:
            DataFrame with columns: symbol, expiry, strike, option_type, lastPrice, 
            volume, openInterest, impliedVolatility, bid, ask
        """
        cache_key = f"options_chain_{symbol}"
        if use_cache and (cached := self._load_from_cache(cache_key)) is not None:
            print(f"Using cached options chain for {symbol}")
            out = cached
            # Apply extra liquidity filter when params > 0 (cache does not store filter)
            if (min_volume > 0 or min_open_interest > 0) and not out.empty and 'volume' in out.columns and 'openInterest' in out.columns:
                v = np.asarray(out['volume'], dtype=float)
                o = np.asarray(out['openInterest'], dtype=float)
                vol = np.where(np.isnan(v), 0, v)
                oi = np.where(np.isnan(o), 0, o)
                out = out.loc[(vol >= min_volume) | (oi >= min_open_interest)].copy()
            return out
        
        print(f"Fetching options chain for {symbol}")
        
        try:
            ticker = yf.Ticker(symbol)
            expirations = ticker.options
            
            if not expirations:
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
                except (AttributeError, KeyError, ValueError):
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
            
            # Filter out illiquid options: (volume>0)|(openInterest>0); cache stores this only
            if 'volume' in df_result.columns and 'openInterest' in df_result.columns:
                v = np.asarray(df_result['volume'], dtype=float)
                o = np.asarray(df_result['openInterest'], dtype=float)
                vol = np.where(np.isnan(v), 0, v)
                oi = np.where(np.isnan(o), 0, o)
                df_result = df_result.loc[((vol > 0) | (oi > 0))].copy()
            
            if use_cache and isinstance(df_result, pd.DataFrame) and not df_result.empty:
                self._save_to_cache(df_result, cache_key)
            # Apply optional min_volume/min_open_interest after cache save so cache stays reusable
            if (min_volume > 0 or min_open_interest > 0) and not df_result.empty and 'volume' in df_result.columns and 'openInterest' in df_result.columns:
                v = np.asarray(df_result['volume'], dtype=float)
                o = np.asarray(df_result['openInterest'], dtype=float)
                vol = np.where(np.isnan(v), 0, v)
                oi = np.where(np.isnan(o), 0, o)
                df_result = df_result.loc[(vol >= min_volume) | (oi >= min_open_interest)].copy()
            if isinstance(df_result, pd.DataFrame) and not df_result.empty:
                print(f"Fetched {len(df_result)} options for {symbol}")
            return df_result if isinstance(df_result, pd.DataFrame) else pd.DataFrame()
            
        except (AttributeError, KeyError, ValueError, TypeError):
            return pd.DataFrame()
    
    def fetch_options_chains_for_hedge_symbols(
        self,
        symbols: List[str],
        use_cache: bool = True,
        min_volume: int = 0,
        min_open_interest: int = 0,
    ) -> pd.DataFrame:
        """
        Fetch and combine options chains for hedge symbols (e.g. SPY, QQQ, IWM) with
        underlying spot and dividend yield. Reuses fetch_stock_data, fetch_options_chain,
        and _get_spot_and_dividend (which uses _get_price).
        
        Args:
            symbols: Underlying symbols to fetch options for (e.g. ['SPY', 'QQQ', 'IWM'])
            use_cache: If True, use cached per-symbol options chains when valid
            min_volume: Minimum volume to keep (passed to fetch_options_chain; 0 = no extra filter)
            min_open_interest: Minimum open interest to keep (passed to fetch_options_chain; 0 = no extra filter)
        
        Returns:
            DataFrame with columns: symbol, expiry, strike, option_type, lastPrice,
            volume, openInterest, impliedVolatility, bid, ask, underlying_spot, dividend_yield
        """
        if not symbols:
            return pd.DataFrame()
        
        # Reuse fetch_stock_data for underlying spot and dividend yield
        spot_by: Dict[str, float] = {}
        div_by: Dict[str, float] = {}
        try:
            stock_df = self.fetch_stock_data(symbols, use_cache=use_cache)
            for _, r in stock_df.iterrows():
                s = str(r['symbol'])
                spot_by[s] = float(r['spot_price'])
                div_by[s] = float(r.get('dividend_yield', 0.0) or 0.0)
        except (ValueError, KeyError, Exception):
            pass
        
        # Fallback for symbols missing from fetch_stock_data: try fetch_stock_data([s]), else _get_spot_and_dividend
        for s in symbols:
            if s in spot_by:
                continue
            try:
                one = self.fetch_stock_data([s], use_cache=use_cache)
                if not one.empty:
                    spot_by[s] = float(one.iloc[0]['spot_price'])
                    div_by[s] = float(one.iloc[0].get('dividend_yield', 0.0) or 0.0)
                else:
                    spot_by[s], div_by[s] = self._get_spot_and_dividend(s)
            except Exception:
                spot_by[s], div_by[s] = self._get_spot_and_dividend(s)
        
        # Reuse fetch_options_chain (with liquidity params) and enrich with underlying_spot, dividend_yield
        all_chains: List[pd.DataFrame] = []
        for symbol in symbols:
            chain = self.fetch_options_chain(
                symbol, use_cache=use_cache, min_volume=min_volume, min_open_interest=min_open_interest
            )
            if chain.empty:
                continue
            chain = chain.copy()
            chain['underlying_spot'] = spot_by.get(symbol, np.nan)
            chain['dividend_yield'] = div_by.get(symbol, 0.0)
            all_chains.append(chain)
        
        if not all_chains:
            return pd.DataFrame()
        return pd.concat(all_chains, ignore_index=True)
    
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
            print(f"Using cached volatility surface for {len(symbols)} symbols")
            return cached
        
        print(f"Building volatility surface for {len(symbols)} symbols")
        
        all_surfaces = []
        failed_symbols = []
        
        # Get spot prices for all symbols
        try:
            market_data = self.fetch_stock_data(symbols)
            spot_prices = dict(zip(market_data['symbol'], market_data['spot_price']))
        except (KeyError, ValueError, AttributeError) as e:
            print(f"Warning: Could not fetch market data for volatility surface: {str(e)}")
            spot_prices = {}
        
        for symbol in symbols:
            try:
                # Fetch options chain
                options_df = self.fetch_options_chain(symbol)
                
                if options_df.empty:
                    failed_symbols.append(f"{symbol} (no options data)")
                    continue
                
                # Get spot price
                spot_price = spot_prices.get(symbol)                
                if spot_price is None or spot_price <= 0:
                    failed_symbols.append(f"{symbol} (invalid spot price)")
                    continue
                
                # Calculate moneyness (strike / spot)
                options_df = options_df.copy()
                options_df['moneyness'] = options_df['strike'] / spot_price
                
                # Extract implied volatility
                if 'impliedVolatility' in options_df.columns:
                    # Filter out invalid implied vols
                    valid_iv = options_df['impliedVolatility'].notna() & (options_df['impliedVolatility'] > 0)
                    options_df = options_df[valid_iv].copy()
                    
                    if options_df.empty:
                        failed_symbols.append(f"{symbol} (no valid implied volatility)")
                        continue
                    
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
                    
                    if not surface_df.empty:
                        all_surfaces.append(surface_df)
                    else:
                        failed_symbols.append(f"{symbol} (no options in moneyness range)")
                else:
                    failed_symbols.append(f"{symbol} (no implied volatility column)")
                    
            except (KeyError, ValueError, AttributeError, TypeError) as e:
                failed_symbols.append(f"{symbol} (error: {str(e)})")
                continue
            except Exception as e:
                failed_symbols.append(f"{symbol} (unexpected error: {str(e)})")
                continue
        
        # Report failed symbols if any
        if failed_symbols:
            print(f"Warning: Could not build volatility surface for {len(failed_symbols)} symbol(s): {', '.join(failed_symbols[:10])}{'...' if len(failed_symbols) > 10 else ''}")
        
        if not all_surfaces:
            return pd.DataFrame({'symbol': [], 'expiry': [], 'strike': [], 'moneyness': [], 'implied_vol': []})
        
        result = pd.concat(all_surfaces, ignore_index=True).sort_values(['symbol', 'expiry', 'strike'])
        if use_cache and not result.empty:
            self._save_to_cache(result, cache_key)
        if not result.empty:
            print(f"Built volatility surface with {len(result)} points")
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
        except (KeyError, ValueError, AttributeError):
            # Fallback: use default prices if fetch fails
            spot_prices = {symbol: 100.0 for symbol in symbols}
        
        positions = []
        
        for i in range(num_positions):
            # Randomly select symbol
            symbol = np.random.choice(symbols)
            spot_price = spot_prices.get(symbol, 100.0)

            equity_to_option_ratio = np.random.random()
            
            is_option = np.random.random() < equity_to_option_ratio
            
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
        print(f"Saved {filename} ({len(data)} rows)")
        
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
    
    def load_all_data(
        self,
        symbols: List[str],
        num_positions: int = 20,
        seed: Optional[int] = None,
        use_cache: bool = True,
        generate_positions: bool = True,
        hedge_option_symbols: Optional[List[str]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Do-it-all function: Fetches all required data and saves to expected file names.
        
        This function orchestrates the entire data loading pipeline:
        1. Fetches stock market data → saves as 'market_data.csv'
        2. Fetches risk-free rates → saves as 'rates.csv'
        3. Builds volatility surface → saves as 'vol_surface.csv'
        4. Generates synthetic positions → saves as 'positions.csv'
        5. Fetches options chains for hedge symbols → saves as 'hedge_options.csv'
        
        Args:
            symbols: List of stock symbols to fetch data for
            num_positions: Number of synthetic positions to generate
            seed: Random seed for position generation (for reproducibility)
            use_cache: If True, use cached data when available
            generate_positions: If True, generate synthetic positions; if False, skip
            hedge_option_symbols: Symbols to fetch options chains for hedging (default: ['SPY', 'QQQ', 'IWM'])
        
        Returns:
            Dictionary with keys: 'market_data', 'rates', 'vol_surface', 'positions', 'hedge_options'
            containing the respective DataFrames
        
        Raises:
            ValueError: If required data cannot be fetched
        """
        print("Loading all required data")
        results: Dict[str, Any] = {}
        
        # 1. Fetch and save market data
        market_data = self.fetch_stock_data(symbols, use_cache=use_cache)
        self.save_data(market_data, "market_data.csv", 
                      metadata={'symbols': symbols, 'num_symbols': len(market_data)})
        results['market_data'] = market_data
        
        # 2. Fetch and save risk-free rates
        rates = self.fetch_risk_free_rates(use_cache=use_cache)
        self.save_data(rates, "rates.csv", 
                      metadata={'num_tenors': len(rates)})
        results['rates'] = rates
        
        # 3. Build and save volatility surface
        vol_surface = self.build_volatility_surface(symbols, use_cache=use_cache)
        if not vol_surface.empty:
            self.save_data(vol_surface, "vol_surface.csv",
                          metadata={'symbols': symbols, 'num_points': len(vol_surface)})
            results['vol_surface'] = vol_surface
        else:
            # Create empty file with correct columns
            empty_vol = pd.DataFrame({'symbol': [], 'expiry': [], 'strike': [], 'moneyness': [], 'implied_vol': []})
            self.save_data(empty_vol, "vol_surface.csv", metadata={'symbols': symbols, 'num_points': 0})
            results['vol_surface'] = empty_vol
        
        # 4. Generate and save synthetic positions
        if generate_positions:
            print(f"Generating {num_positions} synthetic positions")
            positions = self.generate_synthetic_positions(symbols, num_positions, seed)
            self.save_data(positions, "positions.csv",
                          metadata={'num_positions': len(positions), 'seed': seed})
            results['positions'] = positions
            print(f"Generated {len(positions)} positions")
        else:
            results['positions'] = None
        
        # 5. Fetch and save options chains for hedge symbols (SPY, QQQ, IWM by default)
        if hedge_option_symbols is None:
            hedge_option_symbols = ['SPY', 'QQQ', 'IWM']
        if hedge_option_symbols:
            print(f"Fetching hedge options for {hedge_option_symbols}")
            hedge_options = self.fetch_options_chains_for_hedge_symbols(
                hedge_option_symbols, use_cache=use_cache
            )
            if not hedge_options.empty:
                self.save_data(hedge_options, "hedge_options.csv",
                              metadata={'symbols': hedge_option_symbols, 'num_options': len(hedge_options)})
                results['hedge_options'] = hedge_options
                print(f"Saved {len(hedge_options)} hedge options")
            else:
                empty_opts = pd.DataFrame({
                    'symbol': [], 'expiry': [], 'strike': [], 'option_type': [],
                    'lastPrice': [], 'volume': [], 'openInterest': [],
                    'impliedVolatility': [], 'bid': [], 'ask': [],
                    'underlying_spot': [], 'dividend_yield': []
                })
                self.save_data(empty_opts, "hedge_options.csv",
                              metadata={'symbols': hedge_option_symbols, 'num_options': 0})
                results['hedge_options'] = empty_opts
        else:
            results['hedge_options'] = None
        
        print("Data loading complete")
        
        return results
    
    def fetch_treasury_etf_data(self, symbols: List[str], use_cache: bool = True) -> pd.DataFrame:
        """
        Fetch Treasury ETF data including price, yield, and duration information.
        
        Args:
            symbols: List of Treasury ETF symbols (e.g., ['TLT', 'IEF', 'SHY'])
            use_cache: If True, load from cache if available and valid
        
        Returns:
            DataFrame with columns: symbol, spot_price, dividend_yield, borrow_cost_bps,
                                  transaction_cost_bps, yield_to_maturity, duration_years, last_updated
        """
        cache_key = f"treasury_etf_data_{'_'.join(sorted(symbols))}"
        if use_cache and (cached := self._load_from_cache(cache_key)) is not None:
            print(f"Using cached Treasury ETF data for {len(symbols)} symbols")
            return cached
        
        print(f"Fetching Treasury ETF data for {len(symbols)} symbols")
        results = []
        failed_symbols = []
        current_time = datetime.now()
        
        # Duration mapping for Treasury ETFs (approximate, in years)
        duration_map = {
            'TLT': 17.5,  # iShares 20+ Year Treasury Bond ETF
            'IEF': 7.5,   # iShares 7-10 Year Treasury Bond ETF
            'SHY': 2.0,   # iShares 1-3 Year Treasury Bond ETF
            'TBT': 17.5,  # ProShares UltraShort 20+ Year Treasury
            'TBF': 17.5,  # ProShares Short 20+ Year Treasury
            'IEI': 5.0,   # iShares 3-7 Year Treasury Bond ETF
            'VGIT': 5.0,  # Vanguard Intermediate-Term Treasury ETF
        }
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                spot_price = self._get_price(ticker, info)
                if spot_price is None:
                    failed_symbols.append(f"{symbol} (no price data)")
                    continue
                
                dividend_yield = info.get('dividendYield') or 0.0
                borrow = self._estimate_borrow_cost_from_liquidity(info, spot_price, borrow_cap_bps=self.borrow_cap_bps)
                tx = self._estimate_transaction_cost_from_liquidity(info, spot_price)
                borrow_cost_bps = borrow["total_bps"]
                transaction_cost_bps = tx["total_bps"]
                
                print(f"  {symbol}: transaction_cost={transaction_cost_bps} bps (base={tx['tx_base_bps']}, spread={tx['tx_spread_bps']}, vol={tx['tx_volume_bps']}, cap={tx['tx_cap_bps']}, volatility={tx.get('tx_volatility_bps', 0)}); borrow_cost={borrow_cost_bps} bps (base={borrow['borrow_base_bps']}, spread={borrow['borrow_spread_bps']}, vol={borrow['borrow_volume_bps']}, cap={borrow['borrow_cap_bps']}, short={borrow['borrow_short_bps']}, div={borrow.get('borrow_dividend_bps', 0)}, htb={borrow.get('borrow_htb_premium_bps', 0)})")
                
                # Get yield to maturity (30-day SEC yield or trailing yield)
                yield_to_maturity = info.get('yield') or info.get('trailingAnnualDividendYield') or 0.0
                if yield_to_maturity > 1.0:  # If in percentage form, convert to decimal
                    yield_to_maturity = yield_to_maturity / 100.0
                
                # Get duration from mapping or estimate from yield
                duration_years = duration_map.get(symbol)
                if duration_years is None:
                    # Estimate duration: longer maturity ETFs have higher duration
                    # Rough approximation based on symbol characteristics
                    if '20' in symbol.upper() or 'LONG' in symbol.upper():
                        duration_years = 17.0
                    elif '10' in symbol.upper() or '7' in symbol.upper():
                        duration_years = 7.0
                    elif '3' in symbol.upper() or '5' in symbol.upper():
                        duration_years = 5.0
                    elif '1' in symbol.upper() or 'SHORT' in symbol.upper():
                        duration_years = 2.0
                    else:
                        duration_years = 5.0  # Default
                
                row = {
                    'symbol': symbol,
                    'spot_price': float(spot_price),
                    'dividend_yield': float(dividend_yield),
                    'borrow_cost_bps': float(borrow_cost_bps),
                    'transaction_cost_bps': float(transaction_cost_bps),
                    'yield_to_maturity': float(yield_to_maturity),
                    'duration_years': float(duration_years),
                    'last_updated': current_time.isoformat(),
                    'tx_base_bps': tx['tx_base_bps'],
                    'tx_spread_bps': tx['tx_spread_bps'],
                    'tx_volume_bps': tx['tx_volume_bps'],
                    'tx_cap_bps': tx['tx_cap_bps'],
                    'tx_volatility_bps': tx.get('tx_volatility_bps', 0),
                    'tx_vol_52w_bps': tx.get('tx_vol_52w_bps', 0),
                    'tx_vol_beta_bps': tx.get('tx_vol_beta_bps', 0),
                    'spread_bps_raw': tx.get('spread_bps_raw'),
                    'borrow_base_bps': borrow['borrow_base_bps'],
                    'borrow_spread_bps': borrow['borrow_spread_bps'],
                    'borrow_volume_bps': borrow['borrow_volume_bps'],
                    'borrow_cap_bps': borrow['borrow_cap_bps'],
                    'borrow_short_bps': borrow['borrow_short_bps'],
                    'borrow_dividend_bps': borrow.get('borrow_dividend_bps', 0),
                    'borrow_htb_premium_bps': borrow.get('borrow_htb_premium_bps', 0),
                    'dividend_yield_borrow': borrow.get('dividend_yield_borrow'),
                    'short_pct': borrow['short_pct'],
                    'spread_pct': borrow['spread_pct'],
                    'bid': borrow.get('bid') or tx.get('bid'),
                    'ask': borrow.get('ask') or tx.get('ask'),
                    'avg_volume': tx.get('avg_volume') or borrow.get('avg_volume'),
                    'market_cap': borrow.get('market_cap') or tx.get('market_cap'),
                }
                results.append(row)
            except Exception as e:
                failed_symbols.append(f"{symbol} ({str(e)})")
                print(f"Warning: Could not fetch data for {symbol}: {str(e)}")
                continue
        
        # Report failed symbols if any
        if failed_symbols:
            print(f"Warning: Failed to fetch Treasury ETF data for {len(failed_symbols)} symbol(s): {', '.join(failed_symbols)}")
        
        if not results:
            raise ValueError(f"No Treasury ETF data could be fetched. Failed symbols: {', '.join(failed_symbols) if failed_symbols else 'all symbols'}. Please check your symbols and internet connection.")
        
        df = pd.DataFrame(results)
        if use_cache:
            self._save_to_cache(df, cache_key)
        # Also save to permanent CSV file
        self.save_data(df, "treasury_etf_data.csv", 
                      metadata={'symbols': symbols, 'num_symbols': len(df)})
        print(f"Fetched Treasury ETF data for {len(df)} symbols")
        return df