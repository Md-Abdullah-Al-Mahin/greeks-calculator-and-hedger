"""
Dashboard Application

Purpose: User interface to control data, view greeks, run optimizer, and download tickets.
- Initialize Session State: Sets up storage so data persists
- Render Sidebar: Controls for loading data, generating positions, refreshing
- Render Portfolio View: Shows greeks summary, breakdowns, and top risks
- Render Hedge Optimizer: Accepts targets, runs optimization, shows trades, CSV download
- Render Risk Analytics: Compares before/after hedge exposures
- Render Settings: Edits hedge universe, solver settings, data source mode
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import os
import sys
import json

# Add src directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Also add parent directory to support running from project root
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import modules - try multiple import strategies
try:
    # Strategy 1: Direct imports (when running from src/ directory)
    from data_loader import DataLoader
    from greeks_calculator import GreeksCalculator
    from portfolio_aggregator import PortfolioAggregator
    from hedge_optimizer import HedgeOptimizer
    from scenario_analyzer import ScenarioAnalyzer
except ImportError:
    try:
        # Strategy 2: Relative imports (when running as package)
        from .data_loader import DataLoader  # type: ignore
        from .greeks_calculator import GreeksCalculator  # type: ignore
        from .portfolio_aggregator import PortfolioAggregator  # type: ignore
        from .hedge_optimizer import HedgeOptimizer  # type: ignore
        from .scenario_analyzer import ScenarioAnalyzer  # type: ignore
    except ImportError:
        # Strategy 3: Try again with direct imports (parent dir now in path)
        from data_loader import DataLoader  # type: ignore
        from greeks_calculator import GreeksCalculator  # type: ignore
        from portfolio_aggregator import PortfolioAggregator  # type: ignore
        from hedge_optimizer import HedgeOptimizer  # type: ignore
        from scenario_analyzer import ScenarioAnalyzer  # type: ignore


# Default data directory
DEFAULT_DATA_DIR = "notebooks/data"


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    defaults = {
        'data_loaded': False,
        'positions_generated': False,
        'greeks_calculated': False,
        'optimization_complete': False,
        'data_dir': DEFAULT_DATA_DIR,
        'symbols': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'SPY', 'QQQ', 'DIA', 'IWM'],
        'num_positions': 50,
        'portfolio_exposures': None,
        'hedge_recommendations': None,
        'optimization_summary': None,
        'original_exposures': None,
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_data_dir() -> str:
    """Get the data directory from session state or settings."""
    return st.session_state.get('data_dir', DEFAULT_DATA_DIR)


def render_sidebar():
    """Render sidebar controls."""
    st.sidebar.title("📊 Controls")
    
    # Data directory setting
    st.sidebar.subheader("Configuration")
    data_dir = st.sidebar.text_input("Data Directory", value=get_data_dir(), key="sidebar_data_dir")
    st.session_state.data_dir = data_dir
    
    st.sidebar.subheader("Data Loading")
    
    # Display current symbols count
    num_symbols = len(st.session_state.symbols)
    st.sidebar.caption(f"Current: {num_symbols} symbol{'s' if num_symbols != 1 else ''}")
    
    # Use text_area for better multi-line support
    symbols_input = st.sidebar.text_area(
        "Symbols (comma or newline separated)", 
        value=",".join(st.session_state.symbols),
        key="sidebar_symbols",
        height=100,
        help="Enter stock symbols separated by commas or new lines. Example: AAPL, MSFT, GOOGL"
    )
    
    # Parse symbols (handle both comma and newline separators)
    if symbols_input:
        # Replace newlines with commas, then split by comma
        normalized_input = symbols_input.replace('\n', ',').replace(' ', ',')
        symbols_list = normalized_input.split(',')
        
        # Clean and filter symbols
        parsed_symbols = [s.strip().upper() for s in symbols_list if s.strip()]
        st.session_state.symbols = parsed_symbols
        
        # Show parsed count
        if len(parsed_symbols) != num_symbols:
            st.sidebar.caption(f"✓ Parsed {len(parsed_symbols)} symbol{'s' if len(parsed_symbols) != 1 else ''}")
    
    # Display current symbols in an expandable section
    if st.session_state.symbols:
        with st.sidebar.expander(f"📋 View Symbols ({len(st.session_state.symbols)})", expanded=False):
            st.write(", ".join(st.session_state.symbols))
    
    num_positions = st.sidebar.number_input(
        "Number of Positions", 
        min_value=1, 
        max_value=1000, 
        value=st.session_state.num_positions,
        step=1,
        key="sidebar_num_positions"
    )
    st.session_state.num_positions = num_positions
    
    use_cache = st.sidebar.checkbox("Use Cache", value=True, key="sidebar_use_cache")
    
    if st.sidebar.button("🔄 Load Real-Time Data", type="primary", key="sidebar_load_data"):
        with st.spinner("Loading market data..."):
            try:
                data_dir = get_data_dir()
                loader = DataLoader(data_dir=data_dir)
                
                # Load all data
                results = loader.load_all_data(
                    symbols=st.session_state.symbols,
                    num_positions=0,  # Don't generate positions yet
                    use_cache=use_cache,
                    generate_positions=False
                )
                
                st.session_state.data_loaded = True
                st.sidebar.success(f"✅ Loaded data for {len(results.get('market_data', []))} symbols")
            except Exception as e:
                st.sidebar.error(f"❌ Error loading data: {str(e)}")
                st.session_state.data_loaded = False
    
    if st.sidebar.button("📝 Generate Synthetic Positions", key="sidebar_generate_positions"):
        with st.spinner("Generating positions..."):
            try:
                data_dir = get_data_dir()
                loader = DataLoader(data_dir=data_dir)
                
                positions = loader.generate_synthetic_positions(
                    symbols=st.session_state.symbols,
                    num_positions=st.session_state.num_positions,
                    seed=42
                )
                loader.save_data(positions, "positions.csv")
                
                st.session_state.positions_generated = True
                st.sidebar.success(f"✅ Generated {len(positions)} positions")
            except Exception as e:
                st.sidebar.error(f"❌ Error generating positions: {str(e)}")
                st.session_state.positions_generated = False
    
    if st.sidebar.button("🧮 Calculate Greeks", key="sidebar_calculate_greeks"):
        with st.spinner("Calculating greeks..."):
            try:
                data_dir = get_data_dir()
                calculator = GreeksCalculator(data_dir=data_dir)
                positions_with_greeks = calculator.run_pipeline(validate=True)
                
                st.session_state.greeks_calculated = True
                st.sidebar.success(f"✅ Calculated greeks for {len(positions_with_greeks)} positions")
            except Exception as e:
                st.sidebar.error(f"❌ Error calculating greeks: {str(e)}")
                st.session_state.greeks_calculated = False
    
    # Status indicators
    st.sidebar.markdown("---")
    st.sidebar.subheader("Status")
    status_icon = "✅" if st.session_state.data_loaded else "❌"
    st.sidebar.write(f"{status_icon} Data Loaded")
    
    status_icon = "✅" if st.session_state.positions_generated else "❌"
    st.sidebar.write(f"{status_icon} Positions Generated")
    
    status_icon = "✅" if st.session_state.greeks_calculated else "❌"
    st.sidebar.write(f"{status_icon} Greeks Calculated")


def render_positions_view():
    """Render all positions in a table."""
    st.header("📋 All Positions")
    
    if not st.session_state.greeks_calculated:
        st.warning("⚠️ Please calculate greeks first using the sidebar controls.")
        return
    
    try:
        data_dir = get_data_dir()
        aggregator = PortfolioAggregator(data_dir=data_dir)
        
        # Load positions with greeks
        positions = aggregator.load_positions_with_greeks()
        
        if positions.empty:
            st.info("No positions found.")
            return
        
        # Display summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Positions", len(positions))
        with col2:
            equity_count = len(positions[positions['instrument_type'] == 'equity']) if 'instrument_type' in positions.columns else 0
            st.metric("Equities", equity_count)
        with col3:
            option_count = len(positions[positions['instrument_type'] == 'option']) if 'instrument_type' in positions.columns else 0
            st.metric("Options", option_count)
        with col4:
            unique_symbols = positions['symbol'].nunique() if 'symbol' in positions.columns else 0
            st.metric("Unique Symbols", unique_symbols)
        
        # Filter options
        st.subheader("Filters")
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        
        with filter_col1:
            all_symbols = ['All'] + sorted(positions['symbol'].unique().tolist()) if 'symbol' in positions.columns else ['All']
            selected_symbol = st.selectbox("Symbol", all_symbols, key="positions_filter_symbol")
        
        with filter_col2:
            all_types = ['All'] + sorted(positions['instrument_type'].unique().tolist()) if 'instrument_type' in positions.columns else ['All']
            selected_type = st.selectbox("Instrument Type", all_types, key="positions_filter_type")
        
        with filter_col3:
            search_term = st.text_input("Search Position ID", "", key="positions_search")
        
        # Apply filters
        filtered_positions = positions.copy()
        
        if selected_symbol != 'All':
            filtered_positions = filtered_positions[filtered_positions['symbol'] == selected_symbol]
        
        if selected_type != 'All':
            filtered_positions = filtered_positions[filtered_positions['instrument_type'] == selected_type]
        
        if search_term:
            if 'position_id' in filtered_positions.columns:
                filtered_positions = filtered_positions[
                    filtered_positions['position_id'].str.contains(search_term, case=False, na=False)
                ]
        
        # Select columns to display
        base_cols = ['position_id', 'symbol', 'instrument_type', 'quantity']
        option_cols = ['strike', 'expiry', 'option_type']
        market_cols = ['spot_price', 'dividend_yield']
        greek_cols = ['delta', 'gamma', 'vega', 'theta', 'rho']
        position_greek_cols = ['position_delta', 'position_gamma', 'position_vega', 'position_theta', 'position_rho']
        
        display_cols = base_cols.copy()
        
        # Add option-specific columns if any options exist
        if 'instrument_type' in filtered_positions.columns and (filtered_positions['instrument_type'] == 'option').any():
            display_cols.extend([col for col in option_cols if col in filtered_positions.columns])
        
        # Add market data columns
        display_cols.extend([col for col in market_cols if col in filtered_positions.columns])
        
        # Add greeks columns
        display_cols.extend([col for col in greek_cols if col in filtered_positions.columns])
        display_cols.extend([col for col in position_greek_cols if col in filtered_positions.columns])
        
        # Filter to only existing columns
        available_cols = [col for col in display_cols if col in filtered_positions.columns]
        
        # Display table
        st.subheader(f"Positions ({len(filtered_positions)} of {len(positions)})")
        
        if not filtered_positions.empty:
            # Format the dataframe for display
            display_df = filtered_positions[available_cols].copy()
            
            # Format numeric columns
            format_dict = {}
            for col in display_df.columns:
                if col in ['quantity', 'strike', 'spot_price']:
                    format_dict[col] = '{:,.2f}'
                elif col in ['dividend_yield']:
                    format_dict[col] = '{:.4f}'
                elif col in ['delta', 'gamma', 'vega', 'theta', 'rho']:
                    format_dict[col] = '{:,.4f}'
                elif col in position_greek_cols:
                    format_dict[col] = '{:,.2f}'
                elif col in ['position_id']:
                    format_dict[col] = '{}'
            
            st.dataframe(
                display_df.style.format(format_dict),
                width='stretch',
                height=600
            )
            
            # Download button
            csv = filtered_positions.to_csv(index=False)
            st.download_button(
                label="📥 Download Filtered Positions CSV",
                data=csv,
                file_name=f"positions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="download_positions"
            )
        else:
            st.info("No positions match the selected filters.")
            
    except FileNotFoundError as e:
        st.error(f"❌ Positions file not found: {str(e)}")
    except Exception as e:
        st.error(f"❌ Error loading positions: {str(e)}")
        st.exception(e)


def render_portfolio_view():
    """Render portfolio greeks summary and breakdowns."""
    st.header("📈 Portfolio Greeks")
    
    if not st.session_state.greeks_calculated:
        st.warning("⚠️ Please calculate greeks first using the sidebar controls.")
        return
    
    try:
        data_dir = get_data_dir()
        aggregator = PortfolioAggregator(data_dir=data_dir)
        
        # Portfolio Summary
        st.subheader("Portfolio Summary")
        portfolio_summary = aggregator.load_and_aggregate_portfolio_greeks()
        
        # Display key metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Delta", f"{portfolio_summary['total_delta']:,.2f}")
        with col2:
            st.metric("Total Gamma", f"{portfolio_summary['total_gamma']:,.4f}")
        with col3:
            st.metric("Total Vega", f"{portfolio_summary['total_vega']:,.2f}")
        with col4:
            st.metric("Total Theta", f"{portfolio_summary['total_theta']:,.2f}")
        with col5:
            st.metric("Total Rho", f"{portfolio_summary['total_rho']:,.2f}")
        
        col6, col7, col8 = st.columns(3)
        with col6:
            st.metric("Total Notional", f"${portfolio_summary['total_notional']:,.2f}")
        with col7:
            st.metric("Number of Positions", portfolio_summary['num_positions'])
        with col8:
            st.metric("Total Theta (Daily)", f"{portfolio_summary['total_theta']/365:,.2f}")
        
        # Symbol Breakdown
        st.subheader("Symbol Breakdown")
        symbol_breakdown = aggregator.load_and_aggregate_by_symbol()
        if not symbol_breakdown.empty:
            st.dataframe(
                symbol_breakdown.style.format({
                    'delta': '{:,.2f}',
                    'gamma': '{:,.4f}',
                    'vega': '{:,.2f}',
                    'theta': '{:,.2f}',
                    'rho': '{:,.2f}',
                    'notional': '${:,.2f}',
                }),
                width='stretch',
                height=400
            )
        else:
            st.info("No symbol breakdown available.")
        
        # Instrument Type Breakdown
        st.subheader("Instrument Type Breakdown")
        type_breakdown = aggregator.load_and_aggregate_by_instrument_type()
        if not type_breakdown.empty:
            st.dataframe(
                type_breakdown.style.format({
                    'delta': '{:,.2f}',
                    'gamma': '{:,.4f}',
                    'vega': '{:,.2f}',
                    'theta': '{:,.2f}',
                    'rho': '{:,.2f}',
                    'notional': '${:,.2f}',
                }),
                width='stretch'
            )
        else:
            st.info("No instrument type breakdown available.")
        
        # Top Risks
        st.subheader("Top 10 Risky Positions")
        top_risks = aggregator.load_and_identify_top_risks(top_n=10)
        if not top_risks.empty:
            # Select relevant columns for display
            display_cols = ['position_id', 'symbol', 'instrument_type', 'option_type', 'quantity', 
                          'position_delta', 'position_gamma', 'position_vega', 'position_theta', 'position_rho']
            available_cols = [col for col in display_cols if col in top_risks.columns]
            st.dataframe(
                top_risks[available_cols].style.format({
                    'quantity': '{:,.0f}',
                    'position_delta': '{:,.2f}',
                    'position_gamma': '{:,.4f}',
                    'position_vega': '{:,.2f}',
                    'position_theta': '{:,.2f}',
                    'position_rho': '{:,.2f}',
                }),
                width='stretch',
                height=400
            )
        else:
            st.info("No top risks available.")
            
    except Exception as e:
        st.error(f"❌ Error loading portfolio data: {str(e)}")
        st.exception(e)


def render_hedge_optimizer():
    """Render hedge optimizer interface."""
    st.header("🛡️ Hedge Optimizer")
    
    if not st.session_state.greeks_calculated:
        st.warning("⚠️ Please calculate greeks first using the sidebar controls.")
        return
    
    try:
        data_dir = get_data_dir()
        optimizer = HedgeOptimizer(data_dir=data_dir)
        
        # Load current portfolio exposures
        if st.session_state.portfolio_exposures is None:
            st.session_state.portfolio_exposures = optimizer.load_portfolio_exposures()
            st.session_state.original_exposures = st.session_state.portfolio_exposures.copy()
        
        # Display current exposures
        st.subheader("Current Portfolio Exposures")
        exposures = st.session_state.portfolio_exposures
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Delta", f"{exposures['total_delta']:,.2f}")
        with col2:
            st.metric("Current Rho", f"{exposures['total_rho']:,.2f}")
        with col3:
            st.metric("Number of Positions", exposures['num_positions'])
        
        # Target inputs
        st.subheader("Hedge Targets")
        col1, col2 = st.columns(2)
        
        with col1:
            delta_target = st.number_input("Delta Target", value=0.0, step=0.1, format="%.2f", key="hedge_delta_target")
            delta_tolerance = st.number_input("Delta Tolerance", value=0.01, step=0.01, format="%.2f", key="hedge_delta_tolerance")
        
        with col2:
            rho_target = st.number_input("Rho Target", value=0.0, step=1000.0, format="%.0f", key="hedge_rho_target")
            rho_tolerance = st.number_input("Rho Tolerance", value=10000.0, step=1000.0, format="%.0f", key="hedge_rho_tolerance")
        
        # Hedge universe configuration
        with st.expander("Hedge Universe Configuration"):
            include_etfs = st.checkbox("Include ETFs", value=True, key="hedge_include_etfs")
            etf_symbols_input = st.text_input("ETF Symbols (comma-separated)", value="SPY,QQQ,DIA,IWM", key="hedge_etf_symbols")
            include_ir = st.checkbox("Include Interest Rate Instruments (Treasury ETFs)", value=True, key="hedge_include_ir")
            treasury_symbols_input = st.text_input("Treasury Symbols (comma-separated)", value="TLT,IEF,SHY", key="hedge_treasury_symbols")
            
            default_transaction_cost = st.number_input("Default Transaction Cost (bps)", value=5.0, step=0.1, key="hedge_transaction_cost")
            default_max_quantity = st.number_input("Default Max Quantity", value=100000.0, step=1000.0, key="hedge_max_quantity")
        
        targets = {
            'delta_target': delta_target,
            'delta_tolerance': delta_tolerance,
            'rho_target': rho_target,
            'rho_tolerance': rho_tolerance
        }
        
        hedge_config = {
            'include_etfs': include_etfs,
            'etf_symbols': [s.strip().upper() for s in etf_symbols_input.split(",") if s.strip()] if etf_symbols_input else ['SPY', 'QQQ', 'DIA', 'IWM'],
            'include_ir_instruments': include_ir,
            'treasury_symbols': [s.strip().upper() for s in treasury_symbols_input.split(",") if s.strip()] if treasury_symbols_input else [],
            'default_transaction_cost_bps': default_transaction_cost,
            'default_max_quantity': default_max_quantity,
            'use_market_data': True
        }
        
        if st.button("🚀 Optimize Hedge", type="primary", key="hedge_optimize_button"):
            with st.spinner("Optimizing hedge portfolio..."):
                try:
                    # Get symbols from portfolio
                    symbols = optimizer.portfolio_aggregator.get_unique_symbols()
                    if not symbols:
                        symbols = st.session_state.symbols
                    
                    # Build hedge universe
                    hedge_universe = optimizer.build_hedge_universe(symbols, hedge_config)
                    
                    # Load market data
                    try:
                        market_data = optimizer.load_market_data()
                    except FileNotFoundError:
                        market_data = pd.DataFrame()
                    
                    # Optimize
                    hedge_recommendations, optimization_summary = optimizer.optimize_hedge_portfolio(
                        st.session_state.portfolio_exposures,
                        hedge_universe,
                        market_data,
                        targets
                    )
                    
                    st.session_state.hedge_recommendations = hedge_recommendations
                    st.session_state.optimization_summary = optimization_summary
                    st.session_state.optimization_complete = True
                    
                    # Save results
                    optimizer.save_hedge_tickets(hedge_recommendations)
                    optimizer.save_optimization_summary(optimization_summary)
                    
                    st.success("✅ Optimization complete!")
                    
                except Exception as e:
                    st.error(f"❌ Error during optimization: {str(e)}")
                    st.exception(e)
        
        # Display results if optimization is complete
        if st.session_state.optimization_complete and st.session_state.optimization_summary:
            st.subheader("Optimization Results")
            summary = st.session_state.optimization_summary
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                status_color = "🟢" if summary['solver_status'] == 'optimal' else "🟡"
                st.metric("Status", f"{status_color} {summary['solver_status']}")
            with col2:
                st.metric("Total Cost", f"${summary['total_hedge_cost']:,.2f}")
            with col3:
                st.metric("Hedge Effectiveness", f"{summary['hedge_effectiveness_pct']:.1f}%")
            with col4:
                st.metric("Number of Trades", summary['num_hedge_trades'])
            
            col5, col6 = st.columns(2)
            with col5:
                st.metric("Residual Delta", f"{summary['residual_delta']:,.2f}")
            with col6:
                st.metric("Residual Rho", f"{summary['residual_rho']:,.2f}")
            
            # Hedge Recommendations Table
            st.subheader("Hedge Recommendations")
            if st.session_state.hedge_recommendations is not None and not st.session_state.hedge_recommendations.empty:
                recommendations = st.session_state.hedge_recommendations
                st.dataframe(
                    recommendations.style.format({
                        'hedge_quantity': '{:,.0f}',
                        'estimated_cost': '${:,.2f}',
                        'delta_contribution': '{:,.2f}',
                        'rho_contribution': '{:,.2f}',
                    }),
                    width='stretch',
                    height=400
                )
                
                # CSV Download
                csv = recommendations.to_csv(index=False)
                st.download_button(
                    label="📥 Download Hedge Tickets CSV",
                    data=csv,
                    file_name=f"hedge_tickets_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key="download_hedge_tickets"
                )
            else:
                st.info("No hedge recommendations generated.")
                
    except Exception as e:
        st.error(f"❌ Error in hedge optimizer: {str(e)}")
        st.exception(e)


def render_risk_analytics():
    """Render risk analytics and scenario analysis."""
    st.header("📊 Risk Analytics")
    
    if not st.session_state.greeks_calculated:
        st.warning("⚠️ Please calculate greeks first to see analytics.")
        return
    
    try:
        # Before/After Comparison (only show if optimization was run)
        if st.session_state.optimization_complete:
            st.subheader("Before vs After Hedge")
        
        if st.session_state.original_exposures and st.session_state.optimization_summary:
            original = st.session_state.original_exposures
            summary = st.session_state.optimization_summary
            
            comparison_data = {
                'Metric': ['Delta', 'Rho', 'Number of Positions'],
                'Before': [
                    original['total_delta'],
                    original['total_rho'],
                    original['num_positions']
                ],
                'After': [
                    summary['residual_delta'],
                    summary['residual_rho'],
                    original['num_positions']  # Positions don't change
                ],
                'Change': [
                    summary['residual_delta'] - original['total_delta'],
                    summary['residual_rho'] - original['total_rho'],
                    0
                ],
                'Change %': [
                    ((summary['residual_delta'] - original['total_delta']) / original['total_delta'] * 100) if original['total_delta'] != 0 else 0,
                    ((summary['residual_rho'] - original['total_rho']) / original['total_rho'] * 100) if original['total_rho'] != 0 else 0,
                    0
                ]
            }
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(
                comparison_df.style.format({
                    'Before': '{:,.2f}',
                    'After': '{:,.2f}',
                    'Change': '{:,.2f}',
                    'Change %': '{:.2f}%'
                }),
                width='stretch'
            )
            
            # Visual comparison
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Delta Reduction", f"{abs(comparison_df.loc[0, 'Change']):,.2f}")
            with col2:
                st.metric("Rho Reduction", f"{abs(comparison_df.loc[1, 'Change']):,.2f}")
        
        # Hedge Effectiveness Details (only show if optimization was run)
        if st.session_state.optimization_complete:
            st.subheader("Hedge Effectiveness")
            if st.session_state.optimization_summary:
                summary = st.session_state.optimization_summary
                effectiveness = summary['hedge_effectiveness_pct']
                
                st.progress(effectiveness / 100)
                st.write(f"Hedge effectiveness: **{effectiveness:.1f}%**")
                
                if effectiveness >= 70:
                    st.success("✅ Hedge effectiveness meets target (≥70%)")
                else:
                    st.warning(f"⚠️ Hedge effectiveness below target (current: {effectiveness:.1f}%)")
        
        st.markdown("---")
        
        # Scenario Analysis
        st.subheader("Scenario Analysis")
        st.write("Analyze portfolio performance under different market scenarios using greeks approximation.")
        
        try:
            data_dir = get_data_dir()
            analyzer = ScenarioAnalyzer(data_dir=data_dir)
            
            # Scenario inputs
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Market Movement Scenarios")
                price_change_pct = st.slider(
                    "Price Change (%)",
                    min_value=-50.0,
                    max_value=50.0,
                    value=0.0,
                    step=0.5,
                    key="scenario_price_change"
                )
                
                vol_change_pct = st.slider(
                    "Volatility Change (%)",
                    min_value=-50.0,
                    max_value=50.0,
                    value=0.0,
                    step=1.0,
                    key="scenario_vol_change"
                )
            
            with col2:
                st.markdown("#### Interest Rate & Time Scenarios")
                rate_change_bps = st.slider(
                    "Interest Rate Change (basis points)",
                    min_value=-200,
                    max_value=200,
                    value=0,
                    step=10,
                    key="scenario_rate_change"
                )
                
                time_decay_days = st.slider(
                    "Time Decay (days)",
                    min_value=0,
                    max_value=30,
                    value=0,
                    step=1,
                    key="scenario_time_decay"
                )
            
            # Calculate scenario P&L using ScenarioAnalyzer
            scenario_results = analyzer.calculate_scenario_pnl(
                price_change_pct=price_change_pct,
                vol_change_pct=vol_change_pct,
                rate_change_bps=rate_change_bps,
                time_decay_days=time_decay_days
            )
            
            # Extract results
            total_pnl = scenario_results['total_pnl']
            delta_pnl = scenario_results['delta_pnl']
            gamma_pnl = scenario_results['gamma_pnl']
            vega_pnl = scenario_results['vega_pnl']
            theta_pnl = scenario_results['theta_pnl']
            rho_pnl = scenario_results['rho_pnl']
            portfolio_summary = scenario_results['portfolio_summary']
            breakdown_df = scenario_results['breakdown']
            
            # Display results
            st.markdown("---")
            st.markdown("#### Scenario Results")
            
            # Summary metrics
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            
            # Calculate P&L percentage
            pnl_pct = analyzer.calculate_pnl_percentage(total_pnl, portfolio_summary['total_notional'])
            pnl_delta = f"{pnl_pct:.2f}%" if pnl_pct is not None else "N/A"
            
            with col1:
                st.metric("Total P&L", f"${total_pnl:,.2f}", delta=pnl_delta)
            
            with col2:
                st.metric("Delta P&L", f"${delta_pnl:,.2f}")
            
            with col3:
                st.metric("Gamma P&L", f"${gamma_pnl:,.2f}")
            
            with col4:
                st.metric("Vega P&L", f"${vega_pnl:,.2f}")
            
            with col5:
                st.metric("Theta P&L", f"${theta_pnl:,.2f}")
            
            with col6:
                st.metric("Rho P&L", f"${rho_pnl:,.2f}")
            
            # Detailed breakdown
            st.dataframe(
                breakdown_df.style.format({
                    'P&L ($)': '${:,.2f}'
                }),
                width='stretch',
                height=300
            )
            
            # Example scenarios
            st.markdown("---")
            st.markdown("#### Example Scenarios")
            st.caption("Common scenario combinations to test:")
            
            example_col1, example_col2, example_col3 = st.columns(3)
            
            with example_col1:
                st.markdown("**Bull Market**")
                st.write("- Price: +10%")
                st.write("- Vol: -5%")
                st.write("- Time: 0 days")
            
            with example_col2:
                st.markdown("**Bear Market**")
                st.write("- Price: -10%")
                st.write("- Vol: +20%")
                st.write("- Time: 0 days")
            
            with example_col3:
                st.markdown("**Time Decay**")
                st.write("- Price: 0%")
                st.write("- Vol: 0%")
                st.write("- Time: 7-30 days")
            
            # Warning about limitations
            st.info("""
            **Note:** This analysis uses first-order (delta) and second-order (gamma) greeks approximation. 
            For large moves (>20%), higher-order effects may become significant. 
            Results assume linear relationships and may not capture all non-linear effects.
            """)
                
            except FileNotFoundError:
                st.error("❌ Positions with greeks not found. Please calculate greeks first.")
            except Exception as e:
                st.error(f"❌ Error in scenario analysis: {str(e)}")
                st.exception(e)
        
    except Exception as e:
        st.error(f"❌ Error in risk analytics: {str(e)}")
        st.exception(e)


def render_settings():
    """Render settings panel."""
    st.header("⚙️ Settings")
    
    st.subheader("Data Directory")
    data_dir = st.text_input("Data Directory Path", value=get_data_dir(), key="settings_data_dir")
    if st.button("Update Data Directory", key="settings_update_data_dir"):
        st.session_state.data_dir = data_dir
        st.success(f"✅ Data directory updated to: {data_dir}")
    
    st.subheader("Cache Settings")
    cache_expiry = st.number_input("Cache Expiry (hours)", min_value=0.1, max_value=24.0, value=1.0, step=0.1, key="settings_cache_expiry")
    st.info(f"Cache will expire after {cache_expiry} hours")
    
    st.subheader("Default Hedge Universe")
    st.write("Configure default hedge universe settings:")
    
    default_etf_symbols = st.text_input("Default ETF Symbols", value="SPY,QQQ,DIA,IWM", key="settings_etf_symbols")
    default_treasury_symbols = st.text_input("Default Treasury Symbols", value="TLT,IEF,SHY", key="settings_treasury_symbols")
    default_transaction_cost = st.number_input("Default Transaction Cost (bps)", value=5.0, step=0.1, key="settings_transaction_cost")
    default_max_quantity = st.number_input("Default Max Quantity", value=100000.0, step=1000.0, key="settings_max_quantity")
    
    st.subheader("Solver Settings")
    st.write("Optimization solver configuration:")
    max_iter = st.number_input("Max Iterations", min_value=100, max_value=10000, value=1000, step=100, key="settings_max_iter")
    ftol = st.number_input("Function Tolerance", min_value=1e-9, max_value=1e-3, value=1e-6, format="%.0e", key="settings_ftol")
    st.info(f"Solver will use max_iter={max_iter} and ftol={ftol}")
    
    st.subheader("Data Source Mode")
    data_mode = st.radio("Mode", ["Real", "Synthetic", "Mixed"], index=2, key="settings_data_mode")
    st.info(f"Current mode: **{data_mode}** - {'Real' if data_mode == 'Real' else 'Synthetic' if data_mode == 'Synthetic' else 'Mixed (real market data, synthetic positions)'}")


def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="Greeks Calculator & Hedge Optimizer",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📊 Greeks Aggregator & Hedge Optimizer")
    st.markdown("---")
    
    initialize_session_state()
    render_sidebar()
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 Positions", "📈 Portfolio", "🛡️ Hedge Optimizer", "📊 Risk Analytics", "⚙️ Settings"])
    
    with tab1:
        render_positions_view()
    
    with tab2:
        render_portfolio_view()
    
    with tab3:
        render_hedge_optimizer()
    
    with tab4:
        render_risk_analytics()
    
    with tab5:
        render_settings()


if __name__ == "__main__":
    main()
