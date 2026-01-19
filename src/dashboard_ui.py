"""
Dashboard UI

Streamlit UI: sidebar, Positions, Portfolio, Hedge Optimizer, Risk Analytics, Settings.
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from typing import cast

try:
    from data_loader import DataLoader
    from greeks_calculator import GreeksCalculator
    from portfolio_aggregator import PortfolioAggregator
    from hedge_optimizer import HedgeOptimizer
    from scenario_analyzer import ScenarioAnalyzer
except ImportError:
    try:
        from .data_loader import DataLoader  # type: ignore
        from .greeks_calculator import GreeksCalculator  # type: ignore
        from .portfolio_aggregator import PortfolioAggregator  # type: ignore
        from .hedge_optimizer import HedgeOptimizer  # type: ignore
        from .scenario_analyzer import ScenarioAnalyzer  # type: ignore
    except ImportError:
        from data_loader import DataLoader  # type: ignore
        from greeks_calculator import GreeksCalculator  # type: ignore
        from portfolio_aggregator import PortfolioAggregator  # type: ignore
        from hedge_optimizer import HedgeOptimizer  # type: ignore
        from scenario_analyzer import ScenarioAnalyzer  # type: ignore


DEFAULT_DATA_DIR = "notebooks/data"


class DashboardUI:
    """Streamlit dashboard: sidebar, tabs (Positions, Portfolio, Hedge Optimizer, Risk Analytics, Settings)."""

    def __init__(self) -> None:
        pass

    def _get_data_dir(self) -> str:
        return st.session_state.get("data_dir", DEFAULT_DATA_DIR)

    def _initialize_session_state(self) -> None:
        defaults = {
            "data_loaded": False,
            "positions_generated": False,
            "greeks_calculated": False,
            "optimization_complete": False,
            "data_dir": DEFAULT_DATA_DIR,
            "symbols": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "SPY", "QQQ", "DIA", "IWM"],
            "num_positions": 50,
            "portfolio_exposures": None,
            "hedge_recommendations": None,
            "optimization_summary": None,
            "original_exposures": None,
        }
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    def _render_sidebar(self) -> None:
        st.sidebar.title("📊 Controls")
        st.sidebar.subheader("Configuration")
        data_dir = st.sidebar.text_input("Data Directory", value=self._get_data_dir(), key="sidebar_data_dir")
        st.session_state.data_dir = data_dir

        st.sidebar.subheader("Data Loading")
        num_symbols = len(st.session_state.symbols)
        st.sidebar.caption(f"Current: {num_symbols} symbol{'s' if num_symbols != 1 else ''}")

        symbols_input = st.sidebar.text_area(
            "Symbols (comma or newline separated)",
            value=",".join(st.session_state.symbols),
            key="sidebar_symbols",
            height=100,
            help="Enter stock symbols separated by commas or new lines. Example: AAPL, MSFT, GOOGL",
        )

        if symbols_input:
            normalized_input = symbols_input.replace("\n", ",").replace(" ", ",")
            symbols_list = normalized_input.split(",")
            parsed_symbols = [s.strip().upper() for s in symbols_list if s.strip()]
            st.session_state.symbols = parsed_symbols
            if len(parsed_symbols) != num_symbols:
                st.sidebar.caption(f"✓ Parsed {len(parsed_symbols)} symbol{'s' if len(parsed_symbols) != 1 else ''}")

        if st.session_state.symbols:
            with st.sidebar.expander(f"📋 View Symbols ({len(st.session_state.symbols)})", expanded=False):
                st.write(", ".join(st.session_state.symbols))

        num_positions = st.sidebar.number_input(
            "Number of Positions",
            min_value=1,
            max_value=1000,
            value=st.session_state.num_positions,
            step=1,
            key="sidebar_num_positions",
        )
        st.session_state.num_positions = num_positions

        use_cache = st.sidebar.checkbox("Use Cache", value=True, key="sidebar_use_cache")

        if st.sidebar.button("🔄 Load Real-Time Data", type="primary", key="sidebar_load_data"):
            with st.spinner("Loading market data..."):
                try:
                    loader = DataLoader(data_dir=self._get_data_dir())
                    results = loader.load_all_data(
                        symbols=st.session_state.symbols,
                        num_positions=0,
                        use_cache=use_cache,
                        generate_positions=False,
                    )
                    st.session_state.data_loaded = True
                    st.sidebar.success(f"✅ Loaded data for {len(results.get('market_data', []))} symbols")
                except Exception as e:
                    st.sidebar.error(f"❌ Error loading data: {str(e)}")
                    st.session_state.data_loaded = False

        if st.sidebar.button("📝 Generate Synthetic Positions", key="sidebar_generate_positions"):
            with st.spinner("Generating positions..."):
                try:
                    loader = DataLoader(data_dir=self._get_data_dir())
                    positions = loader.generate_synthetic_positions(
                        symbols=st.session_state.symbols,
                        num_positions=st.session_state.num_positions,
                        seed=42,
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
                    calculator = GreeksCalculator(data_dir=self._get_data_dir())
                    positions_with_greeks = calculator.run_pipeline(validate=True)
                    st.session_state.greeks_calculated = True
                    st.sidebar.success(f"✅ Calculated greeks for {len(positions_with_greeks)} positions")
                except Exception as e:
                    st.sidebar.error(f"❌ Error calculating greeks: {str(e)}")
                    st.session_state.greeks_calculated = False

        st.sidebar.markdown("---")
        st.sidebar.subheader("Status")
        st.sidebar.write(f"{'✅' if st.session_state.data_loaded else '❌'} Data Loaded")
        st.sidebar.write(f"{'✅' if st.session_state.positions_generated else '❌'} Positions Generated")
        st.sidebar.write(f"{'✅' if st.session_state.greeks_calculated else '❌'} Greeks Calculated")

    def _render_positions_view(self) -> None:
        st.header("📋 All Positions")
        if not st.session_state.greeks_calculated:
            st.warning("⚠️ Please calculate greeks first using the sidebar controls.")
            return
        try:
            aggregator = PortfolioAggregator(data_dir=self._get_data_dir())
            positions = aggregator.load_positions_with_greeks()
            if positions.empty:
                st.info("No positions found.")
                return

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Positions", len(positions))
            with col2:
                equity_count = len(positions[positions["instrument_type"] == "equity"]) if "instrument_type" in positions.columns else 0
                st.metric("Equities", equity_count)
            with col3:
                option_count = len(positions[positions["instrument_type"] == "option"]) if "instrument_type" in positions.columns else 0
                st.metric("Options", option_count)
            with col4:
                unique_symbols = cast(int, positions["symbol"].nunique() if "symbol" in positions.columns else 0)
                st.metric("Unique Symbols", unique_symbols)

            st.subheader("Filters")
            filter_col1, filter_col2, filter_col3 = st.columns(3)
            with filter_col1:
                all_symbols = ["All"] + sorted(positions["symbol"].unique().tolist()) if "symbol" in positions.columns else ["All"]
                selected_symbol = st.selectbox("Symbol", all_symbols, key="positions_filter_symbol")
            with filter_col2:
                all_types = ["All"] + sorted(positions["instrument_type"].unique().tolist()) if "instrument_type" in positions.columns else ["All"]
                selected_type = st.selectbox("Instrument Type", all_types, key="positions_filter_type")
            with filter_col3:
                search_term = st.text_input("Search Position ID", "", key="positions_search")

            filtered_positions: pd.DataFrame = positions.copy()
            if selected_symbol != "All":
                filtered_positions = cast(pd.DataFrame, filtered_positions[filtered_positions["symbol"] == selected_symbol])
            if selected_type != "All":
                filtered_positions = cast(pd.DataFrame, filtered_positions[filtered_positions["instrument_type"] == selected_type])
            if search_term and "position_id" in filtered_positions.columns:
                filtered_positions = cast(
                    pd.DataFrame,
                    filtered_positions[filtered_positions["position_id"].str.contains(search_term, case=False, na=False)],
                )

            base_cols = ["position_id", "symbol", "instrument_type", "quantity"]
            option_cols = ["strike", "expiry", "option_type"]
            market_cols = ["spot_price", "dividend_yield"]
            greek_cols = ["delta", "gamma", "vega", "theta", "rho"]
            position_greek_cols = ["position_delta", "position_gamma", "position_vega", "position_theta", "position_rho"]
            display_cols = base_cols.copy()
            if "instrument_type" in filtered_positions.columns and (filtered_positions["instrument_type"] == "option").any():
                display_cols.extend([c for c in option_cols if c in filtered_positions.columns])
            display_cols.extend([c for c in market_cols if c in filtered_positions.columns])
            display_cols.extend([c for c in greek_cols if c in filtered_positions.columns])
            display_cols.extend([c for c in position_greek_cols if c in filtered_positions.columns])
            available_cols = [c for c in display_cols if c in filtered_positions.columns]

            st.subheader(f"Positions ({len(filtered_positions)} of {len(positions)})")
            if not filtered_positions.empty:
                display_df = filtered_positions[available_cols].copy()
                format_dict = {}
                for col in display_df.columns:
                    if col in ["quantity", "strike", "spot_price"]:
                        format_dict[col] = "{:,.2f}"
                    elif col in ["dividend_yield"]:
                        format_dict[col] = "{:.4f}"
                    elif col in ["delta", "gamma", "vega", "theta", "rho"]:
                        format_dict[col] = "{:,.4f}"
                    elif col in position_greek_cols:
                        format_dict[col] = "{:,.2f}"
                    else:
                        format_dict[col] = "{}"
                st.dataframe(display_df.style.format(format_dict), width="stretch", height=600)
                st.download_button(
                    label="📥 Download Filtered Positions CSV",
                    data=filtered_positions.to_csv(index=False),
                    file_name=f"positions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key="download_positions",
                )
            else:
                st.info("No positions match the selected filters.")
        except FileNotFoundError as e:
            st.error(f"❌ Positions file not found: {str(e)}")
        except Exception as e:
            st.error(f"❌ Error loading positions: {str(e)}")
            st.exception(e)

    def _render_portfolio_view(self) -> None:
        st.header("📈 Portfolio Greeks")
        if not st.session_state.greeks_calculated:
            st.warning("⚠️ Please calculate greeks first using the sidebar controls.")
            return
        try:
            aggregator = PortfolioAggregator(data_dir=self._get_data_dir())
            st.subheader("Portfolio Summary")
            portfolio_summary = aggregator.load_and_aggregate_portfolio_greeks()
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
                st.metric("Number of Positions", portfolio_summary["num_positions"])
            with col8:
                st.metric("Total Theta (Daily)", f"{portfolio_summary['total_theta']/365:,.2f}")

            st.subheader("Symbol Breakdown")
            symbol_breakdown = aggregator.load_and_aggregate_by_symbol()
            if not symbol_breakdown.empty:
                st.dataframe(
                    symbol_breakdown.style.format(
                        {"delta": "{:,.2f}", "gamma": "{:,.4f}", "vega": "{:,.2f}", "theta": "{:,.2f}", "rho": "{:,.2f}", "notional": "${:,.2f}"}
                    ),
                    width="stretch",
                    height=400,
                )
            else:
                st.info("No symbol breakdown available.")

            st.subheader("Instrument Type Breakdown")
            type_breakdown = aggregator.load_and_aggregate_by_instrument_type()
            if not type_breakdown.empty:
                st.dataframe(
                    type_breakdown.style.format(
                        {"delta": "{:,.2f}", "gamma": "{:,.4f}", "vega": "{:,.2f}", "theta": "{:,.2f}", "rho": "{:,.2f}", "notional": "${:,.2f}"}
                    ),
                    width="stretch",
                )
            else:
                st.info("No instrument type breakdown available.")

            st.subheader("Top 10 Risky Positions")
            top_risks = aggregator.load_and_identify_top_risks(top_n=10)
            if not top_risks.empty:
                display_cols = ["position_id", "symbol", "instrument_type", "option_type", "quantity", "position_delta", "position_gamma", "position_vega", "position_theta", "position_rho"]
                available_cols = [c for c in display_cols if c in top_risks.columns]
                st.dataframe(
                    top_risks[available_cols].style.format(
                        {"quantity": "{:,.0f}", "position_delta": "{:,.2f}", "position_gamma": "{:,.4f}", "position_vega": "{:,.2f}", "position_theta": "{:,.2f}", "position_rho": "{:,.2f}"}
                    ),
                    width="stretch",
                    height=400,
                )
            else:
                st.info("No top risks available.")
        except Exception as e:
            st.error(f"❌ Error loading portfolio data: {str(e)}")
            st.exception(e)

    def _render_hedge_optimizer(self) -> None:
        st.header("🛡️ Hedge Optimizer")
        if not st.session_state.greeks_calculated:
            st.warning("⚠️ Please calculate greeks first using the sidebar controls.")
            return
        try:
            optimizer = HedgeOptimizer(data_dir=self._get_data_dir())
            if st.session_state.portfolio_exposures is None:
                st.session_state.portfolio_exposures = optimizer.load_portfolio_exposures()
                st.session_state.original_exposures = st.session_state.portfolio_exposures.copy()

            st.subheader("Current Portfolio Exposures")
            exposures = st.session_state.portfolio_exposures
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Delta", f"{exposures['total_delta']:,.2f}")
            with col2:
                st.metric("Current Rho", f"{exposures['total_rho']:,.2f}")
            with col3:
                st.metric("Number of Positions", exposures["num_positions"])

            st.subheader("Hedge Targets")
            col1, col2 = st.columns(2)
            with col1:
                delta_target = st.number_input("Delta Target", value=0.0, step=0.1, format="%.2f", key="hedge_delta_target")
                delta_tolerance = st.number_input("Delta Tolerance", value=0.01, step=0.01, format="%.2f", key="hedge_delta_tolerance")
            with col2:
                rho_target = st.number_input("Rho Target", value=0.0, step=1000.0, format="%.0f", key="hedge_rho_target")
                rho_tolerance = st.number_input("Rho Tolerance", value=10000.0, step=1000.0, format="%.0f", key="hedge_rho_tolerance")

            with st.expander("Hedge Universe Configuration"):
                include_etfs = st.checkbox("Include ETFs", value=True, key="hedge_include_etfs")
                etf_symbols_input = st.text_input("ETF Symbols (comma-separated)", value="SPY,QQQ,DIA,IWM", key="hedge_etf_symbols")
                include_ir = st.checkbox("Include Interest Rate Instruments (Treasury ETFs)", value=True, key="hedge_include_ir")
                treasury_symbols_input = st.text_input("Treasury Symbols (comma-separated)", value="TLT,IEF,SHY", key="hedge_treasury_symbols")
                default_transaction_cost = st.number_input("Default Transaction Cost (bps)", value=5.0, step=0.1, key="hedge_transaction_cost")
                default_max_quantity = st.number_input("Default Max Quantity", value=100000.0, step=1000.0, key="hedge_max_quantity")

            targets = {"delta_target": delta_target, "delta_tolerance": delta_tolerance, "rho_target": rho_target, "rho_tolerance": rho_tolerance}
            hedge_config = {
                "include_etfs": include_etfs,
                "etf_symbols": [s.strip().upper() for s in etf_symbols_input.split(",") if s.strip()] if etf_symbols_input else ["SPY", "QQQ", "DIA", "IWM"],
                "include_ir_instruments": include_ir,
                "treasury_symbols": [s.strip().upper() for s in treasury_symbols_input.split(",") if s.strip()] if treasury_symbols_input else [],
                "default_transaction_cost_bps": default_transaction_cost,
                "default_max_quantity": default_max_quantity,
                "use_market_data": True,
            }

            if st.button("🚀 Optimize Hedge", type="primary", key="hedge_optimize_button"):
                with st.spinner("Optimizing hedge portfolio..."):
                    try:
                        symbols = optimizer.portfolio_aggregator.get_unique_symbols() or st.session_state.symbols
                        hedge_universe = optimizer.build_hedge_universe(symbols, hedge_config)
                        try:
                            market_data = optimizer.load_market_data()
                        except FileNotFoundError:
                            market_data = pd.DataFrame()
                        hedge_recommendations, optimization_summary = optimizer.optimize_hedge_portfolio(
                            st.session_state.portfolio_exposures, hedge_universe, market_data, targets
                        )
                        st.session_state.hedge_recommendations = hedge_recommendations
                        st.session_state.optimization_summary = optimization_summary
                        st.session_state.optimization_complete = True
                        optimizer.save_hedge_tickets(hedge_recommendations)
                        optimizer.save_optimization_summary(optimization_summary)
                        st.success("✅ Optimization complete!")
                    except Exception as e:
                        st.error(f"❌ Error during optimization: {str(e)}")
                        st.exception(e)

            if st.session_state.optimization_complete and st.session_state.optimization_summary:
                summary = st.session_state.optimization_summary
                st.subheader("Optimization Results")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Status", f"{'🟢' if summary['solver_status'] == 'optimal' else '🟡'} {summary['solver_status']}")
                with col2:
                    st.metric("Total Cost", f"${summary['total_hedge_cost']:,.2f}")
                with col3:
                    st.metric("Hedge Effectiveness", f"{summary['hedge_effectiveness_pct']:.1f}%")
                with col4:
                    st.metric("Number of Trades", summary["num_hedge_trades"])
                col5, col6 = st.columns(2)
                with col5:
                    st.metric("Residual Delta", f"{summary['residual_delta']:,.2f}")
                with col6:
                    st.metric("Residual Rho", f"{summary['residual_rho']:,.2f}")
                st.subheader("Hedge Recommendations")
                if st.session_state.hedge_recommendations is not None and not st.session_state.hedge_recommendations.empty:
                    rec = st.session_state.hedge_recommendations
                    st.dataframe(
                        rec.style.format({"hedge_quantity": "{:,.0f}", "estimated_cost": "${:,.2f}", "delta_contribution": "{:,.2f}", "rho_contribution": "{:,.2f}"}),
                        width="stretch",
                        height=400,
                    )
                    st.download_button(
                        label="📥 Download Hedge Tickets CSV",
                        data=rec.to_csv(index=False),
                        file_name=f"hedge_tickets_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        key="download_hedge_tickets",
                    )
                else:
                    st.info("No hedge recommendations generated.")
        except Exception as e:
            st.error(f"❌ Error in hedge optimizer: {str(e)}")
            st.exception(e)

    def _render_risk_analytics(self) -> None:
        st.header("📊 Risk Analytics")
        if not st.session_state.greeks_calculated:
            st.warning("⚠️ Please calculate greeks first to see analytics.")
            return
        try:
            if st.session_state.optimization_complete:
                st.subheader("Before vs After Hedge")
            if st.session_state.original_exposures and st.session_state.optimization_summary:
                orig = st.session_state.original_exposures
                summary = st.session_state.optimization_summary
                comparison_data = {
                    "Metric": ["Delta", "Rho", "Number of Positions"],
                    "Before": [orig["total_delta"], orig["total_rho"], orig["num_positions"]],
                    "After": [summary["residual_delta"], summary["residual_rho"], orig["num_positions"]],
                    "Change": [summary["residual_delta"] - orig["total_delta"], summary["residual_rho"] - orig["total_rho"], 0],
                    "Change %": [
                        ((summary["residual_delta"] - orig["total_delta"]) / orig["total_delta"] * 100) if orig["total_delta"] != 0 else 0,
                        ((summary["residual_rho"] - orig["total_rho"]) / orig["total_rho"] * 100) if orig["total_rho"] != 0 else 0,
                        0,
                    ],
                }
                comp_df = pd.DataFrame(comparison_data)
                st.dataframe(comp_df.style.format({"Before": "{:,.2f}", "After": "{:,.2f}", "Change": "{:,.2f}", "Change %": "{:.2f}%"}), width="stretch")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Delta Reduction", f"{abs(comp_df.loc[0, 'Change']):,.2f}")
                with col2:
                    st.metric("Rho Reduction", f"{abs(comp_df.loc[1, 'Change']):,.2f}")

            if st.session_state.optimization_complete and st.session_state.optimization_summary:
                st.subheader("Hedge Effectiveness")
                eff = st.session_state.optimization_summary["hedge_effectiveness_pct"]
                st.progress(eff / 100)
                st.write(f"Hedge effectiveness: **{eff:.1f}%**")
                if eff >= 70:
                    st.success("✅ Hedge effectiveness meets target (≥70%)")
                else:
                    st.warning(f"⚠️ Hedge effectiveness below target (current: {eff:.1f}%)")

            st.markdown("---")
            st.subheader("Scenario Analysis")
            st.write("Analyze portfolio performance under different market scenarios using greeks approximation.")
            try:
                analyzer = ScenarioAnalyzer(data_dir=self._get_data_dir())
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### Market Movement Scenarios")
                    price_change_pct = st.slider("Price Change (%)", min_value=-50.0, max_value=50.0, value=0.0, step=0.5, key="scenario_price_change")
                    vol_change_pct = st.slider("Volatility Change (%)", min_value=-50.0, max_value=50.0, value=0.0, step=1.0, key="scenario_vol_change")
                with col2:
                    st.markdown("#### Interest Rate & Time Scenarios")
                    rate_change_bps = st.slider("Interest Rate Change (basis points)", min_value=-200, max_value=200, value=0, step=10, key="scenario_rate_change")
                    time_decay_days = st.slider("Time Decay (days)", min_value=0, max_value=30, value=0, step=1, key="scenario_time_decay")

                res = analyzer.calculate_scenario_pnl(
                    price_change_pct=price_change_pct,
                    vol_change_pct=vol_change_pct,
                    rate_change_bps=rate_change_bps,
                    time_decay_days=time_decay_days,
                )
                total_pnl = res["total_pnl"]
                pnl_pct = analyzer.calculate_pnl_percentage(total_pnl, res["portfolio_summary"]["total_notional"])
                pnl_delta = f"{pnl_pct:.2f}%" if pnl_pct is not None else "N/A"

                st.markdown("---")
                st.markdown("#### Scenario Results")
                c1, c2, c3, c4, c5, c6 = st.columns(6)
                with c1:
                    st.metric("Total P&L", f"${total_pnl:,.2f}", delta=pnl_delta)
                with c2:
                    st.metric("Delta P&L", f"${res['delta_pnl']:,.2f}")
                with c3:
                    st.metric("Gamma P&L", f"${res['gamma_pnl']:,.2f}")
                with c4:
                    st.metric("Vega P&L", f"${res['vega_pnl']:,.2f}")
                with c5:
                    st.metric("Theta P&L", f"${res['theta_pnl']:,.2f}")
                with c6:
                    st.metric("Rho P&L", f"${res['rho_pnl']:,.2f}")
                st.dataframe(res["breakdown"].style.format({"P&L ($)": "${:,.2f}"}), width="stretch", height=300)

                st.markdown("---")
                st.markdown("#### Example Scenarios")
                st.caption("Common scenario combinations to test:")
                e1, e2, e3 = st.columns(3)
                with e1:
                    st.markdown("**Bull Market**")
                    st.write("- Price: +10%")
                    st.write("- Vol: -5%")
                    st.write("- Time: 0 days")
                with e2:
                    st.markdown("**Bear Market**")
                    st.write("- Price: -10%")
                    st.write("- Vol: +20%")
                    st.write("- Time: 0 days")
                with e3:
                    st.markdown("**Time Decay**")
                    st.write("- Price: 0%")
                    st.write("- Vol: 0%")
                    st.write("- Time: 7-30 days")
                st.info(
                    "**Note:** This analysis uses first-order (delta) and second-order (gamma) greeks approximation. "
                    "For large moves (>20%), higher-order effects may become significant. "
                    "Results assume linear relationships and may not capture all non-linear effects."
                )
            except FileNotFoundError:
                st.error("❌ Positions with greeks not found. Please calculate greeks first.")
            except Exception as e:
                st.error(f"❌ Error in scenario analysis: {str(e)}")
                st.exception(e)
        except Exception as e:
            st.error(f"❌ Error in risk analytics: {str(e)}")
            st.exception(e)

    def _render_settings(self) -> None:
        st.header("⚙️ Settings")
        st.subheader("Data Directory")
        data_dir = st.text_input("Data Directory Path", value=self._get_data_dir(), key="settings_data_dir")
        if st.button("Update Data Directory", key="settings_update_data_dir"):
            st.session_state.data_dir = data_dir
            st.success(f"✅ Data directory updated to: {data_dir}")
        st.subheader("Cache Settings")
        cache_expiry = st.number_input("Cache Expiry (hours)", min_value=0.1, max_value=24.0, value=1.0, step=0.1, key="settings_cache_expiry")
        st.info(f"Cache will expire after {cache_expiry} hours")
        st.subheader("Default Hedge Universe")
        st.write("Configure default hedge universe settings:")
        st.text_input("Default ETF Symbols", value="SPY,QQQ,DIA,IWM", key="settings_etf_symbols")
        st.text_input("Default Treasury Symbols", value="TLT,IEF,SHY", key="settings_treasury_symbols")
        st.number_input("Default Transaction Cost (bps)", value=5.0, step=0.1, key="settings_transaction_cost")
        st.number_input("Default Max Quantity", value=100000.0, step=1000.0, key="settings_max_quantity")
        st.subheader("Solver Settings")
        st.write("Optimization solver configuration:")
        max_iter = st.number_input("Max Iterations", min_value=100, max_value=10000, value=1000, step=100, key="settings_max_iter")
        ftol = st.number_input("Function Tolerance", min_value=1e-9, max_value=1e-3, value=1e-6, format="%.0e", key="settings_ftol")
        st.info(f"Solver will use max_iter={max_iter} and ftol={ftol}")
        st.subheader("Data Source Mode")
        data_mode = st.radio("Mode", ["Real", "Synthetic", "Mixed"], index=2, key="settings_data_mode")
        st.info(f"Current mode: **{data_mode}** - {'Real' if data_mode == 'Real' else 'Synthetic' if data_mode == 'Synthetic' else 'Mixed (real market data, synthetic positions)'}")

    def run(self) -> None:
        st.set_page_config(page_title="Greeks Calculator & Hedge Optimizer", page_icon="📊", layout="wide", initial_sidebar_state="expanded")
        st.title("📊 Greeks Aggregator & Hedge Optimizer")
        st.markdown("---")
        self._initialize_session_state()
        self._render_sidebar()
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 Positions", "📈 Portfolio", "🛡️ Hedge Optimizer", "📊 Risk Analytics", "⚙️ Settings"])
        with tab1:
            self._render_positions_view()
        with tab2:
            self._render_portfolio_view()
        with tab3:
            self._render_hedge_optimizer()
        with tab4:
            self._render_risk_analytics()
        with tab5:
            self._render_settings()
