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

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import DataLoader
from greeks_calculator import GreeksCalculator
from portfolio_aggregator import PortfolioAggregator
from hedge_optimizer import HedgeOptimizer


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'positions_generated' not in st.session_state:
        st.session_state.positions_generated = False
    if 'greeks_calculated' not in st.session_state:
        st.session_state.greeks_calculated = False


def render_sidebar():
    """Render sidebar controls."""
    st.sidebar.title("Controls")
    
    st.sidebar.subheader("Data Loading")
    symbols_input = st.sidebar.text_input("Symbols (comma-separated)", value="AAPL,MSFT,GOOGL,SPY")
    
    if st.sidebar.button("Load Real-Time Data"):
        # TODO: Implement data loading
        st.session_state.data_loaded = True
        st.sidebar.success("Data loaded!")
    
    if st.sidebar.button("Generate Synthetic Positions"):
        # TODO: Implement position generation
        st.session_state.positions_generated = True
        st.sidebar.success("Positions generated!")
    
    if st.sidebar.button("Calculate Greeks"):
        # TODO: Implement greeks calculation
        st.session_state.greeks_calculated = True
        st.sidebar.success("Greeks calculated!")


def render_portfolio_view():
    """Render portfolio greeks summary and breakdowns."""
    st.header("Portfolio Greeks")
    
    # TODO: Load and display portfolio summary
    # TODO: Display symbol breakdown
    # TODO: Display top risks
    
    st.info("Portfolio view will display here once data is loaded and greeks are calculated.")


def render_hedge_optimizer():
    """Render hedge optimizer interface."""
    st.header("Hedge Optimizer")
    
    col1, col2 = st.columns(2)
    
    with col1:
        delta_target = st.number_input("Delta Target", value=0.0, step=0.1)
        delta_tolerance = st.number_input("Delta Tolerance", value=0.01, step=0.01)
    
    with col2:
        rho_target = st.number_input("Rho Target", value=0.0, step=1000.0)
        rho_tolerance = st.number_input("Rho Tolerance", value=10000.0, step=1000.0)
    
    if st.button("Optimize Hedge"):
        # TODO: Implement optimization
        st.success("Optimization complete!")
    
    # TODO: Display hedge recommendations
    # TODO: Add CSV download button


def render_risk_analytics():
    """Render risk analytics and scenario analysis."""
    st.header("Risk Analytics")
    
    # TODO: Display before/after hedge comparison
    # TODO: Add scenario analysis
    
    st.info("Risk analytics will display here once hedge optimization is complete.")


def render_settings():
    """Render settings panel."""
    st.header("Settings")
    
    st.subheader("Hedge Universe")
    # TODO: Add hedge universe configuration
    
    st.subheader("Solver Settings")
    # TODO: Add solver configuration
    
    st.subheader("Data Source Mode")
    data_mode = st.radio("Mode", ["Real", "Synthetic", "Mixed"], index=2)
    st.info(f"Current mode: {data_mode}")


def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="Greeks Calculator & Hedge Optimizer",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("Greeks Aggregator & Hedge Optimizer")
    
    initialize_session_state()
    render_sidebar()
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Portfolio", "Hedge Optimizer", "Risk Analytics", "Settings"])
    
    with tab1:
        render_portfolio_view()
    
    with tab2:
        render_hedge_optimizer()
    
    with tab3:
        render_risk_analytics()
    
    with tab4:
        render_settings()


if __name__ == "__main__":
    main()
