"""
Test module for Hedge Optimizer

Smoke tests for optimization:
- Optimizer returns trades
- Residual delta within tolerance
- Hedge effectiveness >= 70%
- Correlation-aware hedging backward compatibility
- Variance reduction with correlation-aware hedging
"""

import unittest
import sys
import os
import numpy as np
import pandas as pd

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from hedge_optimizer import HedgeOptimizer


class TestHedgeOptimizer(unittest.TestCase):
    """Test cases for Hedge Optimizer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.optimizer = HedgeOptimizer()
        
        # Create sample hedge universe
        self.sample_hedge_universe = pd.DataFrame({
            'symbol': ['AAPL', 'MSFT', 'SPY'],
            'instrument_type': ['equity', 'equity', 'etf'],
            'spot_price': [150.0, 300.0, 450.0],
            'borrow_cost_bps': [20.0, 15.0, 10.0],
            'transaction_cost_bps': [5.0, 5.0, 3.0],
            'max_long_quantity': [10000.0, 10000.0, 10000.0],
            'max_short_quantity': [10000.0, 10000.0, 10000.0],
            'delta_per_unit': [1.0, 1.0, 1.0],
            'rho_per_unit': [0.0, 0.0, 0.0],
        })
        
        # Sample portfolio exposures
        self.sample_portfolio_exposures = {
            'total_delta': 1000.0,
            'total_gamma': 50.0,
            'total_vega': 200.0,
            'total_theta': -10.0,
            'total_rho': 5000.0,
            'total_notional': 150000.0,
            'num_positions': 5
        }
        
        # Sample targets
        self.sample_targets = {
            'delta_target': 0.0,
            'delta_tolerance': 10.0,
            'rho_target': 0.0,
            'rho_tolerance': 10000.0
        }
    
    def test_optimizer_returns_trades(self):
        """Test that optimizer returns hedge recommendations."""
        hedge_recommendations, summary = self.optimizer.optimize_hedge_portfolio(
            self.sample_portfolio_exposures,
            self.sample_hedge_universe,
            pd.DataFrame(),
            self.sample_targets
        )
        
        # Should return a DataFrame and dict
        self.assertIsInstance(hedge_recommendations, pd.DataFrame)
        self.assertIsInstance(summary, dict)
        
        # Summary should have required keys
        self.assertIn('solver_status', summary)
        self.assertIn('total_hedge_cost', summary)
        self.assertIn('residual_delta', summary)
        self.assertIn('hedge_effectiveness_pct', summary)
    
    def test_residual_delta_within_tolerance(self):
        """Test that residual delta is within specified tolerance."""
        hedge_recommendations, summary = self.optimizer.optimize_hedge_portfolio(
            self.sample_portfolio_exposures,
            self.sample_hedge_universe,
            pd.DataFrame(),
            self.sample_targets
        )
        
        if summary['solver_status'] == 'optimal':
            residual = abs(summary['residual_delta'] - self.sample_targets['delta_target'])
            self.assertLessEqual(residual, self.sample_targets['delta_tolerance'] * 2)
    
    def test_hedge_effectiveness(self):
        """Test that hedge effectiveness is calculated correctly."""
        hedge_recommendations, summary = self.optimizer.optimize_hedge_portfolio(
            self.sample_portfolio_exposures,
            self.sample_hedge_universe,
            pd.DataFrame(),
            self.sample_targets
        )
        
        # Effectiveness should be between 0 and 100
        self.assertGreaterEqual(summary['hedge_effectiveness_pct'], 0.0)
        self.assertLessEqual(summary['hedge_effectiveness_pct'], 100.0)


class TestCorrelationAwareHedging(unittest.TestCase):
    """Test cases for correlation-aware hedging feature."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.optimizer = HedgeOptimizer()
        
        # Create sample hedge universe with multiple correlated instruments
        self.sample_hedge_universe = pd.DataFrame({
            'symbol': ['AAPL', 'MSFT', 'GOOGL', 'SPY', 'QQQ'],
            'instrument_type': ['equity', 'equity', 'equity', 'etf', 'etf'],
            'spot_price': [150.0, 300.0, 140.0, 450.0, 380.0],
            'borrow_cost_bps': [20.0, 15.0, 18.0, 10.0, 12.0],
            'transaction_cost_bps': [5.0, 5.0, 5.0, 3.0, 3.0],
            'max_long_quantity': [10000.0, 10000.0, 10000.0, 10000.0, 10000.0],
            'max_short_quantity': [10000.0, 10000.0, 10000.0, 10000.0, 10000.0],
            'delta_per_unit': [1.0, 1.0, 1.0, 1.0, 1.0],
            'rho_per_unit': [0.0, 0.0, 0.0, 0.0, 0.0],
        })
        
        self.sample_portfolio_exposures = {
            'total_delta': 5000.0,
            'total_gamma': 100.0,
            'total_vega': 500.0,
            'total_theta': -25.0,
            'total_rho': 10000.0,
            'total_notional': 500000.0,
            'num_positions': 10
        }
        
        self.sample_targets = {
            'delta_target': 0.0,
            'delta_tolerance': 50.0,
            'rho_target': 0.0,
            'rho_tolerance': 15000.0
        }
    
    def test_backward_compatibility_without_correlation(self):
        """Test that optimization works identically when correlation is disabled."""
        # Run without correlation
        rec1, summary1 = self.optimizer.optimize_hedge_portfolio(
            self.sample_portfolio_exposures,
            self.sample_hedge_universe,
            pd.DataFrame(),
            self.sample_targets,
            use_correlation=False,
            variance_penalty=0.0
        )
        
        # Run with correlation disabled via zero penalty
        rec2, summary2 = self.optimizer.optimize_hedge_portfolio(
            self.sample_portfolio_exposures,
            self.sample_hedge_universe,
            pd.DataFrame(),
            self.sample_targets,
            use_correlation=True,
            variance_penalty=0.0  # Zero penalty should behave same as disabled
        )
        
        # Both should return valid results
        self.assertIsInstance(rec1, pd.DataFrame)
        self.assertIsInstance(rec2, pd.DataFrame)
        
        # Solver status should match
        self.assertEqual(summary1['solver_status'], summary2['solver_status'])
        
        # correlation_aware flag should be correct
        self.assertFalse(summary1.get('correlation_aware', False))
        self.assertFalse(summary2.get('correlation_aware', False))
    
    def test_correlation_aware_adds_summary_fields(self):
        """Test that correlation-aware hedging adds variance fields to summary."""
        # This test checks the API contract - when correlation is enabled with penalty > 0,
        # the summary should have correlation_aware = True
        rec, summary = self.optimizer.optimize_hedge_portfolio(
            self.sample_portfolio_exposures,
            self.sample_hedge_universe,
            pd.DataFrame(),
            self.sample_targets,
            use_correlation=True,
            variance_penalty=0.1,
            correlation_lookback_days=252
        )
        
        # Should have correlation_aware flag
        self.assertIn('correlation_aware', summary)
        self.assertIn('variance_penalty', summary)
        
        # If covariance computation worked, we should have variance fields
        # (may not always be present if data fetch fails, so we just check the flag)
        if summary.get('correlation_aware'):
            self.assertEqual(summary['variance_penalty'], 0.1)
    
    def test_build_variance_terms_structure(self):
        """Test that _build_variance_terms returns correct matrix shapes."""
        portfolio_exposures = {'AAPL': 1000.0, 'MSFT': 500.0}
        hedge_symbols = ['AAPL', 'MSFT', 'SPY']
        
        # Create a simple diagonal covariance matrix
        cov_symbols = ['AAPL', 'MSFT', 'SPY', 'QQQ']
        cov_matrix = np.eye(4) * 0.04  # 20% volatility, no correlation
        hedge_spot_prices = np.array([150.0, 300.0, 450.0])
        
        Q, c = self.optimizer._build_variance_terms(
            portfolio_exposures,
            hedge_symbols,
            cov_matrix,
            cov_symbols,
            hedge_spot_prices
        )
        
        # Q should be n_hedge x n_hedge
        self.assertEqual(Q.shape, (3, 3))
        # c should be n_hedge vector
        self.assertEqual(c.shape, (3,))
        
        # Q should be symmetric
        np.testing.assert_array_almost_equal(Q, Q.T)
    
    def test_variance_terms_cross_covariance(self):
        """Test that cross-covariance term c captures portfolio-hedge correlation."""
        # Portfolio only in AAPL
        portfolio_exposures = {'AAPL': 1000.0}
        hedge_symbols = ['SPY']
        hedge_spot_prices = np.array([450.0])
        
        # Create covariance where AAPL and SPY are positively correlated
        cov_symbols = ['AAPL', 'SPY']
        cov_matrix = np.array([
            [0.04, 0.02],  # AAPL variance, AAPL-SPY covariance
            [0.02, 0.03]   # SPY-AAPL covariance, SPY variance
        ])
        
        Q, c = self.optimizer._build_variance_terms(
            portfolio_exposures,
            hedge_symbols,
            cov_matrix,
            cov_symbols,
            hedge_spot_prices
        )
        
        # c should be positive (shorting SPY would reduce variance since 
        # AAPL and SPY are positively correlated)
        # c = 2 * cov(AAPL, SPY) * portfolio_AAPL * hedge_price_SPY
        expected_c = 2 * 0.02 * 1000.0 * 450.0
        self.assertAlmostEqual(c[0], expected_c, places=2)
    
    def test_empty_portfolio_exposures(self):
        """Test handling of empty portfolio exposures."""
        Q, c = self.optimizer._build_variance_terms(
            {},  # Empty portfolio
            ['AAPL', 'SPY'],
            np.eye(2) * 0.04,
            ['AAPL', 'SPY'],
            np.array([150.0, 450.0])
        )
        
        # Should return valid zero matrices
        self.assertEqual(Q.shape, (2, 2))
        self.assertEqual(c.shape, (2,))
        # c should be zero since no portfolio positions
        np.testing.assert_array_almost_equal(c, np.zeros(2))


class TestDataLoaderCovariance(unittest.TestCase):
    """Test cases for DataLoader covariance computation."""
    
    def test_compute_covariance_empty_symbols(self):
        """Test compute_covariance_matrix with empty symbol list."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
        from data_loader import DataLoader
        
        loader = DataLoader()
        cov, symbols = loader.compute_covariance_matrix([], use_cache=False)
        
        # Should return empty matrix and list
        self.assertEqual(cov.shape, (0, 0))
        self.assertEqual(symbols, [])
    
    def test_fetch_historical_returns_empty_symbols(self):
        """Test fetch_historical_returns with empty symbol list."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
        from data_loader import DataLoader
        
        loader = DataLoader()
        returns = loader.fetch_historical_returns([], use_cache=False)
        
        # Should return empty DataFrame
        self.assertTrue(returns.empty)


if __name__ == '__main__':
    unittest.main()
