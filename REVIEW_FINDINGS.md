# Code Review: Greeks and Cost Calculation Accuracy

## Executive Summary

This document reviews the accuracy of Greeks calculation methods and cost calculation methods in the greeks-calculator-and-hedger project. The review covers:

1. **Black-Scholes Greeks formulas with dividend adjustment (Merton model)**
2. **Cost calculation methods (transaction costs and borrow costs)**
3. **Bond rho calculation**

## 1. Greeks Calculation Review

### 1.1 Black-Scholes Greeks with Dividend Adjustment

**Status: ✅ CORRECT**

The implementation in `greeks_calculator.py` correctly implements the Merton model (dividend-adjusted Black-Scholes) formulas.

#### Verified Formulas:

**d1 and d2 calculation:**
```python
d1 = (np.log(spot / strike) + (rate - dividend_yield + 0.5 * volatility ** 2) * time_to_expiry) / (volatility * sqrt_T)
d2 = d1 - volatility * sqrt_T
```
✅ Correct: Uses `(r - q)` adjustment for dividend yield.

**Delta (Call):**
```python
delta = discount_dividend * N_d1  # where discount_dividend = e^(-q*T)
```
✅ Correct: Matches standard formula `Δ_call = e^(-q*T) * Φ(d1)`

**Delta (Put):**
```python
delta = discount_dividend * (N_d1 - 1.0)
```
✅ Correct: Matches standard formula `Δ_put = e^(-q*T) * (Φ(d1) - 1)`

**Gamma:**
```python
gamma = discount_dividend * n_d1 / (spot * volatility * sqrt_T)
```
✅ Correct: Matches standard formula `Γ = e^(-q*T) * φ(d1) / (S * σ * √T)`

**Vega:**
```python
vega = spot * discount_dividend * n_d1 * sqrt_T
```
✅ Correct: Matches standard formula `ν = S * e^(-q*T) * φ(d1) * √T`

**Theta (Call):**
```python
theta = (theta_vol_term 
        - rate * strike * discount_rate * N_d2
        + dividend_yield * spot * discount_dividend * N_d1)
```
Where `theta_vol_term = -spot * discount_dividend * n_d1 * volatility / (2 * sqrt_T)`

✅ Correct: Matches standard formula:
```
Θ_call = -e^(-q*T) * S * φ(d1) * σ / (2*√T) - r*K*e^(-r*T)*Φ(d2) + q*S*e^(-q*T)*Φ(d1)
```

**Theta (Put):**
```python
theta = (theta_vol_term 
        + rate * strike * discount_rate * N_neg_d2
        - dividend_yield * spot * discount_dividend * N_neg_d1)
```
✅ Correct: Matches standard formula:
```
Θ_put = -e^(-q*T) * S * φ(d1) * σ / (2*√T) + r*K*e^(-r*T)*Φ(-d2) - q*S*e^(-q*T)*Φ(-d1)
```

**Rho (Call):**
```python
rho = strike * time_to_expiry * discount_rate * N_d2
```
✅ Correct: Matches standard formula `ρ_call = K*T*e^(-r*T)*Φ(d2)`

**Rho (Put):**
```python
rho = -strike * time_to_expiry * discount_rate * N_neg_d2
```
✅ Correct: Matches standard formula `ρ_put = -K*T*e^(-r*T)*Φ(-d2)`

### 1.2 Bond Rho Calculation

**Status: ✅ CORRECT**

```python
modified_duration = duration_years / (1.0 + yield_to_maturity)
rho = -modified_duration * spot_price * 0.01
```

✅ Correct: Uses standard bond duration formula. Rho represents the change in bond price for a 1% (0.01) change in interest rates.

## 2. Cost Calculation Review

### 2.1 Transaction Cost Calculation

**Status: ✅ CORRECT (with minor clarification needed)**

Transaction costs are calculated correctly in the objective function:

```python
notional = np.abs(x) * spot_prices
transaction_cost = np.sum(transaction_costs_bps * notional / BPS_TO_DECIMAL)
```

✅ Correct: Transaction costs are one-time costs paid upfront, calculated as:
- Cost = (basis_points / 10000) * notional

The transaction cost estimation in `data_loader.py` considers:
- Bid-ask spread (one-way cost)
- Volume factor (market impact)
- Market cap factor
- Base commission

✅ This is appropriate for estimating one-time transaction costs.

### 2.2 Borrow Cost Calculation

**Status: ⚠️ POTENTIAL ISSUE - Needs Clarification**

**Issue Identified:**

Borrow costs are estimated as **annualized basis points** in `data_loader.py`:
```python
def _estimate_borrow_cost_from_liquidity(self, info: Dict, spot_price: float) -> float:
    """
    Estimate borrow cost (in basis points) from liquidity metrics.
    """
    # Returns annualized basis points (e.g., 20 bps = 0.2% per year)
```

However, in the hedge optimizer objective function, borrow costs are treated as **one-time costs**:
```python
borrow_cost = np.sum(borrow_costs_bps[short_mask] * notional[short_mask] / BPS_TO_DECIMAL)
```

**Problem:**
- Borrow costs are annualized rates (cost per year)
- But they're being applied as if they're one-time costs
- This underestimates the true cost for longer holding periods
- This overestimates the cost if we're only considering immediate hedge establishment

**Recommendation:**

The borrow cost calculation should account for the holding period. Options:

1. **Option A: Add holding period parameter**
   ```python
   # If borrow costs are annualized, scale by holding period
   holding_period_years = 1.0  # Default to 1 year, or make it configurable
   borrow_cost = np.sum(borrow_costs_bps[short_mask] * notional[short_mask] / BPS_TO_DECIMAL * holding_period_years)
   ```

2. **Option B: Document that borrow costs are treated as one-time**
   - If the optimizer is designed to minimize immediate hedge establishment costs only
   - Then borrow costs should be re-estimated as one-time costs (e.g., for a standard period like 1 year)
   - Or document that borrow costs represent the cost for a 1-year holding period

3. **Option C: Separate immediate vs. ongoing costs**
   - Transaction costs: immediate (one-time)
   - Borrow costs: ongoing (annualized, needs holding period)
   - Optimize for immediate costs, but report ongoing costs separately

**Current Behavior:**
The current implementation treats borrow costs as if they're one-time costs equal to the annualized rate, which is inconsistent. For a 1-year holding period, this happens to be correct, but it's not explicitly documented.

### 2.3 Cost Calculation in Hedge Recommendations

**Status: ⚠️ CONSISTENT WITH OBJECTIVE FUNCTION (but inherits the borrow cost issue)**

In `_create_hedge_recommendations`:
```python
cost_bps = float(row['transaction_cost_bps']) + (float(row['borrow_cost_bps']) if quantity < 0 else 0)
estimated_cost = cost_bps * notional / BPS_TO_DECIMAL
```

This is consistent with how costs are calculated in the objective function, so the optimization and reporting are aligned. However, it inherits the same borrow cost issue mentioned above.

## 3. Summary of Findings

### ✅ Correct Implementations:
1. **All Black-Scholes Greeks formulas** - Correctly implement Merton model with dividend adjustment
2. **Bond rho calculation** - Correctly uses modified duration
3. **Transaction cost calculation** - Correctly treats as one-time costs
4. **Formula consistency** - All formulas match standard financial literature

### ⚠️ Issues Requiring Attention:
1. **Borrow cost treatment** - Annualized borrow costs are being treated as one-time costs
   - **Impact**: May underestimate/overestimate true hedge costs depending on holding period
   - **Status**: ✅ **FIXED** - Added `holding_period_years` parameter (default 1.0) to properly scale annualized borrow costs

### 📝 Recommendations:
1. **Document borrow cost assumption**: Clearly state whether borrow costs are:
   - Treated as one-time costs (for immediate hedge establishment)
   - Or scaled by a holding period (for ongoing costs)
   
2. **Add holding period parameter**: If the optimizer should consider ongoing costs, add a `holding_period_years` parameter to scale borrow costs appropriately.

3. **Separate cost reporting**: Consider reporting:
   - Immediate costs (transaction costs)
   - Ongoing costs (borrow costs, annualized)
   - Total cost for a given holding period

## 4. Validation Checks

The code includes built-in validation checks:
- ✅ Gamma non-negativity
- ✅ ATM call delta ≈ 0.5
- ✅ Vega non-negativity
- ✅ Theta sign validation for long positions
- ✅ Equity delta = 1.0

These validation checks are appropriate and help ensure Greeks are calculated correctly.

## 5. Fixes Applied

### Borrow Cost Calculation Fix

**Issue**: Annualized borrow costs were being treated as one-time costs, which was inconsistent.

**Solution**: Added `holding_period_years` parameter (default: 1.0 year) to properly scale annualized borrow costs:

1. **Updated `_build_objective_function`**: Now scales borrow costs by holding period
   ```python
   borrow_cost = np.sum(borrow_costs_bps[short_mask] * notional[short_mask] / BPS_TO_DECIMAL * holding_period_years)
   ```

2. **Updated `_create_hedge_recommendations`**: Separates transaction costs and borrow costs, scales borrow costs appropriately
   - Added `transaction_cost` and `borrow_cost` columns to provide detailed cost breakdown
   - `estimated_cost` remains as total cost for backward compatibility

3. **Updated API methods**: Added `holding_period_years` parameter to:
   - `optimize_hedge_portfolio()`
   - `run_pipeline()`
   - `run_end_to_end()`

**Backward Compatibility**: Default value of 1.0 year maintains existing behavior (borrow costs treated as one-time costs for a 1-year holding period).

## 6. Conclusion

**Overall Assessment: ✅ All Greeks calculations are mathematically correct and match standard financial formulas. Cost calculations have been fixed to properly handle annualized borrow costs with a configurable holding period.**

The codebase demonstrates a solid understanding of Black-Scholes-Merton model implementation and correctly handles dividend adjustments. The borrow cost calculation issue has been resolved with the addition of a holding period parameter, making the cost calculations accurate and flexible.
