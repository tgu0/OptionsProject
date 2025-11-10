import math
from statistics import NormalDist
from scipy.stats import norm
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d,PchipInterpolator


def _delta_to_strike_forward(S, T, r, sigma, delta_abs):
    """Map absolute forward-delta (0,1) and vol to strike (uses each point's own vol)."""
    F = S * math.exp(r * T)
    dc = delta_abs if delta_abs >= 0.5 else (1.0 - delta_abs)
    dc = min(max(dc, 1e-8), 1 - 1e-8)
    d1 = norm.ppf(dc)
    return F * math.exp(-sigma * math.sqrt(T) * d1 + 0.5 * (sigma**2) * T)
def black_scholes_price(S, K, T, r, sigma, option_type="C"):
    """
    Compute the Black-Scholes price of a European option.

    Parameters
    ----------
    S : float
        Current spot price of the underlying asset
    K : float
        Strike price
    T : float
        Time to maturity in years (e.g. 30 days = 30/365)
    r : float
        Annualized risk-free rate (continuously compounded)
    sigma : float
        Volatility of the underlying asset (annualized standard deviation)
    option_type : str, default "call"
        Either "call" or "put"

    Returns
    -------
    float
        Theoretical option price
    """
    if sigma > 10:
        raise ValueError("sigma must be in decimal form, 1.0 = 100%")
    if T <= 0 or sigma <= 0:
        # Option has expired or zero volatility
        return max(0.0, (S - K) if option_type == "call" else (K - S))
    if K <= 1:
        raise ValueError("K <= 1")
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if option_type.upper()[0] == "C":
        price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    elif option_type.upper()[0] == "P":
        price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return price


def strikes_at_delta(atm_iv: float,
                     spot_price: float,
                     days_to_expiry: float,
                     delta_abs: float,
                     annual_rate: float = 0.05):
    """
    Estimate call/put strikes for a target delta under Black-Scholes

    Parameters
    ----------
    atm_iv : float
        ATM implied volatility (annualized, e.g. 0.60 for 60%).
    spot_price : float
        Current spot price of the underlying.
    annual_rate : float
        Continuously compounded annual risk-free rate (e.g. 0.03).
    days_to_expiry : float
        Days to expiry.
    delta_abs : float
        Absolute delta target (e.g. 0.25 for 25Δ).

    Returns
    -------
    call_strike, put_strike : float, float
        Estimated strikes corresponding to +delta_abs call and -delta_abs put.
    """
    if not (0.0 < delta_abs < 0.5):
        raise ValueError("delta_abs must be in (0, 0.5), e.g. 0.25 for 25Δ")
    if atm_iv > 10:
        raise ValueError("atm iv should be in decimal form, 1.0 = 100%")
    if days_to_expiry < 1:
        raise ValueError("days to expiry should be in days not years")
    T = days_to_expiry / 365.0
    sigma = atm_iv
    F = spot_price * math.exp(annual_rate * T)  # forward price
    rootT = math.sqrt(T)
    sig2T = (sigma ** 2) * T

    # Forward delta convention (premium-unadjusted)
    z = NormalDist().inv_cdf(1.0 - delta_abs)  # e.g. Φ^{-1}(0.75) ≈ 0.674 for 25Δ

    # Solve d1 = (ln(F/K) + 0.5σ²T) / (σ√T)
    K_call = F * math.exp(+z * sigma * rootT + 0.5 * sig2T)
    K_put  = F * math.exp(-z * sigma * rootT + 0.5 * sig2T)

    return int(K_call//1000 * 1000), int(K_put//1000*1000)

def delta_to_strike(S, T, r, sigma, delta, option_type="call"):
    """
    Convert delta to strike using Black-Scholes (inverse of delta formula).
    """
    # ensure delta in (0,1)
    delta = np.clip(delta, 1e-6, 1 - 1e-6)
    d1 = norm.ppf(delta) if option_type == "call" else norm.ppf(1 - delta)
    K = S * np.exp(- (d1 * sigma * np.sqrt(T)) + (r + 0.5 * sigma**2) * T)
    return K

def interpolate_vol_from_strike(S, T, r, deltas_pct, vols, query_strikes,vol_floor=None,):
    """
    Interpolate IV(K) using PCHIP on total variance inside the smile,
    and your custom wing extrapolation rule outside:
      - Left wing (K < Kmin): if v0 < v1 -> flat at v0, else linear from (K0,v0)-(K1,v1)
      - Right wing (K > Kmax): if vN < vN-1 -> flat at vN, else linear from (K_{N-2},v_{N-1})
    Inputs:
      deltas_pct : e.g. [10,25,50,75,90] as absolute deltas in percent
      vols       : IVs corresponding to those deltas (same length)
      query_strikes : scalar or array of strikes to evaluate
    """
    if vol_floor is None:
        vol_floor = min(vols)-0.1
    d_abs = np.asarray(deltas_pct, dtype=float) / 100.0
    v     = np.asarray(vols, dtype=float)
    kq    = np.atleast_1d(np.asarray(query_strikes, dtype=float))

    # 1) Map (Δ, σ) -> strikes with each point's own σ
    strikes = np.array([_delta_to_strike_forward(S, T, r, sig, d) for d, sig in zip(d_abs, v)])

    # 2) Sort and de-duplicate by strike
    order = np.argsort(strikes)
    K = strikes[order]
    v = v[order]
    uniq, idx = np.unique(np.round(K, 10), return_index=True)
    K, v = K[idx], v[idx]
    n = len(K)
    if n == 0:
        raise ValueError("No valid smile points after deduplication.")
    if n == 1:
        out = np.full_like(kq, np.clip(v[0], vol_floor, None), dtype=float)
        return out if kq.ndim else float(out)

    # 3) Build interior interpolator on total variance (positivity)
    w = np.maximum(v, vol_floor)**2 * max(T, 1e-12)  # total variance
    pchip = PchipInterpolator(K, w, extrapolate=False)

    # Helper: linear extrapolation in IV space between (K_a, v_a) and (K_b, v_b)
    def _linear_iv(k, Ka, va, Kb, vb):
        slope = (vb - va) / (Kb - Ka)
        return va + slope * (k - Ka)

    # 4) Evaluate with custom wings
    out = np.empty_like(kq, dtype=float)
    Kmin, Kmax = K[0], K[-1]

    # Masks
    m_left  = (kq < Kmin)
    m_right = (kq > Kmax)
    m_mid   = ~(m_left | m_right)

    # Left wing rule
    if np.any(m_left):
        K0, v0 = K[0], v[0]
        K1, v1 = (K[1], v[1]) if n >= 2 else (K[0], v[0])
        if v0 < v1:
            # flat at boundary vol
            out[m_left] = v0
        else:
            out[m_left] = _linear_iv(kq[m_left], K0, v0, K1, v1)

    # Right wing rule
    if np.any(m_right):
        KNm1, vNm1 = K[-1], v[-1]
        KNm2, vNm2 = (K[-2], v[-2]) if n >= 2 else (K[-1], v[-1])
        if vNm1 < vNm2:
            out[m_right] = vNm1
        else:
            out[m_right] = _linear_iv(kq[m_right], KNm2, vNm2, KNm1, vNm1)

    # Interior via PCHIP(total variance) -> IV
    if np.any(m_mid):
        w_mid = pchip(kq[m_mid])
        # pchip returns NaN outside; inside it's fine
        sigma_mid = np.sqrt(np.maximum(w_mid, 0.0) / max(T, 1e-12))
        out[m_mid] = sigma_mid

    # Numerical floor to avoid tiny negatives from fp noise
    out = np.clip(out, vol_floor, None)
    return out if out.ndim > 1 else float(out)