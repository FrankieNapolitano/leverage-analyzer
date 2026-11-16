"""
engine.py - the brain of the Leverage Analyzer.

Flow:  fetch_raw(ticker)  ->  analyze(raw)  ->  dict of results the dashboard shows.

1. fetch_raw     pulls financial statements, prices, the 10-year yield and an options chain from Yahoo Finance
2. leverage      debt load, interest coverage, free cash flow, capex vs cash flow, debt growth, Altman Z
3. market        beta, volatility, trend, and how the stock reacts to interest-rate moves
4. risk score    0-100 (higher = more fragile) -> tier: Fortress / Healthy / Stretched / Fragile
5. bias          leverage + trend + rates -> Bullish / Neutral / Bearish
6. strategy      bias + how expensive options are -> a specific trade with real strikes, cost, max loss, breakeven

Python does every calculation. The AI (ai_writer.py) only explains the numbers.
Educational tool, not financial advice.
"""

from datetime import datetime, date
import math

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------

def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _num(x):
    """Turn anything into a float or None (handles NaN, None, strings)."""
    try:
        v = float(x)
        return None if math.isnan(v) or math.isinf(v) else v
    except (TypeError, ValueError):
        return None


def _row(df, names):
    """Return the first matching row (as a Series, newest column first) from a Yahoo statement.
    Yahoo names the same line differently across companies, so we try several names."""
    if df is None or df.empty:
        return None
    for n in names:
        if n in df.index:
            s = pd.to_numeric(df.loc[n], errors="coerce")
            if isinstance(s, pd.DataFrame):  # duplicate index labels
                s = s.iloc[0]
            if s.notna().any():
                return s
    return None


def _latest(df, names):
    s = _row(df, names)
    if s is None:
        return None
    s = s.dropna()
    return _num(s.iloc[0]) if len(s) else None


def _ttm(q_df, a_df, names):
    """Trailing-twelve-month value: sum of last 4 quarters if available, else latest annual."""
    s = _row(q_df, names)
    if s is not None:
        s = s.dropna()
        if len(s) >= 4:
            return _num(s.iloc[:4].sum())
    return _latest(a_df, names)


def _interp(x, points):
    """Piecewise-linear map from a metric to a 0-100 risk score. points = [(x, score), ...] sorted by x."""
    if x is None:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return float(np.interp(x, xs, ys))


def _safe_div(a, b):
    if a is None or b is None or b == 0:
        return None
    return a / b


# Line-item names Yahoo uses (first match wins)
N = {
    "debt": ["Total Debt"],
    "cash": ["Cash Cash Equivalents And Short Term Investments", "Cash And Cash Equivalents",
             "Cash Financial"],
    "equity": ["Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest"],
    "assets": ["Total Assets"],
    "liab": ["Total Liabilities Net Minority Interest", "Total Liabilities"],
    "cur_assets": ["Current Assets"],
    "cur_liab": ["Current Liabilities"],
    "ret_earn": ["Retained Earnings"],
    "revenue": ["Total Revenue", "Operating Revenue"],
    "ebit": ["EBIT", "Operating Income"],
    "ebitda": ["EBITDA", "Normalized EBITDA"],
    "interest": ["Interest Expense", "Interest Expense Non Operating"],
    "net_income": ["Net Income", "Net Income Common Stockholders"],
    "ocf": ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"],
    "capex": ["Capital Expenditure"],
    "fcf": ["Free Cash Flow"],
    "dividends": ["Cash Dividends Paid", "Common Stock Dividend Paid"],
    "buybacks": ["Repurchase Of Capital Stock", "Common Stock Payments"],
}


# --------------------------------------------------------------------------------------
# 1. Fetch
# --------------------------------------------------------------------------------------

def fetch_raw(ticker, target_dte=45):
    """Download everything needed for one ticker. Returns a plain dict (easy to cache)."""
    import yfinance as yf

    ticker = ticker.strip().upper()
    t = yf.Ticker(ticker)

    px = t.history(period="2y", interval="1d", auto_adjust=True)
    if px is None or px.empty:
        raise ValueError(f"No price data for '{ticker}'. Check the ticker symbol.")
    close = px["Close"]
    close.index = pd.to_datetime(close.index).tz_localize(None)

    def _series(sym):
        h = yf.Ticker(sym).history(period="2y", interval="1d", auto_adjust=True)
        s = h["Close"]
        s.index = pd.to_datetime(s.index).tz_localize(None)
        return s

    try:
        info = t.info or {}
    except Exception:
        info = {}

    raw = {
        "ticker": ticker,
        "info": info,
        "bs_a": t.balance_sheet, "inc_a": t.income_stmt, "cf_a": t.cashflow,
        "bs_q": t.quarterly_balance_sheet, "inc_q": t.quarterly_income_stmt, "cf_q": t.quarterly_cashflow,
        "px": close,
        "spx": _series("^GSPC"),
        "tnx": _series("^TNX"),       # 10-year Treasury yield, in percent (e.g. 5.3)
        "expiries": [], "expiry": None, "calls": None, "puts": None,
        "fetched": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }

    # Options: pick the expiry closest to target_dte (between ~3 weeks and ~3 months if possible)
    try:
        exps = list(t.options)
    except Exception:
        exps = []
    raw["expiries"] = exps
    exp = choose_expiry(exps, target_dte)
    if exp:
        chain = t.option_chain(exp)
        raw.update(expiry=exp, calls=chain.calls, puts=chain.puts)
    return raw


def choose_expiry(expiries, target_dte=45, today=None):
    today = today or date.today()
    best, best_gap = None, None
    for e in expiries:
        dte = (datetime.strptime(e, "%Y-%m-%d").date() - today).days
        if dte < 14:
            continue
        penalty = 0 if 21 <= dte <= 75 else 1000
        gap = abs(dte - target_dte) + penalty
        if best_gap is None or gap < best_gap:
            best, best_gap = e, gap
    return best


# --------------------------------------------------------------------------------------
# 2. Leverage metrics
# --------------------------------------------------------------------------------------

def leverage_metrics(raw):
    bs_q, bs_a = raw["bs_q"], raw["bs_a"]
    inc_q, inc_a = raw["inc_q"], raw["inc_a"]
    cf_q, cf_a = raw["cf_q"], raw["cf_a"]
    info = raw.get("info", {}) or {}

    def bal(key):  # latest balance-sheet value (quarterly first)
        v = _latest(bs_q, N[key])
        return v if v is not None else _latest(bs_a, N[key])

    debt = bal("debt") or 0.0
    cash = bal("cash") or 0.0
    equity = bal("equity")
    assets = bal("assets")
    liab = bal("liab")
    cur_a, cur_l = bal("cur_assets"), bal("cur_liab")
    ret_earn = bal("ret_earn")

    revenue = _ttm(inc_q, inc_a, N["revenue"])
    ebit = _ttm(inc_q, inc_a, N["ebit"])
    ebitda = _ttm(inc_q, inc_a, N["ebitda"])
    interest = _ttm(inc_q, inc_a, N["interest"])
    interest = abs(interest) if interest is not None else None
    net_income = _ttm(inc_q, inc_a, N["net_income"])
    ocf = _ttm(cf_q, cf_a, N["ocf"])
    capex = _ttm(cf_q, cf_a, N["capex"])
    capex = abs(capex) if capex is not None else None
    fcf = _ttm(cf_q, cf_a, N["fcf"])
    if fcf is None and ocf is not None and capex is not None:
        fcf = ocf - capex
    dividends = _ttm(cf_q, cf_a, N["dividends"])
    buybacks = _ttm(cf_q, cf_a, N["buybacks"])

    net_debt = debt - cash
    mcap = _num(info.get("marketCap"))
    if mcap is None:
        shares = _num(info.get("sharesOutstanding"))
        mcap = shares * float(raw["px"].iloc[-1]) if shares else None

    # Debt growth over the last year (annual statements)
    debt_hist = _row(bs_a, N["debt"])
    debt_growth = None
    if debt_hist is not None:
        d = debt_hist.dropna()
        if len(d) >= 2 and d.iloc[1] > 0:
            debt_growth = d.iloc[0] / d.iloc[1] - 1

    # Altman Z-score (original public-company version)
    z = None
    if all(v is not None for v in [cur_a, cur_l, assets, ret_earn, ebit, mcap, liab, revenue]) and assets and liab:
        z = (1.2 * (cur_a - cur_l) / assets + 1.4 * ret_earn / assets + 3.3 * ebit / assets
             + 0.6 * mcap / liab + 1.0 * revenue / assets)

    m = {
        "total_debt": debt, "cash": cash, "net_debt": net_debt, "equity": equity,
        "total_assets": assets, "market_cap": mcap,
        "revenue_ttm": revenue, "ebit_ttm": ebit, "ebitda_ttm": ebitda, "interest_ttm": interest,
        "net_income_ttm": net_income, "ocf_ttm": ocf, "capex_ttm": capex, "fcf_ttm": fcf,
        "dividends_ttm": abs(dividends) if dividends else None,
        "buybacks_ttm": abs(buybacks) if buybacks else None,
        "debt_to_equity": _safe_div(debt, equity) if equity and equity > 0 else None,
        "negative_equity": equity is not None and equity <= 0,
        "debt_to_assets": _safe_div(debt, assets),
        "net_debt_to_ebitda": None,
        "interest_coverage": None,
        "current_ratio": _safe_div(cur_a, cur_l),
        "fcf_margin": _safe_div(fcf, revenue),
        "capex_to_ocf": None,
        "debt_growth_1y": debt_growth,
        "altman_z": z,
        "cash_runway_yrs": None,
    }

    if ebitda is not None:
        if net_debt <= 0:
            m["net_debt_to_ebitda"] = net_debt / ebitda if ebitda > 0 else 0.0  # net cash -> <= 0
        elif ebitda > 0:
            m["net_debt_to_ebitda"] = net_debt / ebitda
        else:
            m["net_debt_to_ebitda"] = float("inf")   # debt but no earnings
    if ebit is not None:
        if interest is None or interest < 1e6:
            m["interest_coverage"] = float("inf")     # basically no interest bill
        else:
            m["interest_coverage"] = ebit / interest
    if ocf is not None and capex is not None:
        m["capex_to_ocf"] = capex / ocf if ocf > 0 else float("inf")
    if fcf is not None and fcf < 0 and cash:
        m["cash_runway_yrs"] = cash / -fcf

    return m


def debt_fcf_history(raw):
    """Annual Total Debt and Free Cash Flow (in $B) for the history chart."""
    debt = _row(raw["bs_a"], N["debt"])
    fcf = _row(raw["cf_a"], N["fcf"])
    if fcf is None:
        ocf, capex = _row(raw["cf_a"], N["ocf"]), _row(raw["cf_a"], N["capex"])
        if ocf is not None and capex is not None:
            fcf = ocf + capex  # Yahoo capex is negative
    df = pd.DataFrame({"Total debt": debt, "Free cash flow": fcf}) / 1e9
    df = df.dropna(how="all")
    df.index = [pd.to_datetime(i).year for i in df.index]
    return df.sort_index()


# --------------------------------------------------------------------------------------
# 3. Market metrics
# --------------------------------------------------------------------------------------

def market_metrics(raw):
    px, spx, tnx = raw["px"], raw["spx"], raw["tnx"]
    out = {"price": float(px.iloc[-1])}

    # Volatility (annualized) from daily returns
    r = px.pct_change().dropna()
    out["vol_1y"] = float(r.iloc[-252:].std() * math.sqrt(252))
    out["vol_3m"] = float(r.iloc[-63:].std() * math.sqrt(252))

    # Trend
    sma50, sma200 = px.rolling(50).mean(), px.rolling(200).mean()
    out["sma50"], out["sma200"] = _num(sma50.iloc[-1]), _num(sma200.iloc[-1])
    out["ret_6m"] = float(px.iloc[-1] / px.iloc[-min(126, len(px) - 1) - 1] - 1)
    hi52 = px.iloc[-252:].max()
    out["off_52w_high"] = float(px.iloc[-1] / hi52 - 1)

    # Weekly data for regressions (less noise than daily)
    wk = pd.concat({"s": px, "m": spx, "y": tnx}, axis=1).dropna().resample("W-FRI").last().dropna()
    rs, rm = wk["s"].pct_change(), wk["m"].pct_change()
    dy = wk["y"].diff()                        # change in yield, percentage points
    reg = pd.concat({"rs": rs, "rm": rm, "dy": dy}, axis=1).dropna()

    out["beta"], out["rate_sens_25bp"], out["corr_mkt"], out["rate_tstat"] = None, None, None, None
    if len(reg) >= 30:
        out["beta"] = float(np.polyfit(reg["rm"], reg["rs"], 1)[0])
        out["corr_mkt"] = float(reg["rs"].corr(reg["rm"]))
        # Multiple regression: stock return = a + b_mkt*market return + b_rate*yield change
        # b_rate = how the stock moves when rates rise, AFTER removing the general market move.
        X = np.column_stack([np.ones(len(reg)), reg["rm"], reg["dy"]])
        y = reg["rs"].values
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ coef
        sigma2 = resid @ resid / (len(y) - X.shape[1])
        if reg["dy"].std() > 1e-6 and reg["rm"].std() > 1e-9:
            with np.errstate(all="ignore"):
                se = np.sqrt(np.abs(np.diag(sigma2 * np.linalg.pinv(X.T @ X))))
            out["rate_sens_25bp"] = float(coef[2] * 0.25)   # % move per +0.25pp (25 bps) in the 10-yr
            out["rate_tstat"] = float(coef[2] / se[2]) if np.isfinite(se[2]) and se[2] > 0 else 0.0

    out["tnx_now"] = float(tnx.iloc[-1])
    out["tnx_chg_3m"] = float(tnx.iloc[-1] - tnx.iloc[-min(63, len(tnx) - 1) - 1])
    return out


# --------------------------------------------------------------------------------------
# 4. Leverage risk score
# --------------------------------------------------------------------------------------

COMPONENTS = [
    # key, label, weight, description
    # Weights re-set after the Oct 2026 backtest (199 stocks, 2022-2026): debt growth, interest coverage,
    # FCF margin and capex/OCF tracked deeper drawdowns; net debt/EBITDA and Altman Z did not.
    ("net_debt_to_ebitda", "Net debt / EBITDA", 0.08, "Years of earnings needed to pay off debt"),
    ("interest_coverage", "Interest coverage", 0.22, "How many times profits cover the interest bill"),
    ("fcf_margin", "Free cash flow margin", 0.20, "Cash left after all spending, per $ of sales"),
    ("capex_to_ocf", "Capex / operating cash", 0.15, "Is building being paid for with own cash or borrowing?"),
    ("debt_growth_1y", "Debt growth (1y)", 0.22, "How fast borrowing is rising"),
    ("altman_z", "Altman Z-score", 0.05, "Classic bankruptcy-risk score"),
    ("rate_sens_25bp", "Rate sensitivity", 0.08, "Move per +25 bps in the 10-yr, beyond the market"),
]


def component_scores(lev, mkt, sector=""):
    s = {}
    nde = lev["net_debt_to_ebitda"]
    ic = lev["interest_coverage"]
    if sector == "Utilities":
        # Regulated utilities normally run ~5x net debt/EBITDA and ~2-3x coverage with stable cash flows.
        # The backtest found them making up ~45% of "Fragile" names, so grade them on a curve.
        if nde not in (None, float("inf")):
            nde = nde * 0.6
        if ic not in (None, float("inf")):
            ic = ic * 1.8
    s["net_debt_to_ebitda"] = 100.0 if nde == float("inf") else _interp(
        nde, [(0, 0), (1, 15), (2, 35), (3, 55), (4, 75), (5, 95), (6, 100)])
    s["interest_coverage"] = 0.0 if ic == float("inf") else _interp(
        ic, [(0, 100), (1, 100), (2, 70), (4, 40), (8, 15), (15, 0)])
    s["fcf_margin"] = _interp(lev["fcf_margin"], [(-0.25, 100), (-0.10, 80), (0, 50), (0.05, 20), (0.15, 0)])
    c2o = lev["capex_to_ocf"]
    s["capex_to_ocf"] = 100.0 if c2o == float("inf") else _interp(
        c2o, [(0.5, 0), (0.8, 25), (1.0, 50), (1.3, 75), (1.7, 100)])
    s["debt_growth_1y"] = _interp(lev["debt_growth_1y"], [(0, 0), (0.10, 20), (0.25, 50), (0.5, 80), (1.0, 100)])
    s["altman_z"] = _interp(lev["altman_z"], [(1.1, 100), (1.8, 60), (2.5, 25), (3.0, 0)])
    # Only count rate sensitivity if it's statistically real (t-stat <= -2); otherwise it's noise -> 0
    rs, t = mkt.get("rate_sens_25bp"), mkt.get("rate_tstat")
    if rs is None:
        s["rate_sens_25bp"] = None
    elif t is not None and t <= -2:
        s["rate_sens_25bp"] = _interp(rs, [(-0.03, 100), (-0.02, 85), (-0.01, 55), (-0.005, 30), (0, 0)])
    else:
        s["rate_sens_25bp"] = 0.0
    if lev.get("negative_equity") and s["net_debt_to_ebitda"] is not None:
        s["net_debt_to_ebitda"] = max(s["net_debt_to_ebitda"], 60.0)
    return s


def risk_score(scores):
    """Weighted average of available component scores (missing ones are skipped and weights rescaled)."""
    tot, w = 0.0, 0.0
    for key, _, weight, _ in COMPONENTS:
        v = scores.get(key)
        if v is not None:
            tot += v * weight
            w += weight
    if w == 0:
        return None, None, 0.0
    score = tot / w
    if score < 25:
        tier = "Fortress"
    elif score < 45:
        tier = "Healthy"
    elif score < 65:
        tier = "Stretched"
    else:
        tier = "Fragile"
    return score, tier, w


# --------------------------------------------------------------------------------------
# 5. Directional bias
# --------------------------------------------------------------------------------------

def directional_bias(tier, mkt):
    reasons = []
    trend = 0
    p = mkt["price"]
    if mkt["sma200"]:
        trend += 1 if p > mkt["sma200"] else -1
        reasons.append(f"price {'above' if p > mkt['sma200'] else 'below'} 200-day average")
    if mkt["sma50"] and mkt["sma200"]:
        trend += 1 if mkt["sma50"] > mkt["sma200"] else -1
        reasons.append(f"50-day {'above' if mkt['sma50'] > mkt['sma200'] else 'below'} 200-day")
    trend += 1 if mkt["ret_6m"] > 0 else -1
    reasons.append(f"6-month return {mkt['ret_6m']:+.1%}")

    # Backtest: trend predicted returns (strong uptrends beat strong downtrends by ~11 pts/yr); leverage tier
    # did not (Fragile names actually outperformed in 2024). So leverage only nudges the call now.
    lev_adj = {"Fortress": 0.25, "Healthy": 0.0, "Stretched": -0.25, "Fragile": -0.5}.get(tier, 0)
    reasons.append(f"{tier} balance sheet ({lev_adj:+.1f})")

    rate_adj = 0.0
    rs = mkt.get("rate_sens_25bp")
    if rs is not None and rs < -0.005 and (mkt.get("rate_tstat") or 0) <= -2 and mkt["tnx_chg_3m"] > 0.25:
        rate_adj = -0.5
        reasons.append("rates rising and this stock is rate-sensitive (-0.5)")

    score = 0.5 * trend + lev_adj + rate_adj
    if score >= 1:
        bias = "Bullish"
    elif score <= -1:
        bias = "Bearish"
    else:
        bias = "Neutral"
    conviction = "High" if abs(score) >= 2 else "Medium" if abs(score) >= 1 else "Low"
    return {"bias": bias, "conviction": conviction, "score": score, "trend": trend, "reasons": reasons}


# --------------------------------------------------------------------------------------
# 6. Options: pricing helpers + strategy picker
# --------------------------------------------------------------------------------------

def bs_delta(S, K, T, sigma, r, kind):
    if not sigma or sigma <= 0 or T <= 0:
        return None
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return _norm_cdf(d1) if kind == "call" else _norm_cdf(d1) - 1


def prob_above(S, K, T, sigma, r=0.0):
    """Risk-neutral probability the stock finishes above K (lognormal)."""
    if K <= 0:
        return 1.0
    d2 = (math.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return _norm_cdf(d2)


def _prep_chain(df):
    if df is None or df.empty:
        return None
    d = df.copy()
    for c in ["strike", "bid", "ask", "lastPrice", "impliedVolatility", "openInterest", "volume"]:
        if c not in d:
            d[c] = np.nan
        d[c] = pd.to_numeric(d[c], errors="coerce")
    has_quote = (d["bid"] > 0) & (d["ask"] > 0)
    d["mid"] = np.where(has_quote, (d["bid"] + d["ask"]) / 2, d["lastPrice"])
    d["spread_pct"] = np.where(has_quote, (d["ask"] - d["bid"]) / d["mid"], np.nan)
    d = d[d["mid"] > 0].sort_values("strike").reset_index(drop=True)
    return d


def options_snapshot(raw, mkt):
    calls, puts = _prep_chain(raw.get("calls")), _prep_chain(raw.get("puts"))
    if calls is None or puts is None or raw.get("expiry") is None:
        return None
    S = mkt["price"]
    T = max((datetime.strptime(raw["expiry"], "%Y-%m-%d").date() - date.today()).days, 1) / 365
    # ATM implied vol: average of call and put IV at the strike closest to spot
    ivs = []
    for d in (calls, puts):
        near = d.iloc[(d["strike"] - S).abs().argsort()[:2]]
        iv = near["impliedVolatility"].replace(0, np.nan).mean()
        if not np.isnan(iv):
            ivs.append(iv)
    atm_iv = float(np.mean(ivs)) if ivs else mkt["vol_3m"]
    rv = mkt["vol_3m"] if mkt["vol_3m"] > 0 else mkt["vol_1y"]
    ratio = atm_iv / rv if rv else None
    regime = "cheap" if ratio is not None and ratio < 1.0 else "rich" if ratio is not None and ratio > 1.3 else "fair"
    r = mkt["tnx_now"] / 100
    for d, kind in ((calls, "call"), (puts, "put")):
        d["delta"] = [bs_delta(S, k, T, iv if iv and iv > 0.01 else atm_iv, r, kind)
                      for k, iv in zip(d["strike"], d["impliedVolatility"])]
    return {"calls": calls, "puts": puts, "expiry": raw["expiry"], "T": T, "dte": round(T * 365),
            "atm_iv": atm_iv, "rv": rv, "iv_rv": ratio, "regime": regime,
            "exp_move": S * atm_iv * math.sqrt(T), "r": r}


def _nearest(d, target, col="strike"):
    return d.iloc[(d[col] - target).abs().argsort().iloc[0]]


def _by_delta(d, target):
    dd = d.dropna(subset=["delta"])
    return dd.iloc[(dd["delta"] - target).abs().argsort().iloc[0]]


def _wing(d, strike, direction, width):
    """Pick a strike roughly `width` away in `direction` (+1 up, -1 down), strictly beyond `strike`."""
    cand = d[d["strike"] > strike] if direction > 0 else d[d["strike"] < strike]
    if cand.empty:
        return None
    return _nearest(cand, strike + direction * width)


def _leg(action, kind, row):
    return {"action": action, "kind": kind, "strike": float(row["strike"]), "price": float(row["mid"]),
            "iv": _num(row["impliedVolatility"]), "delta": _num(row.get("delta")),
            "oi": _num(row.get("openInterest")), "spread_pct": _num(row.get("spread_pct"))}


def payoff(legs, grid, stock_shares=0, stock_cost=0.0):
    """P/L at expiry per 1 contract set (x100 shares) across a price grid."""
    pl = np.zeros_like(grid, dtype=float)
    for L in legs:
        sign = 1 if L["action"] == "BUY" else -1
        intrinsic = np.maximum(grid - L["strike"], 0) if L["kind"] == "call" else np.maximum(L["strike"] - grid, 0)
        pl += sign * (intrinsic - L["price"]) * 100
    if stock_shares:
        pl += stock_shares * (grid - stock_cost)
    return pl


def _finish(name, legs, S, snap, why, stock_shares=0):
    grid = np.linspace(S * 0.6, S * 1.4, 801)
    pl = payoff(legs, grid, stock_shares, S)
    # + credit, - debit (options only; for a hedge, the 100 shares you already own aren't counted as a cost)
    net = sum((-1 if L["action"] == "BUY" else 1) * L["price"] for L in legs) * 100
    max_profit, max_loss = float(pl.max()), float(pl.min())
    # Unbounded if the payoff is still sloping at the grid edges
    unbounded_up = (pl[-1] - pl[-2]) > 1e-6 and pl[-1] >= max_profit - 1e-6
    # breakevens: sign changes
    be = []
    for i in range(1, len(grid)):
        if (pl[i - 1] < 0 <= pl[i]) or (pl[i - 1] >= 0 > pl[i]):
            x0, x1, y0, y1 = grid[i - 1], grid[i], pl[i - 1], pl[i]
            be.append(float(x0 - y0 * (x1 - x0) / (y1 - y0)))
    # probability of profit (risk-neutral, lognormal, ATM IV)
    sig, T, r = snap["atm_iv"], snap["T"], snap["r"]
    pop = None
    if be:
        mid_profit = pl[len(grid) // 2] > 0
        pts = [0.0] + be + [float("inf")]
        pop = 0.0
        for a, b in zip(pts[:-1], pts[1:]):
            x = (a + b) / 2 if b != float("inf") else a * 1.1 + 1
            if np.interp(x, grid, pl) > 0:
                pa = 1.0 if a == 0 else prob_above(S, a, T, sig, r)
                pb = 0.0 if b == float("inf") else prob_above(S, b, T, sig, r)
                pop += pa - pb
    warnings = []
    for L in legs:
        if L["spread_pct"] is not None and L["spread_pct"] > 0.25:
            warnings.append(f"Wide bid/ask on the {L['strike']:g} {L['kind']} ({L['spread_pct']:.0%}) - use a limit order at mid.")
        if L["oi"] is not None and L["oi"] < 100:
            warnings.append(f"Low open interest on the {L['strike']:g} {L['kind']} ({L['oi']:.0f}) - may be hard to exit.")
    return {"name": name, "legs": legs, "net": net, "max_profit": max_profit, "max_loss": max_loss,
            "unbounded_up": bool(unbounded_up), "breakevens": be, "pop": pop, "why": why,
            "grid": grid, "pl": pl, "warnings": warnings, "stock_shares": stock_shares}


def pick_strategy(bias, tier, snap, S):
    """Map (bias x option price regime) to a defined-risk trade with real strikes from the chain."""
    if snap is None:
        return None
    calls, puts, em, regime = snap["calls"], snap["puts"], snap["exp_move"], snap["regime"]
    width = max(S * 0.05, em * 0.5)
    conv_note = {"cheap": "Options are cheap relative to how much the stock actually moves, so buying them is favored.",
                 "fair": "Options are fairly priced, so a debit spread keeps cost down while defining risk.",
                 "rich": "Options are expensive relative to actual moves, so selling premium (collecting it) is favored."}[regime]

    if bias == "Bullish":
        if regime == "rich":
            short = _by_delta(puts, -0.30)
            long = _wing(puts, short["strike"], -1, width)
            if long is None:
                return None
            legs = [_leg("SELL", "put", short), _leg("BUY", "put", long)]
            return _finish("Bull put credit spread", legs, S, snap,
                           f"Bullish view + {conv_note} You collect cash up front and keep it if the stock stays above "
                           f"{short['strike']:g} at expiry.")
        long = _nearest(calls, S)
        short = _wing(calls, long["strike"], +1, em)
        if short is None:
            return None
        legs = [_leg("BUY", "call", long), _leg("SELL", "call", short)]
        return _finish("Bull call debit spread", legs, S, snap,
                       f"Bullish view. {conv_note} Profit if the stock rises toward {short['strike']:g} "
                       f"(about one expected move) by expiry; the most you can lose is what you pay.")

    if bias == "Bearish":
        if regime == "rich":
            short = _by_delta(calls, 0.30)
            long = _wing(calls, short["strike"], +1, width)
            if long is None:
                return None
            legs = [_leg("SELL", "call", short), _leg("BUY", "call", long)]
            return _finish("Bear call credit spread", legs, S, snap,
                           f"Bearish view + {conv_note} You collect cash up front and keep it if the stock stays below "
                           f"{short['strike']:g}. Defined-risk alternative to shorting the stock.")
        long = _nearest(puts, S)
        short = _wing(puts, long["strike"], -1, em)
        if short is None:
            return None
        legs = [_leg("BUY", "put", long), _leg("SELL", "put", short)]
        return _finish("Bear put debit spread", legs, S, snap,
                       f"Bearish view. {conv_note} Profit if the stock falls toward {short['strike']:g}. "
                       f"Unlike shorting the stock, your loss is capped at what you pay.")

    # Neutral
    if regime == "rich":
        sp, sc = _by_delta(puts, -0.20), _by_delta(calls, 0.20)
        lp, lc = _wing(puts, sp["strike"], -1, width), _wing(calls, sc["strike"], +1, width)
        if lp is None or lc is None:
            return None
        legs = [_leg("BUY", "put", lp), _leg("SELL", "put", sp), _leg("SELL", "call", sc), _leg("BUY", "call", lc)]
        return _finish("Iron condor", legs, S, snap,
                       f"No strong direction + {conv_note} You profit if the stock stays between "
                       f"{sp['strike']:g} and {sc['strike']:g} through expiry.")
    if tier in ("Stretched", "Fragile"):
        p = _nearest(puts, S * 0.95)
        legs = [_leg("BUY", "put", p)]
        return _finish("Protective put (hedge for shares you own)", legs, S, snap,
                       "No clear trade edge, but the balance sheet is weak. If you already own shares, a ~5% "
                       "out-of-the-money put acts as insurance against a sharp drop. If you don't own it, stay out.",
                       stock_shares=100)
    return {"name": "No trade - wait for a better setup", "legs": [], "why":
            "Signals are mixed and options aren't expensive enough to sell. If you like the company long-term, "
            "buying shares slowly is fine; otherwise wait for the trend or option prices to give an edge.",
            "net": 0, "max_profit": None, "max_loss": None, "breakevens": [], "pop": None,
            "grid": None, "pl": None, "warnings": [], "unbounded_up": False, "stock_shares": 0}


def stock_view(tier, bias):
    """Plain stock-position guidance (no options)."""
    table = {
        ("Fortress", "Bullish"): "Long - core holding candidate",
        ("Healthy", "Bullish"): "Long",
        ("Stretched", "Bullish"): "Long, smaller size (balance sheet adds risk)",
        ("Fragile", "Bullish"): "Trade only, small size - momentum is fighting a weak balance sheet",
        ("Fortress", "Neutral"): "Hold / accumulate slowly",
        ("Healthy", "Neutral"): "Hold",
        ("Stretched", "Neutral"): "Avoid new buys; hedge if you own it",
        ("Fragile", "Neutral"): "Avoid; hedge if you own it",
        ("Fortress", "Bearish"): "Avoid for now - strong company, weak trend (often a buy-the-dip watchlist name)",
        ("Healthy", "Bearish"): "Avoid / reduce",
        ("Stretched", "Bearish"): "Avoid; bearish only with small defined-risk put spreads",
        ("Fragile", "Bearish"): "Avoid; bearish trades small and defined-risk only (leverage alone hasn't predicted underperformance)",
    }
    return table.get((tier, bias), "Neutral")


# Fit on the backtest (out-of-sample test 2024-25: corr 0.63, error 8.3 pts vs 10.8 naive):
#   typical worst drawdown over the next 12 months = -12.5% - 0.458 x (annualized 1-year volatility)
DD_INTERCEPT, DD_VOL, DD_BAND = -0.1247, -0.4578, 0.14


def expected_drawdown(mkt):
    vol = mkt.get("vol_1y")
    if vol is None:
        return None
    mid = max(DD_INTERCEPT + DD_VOL * vol, -0.95)
    return {"mid": mid, "bad": max(mid - DD_BAND, -0.98), "mild": min(mid + DD_BAND, -0.02)}


def stock_position_cap(account, risk_pct, exp_dd):
    """Largest stock position where a typical drawdown costs no more than risk_pct of the account."""
    if not exp_dd or exp_dd["mid"] >= 0:
        return None
    return account * risk_pct / abs(exp_dd["mid"])


def size_position(strategy, account, risk_pct):
    """How many contracts so the max loss is <= risk_pct of the account."""
    if not strategy or not strategy.get("legs") or strategy.get("max_loss") is None:
        return None
    if strategy.get("stock_shares"):
        # hedge: size = cost of the put per 100 shares owned
        return None
    loss = abs(strategy["max_loss"])
    if loss <= 0:
        return None
    return int((account * risk_pct) // loss)


# --------------------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------------------

def analyze(raw):
    info = raw.get("info", {}) or {}
    lev = leverage_metrics(raw)
    mkt = market_metrics(raw)
    sector = info.get("sector", "") or ""
    scores = component_scores(lev, mkt, sector)
    score, tier, coverage = risk_score(scores)
    bias = directional_bias(tier, mkt)
    snap = options_snapshot(raw, mkt)
    strat = pick_strategy(bias["bias"], tier, snap, mkt["price"]) if snap else None

    notes = []
    if sector == "Utilities":
        notes.append("Utility: debt ratios are graded on a curve because regulated utilities normally carry more debt.")
    if sector == "Financial Services":
        notes.append("Banks, insurers and lenders use debt as their raw material, so these leverage ratios "
                     "overstate their risk. Treat the score with caution.")
    if coverage < 0.6:
        notes.append("Several metrics were missing from Yahoo's data, so the score rests on fewer inputs.")
    if lev["negative_equity"]:
        notes.append("Negative shareholders' equity (often from big buybacks). Debt/equity isn't meaningful here.")

    return {
        "ticker": raw["ticker"], "name": info.get("longName") or info.get("shortName") or raw["ticker"],
        "sector": sector, "industry": info.get("industry", ""), "fetched": raw.get("fetched"),
        "lev": lev, "mkt": mkt, "scores": scores, "risk_score": score, "tier": tier,
        "score_coverage": coverage, "bias": bias, "options": snap, "strategy": strat,
        "stock_view": stock_view(tier, bias["bias"]) if tier else None,
        "exp_drawdown": expected_drawdown(mkt),
        "history": debt_fcf_history(raw), "px": raw["px"], "tnx": raw["tnx"], "notes": notes,
    }


def scan(tickers):
    """Quick leverage scan for a watchlist (no options) - used by the AI-chain page."""
    rows = []
    for tk in tickers:
        try:
            raw = fetch_raw(tk)
            a = analyze(raw)
            L, M = a["lev"], a["mkt"]
            rows.append({"Ticker": tk, "Name": a["name"], "Risk score": a["risk_score"], "Tier": a["tier"],
                         "Bias": a["bias"]["bias"], "Net debt/EBITDA": L["net_debt_to_ebitda"],
                         "Interest cover": L["interest_coverage"], "FCF margin": L["fcf_margin"],
                         "Capex/OCF": L["capex_to_ocf"], "Debt growth 1y": L["debt_growth_1y"],
                         "Rate sens /25bp": M["rate_sens_25bp"], "6m return": M["ret_6m"],
                         "Stock view": a["stock_view"]})
        except Exception as e:
            rows.append({"Ticker": tk, "Name": f"Error: {e}"})
    return pd.DataFrame(rows)
