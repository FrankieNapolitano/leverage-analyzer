"""
montecarlo.py - four ways to simulate the next 12 months of the S&P 500, and an honest test of
which one actually predicted drawdowns better.

Models (all pure numpy so the app can run them live):
  normal      - the classroom version: every day is an independent draw from one bell curve.
  bootstrap   - "stationary block bootstrap": replay real chunks of history (avg 63 trading days),
                which keeps fat tails and the way bad days cluster together.
  regime      - two-state hidden Markov model (calm vs stress). The market flips between states at
                historically estimated rates; returns in each state are resampled from real days.
                Starts from the state the market is most likely in TODAY.
  analog      - only replays what happened after the historical months that looked most like
                today (rates, curve, credit, inflation, valuation, momentum...).

Output: probability of a 10% / 20% / 30% peak-to-trough drawdown within the horizon.

Validation (walk-forward, no peeking): at every month-end from 1965 to 2025, each model is fitted
ONLY on data available up to that month, makes its forecast, and is scored against what actually
happened over the next 12 months. Scores are Brier scores (lower = better) compared with simply
guessing the historical base rate.
"""

import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
RAW = os.path.join(DATA, "raw")
THRESHOLDS = (0.10, 0.20, 0.30)
H = 252


# ----------------------------------------------------------------------------- data

def daily_returns():
    d = pd.read_csv(os.path.join(RAW, "gspc_daily.csv"), parse_dates=["date"]).dropna(subset=["close"])
    d = d.set_index("date")["close"].sort_index()
    return np.log(d).diff().dropna()


# ----------------------------------------------------------------------------- path tools

def max_drawdown_paths(logret_paths):
    """Peak-to-trough drawdown within each simulated path (starting level counts as a peak)."""
    cum = np.cumsum(logret_paths, axis=1)
    cum = np.concatenate([np.zeros((cum.shape[0], 1)), cum], axis=1)
    peak = np.maximum.accumulate(cum, axis=1)
    return 1 - np.exp((cum - peak).min(axis=1))


def probs(dd):
    return {f"p_dd_{int(t*100)}": float((dd >= t).mean()) for t in THRESHOLDS}


def stationary_bootstrap(r, n_paths, horizon, mean_block, rng):
    """Politis-Romano stationary bootstrap: random start points, geometric block lengths."""
    n = len(r)
    p = 1.0 / mean_block
    idx = np.empty((n_paths, horizon), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, n_paths)
    jump = rng.random((n_paths, horizon)) < p
    starts = rng.integers(0, n, (n_paths, horizon))
    for t in range(1, horizon):
        nxt = (idx[:, t - 1] + 1) % n
        idx[:, t] = np.where(jump[:, t], starts[:, t], nxt)
    return r[idx]


# ----------------------------------------------------------------------------- models

def sim_normal(r, n_paths=4000, horizon=H, rng=None):
    rng = rng or np.random.default_rng(0)
    return rng.normal(r.mean(), r.std(), (n_paths, horizon))


def sim_bootstrap(r, n_paths=4000, horizon=H, mean_block=63, rng=None):
    rng = rng or np.random.default_rng(0)
    return stationary_bootstrap(np.asarray(r), n_paths, horizon, mean_block, rng)


def fit_hmm(r, n_iter=60, tol=1e-6):
    """2-state Gaussian HMM via Baum-Welch (scaled). Returns params + smoothed state probs."""
    x = np.asarray(r, dtype=float)
    T = len(x)
    mu = np.array([x.mean() + 0.0002, x.mean() - 0.0005])
    sd = np.array([x.std() * 0.7, x.std() * 1.8])
    A = np.array([[0.99, 0.01], [0.03, 0.97]])
    pi = np.array([0.8, 0.2])
    prev = -np.inf
    for _ in range(n_iter):
        B = np.exp(-0.5 * ((x[:, None] - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi)) + 1e-300
        alpha = np.empty((T, 2)); c = np.empty(T)
        alpha[0] = pi * B[0]; c[0] = alpha[0].sum(); alpha[0] /= c[0]
        for t in range(1, T):
            alpha[t] = (alpha[t - 1] @ A) * B[t]
            c[t] = alpha[t].sum(); alpha[t] /= c[t]
        beta = np.empty((T, 2)); beta[-1] = 1
        for t in range(T - 2, -1, -1):
            beta[t] = (A @ (B[t + 1] * beta[t + 1])) / c[t + 1]
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True)
        xi = (alpha[:-1, :, None] * A[None] * (B[1:] * beta[1:])[:, None, :]) / c[1:, None, None]
        A = xi.sum(axis=0); A /= A.sum(axis=1, keepdims=True)
        pi = gamma[0]
        w = gamma / gamma.sum(axis=0)
        mu = (w * x[:, None]).sum(axis=0)
        sd = np.sqrt((w * (x[:, None] - mu) ** 2).sum(axis=0))
        ll = np.log(c).sum()
        if ll - prev < tol:
            break
        prev = ll
    order = np.argsort(sd)  # state 0 = calm (low vol), state 1 = stress
    return {"mu": mu[order], "sd": sd[order], "A": A[np.ix_(order, order)], "filtered_last": alpha[-1][order],
            "smoothed": gamma[:, order], "x": x}


def filter_last(r, params):
    """Probability of each state on the last day, given fixed params (forward filter only)."""
    x = np.asarray(r, dtype=float)
    mu, sd, A = params["mu"], params["sd"], params["A"]
    a = np.array([0.5, 0.5])
    for t in range(len(x)):
        b = np.exp(-0.5 * ((x[t] - mu) / sd) ** 2) / sd
        a = (a @ A) * b
        a /= a.sum()
    return a


def sim_regime(r, params, start_probs, n_paths=4000, horizon=H, rng=None):
    """Simulate state paths from the transition matrix, resampling real returns from each state."""
    rng = rng or np.random.default_rng(0)
    # resample from the returns the HMM was fitted on (all dated before the forecast date)
    x = np.asarray(params["x"]) if "x" in params else np.asarray(r)
    sm = params["smoothed"]
    pools = [x[sm[:, s] > 0.5] for s in (0, 1)]
    if min(len(p) for p in pools) < 50:
        pools = [x[np.argsort(sm[:, 1])[: len(x) // 2]], x[np.argsort(sm[:, 1])[len(x) // 2:]]]
    A = params["A"]
    state = (rng.random(n_paths) < start_probs[1]).astype(int)
    out = np.empty((n_paths, horizon))
    u = rng.random((n_paths, horizon))
    for t in range(horizon):
        for s in (0, 1):
            m = state == s
            k = m.sum()
            if k:
                out[m, t] = pools[s][rng.integers(0, len(pools[s]), k)]
        stay = A[state, state]
        state = np.where(u[:, t] < stay, state, 1 - state)
    return out


def analog_windows(w, feats, asof, k=25, horizon_months=12, min_gap_months=3, start="1950-01-01"):
    """The k historical month-ends most similar to `asof` (standardized with data up to asof only),
    whose following 12 months were already fully known at asof."""
    hist = w.loc[:asof]
    mu, sd = hist[feats].mean(), hist[feats].std()
    Z = (hist[feats] - mu) / sd
    zt = Z.loc[asof]
    cand = Z.loc[pd.Timestamp(start): asof - pd.DateOffset(months=horizon_months)]
    diff = cand.sub(zt, axis=1)
    valid = diff.notna()
    dist = np.sqrt((diff ** 2).where(valid).sum(axis=1) / valid.sum(axis=1))
    dist = dist[valid.sum(axis=1) >= max(4, int(0.6 * len(feats)))].sort_values()
    chosen = []
    for m in dist.index:
        if all(abs((m.year - c.year) * 12 + m.month - c.month) >= min_gap_months for c in chosen):
            chosen.append(m)
        if len(chosen) >= k:
            break
    return chosen, dist


def sim_analog(r, months, n_paths=4000, horizon=H, mean_block=21, rng=None):
    """Pool the daily returns from the 12 months after each analog month and bootstrap from that pool."""
    rng = rng or np.random.default_rng(0)
    pool = []
    for m in months:
        seg = r[(r.index > m) & (r.index <= m + pd.DateOffset(months=12))]
        pool.append(seg.values)
    if not pool:
        return sim_bootstrap(r, n_paths, horizon, rng=rng)
    pool = np.concatenate(pool)
    return stationary_bootstrap(pool, n_paths, horizon, mean_block, rng)


def actual_forward_dd(r, asof, horizon=H):
    seg = r[r.index > asof].iloc[:horizon]
    if len(seg) < horizon * 0.95:
        return None
    return float(max_drawdown_paths(seg.values[None, :])[0])


ANALOG_FEATS = ["short_rate", "short_rate_chg_12m", "t10y", "t10y_chg_12m", "curve_10y_3m", "baa_aaa_spread",
                "baa_aaa_chg_12m", "cpi_yoy", "cape", "sp500_ret_12m", "realized_vol", "indpro_yoy", "unemployment_chg_12m"]


# ----------------------------------------------------------------------------- today

def forecast_today(n_paths=20000, seed=7):
    r = daily_returns()
    w = pd.read_csv(os.path.join(DATA, "timeline.csv"), parse_dates=["asof"], index_col="asof")
    asof = w.index.max()
    rng = np.random.default_rng(seed)
    hmm = fit_hmm(r.values[-252 * 40:])   # same 40-year window the validation test uses
    start = hmm["filtered_last"]
    months, dist = analog_windows(w, ANALOG_FEATS, asof)
    sims = {
        "normal": sim_normal(r.values, n_paths, rng=rng),
        "bootstrap": sim_bootstrap(r.values, n_paths, rng=rng),
        "regime": sim_regime(r.values[-252 * 40:], hmm, start, n_paths, rng=rng),
        "analog": sim_analog(r, months, n_paths, rng=rng),
    }
    res = {}
    for k, s in sims.items():
        dd = max_drawdown_paths(s)
        ret = np.exp(s.sum(axis=1)) - 1
        res[k] = {**probs(dd), "median_dd": float(np.median(dd)), "p5_return": float(np.percentile(ret, 5)),
                  "median_return": float(np.median(ret)), "p95_return": float(np.percentile(ret, 95))}
        np.save(os.path.join(DATA, f"mc_dd_{k}.npy"), dd[:5000].astype(np.float32))
    params = {
        "asof": str(asof.date()), "last_daily_date": str(r.index.max().date()),
        "hmm": {"mu": hmm["mu"].tolist(), "sd": hmm["sd"].tolist(), "A": hmm["A"].tolist(),
                "p_stress_today": float(start[1]),
                "calm_daily_vol_annual": float(hmm["sd"][0] * np.sqrt(252)),
                "stress_daily_vol_annual": float(hmm["sd"][1] * np.sqrt(252))},
        "analog_months": [str(m.date()) for m in months],
        "results": res,
    }
    # empirical base rate since 1950 (share of month-ends followed by a >=X% drawdown within 12m)
    me = r.groupby(r.index.to_period("M")).apply(lambda s: s.index.max()).values
    hits = {t: [] for t in THRESHOLDS}
    for m in me:
        dd = actual_forward_dd(r, pd.Timestamp(m))
        if dd is None:
            continue
        for t in THRESHOLDS:
            hits[t].append(dd >= t)
    params["base_rate_since_1950"] = {f"p_dd_{int(t*100)}": float(np.mean(v)) for t, v in hits.items()}
    with open(os.path.join(DATA, "mc_today.json"), "w") as f:
        json.dump(params, f, indent=1)
    return params


# ----------------------------------------------------------------------------- walk-forward test

def walk_forward(start="1965-01-31", end="2025-09-30", n_paths=1500, refit_every_years=5, seed=11):
    r = daily_returns()
    w = pd.read_csv(os.path.join(DATA, "timeline.csv"), parse_dates=["asof"], index_col="asof")
    rng = np.random.default_rng(seed)
    dates = w.loc[start:end].index
    hmm, last_fit = None, None
    rows = []
    for asof in dates:
        hist = r[r.index <= asof]
        if hmm is None or (asof - last_fit).days > 365 * refit_every_years:
            hmm = fit_hmm(hist.values[-252 * 40:], n_iter=40)   # fit on up to 40 years before asof
            last_fit = asof
        actual = actual_forward_dd(r, asof)
        if actual is None:
            continue
        start_p = filter_last(hist.values[-252:], hmm)
        months, _ = analog_windows(w, ANALOG_FEATS, asof)
        sims = {
            "normal": sim_normal(hist.values, n_paths, rng=rng),
            "bootstrap": sim_bootstrap(hist.values, n_paths, rng=rng),
            "regime": sim_regime(hist.values[-252 * 40:], hmm, start_p, n_paths, rng=rng),
            "analog": sim_analog(hist, months, n_paths, rng=rng),
        }
        row = {"asof": asof, "actual_dd": actual}
        for k, s in sims.items():
            p = probs(max_drawdown_paths(s))
            for kk, v in p.items():
                row[f"{k}_{kk}"] = v
        # base-rate benchmark: share of past month-ends (with known outcomes) followed by a drawdown
        rows.append(row)
    df = pd.DataFrame(rows)
    # expanding base rate using only outcomes known at the time (12 months earlier)
    for t in THRESHOLDS:
        hit = (df.actual_dd >= t).astype(float)
        # seed with outcomes from 1950 up to 12 months before the first forecast (all known at the time)
        seed_dates = w.loc["1950-01-31":pd.Timestamp(start) - pd.DateOffset(months=13)].index
        seed_hits = [float(d >= t) for d in (actual_forward_dd(r, a) for a in seed_dates) if d is not None]
        hv, n0, s0 = hit.values, len(seed_hits), sum(seed_hits)
        csum = np.concatenate([[0.0], np.cumsum(hv)])
        base = [(s0 + csum[max(i - 12, 0)]) / (n0 + max(i - 12, 0)) for i in range(len(hv))]
        df[f"base_p_dd_{int(t*100)}"] = base
    df.to_csv(os.path.join(DATA, "mc_walkforward.csv"), index=False)
    return df


def score(df):
    out = {}
    models = ["normal", "bootstrap", "regime", "analog", "base"]
    for period, sub in (("1965-2025", df), ("1965-1994", df[df["asof"] < "1995-01-01"]),
                        ("1995-2025", df[df["asof"] >= "1995-01-01"])):
        out[period] = {}
        for t in THRESHOLDS:
            y = (sub.actual_dd >= t).astype(float)
            key = f"p_dd_{int(t*100)}"
            s = {}
            for m in models:
                p = sub[f"{m}_{key}"]
                s[m] = {"brier": float(((p - y) ** 2).mean()), "avg_forecast": float(p.mean())}
            base = s["base"]["brier"]
            for m in models:
                s[m]["skill_vs_base"] = float(1 - s[m]["brier"] / base) if base > 0 else None
            s["actual_rate"] = float(y.mean())
            out[period][key] = s
    # calibration for the 10% event: forecast buckets vs actual frequency
    cal = {}
    y = (df.actual_dd >= 0.10).astype(float)
    for m in ["normal", "bootstrap", "regime", "analog"]:
        p = df[f"{m}_p_dd_10"]
        bins = pd.cut(p, [0, 0.2, 0.35, 0.5, 0.65, 1.0])
        cal[m] = (pd.DataFrame({"p": p, "y": y, "b": bins}).groupby("b", observed=True)
                  .agg(avg_forecast=("p", "mean"), actual=("y", "mean"), n=("y", "size")).round(3)
                  .reset_index().astype({"b": str}).to_dict(orient="records"))
    out["calibration_dd10"] = cal
    with open(os.path.join(DATA, "mc_validation.json"), "w") as f:
        json.dump(out, f, indent=1)
    return out


if __name__ == "__main__":
    import sys, time
    t0 = time.time()
    p = forecast_today()
    print(json.dumps({k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in p["results"].items()}, indent=1))
    print("base rate:", p["base_rate_since_1950"], "p_stress_today:", round(p["hmm"]["p_stress_today"], 3))
    print("analogs:", p["analog_months"][:12])
    print(f"today done in {time.time()-t0:.0f}s")
    if "--validate" in sys.argv:
        df = walk_forward()
        s = score(df)
        print(json.dumps({per: {k: {m: round(v["brier"], 4) if isinstance(v, dict) else v for m, v in d.items()} for k, d in s[per].items()} for per in ("1965-2025", "1965-1994", "1995-2025")}, indent=1))
        print(f"total {time.time()-t0:.0f}s")
