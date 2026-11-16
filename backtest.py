"""
backtest.py - does the leverage risk score actually predict anything?

For ~230 large non-financial US stocks, for each of the last ~4 fiscal years:
  1. Rebuild the score using ONLY what was public at the time
     (that year's annual statements + prices up to 75 days after fiscal year-end, when the 10-K is out).
  2. Record what happened next: 6- and 12-month return vs. the S&P 500, and the worst drawdown.

Output:
  backtest_results.csv   one row per stock-year (send this to Claude for the deep analysis)
  backtest_summary.txt   quick read: do Fragile stocks underperform / crash harder than Fortress stocks?

It saves as it goes and skips tickers already done, so you can stop and restart any time.
Takes roughly 20-45 minutes. Leave the window open.

Known limits (be honest about these when presenting results):
  - Survivorship bias: the list is today's companies, so firms that went bankrupt aren't in it.
  - Only ~4 years of annual statements are free from Yahoo, which covers 1-2 market regimes.
  - Options trades can't be backtested here (no free historical options data).
"""

import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

import engine as E

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_CSV = os.path.join(HERE, "backtest_results.csv")
OUT_TXT = os.path.join(HERE, "backtest_summary.txt")
FILING_LAG_DAYS = 75

UNIVERSE = """
AAPL MSFT NVDA GOOGL AMZN META AVGO ORCL AMD INTC QCOM TXN MU AMAT LRCX KLAC ADI MRVL CRM ADBE NOW INTU IBM
CSCO DELL HPE HPQ SMCI ANET NBIS PLTR SNOW DDOG NET CRWD PANW FTNT ZS MDB WDAY TEAM SHOP UBER ABNB DASH
SPOT NFLX APLD IREN WDC STX
T VZ TMUS CMCSA CHTR DIS WBD EA TTWO LUMN SIRI
WMT COST HD LOW TGT NKE SBUX MCD CMG YUM DPZ KO PEP PG CL KMB MDLZ GIS KHC HSY MO PM EL TSLA F GM RIVN
LCID BKNG MAR HLT CCL RCL NCLH DAL UAL AAL LUV LULU ROST TJX DG DLTR BBY KR MGM WYNN LVS CZR
JNJ PFE MRK ABBV LLY BMY AMGN GILD REGN VRTX BIIB MRNA CVS UNH CI ELV HUM ABT TMO DHR MDT BSX SYK ISRG
ZBH BAX BDX HCA
BA CAT DE GE HON LMT RTX NOC GD UPS FDX UNP CSX NSC MMM EMR ETN PH ITW WM RSG URI CARR OTIS JCI PWR VRT
XOM CVX COP EOG OXY SLB HAL DVN MPC VLO PSX KMI WMB OKE
NEE DUK SO D AEP EXC SRE XEL PCG CEG VST
LIN APD SHW ECL NEM FCX DOW DD NUE ALB
""".split()


def history(sym):
    h = yf.Ticker(sym).history(period="10y", interval="1d", auto_adjust=True)
    s = h["Close"]
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s


def max_drawdown(path):
    run_max = path.cummax()
    return float((path / run_max - 1).min())


def shares_at(bs_col):
    for n in ["Ordinary Shares Number", "Share Issued"]:
        if n in bs_col.index and pd.notna(bs_col[n]):
            return float(bs_col[n])
    return None


def score_ticker(tk, spx, tnx):
    t = yf.Ticker(tk)
    px = history(tk)
    bs, inc, cf = t.balance_sheet, t.income_stmt, t.cashflow
    try:
        sector = (t.info or {}).get("sector", "")
    except Exception:
        sector = ""
    if bs is None or bs.empty or inc is None or inc.empty or len(px) < 400:
        return []

    empty = pd.DataFrame()
    rows = []
    cols = list(bs.columns)
    for i, fye in enumerate(cols):
        fye = pd.Timestamp(fye)
        sig_target = fye + pd.Timedelta(days=FILING_LAG_DAYS)
        after = px.index[px.index >= sig_target]
        if len(after) == 0:
            continue
        sig = after[0]
        hist = px[:sig]
        if len(hist) < 260:
            continue
        # Align statements: keep columns from this fiscal year back
        def upto(df):
            if df is None or df.empty:
                return empty
            keep = [c for c in df.columns if pd.Timestamp(c) <= fye]
            return df[keep]
        bs_i, inc_i, cf_i = upto(bs), upto(inc), upto(cf)
        if bs_i.empty or inc_i.empty:
            continue
        p0 = float(hist.iloc[-1])
        sh = shares_at(bs_i.iloc[:, 0])
        raw = {"ticker": tk, "info": {"marketCap": sh * p0 if sh else None, "sector": sector},
               "bs_a": bs_i, "inc_a": inc_i, "cf_a": cf_i, "bs_q": empty, "inc_q": empty, "cf_q": empty,
               "px": hist, "spx": spx[:sig], "tnx": tnx[:sig]}
        lev = E.leverage_metrics(raw)
        mkt = E.market_metrics(raw)
        comps = E.component_scores(lev, mkt, sector)
        score, tier, cov = E.risk_score(comps)
        if score is None:
            continue
        bias = E.directional_bias(tier, mkt)

        fwd = px[sig:]
        sfwd = spx[sig:]
        row = {"ticker": tk, "sector": sector, "fiscal_year_end": fye.date(), "signal_date": sig.date(),
               "risk_score": score, "tier": tier, "score_coverage": cov,
               "bias": bias["bias"], "bias_score": bias["score"], "trend": bias["trend"]}
        row.update({f"c_{k}": v for k, v in comps.items()})
        for k in ["net_debt_to_ebitda", "interest_coverage", "fcf_margin", "capex_to_ocf", "debt_growth_1y",
                  "altman_z", "debt_to_assets", "current_ratio", "cash_runway_yrs"]:
            v = lev.get(k)
            row[k] = None if v is None or v == float("inf") else v
        for k in ["beta", "rate_sens_25bp", "rate_tstat", "vol_1y", "ret_6m", "off_52w_high", "tnx_now", "tnx_chg_3m"]:
            row[k] = mkt.get(k)
        for label, n in (("6m", 126), ("12m", 252)):
            if len(fwd) > n and len(sfwd) > n:
                r = float(fwd.iloc[n] / fwd.iloc[0] - 1)
                m = float(sfwd.iloc[n] / sfwd.iloc[0] - 1)
                row[f"fwd_{label}"], row[f"spx_{label}"], row[f"excess_{label}"] = r, m, r - m
                row[f"maxdd_{label}"] = max_drawdown(fwd.iloc[: n + 1])
            else:
                row[f"fwd_{label}"] = row[f"spx_{label}"] = row[f"excess_{label}"] = row[f"maxdd_{label}"] = None
        rows.append(row)
    return rows


def ols(y, x):
    """slope, t-stat, r2 of y = a + b*x"""
    d = pd.concat([y, x], axis=1).dropna()
    if len(d) < 20:
        return None, None, None, len(d)
    Y, Xv = d.iloc[:, 0].values.astype(float), d.iloc[:, 1].values.astype(float)
    if np.std(Xv) < 1e-9:
        return None, None, None, len(d)
    X = np.column_stack([np.ones(len(Xv)), Xv])
    b, *_ = np.linalg.lstsq(X, Y, rcond=None)
    res = Y - X @ b
    s2 = res @ res / (len(Y) - 2)
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    r2 = 1 - (res @ res) / ((Y - Y.mean()) @ (Y - Y.mean()))
    return b[1], b[1] / se[1], r2, len(d)


def summarize(df):
    L = []
    p = L.append
    p(f"LEVERAGE SCORE BACKTEST  -  {datetime.now():%Y-%m-%d %H:%M}")
    p(f"{df['ticker'].nunique()} stocks, {len(df)} stock-years, signals {df['signal_date'].min()} to {df['signal_date'].max()}")
    p("")
    d12 = df.dropna(subset=["excess_12m"])
    p(f"Observations with a full 12 months of follow-up: {len(d12)}")
    if len(d12) >= 40:
        d12 = d12.copy()
        d12["quintile"] = pd.qcut(d12["risk_score"].rank(method="first"), 5, labels=["Q1 safest", "Q2", "Q3", "Q4", "Q5 riskiest"])
        g = d12.groupby("quintile", observed=True).agg(
            n=("ticker", "size"), avg_score=("risk_score", "mean"),
            excess_12m=("excess_12m", "mean"), median_excess=("excess_12m", "median"),
            worst_drawdown=("maxdd_12m", "mean"), pct_beat_spx=("excess_12m", lambda s: (s > 0).mean()))
        p("\nBY RISK-SCORE QUINTILE (next 12 months, vs S&P 500):")
        p(g.to_string(float_format=lambda v: f"{v:,.3f}"))
        spread = g["excess_12m"].iloc[-1] - g["excess_12m"].iloc[0]
        p(f"\nRiskiest minus safest quintile, avg 12m excess return: {spread:+.1%}")
        dd = g["worst_drawdown"].iloc[-1] - g["worst_drawdown"].iloc[0]
        p(f"Riskiest minus safest quintile, avg max drawdown:     {dd:+.1%}")

        p("\nBY TIER:")
        p(d12.groupby("tier").agg(n=("ticker", "size"), excess_12m=("excess_12m", "mean"),
                                  worst_drawdown=("maxdd_12m", "mean")).to_string(float_format=lambda v: f"{v:,.3f}"))
        p("\nBY BIAS (does the bullish/bearish call work?):")
        p(d12.groupby("bias").agg(n=("ticker", "size"), excess_12m=("excess_12m", "mean"),
                                  pct_beat_spx=("excess_12m", lambda s: (s > 0).mean()),
                                  worst_drawdown=("maxdd_12m", "mean")).to_string(float_format=lambda v: f"{v:,.3f}"))

        p("\nREGRESSIONS (slope per 10 points of risk score; |t| >= 2 means statistically meaningful):")
        for ycol, name in (("excess_12m", "12m excess return"), ("maxdd_12m", "12m max drawdown"),
                           ("excess_6m", "6m excess return")):
            b, t, r2, n = ols(df[ycol], df["risk_score"])
            if b is not None:
                p(f"  {name:20s} slope {b * 10:+.2%} per 10 pts   t = {t:+.2f}   R2 = {r2:.3f}   n = {n}")
        p("\nWHICH COMPONENTS PREDICT DRAWDOWNS? (slope per 10 pts of each component score)")
        for c in [c for c in df.columns if c.startswith("c_")]:
            b, t, r2, n = ols(df["maxdd_12m"], df[c])
            if b is not None:
                p(f"  {c[2:]:22s} {b * 10:+.2%}   t = {t:+.2f}   n = {n}")
    p("\nCaveats: survivorship bias (today's companies only), ~4 years of data, one or two market regimes.")
    p("Send backtest_results.csv to Claude for the full analysis and re-weighting.")
    txt = "\n".join(L)
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write(txt)
    print("\n" + txt)


def main():
    done = set()
    rows = []
    if os.path.exists(OUT_CSV):
        prev = pd.read_csv(OUT_CSV)
        rows = prev.to_dict("records")
        done = set(prev["ticker"].unique())
        print(f"Resuming: {len(done)} tickers already done.")
    todo = [t for t in UNIVERSE if t not in done]
    print("Downloading S&P 500 and 10-year yield history...")
    spx, tnx = history("^GSPC"), history("^TNX")
    start = time.time()
    failed = []
    for k, tk in enumerate(todo, 1):
        try:
            new = score_ticker(tk, spx, tnx)
            rows.extend(new)
            status = f"{len(new)} years"
        except Exception as e:
            failed.append(tk)
            status = f"skipped ({str(e)[:60]})"
        el = time.time() - start
        eta = el / k * (len(todo) - k) / 60
        print(f"[{k}/{len(todo)}] {tk:6s} {status:40s} ~{eta:.0f} min left", flush=True)
        if k % 10 == 0 or k == len(todo):
            pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
        time.sleep(0.4)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    if failed:
        print(f"\nSkipped {len(failed)}: {' '.join(failed)}")
    if df.empty:
        print("No results - check your internet connection.")
        return
    summarize(df)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--summary":
        summarize(pd.read_csv(OUT_CSV))
    else:
        main()
