"""
sector_history.py - the thesis test, done as rigorously as the data allow.

Question: during S&P 500 declines, do capital-heavy / leveraged industries fall MORE than asset-light ones,
and is that gap bigger when credit conditions tighten (BAA-AAA spread widens during the fall)?

1. ETF version (1999+, always available): permutation test on data/sector_test.csv.
2. Full-history version (1928+): needs Ken French's 49-industry file, which update_data.py downloads
   on your laptop (data/raw/french_49_industries_monthly.csv). Uses month-end returns from the month
   before the peak to the trough month.

Outputs: data/sector_stats.json and, if French data exist, data/sector_test_long.csv
"""

import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
RAW = os.path.join(DATA, "raw")

HEAVY = ["Util", "Oil", "Steel", "Chems", "Mach", "Trans", "Banks", "RlEst", "Telcm", "Mines", "Coal", "Paper",
         "Autos", "Ships", "Aero", "BldMt", "Cnstr", "Rubbr", "Boxes"]
LIGHT = ["Softw", "Hardw", "Chips", "Drugs", "MedEq", "Hlth", "Food", "Beer", "Smoke", "Hshld", "Rtail",
         "PerSv", "BusSv", "Fun", "Books", "Clths", "Meals", "Toys", "Soda"]
CREDIT_RULE = 0.15   # spread widened by more than 0.15 percentage points during the fall


def perm_test(gap, flag, n=20000, seed=0):
    """Difference in mean gap (tightened minus not) and a one-sided permutation p-value."""
    rng = np.random.default_rng(seed)
    gap, flag = np.asarray(gap, float), np.asarray(flag, bool)
    obs = gap[flag].mean() - gap[~flag].mean()
    cnt = 0
    for _ in range(n):
        f = rng.permutation(flag)
        if gap[f].mean() - gap[~f].mean() <= obs:
            cnt += 1
    return float(obs), float(cnt / n)


def corr_test(gap, spread_chg, n=20000, seed=1):
    rng = np.random.default_rng(seed)
    gap, x = np.asarray(gap, float), np.asarray(spread_chg, float)
    obs = np.corrcoef(gap, x)[0, 1]
    null = np.array([np.corrcoef(gap, rng.permutation(x))[0, 1] for _ in range(n)])
    return float(obs), float((null <= obs).mean())


def long_test(w, lib):
    p = os.path.join(RAW, "french_49_industries_monthly.csv")
    if not os.path.exists(p):
        return None
    f = pd.read_csv(p, parse_dates=["month"], index_col="month") / 100.0
    heavy = [c for c in HEAVY if c in f.columns]
    light = [c for c in LIGHT if c in f.columns]
    rows = []
    for _, e in lib.iterrows():
        pk, tr = pd.Timestamp(e.peak_date), pd.Timestamp(e.trough_date)
        pm = pk.to_period("M").to_timestamp("M") - pd.offsets.MonthEnd(1)
        tm = tr.to_period("M").to_timestamp("M")
        seg = f.loc[(f.index > pm) & (f.index <= tm)]
        if seg.empty or pm not in w.index or tm not in w.index:
            continue
        cum = (1 + seg).prod() - 1
        rows.append({"id": e.id, "peak_date": e.peak_date, "sp500_depth": e.depth_pct,
                     "heavy_avg": cum[heavy].mean() * 100, "light_avg": cum[light].mean() * 100,
                     "gap_heavy_minus_light": (cum[heavy].mean() - cum[light].mean()) * 100,
                     "credit_spread_change_during": w.loc[tm, "baa_aaa_spread"] - w.loc[pm, "baa_aaa_spread"],
                     "short_rate_change_12m_before": w.loc[pm, "short_rate_chg_12m"]})
    t = pd.DataFrame(rows).round(2)
    t["credit_tightened"] = t.credit_spread_change_during > CREDIT_RULE
    t.to_csv(os.path.join(DATA, "sector_test_long.csv"), index=False)
    return t


def investment_test(w, lib):
    """Top 20% vs bottom 20% asset-growth portfolios (Ken French, value-weighted, 1963+)."""
    p = os.path.join(RAW, "french_inv_monthly.csv")
    if not os.path.exists(p):
        return None
    f = pd.read_csv(p, parse_dates=["month"], index_col="month") / 100.0
    rows = []
    for _, e in lib.iterrows():
        pk, tr = pd.Timestamp(e.peak_date), pd.Timestamp(e.trough_date)
        pm = pk.to_period("M").to_timestamp("M") - pd.offsets.MonthEnd(1)
        tm = tr.to_period("M").to_timestamp("M")
        if pm < f.index.min():
            continue
        seg = f.loc[(f.index > pm) & (f.index <= tm)]
        c = (1 + seg).prod() - 1
        rows.append({"id": e.id, "peak_date": e.peak_date, "sp500_depth": e.depth_pct,
                     "heavy_avg": c["Hi 20"] * 100, "light_avg": c["Lo 20"] * 100,
                     "hi10": c["Hi 10"] * 100, "lo10": c["Lo 10"] * 100,
                     "credit_spread_change_during": w.loc[tm, "baa_aaa_spread"] - w.loc[pm, "baa_aaa_spread"],
                     "short_rate_change_12m_before": w.loc[pm, "short_rate_chg_12m"]})
    t = pd.DataFrame(rows)
    t["gap_heavy_minus_light"] = t.heavy_avg - t.light_avg
    t["gap10"] = t.hi10 - t.lo10
    t = t.round(2)
    t["credit_tightened"] = t.credit_spread_change_during > CREDIT_RULE
    t.to_csv(os.path.join(DATA, "sector_test_investment.csv"), index=False)
    return t


def summarize(t, label):
    d, p = perm_test(t.gap_heavy_minus_light, t.credit_tightened)
    r, pr = corr_test(t.gap_heavy_minus_light, t.credit_spread_change_during)
    return {"label": label, "n": int(len(t)), "n_tightened": int(t.credit_tightened.sum()),
            "gap_when_tightened": float(t[t.credit_tightened].gap_heavy_minus_light.mean()),
            "gap_when_not": float(t[~t.credit_tightened].gap_heavy_minus_light.mean()),
            "difference": d, "perm_p_value": p, "corr_gap_vs_spread_change": r, "corr_p_value": pr,
            "share_heavy_worse": float((t.gap_heavy_minus_light < 0).mean())}


def build():
    w = pd.read_csv(os.path.join(DATA, "timeline.csv"), parse_dates=["asof"], index_col="asof")
    lib = pd.read_csv(os.path.join(DATA, "crash_library.csv"))
    out = {"etf_1999_plus": summarize(pd.read_csv(os.path.join(DATA, "sector_test.csv")), "Sector ETFs, 1999+")}
    lt = long_test(w, lib)
    lp = os.path.join(DATA, "sector_test_long.csv")
    if lt is None and os.path.exists(lp):   # results saved from an earlier download of the French data
        lt = pd.read_csv(lp)
    if lt is not None and len(lt) >= 8:
        out["french_1928_plus"] = summarize(lt, "Ken French 49 industries, 1928+")
        out["french_1950_plus"] = summarize(lt[pd.to_datetime(lt.peak_date) >= "1950-01-01"], "Ken French 49 industries, 1950+")
        rs = lt.short_rate_change_12m_before > 0.5
        out["rate_shock_only"] = {"n_rate_shock": int(rs.sum()),
                                  "gap_after_rate_rise": float(lt[rs].gap_heavy_minus_light.mean()),
                                  "gap_otherwise": float(lt[~rs].gap_heavy_minus_light.mean()),
                                  "perm_p_value": perm_test(lt.gap_heavy_minus_light, rs)[1]}
    ip = os.path.join(DATA, "sector_test_investment.csv")   # Ken French portfolios sorted on asset growth (1963+)
    it = investment_test(w, lib)          # recomputed when the raw file exists (update_data.py downloads it)
    if it is None and os.path.exists(ip):
        it = pd.read_csv(ip)
    if it is not None:
        out["investment_1963_plus"] = summarize(it, "Aggressive vs conservative investors (top vs bottom 20% asset growth), 1963+")
        out["investment_1963_plus"]["median_when_tightened"] = float(it[it.credit_tightened].gap_heavy_minus_light.median())
        out["investment_1963_plus"]["median_when_not"] = float(it[~it.credit_tightened].gap_heavy_minus_light.median())
        ex = it[it.id != 31]
        out["investment_1963_plus"]["gap_when_tightened_ex2000"] = float(ex[ex.credit_tightened].gap_heavy_minus_light.mean())
    with open(os.path.join(DATA, "sector_stats.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    return out


if __name__ == "__main__":
    print(json.dumps(build(), indent=1))
