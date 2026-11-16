"""
analysis.py - what kinds of crashes are there, which past moments look like today, and
did leveraged / capex-heavy sectors get hit harder when credit tightened?

Inputs : data/timeline.csv, data/crash_library.csv, data/crash_fingerprints.csv, raw sector ETFs
Outputs: data/crash_types.csv, data/analogs_crashes.csv, data/analogs_months.csv,
         data/sector_test.csv, data/analysis_summary.json
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
RAW = os.path.join(DATA, "raw")

# Features with history back to 1928 (so every crash can be compared on the same footing)
CORE = ["short_rate_chg_12m", "real_short_rate", "t10y", "t10y_chg_12m", "curve_10y_3m",
        "baa_aaa_spread", "baa_aaa_chg_12m", "cpi_yoy", "cape", "sp500_ret_12m", "realized_vol", "indpro_yoy"]
# For grouping crashes into TYPES we use era-neutral versions: each condition vs its own trailing norm,
# so a 1950s crash and a 2020s crash are judged on the same footing (CAPE has drifted up for a century).
TYPE_FEATS = ["cape_vs_20y", "sp500_ret_12m", "short_rate_chg_12m", "t10y_chg_12m", "real_short_rate",
              "curve_10y_3m", "spread_vs_10y", "baa_aaa_chg_12m", "vol_vs_10y", "cpi_chg_12m", "indpro_yoy"]


def add_relative(w):
    """Trailing-window (point-in-time) relative features."""
    w = w.copy()
    w["cape_vs_20y"] = w.cape / w.cape.rolling(240, min_periods=60).median() - 1
    w["spread_vs_10y"] = w.baa_aaa_spread - w.baa_aaa_spread.rolling(120, min_periods=36).median()
    w["vol_vs_10y"] = w.realized_vol - w.realized_vol.rolling(120, min_periods=36).median()
    w["cpi_chg_12m"] = w.cpi_yoy - w.cpi_yoy.shift(12)
    return w
# Extra features used for analog matching when available (post-war)
EXTRA = ["unemployment", "unemployment_chg_12m", "oil_yoy"]

LABELS = {
    "short_rate_chg_12m": "Fed tightening (12m)", "real_short_rate": "Real short rate", "t10y": "10-yr yield",
    "t10y_chg_12m": "10-yr change (12m)", "curve_10y_3m": "Yield curve", "baa_aaa_spread": "Credit spread",
    "baa_aaa_chg_12m": "Credit spread change", "cpi_yoy": "Inflation", "cape": "CAPE valuation",
    "sp500_ret_12m": "Prior 12m return", "realized_vol": "Volatility", "indpro_yoy": "Industrial production",
    "unemployment": "Unemployment", "unemployment_chg_12m": "Unemployment change", "oil_yoy": "Oil price change",
    "cape_vs_20y": "Valuation vs 20-yr norm", "spread_vs_10y": "Credit spread vs 10-yr norm",
    "vol_vs_10y": "Volatility vs 10-yr norm", "cpi_chg_12m": "Inflation acceleration",
}

# Names assigned after inspecting each cluster's average conditions (see report). Keyed by the
# cluster's dominant feature so they stay stable if the data are refreshed.


def zscore_params(w, cols):
    """Standardize using the whole 1928-today distribution of months (not just crash months)."""
    return w[cols].mean(), w[cols].std()


def _unused_name_cluster(center):
    """Turn a cluster's average z-scores into a plain-English type name."""
    c = center
    if c.get("cape", 0) > 0.8 and c.get("sp500_ret_12m", 0) > 0.3:
        return "Valuation bust (expensive market, strong run-up)"
    if c.get("cpi_yoy", 0) > 0.7 or c.get("short_rate_chg_12m", 0) > 0.7:
        return "Inflation / Fed-tightening squeeze"
    if c.get("baa_aaa_spread", 0) > 0.7 or c.get("realized_vol", 0) > 0.8:
        return "Credit stress / fragile market"
    return "Outside shock in a calm market"


TYPE_NAMES = {
    "squeeze": "Fed-tightening / inflation squeeze",
    "expensive": "Expensive market meets a shock",
    "credit": "Credit stress in a slowing economy",
    "aftershock": "Aftershock in a young recovery",
}
TYPE_BLURBS = {
    "Fed-tightening / inflation squeeze": "Short and long rates rising, the yield curve flat or inverted, inflation speeding up and credit spreads starting to widen. Historically the deepest type.",
    "Expensive market meets a shock": "Valuations well above their own recent norm after a strong run, with rates and credit looking normal. A shock (war, virus, policy, a crash in one corner) does the damage.",
    "Credit stress in a slowing economy": "Industrial production slowing, credit spreads wider than normal and widening, the Fed already easing. Lenders are pulling back.",
    "Aftershock in a young recovery": "Cheap valuations after a big rebound from a prior crash; the recovery stumbles before it's established.",
}


def assign_names(C):
    """Map each cluster (centres in percentile units) to the type it most resembles."""
    score = pd.DataFrame({
        "squeeze": C.short_rate_chg_12m + C.t10y_chg_12m - C.curve_10y_3m + C.cpi_chg_12m,
        "expensive": 2 * C.cape_vs_20y - C.spread_vs_10y,
        "credit": C.spread_vs_10y + C.baa_aaa_chg_12m - C.indpro_yoy - C.short_rate_chg_12m,
        "aftershock": -C.cape_vs_20y + C.sp500_ret_12m - C.baa_aaa_chg_12m,
    })
    names, used = {}, set()
    for _ in range(len(C)):
        best = None
        for i in C.index:
            if i in names:
                continue
            for t in score.columns:
                if t in used:
                    continue
                if best is None or score.loc[i, t] > best[2]:
                    best = (i, t, score.loc[i, t])
        names[best[0]] = TYPE_NAMES[best[1]]
        used.add(best[1])
    return names


def type_matrix(w, months):
    """Percentile (vs every month since 1928) of each era-neutral feature, centred at 0."""
    P = w[TYPE_FEATS].rank(pct=True)
    return pd.DataFrame({k: P.loc[m] for k, m in months.items()}).T.astype(float).fillna(0.5) - 0.5


def cluster_crashes(w, lib, fp):
    w = add_relative(w)
    peaks = pd.to_datetime(lib.set_index("id").peak_date)
    m0 = peaks.map(lambda d: d.to_period("M").to_timestamp("M") - pd.offsets.MonthEnd(1))
    X = type_matrix(w, m0)
    km = KMeans(n_clusters=4, n_init=100, random_state=0).fit(X)
    C = pd.DataFrame(km.cluster_centers_, columns=TYPE_FEATS)
    names = assign_names(C)
    out = lib.set_index("id").join(pd.Series(km.labels_, index=X.index, name="cluster"))
    out["crash_type"] = out.cluster.map(names)
    for c in TYPE_FEATS:
        out["pct_" + c] = (X[c] + 0.5).round(2)
    profile = (out.groupby("crash_type")
               .agg(n=("depth_pct", "size"), avg_depth=("depth_pct", "mean"), worst=("depth_pct", "min"),
                    median_days_to_trough=("days_to_trough", "median"),
                    recession_rate=("recession_followed", "mean"))
               .round(2))
    C.index = [names[i] for i in C.index]
    # which type does TODAY look like? (nearest cluster centre)
    last = w.index.max()
    xt = type_matrix(w, {"today": last}).loc["today"]
    d = ((C - xt) ** 2).sum(axis=1) ** 0.5
    today_type = {"nearest_type": d.idxmin(), "distances": d.round(3).to_dict(),
                  "today_percentiles": (xt + 0.5).round(2).to_dict()}
    info = {"k": 4, "silhouette": float(silhouette_score(X, km.labels_)), "today": today_type}
    return out.reset_index(), profile, C, info


def today_vector(w):
    last = w.index.max()
    return last, w.loc[last]


def analogs_to_crashes(w, fp, lib, today):
    feats = CORE + EXTRA
    mu, sd = zscore_params(w, feats)
    f0 = fp[fp.lag_months == 0].set_index("id")
    zt = (today[feats] - mu) / sd
    rows = []
    for cid, r in f0.iterrows():
        zc = (r[feats] - mu) / sd
        both = zt.notna() & zc.astype(float).notna()
        diff = (zt[both] - zc[both].astype(float))
        dist = float(np.sqrt((diff ** 2).mean()))   # RMS z-distance over shared features
        close = diff.abs().sort_values().index[:3].tolist()
        far = diff.abs().sort_values(ascending=False).index[:3].tolist()
        rows.append({"id": cid, "distance": round(dist, 3), "features_compared": int(both.sum()),
                     "most_similar_on": ", ".join(LABELS[c] for c in close),
                     "most_different_on": ", ".join(LABELS[c] for c in far)})
    a = pd.DataFrame(rows).merge(lib[["id", "peak_date", "trough_date", "depth_pct", "severity", "recession_followed"]], on="id")
    a["similarity"] = (100 * np.exp(-a.distance)).round(0)
    return a.sort_values("distance")


def analog_months(w, today, horizon=12):
    """Nearest historical MONTHS to today (any month, not just crash peaks), with what happened next."""
    feats = CORE + EXTRA
    mu, sd = zscore_params(w, feats)
    Z = (w[feats] - mu) / sd
    zt = (today[feats] - mu) / sd
    hist = Z.loc[: w.index.max() - pd.DateOffset(months=horizon)]
    diff = hist.sub(zt, axis=1)
    valid = diff.notna()
    dist = np.sqrt((diff ** 2).where(valid).sum(axis=1) / valid.sum(axis=1))
    dist = dist[valid.sum(axis=1) >= 8]
    # forward outcomes (hindsight - used only as the answer key)
    d = pd.read_csv(os.path.join(RAW, "gspc_daily.csv"), parse_dates=["date"]).set_index("date")["close"]
    sp = w["sp500"]
    outs = []
    for m, dv in dist.sort_values().items():
        if any(abs((m - o).days) < 180 for o, *_ in outs):   # keep analogs at least 6 months apart
            continue
        end = m + pd.DateOffset(months=horizon)
        if m >= pd.Timestamp("1950-01-01"):
            path = d[(d.index > m) & (d.index <= end)]
            start = d[d.index <= m].iloc[-1]
        else:
            path = sp[(sp.index > m) & (sp.index <= end)]
            start = sp.loc[m]
        if len(path) == 0:
            continue
        maxdd = float((np.minimum(path.min(), start) / start - 1) * 100)
        # deepest peak-to-trough inside the window (including the start)
        full = pd.concat([pd.Series([start]), path.reset_index(drop=True)])
        dd = float(((full / full.cummax()) - 1).min() * 100)
        ret = float((path.iloc[-1] / start - 1) * 100)
        outs.append((m, dv, ret, dd))
        if len(outs) >= 15:
            break
    return pd.DataFrame(outs, columns=["month", "distance", "next_12m_return", "next_12m_max_drawdown"])


def sector_test(w, lib):
    """Since 1999: in each drawdown, how did leveraged / capex-heavy sectors do vs asset-light ones,
    and does the gap widen when credit spreads widened during the decline?"""
    s = pd.read_csv(os.path.join(RAW, "sector_etfs_weekly.csv"), parse_dates=["date"]).set_index("date")
    heavy = ["XLU", "XLE", "XLI", "XLB", "XLF"]     # capital-intensive and/or balance-sheet-heavy
    light = ["XLK", "XLP", "XLV", "XLY"]
    rows = []
    for _, e in lib[pd.to_datetime(lib.peak_date) >= "1999-01-01"].iterrows():
        pk, tr = pd.Timestamp(e.peak_date), pd.Timestamp(e.trough_date)
        a = s[s.index <= pk + pd.Timedelta(days=3)].iloc[-1]
        b = s[s.index <= tr + pd.Timedelta(days=3)].iloc[-1]
        r = (b / a - 1) * 100
        pm = pk.to_period("M").to_timestamp("M") - pd.offsets.MonthEnd(1)
        tm = tr.to_period("M").to_timestamp("M")
        spread_chg = w.loc[tm, "baa_aaa_spread"] - w.loc[pm, "baa_aaa_spread"]
        rows.append({"id": e.id, "peak_date": e.peak_date, "sp500_depth": e.depth_pct,
                     "heavy_avg": r[heavy].mean(), "light_avg": r[light].mean(),
                     "gap_heavy_minus_light": r[heavy].mean() - r[light].mean(),
                     "credit_spread_change_during": spread_chg,
                     "t10y_change_12m_before": w.loc[pm, "t10y_chg_12m"],
                     **{k: r[k] for k in heavy + light}})
    t = pd.DataFrame(rows).round(2)
    t["credit_tightened"] = t.credit_spread_change_during > 0.15
    return t


def build():
    w = pd.read_csv(os.path.join(DATA, "timeline.csv"), parse_dates=["asof"], index_col="asof")
    lib = pd.read_csv(os.path.join(DATA, "crash_library.csv"))
    fp = pd.read_csv(os.path.join(DATA, "crash_fingerprints.csv"))

    types, profile, centers, kinfo = cluster_crashes(w, lib, fp)
    types.to_csv(os.path.join(DATA, "crash_types.csv"), index=False)

    asof, today = today_vector(w)
    ac = analogs_to_crashes(w, fp, types.rename(columns={}), today)
    ac = ac.merge(types[["id", "crash_type"]], on="id")
    ac.to_csv(os.path.join(DATA, "analogs_crashes.csv"), index=False)

    am = analog_months(w, today)
    am.to_csv(os.path.join(DATA, "analogs_months.csv"), index=False)

    st = sector_test(w, lib)
    st.to_csv(os.path.join(DATA, "sector_test.csv"), index=False)

    g = st.groupby("credit_tightened").gap_heavy_minus_light.agg(["mean", "count"]).round(2)
    summary = {
        "asof": str(asof.date()),
        "clusters": kinfo,
        "type_profile": profile.reset_index().to_dict(orient="records"),
        "type_centers_z": centers.round(2).reset_index().rename(columns={"index": "crash_type"}).to_dict(orient="records"),
        "top_crash_analogs": ac.head(8)[["id", "peak_date", "depth_pct", "crash_type", "similarity", "most_similar_on", "most_different_on"]].to_dict(orient="records"),
        "analog_months": am.assign(month=am.month.dt.strftime("%Y-%m")).round(2).to_dict(orient="records"),
        "analog_months_stats": {
            "n": int(len(am)),
            "median_next_12m_return": float(am.next_12m_return.median()),
            "share_with_10pct_drawdown": float((am.next_12m_max_drawdown <= -10).mean()),
            "share_with_20pct_drawdown": float((am.next_12m_max_drawdown <= -20).mean()),
        },
        "sector_gap_by_credit": g.reset_index().to_dict(orient="records"),
        "type_blurbs": TYPE_BLURBS,
    }
    with open(os.path.join(DATA, "analysis_summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=str)
    return types, profile, centers, ac, am, st, summary


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    types, profile, centers, ac, am, st, s = build()
    print("k/silhouette:", s["clusters"])
    print(profile.to_string())
    print(centers.round(2).to_string())
    print(types[["id", "peak_date", "depth_pct", "crash_type"]].to_string(index=False))
    print(ac.head(10)[["id", "peak_date", "depth_pct", "crash_type", "similarity", "most_similar_on", "most_different_on"]].to_string(index=False))
    print(am.to_string(index=False))
    print(s["analog_months_stats"])
    print(st[["peak_date", "sp500_depth", "heavy_avg", "light_avg", "gap_heavy_minus_light", "credit_spread_change_during", "credit_tightened"]].to_string(index=False))
    print(s["sector_gap_by_credit"])
