"""
early_warning.py - which warning lights, as they looked at the time, actually came before big drops?

For every month-end since 1955 we check a set of simple yes/no "warning lights" built only from the
point-in-time timeline (no hindsight), then look at what the S&P 500 did over the next 12 months.

  hit rate  = when the light was on, how often a 20%+ drop (from that month's level or a later high) followed
  base rate = the same for all months
  lift      = hit rate / base rate (above 1 = the light carried information)
  caught    = of the 20%+ declines since 1955, how many had the light on at some point in the 12 months before the peak
  p-value   = block-permutation test: shift the light's on/off history in time by a random amount
              (keeps its streaks intact), to see how often a lift this big shows up by chance

Outputs: data/early_warning.csv, data/early_warning_today.json
"""

import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
RAW = os.path.join(DATA, "raw")
START, HORIZON = "1955-01-31", 252


def expanding_pct(s, min_periods=120):
    """Percentile of each value vs all values up to that date (uses only the past)."""
    out = pd.Series(np.nan, index=s.index)
    vals = s.values
    for i in range(len(s)):
        hist = vals[: i + 1]
        hist = hist[~np.isnan(hist)]
        if len(hist) >= min_periods and not np.isnan(vals[i]):
            out.iloc[i] = (hist <= vals[i]).mean()
    return out


def lights(w):
    L = {}
    L["Yield curve inverted"] = w.curve_10y_3m < 0
    L["Fed hiked 1+ pt in a year"] = w.short_rate_chg_12m >= 1.0
    L["10-year up 1+ pt in a year"] = w.t10y_chg_12m >= 1.0
    L["10-year at or above 5%"] = w.t10y >= 5.0
    L["Credit spread widened 0.2+ pt"] = w.baa_aaa_chg_12m >= 0.2
    L["Valuation in top 10% of history so far"] = expanding_pct(w.cape) >= 0.9
    L["Real short rate above 2%"] = w.real_short_rate >= 2.0
    L["Inflation up 1.5+ pt in a year"] = (w.cpi_yoy - w.cpi_yoy.shift(12)) >= 1.5
    L["Oil up 50%+ in a year"] = w.oil_yoy >= 50
    L["Unemployment up 0.5+ pt"] = w.unemployment_chg_12m >= 0.5
    L["Industrial production shrinking"] = w.indpro_yoy < 0
    L["Margin debt up 30%+ in a year"] = w.margin_debt_yoy >= 30
    L["Banks tightening loans (20%+ net)"] = w.lending_standards >= 20
    L["Market calm (vol in bottom 20%)"] = expanding_pct(w.realized_vol) <= 0.2
    avail = {
        "Oil up 50%+ in a year": w.oil_yoy.notna(), "Unemployment up 0.5+ pt": w.unemployment_chg_12m.notna(),
        "Margin debt up 30%+ in a year": w.margin_debt_yoy.notna(),
        "Banks tightening loans (20%+ net)": w.lending_standards.notna(),
    }
    return {k: (v.fillna(False).astype(bool), avail.get(k, pd.Series(True, index=w.index))) for k, v in L.items()}


def forward_dd(daily, months):
    out = {}
    for m in months:
        seg = daily[daily.index > m].iloc[:HORIZON]
        if len(seg) < HORIZON * 0.95:
            out[m] = np.nan
            continue
        start = daily[daily.index <= m].iloc[-1]
        path = np.concatenate([[start], seg.values])
        peak = np.maximum.accumulate(path)
        out[m] = float((1 - path / peak).max())
    return pd.Series(out)


def build(n_perm=2000, seed=3):
    w = pd.read_csv(os.path.join(DATA, "timeline.csv"), parse_dates=["asof"], index_col="asof")
    daily = pd.read_csv(os.path.join(RAW, "gspc_daily.csv"), parse_dates=["date"]).set_index("date")["close"]
    lib = pd.read_csv(os.path.join(DATA, "crash_library.csv"), parse_dates=["peak_date"])
    big = lib[(lib.depth_pct <= -20) & (lib.peak_date >= START)]
    months = w.loc[START:].index
    fdd = forward_dd(daily, months)
    known = fdd.dropna().index
    y = (fdd[known] >= 0.20).astype(float)
    base = y.mean()
    rng = np.random.default_rng(seed)
    rows = []
    for name, (on, ok) in lights(w).items():
        mask = ok.loc[known].values
        o = on.loc[known].values[mask]
        yy = y.values[mask]
        if o.sum() < 6:
            continue
        hit = yy[o].mean()
        b = yy.mean()
        lift = hit / b if b > 0 else np.nan
        null = []
        for _ in range(n_perm):
            k = rng.integers(12, len(o) - 12)
            null.append(yy[np.roll(o, k)].mean() / b)
        p = float((np.array(null) >= lift).mean())
        caught = 0
        eligible = 0
        for pk in big.peak_date:
            win = on.loc[pk - pd.DateOffset(months=12): pk - pd.offsets.MonthEnd(1)]
            av = ok.loc[win.index]
            if av.any():
                eligible += 1
                caught += int(win[av].any())
        halves = []
        for a, bnd in (("1955", "1990"), ("1991", "2025")):
            idx = y.loc[a:bnd].index
            mm = ok.loc[idx]
            oo, yh = on.loc[idx][mm], y.loc[idx][mm]
            halves.append(round(float(yh[oo].mean() / yh.mean()), 2) if oo.sum() >= 6 and yh.mean() > 0 else None)
        rows.append({"light": name, "lift_1955_1990": halves[0], "lift_1991_2025": halves[1], "months_on_pct": round(o.mean() * 100, 1), "hit_rate_pct": round(hit * 100, 1),
                     "base_rate_pct": round(b * 100, 1), "lift": round(lift, 2), "p_value": round(p, 3),
                     "crashes_caught": f"{caught}/{eligible}", "since": str(known[mask][0].year),
                     "on_today": bool(on.iloc[-1]) if ok.iloc[-1] else None})
    t = pd.DataFrame(rows).sort_values("lift", ascending=False)
    t.to_csv(os.path.join(DATA, "early_warning.csv"), index=False)
    # composite of the lights that carried information in BOTH halves of the sample (1955-90 and 1991-2025)
    core = [r.light for r in t.itertuples() if r.since <= "1960" and (r.lift_1955_1990 or 0) > 1.3
            and (r.lift_1991_2025 or 0) > 1.3]
    L = lights(w)
    cnt = sum(L[k][0].astype(int) for k in core)
    cc = cnt.loc[known]
    comp = []
    for lo, hi in ((0, 0), (1, 1), (2, 99)):
        m = (cc >= lo) & (cc <= hi)
        comp.append({"lights_on": f"{lo}" if lo == hi else f"{lo}+", "months": int(m.sum()),
                     "hit_rate_pct": round(float(y[m].mean()) * 100, 1) if m.sum() else None})
    # honest out-of-sample check: pick lights using 1955-1990 only (lift > 1.3), test on 1991-2025
    pick = [r.light for r in t.itertuples() if r.since <= "1960" and (r.lift_1955_1990 or 0) > 1.3]
    c2 = sum(L[k][0].astype(int) for k in pick).loc[known].clip(upper=2)
    oos = {"lights_picked_on_1955_1990": pick, "today_count": int(sum(L[k][0].astype(int) for k in pick).iloc[-1])}
    for lab, a, bnd in (("in_sample_1955_1990", "1955", "1990"), ("out_of_sample_1991_2025", "1991", "2025")):
        cc2, yy2 = c2.loc[a:bnd], y.loc[a:bnd]
        oos[lab] = [{"lights_on": ("2+" if k == 2 else str(k)), "months": int((cc2 == k).sum()),
                     "hit_rate_pct": round(float(yy2[cc2 == k].mean()) * 100, 1)} for k in (0, 1, 2)]
    today = {"oos_test": oos, "asof": str(w.index[-1].date()), "base_rate_pct": round(base * 100, 1),
             "lights_on": [k for k, (on, ok) in L.items() if ok.iloc[-1] and on.iloc[-1]],
             "core_lights": core, "core_count_today": int(cnt.iloc[-1]), "composite": comp,
             "n_months": int(len(known)), "n_big_declines": int(len(big))}
    with open(os.path.join(DATA, "early_warning_today.json"), "w") as fh:
        json.dump(today, fh, indent=1)
    return t, today


if __name__ == "__main__":
    t, today = build()
    pd.set_option("display.width", 220)
    print(t.to_string(index=False))
    print(json.dumps(today, indent=1))
