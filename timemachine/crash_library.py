"""
crash_library.py - finds every S&P 500 decline of 10% or more and attaches its "fingerprint".

How a decline is defined
  - Start from a peak. If the index falls 10%+ below it, that's a drawdown episode.
  - The episode's trough is the lowest close before either
      (a) the index gets all the way back to the old peak (full recovery), or
      (b) it rallies 25%+ off the low (a new bull market has clearly started).
    Rule (b) keeps decades-long slumps like 1929-1954 from being one giant "event",
    while keeping the classic bear markets (2000-02, 2007-09) as single episodes.
  - Daily closes from 1950; Shiller monthly averages for 1928-1949 (understates intramonth drops).

Fingerprint: the point-in-time timeline values 12, 6 and 0 months before the peak month
(month-end strictly BEFORE the peak, so nothing from the decline itself leaks in).

Outputs: data/crash_library.csv, data/crash_fingerprints.csv
"""

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
RAW = os.path.join(DATA, "raw")

THRESH = 0.10
NEW_BULL = 0.25

FINGERPRINT_VARS = [
    "short_rate", "real_short_rate", "short_rate_chg_12m", "t10y", "t10y_chg_12m", "curve_10y_3m",
    "baa_aaa_spread", "baa_aaa_chg_12m", "baa_10y_spread", "cpi_yoy", "unemployment",
    "unemployment_chg_12m", "indpro_yoy", "cape", "sp500_ret_12m", "realized_vol", "oil_yoy",
    "home_price_yoy", "m2_yoy", "consumer_sentiment", "vix", "t10y_above_4", "t10y_above_5",
    "curve_inverted", "gdp_growth", "payrolls_yoy", "food_home_yoy", "lending_standards",
]


def price_series():
    d = pd.read_csv(os.path.join(RAW, "gspc_daily.csv"), parse_dates=["date"]).dropna(subset=["close"])
    d = d.set_index("date")["close"].sort_index()
    sh = pd.read_csv(os.path.join(RAW, "shiller_sp500.csv"), parse_dates=["Date"])
    sh = sh[(sh.SP500 > 0) & (sh.Date >= "1926-01-01") & (sh.Date < "1950-01-01")]
    m = pd.Series(sh.SP500.values, index=sh.Date + pd.offsets.MonthEnd(0))
    return pd.concat([m, d]), d.index.min()


def find_episodes(p):
    vals, idx = p.values, p.index
    episodes = []
    peak_i = 0
    i = 1
    n = len(vals)
    while i < n:
        if vals[i] >= vals[peak_i]:
            peak_i = i
            i += 1
            continue
        if vals[i] <= vals[peak_i] * (1 - THRESH):
            # inside a drawdown: find trough until recovery or new bull
            trough_i = i
            j = i
            end_reason = "open"
            while j < n:
                if vals[j] < vals[trough_i]:
                    trough_i = j
                if vals[j] >= vals[peak_i]:
                    end_reason = "recovered"
                    break
                if vals[j] >= vals[trough_i] * (1 + NEW_BULL):
                    end_reason = "new_bull"
                    break
                j += 1
            episodes.append((peak_i, trough_i, j if j < n else None, end_reason))
            # restart: new peak search from the end point
            if j >= n:
                break
            peak_i = j
            i = j + 1
            continue
        i += 1
    return episodes


def build():
    p, daily_start = price_series()
    w = pd.read_csv(os.path.join(DATA, "timeline.csv"), parse_dates=["asof"], index_col="asof")
    eps = find_episodes(p)
    rows, fps = [], []
    for k, (pi, ti, ei, why) in enumerate(eps, 1):
        pk_d, tr_d = p.index[pi], p.index[ti]
        pk, tr = p.iloc[pi], p.iloc[ti]
        after = p.iloc[ti:]
        rec = after[after >= pk]
        rec_d = rec.index[0] if len(rec) else None
        ath_before = p.iloc[: pi + 1].max()
        rec_window = w.loc[pk_d - pd.offsets.MonthEnd(1): tr_d + pd.DateOffset(months=6), "recession_hindsight"]
        row = {
            "id": k,
            "peak_date": pk_d.date(), "trough_date": tr_d.date(),
            "recovery_date": rec_d.date() if rec_d is not None else None,
            "peak": round(pk, 2), "trough": round(tr, 2),
            "depth_pct": round((tr / pk - 1) * 100, 1),
            "days_to_trough": (tr_d - pk_d).days,
            "days_to_recover": (rec_d - tr_d).days if rec_d is not None else None,
            "ended_by": why,
            "from_record_high": bool(pk >= ath_before - 1e-9),
            "recession_followed": bool(rec_window.max() == 1) if len(rec_window) else None,
            "data": "daily" if pk_d >= daily_start else "monthly avg",
        }
        d = row["depth_pct"]
        row["severity"] = ("Correction (10-20%)" if d > -20 else "Bear (20-30%)" if d > -30
                           else "Severe bear (30-50%)" if d > -50 else "Crash (50%+)")
        rows.append(row)
        # fingerprints at the last month-end strictly before the peak
        m0 = (pk_d - pd.offsets.MonthEnd(1)) if pk_d != pk_d + pd.offsets.MonthEnd(0) else pk_d - pd.offsets.MonthEnd(1)
        m0 = pk_d.to_period("M").to_timestamp("M") - pd.offsets.MonthEnd(1)
        for lag in (0, 6, 12):
            asof = m0 - pd.offsets.MonthEnd(lag) if lag else m0
            if asof in w.index:
                fp = {"id": k, "lag_months": lag, "asof": asof.date()}
                fp.update({v: w.loc[asof, v] for v in FINGERPRINT_VARS if v in w.columns})
                fps.append(fp)
    lib = pd.DataFrame(rows)
    fp = pd.DataFrame(fps)
    lib.to_csv(os.path.join(DATA, "crash_library.csv"), index=False)
    fp.to_csv(os.path.join(DATA, "crash_fingerprints.csv"), index=False)
    return lib, fp


if __name__ == "__main__":
    lib, fp = build()
    pd.set_option("display.width", 200)
    print(lib[["id", "peak_date", "trough_date", "depth_pct", "days_to_trough", "days_to_recover", "ended_by",
               "from_record_high", "recession_followed", "data"]].to_string(index=False))
    print(len(lib), "episodes")
