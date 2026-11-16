"""
build_timeline.py - builds the point-in-time ("no hindsight") monthly timeline.

For every month-end from 1928 to today, each indicator holds the value that was
actually PUBLISHED by that date:

  method = "first_release"  -> the number as first published (FRED/ALFRED initial release),
                               available from its real release date
  method = "revised_lagged" -> no vintage archive exists for that era, so we use today's
                               (revised) number but only make it available after the
                               normal publication delay. Flagged so you know.
  method = "market"         -> market prices/yields (not revised); known at month end

Outputs (data/):
  timeline.csv        wide table: one row per month-end, one column per indicator
  timeline_meta.csv   for every month x indicator: which period the number refers to,
                      when it was published, and which method was used
  indicators.json     labels, groups, units, descriptions for the app
"""

import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data", "raw")
OUT = os.path.join(HERE, "data")
START, END = "1928-01-31", None  # END = last complete month

# ----------------------------------------------------------------------------------- loaders

def cur(sid):
    df = pd.read_csv(os.path.join(RAW, f"{sid}.csv"), parse_dates=["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna()


def first(sid):
    p = os.path.join(RAW, f"{sid}_first.csv")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p, parse_dates=["date", "release_date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna()
    # A "first release" more than ~400 days after the period isn't a real first print
    # (it's the oldest vintage FRED archived), so we don't trust it as point-in-time.
    return df[(df.release_date - df.date).dt.days < 400]


def month_end(d):
    return d + pd.offsets.MonthEnd(0)


def pit_macro(sid, lag_days, freq="m"):
    """Point-in-time table for a revised macro series: first releases where they exist,
    revised values + publication lag before that."""
    c = cur(sid)
    f = first(sid)
    rows = []
    cutoff = f.date.min() if f is not None and len(f) else None
    for _, r in c.iterrows():
        if cutoff is not None and r.date >= cutoff:
            continue
        rows.append((r.date, r.value, r.date + pd.Timedelta(days=lag_days), "revised_lagged"))
    if f is not None:
        for _, r in f.iterrows():
            rows.append((r.date, r.value, r.release_date, "first_release"))
    t = pd.DataFrame(rows, columns=["obs_date", "value", "available", "method"])
    return t.sort_values("obs_date").reset_index(drop=True)


def pit_market(df, date_col="date", value_col="value", monthly_avg=True):
    """Market data isn't revised. A monthly average for month M is known at the end of M."""
    t = pd.DataFrame({"obs_date": df[date_col], "value": df[value_col]})
    t["available"] = t.obs_date.map(month_end) if monthly_avg else t.obs_date
    t["method"] = "market"
    return t.dropna().sort_values("obs_date").reset_index(drop=True)


def yoy(t, periods=12):
    """Year-over-year % change on a point-in-time table (both points as published)."""
    t = t.sort_values("obs_date").reset_index(drop=True).copy()
    # match on calendar month, not exact date (month-end trading days differ year to year)
    key = t.obs_date.dt.to_period("M")
    prev = pd.Series(t["value"].values, index=key).groupby(level=0).last()
    base = (key - periods).map(prev)
    out = t.copy()
    out["value"] = (t["value"] / base.values - 1) * 100
    return out.dropna(subset=["value"])


def change(t, periods=12):
    t = t.sort_values("obs_date").reset_index(drop=True).copy()
    key = t.obs_date.dt.to_period("M")
    prev = pd.Series(t["value"].values, index=key).groupby(level=0).last()
    base = (key - periods).map(prev)
    out = t.copy()
    out["value"] = t["value"] - base.values
    return out.dropna(subset=["value"])


def splice(*tables):
    """Use the first table where it has data, fall back to later ones for earlier dates."""
    out = None
    for t in tables:
        if t is None or t.empty:
            continue
        if out is None:
            out = t
        else:
            out = pd.concat([out, t[t.obs_date < out.obs_date.min()]])
    return out.sort_values("obs_date").reset_index(drop=True)


# ----------------------------------------------------------------------------------- S&P 500

def sp500_tables():
    """Month-end S&P 500 close (daily data from 1950; Shiller monthly averages before)."""
    d = pd.read_csv(os.path.join(RAW, "gspc_daily.csv"), parse_dates=["date"]).dropna(subset=["close"])
    d = d.sort_values("date")
    me = d.groupby(d.date.dt.to_period("M")).last()
    daily_m = pd.DataFrame({"obs_date": me.date.values, "value": me.close.values})
    daily_m["available"] = daily_m.obs_date
    daily_m["method"] = "market"

    sh = pd.read_csv(os.path.join(RAW, "shiller_sp500.csv"), parse_dates=["Date"])
    sh = sh[(sh.SP500 > 0) & (sh.Date < "1950-01-01")]
    shm = pd.DataFrame({"obs_date": sh.Date.map(month_end), "value": sh.SP500})
    shm["available"] = shm.obs_date
    shm["method"] = "market_monthly_avg"

    level = splice(daily_m, shm)

    # realized volatility: daily (63 trading days) from 1950; monthly returns x sqrt(12) before
    d["ret"] = d.close.pct_change()
    d["vol3m"] = d.ret.rolling(63).std() * np.sqrt(252) * 100
    vm = d.groupby(d.date.dt.to_period("M")).last()
    vol_d = pd.DataFrame({"obs_date": vm.date.values, "value": vm.vol3m.values, "available": vm.date.values,
                          "method": "market"}).dropna()
    shr = shm.copy()
    shr["value"] = shm.value.pct_change().rolling(12).std() * np.sqrt(12) * 100
    vol = splice(vol_d, shr.dropna())

    # drawdown from all-time high (using daily closes where available)
    d["ath"] = d.close.cummax()
    d["dd"] = (d.close / d.ath - 1) * 100
    dm = d.groupby(d.date.dt.to_period("M")).last()
    dd_d = pd.DataFrame({"obs_date": dm.date.values, "value": dm.dd.values, "available": dm.date.values,
                         "method": "market"})
    # pre-1950: ATH from Shiller monthly back to 1871
    sh_all = pd.read_csv(os.path.join(RAW, "shiller_sp500.csv"), parse_dates=["Date"])
    sh_all = sh_all[sh_all.SP500 > 0]
    sh_all["dd"] = (sh_all.SP500 / sh_all.SP500.cummax() - 1) * 100
    pre = sh_all[sh_all.Date < "1950-01-01"]
    dd_m = pd.DataFrame({"obs_date": pre.Date.map(month_end), "value": pre.dd, "available": pre.Date.map(month_end),
                         "method": "market_monthly_avg"})
    # daily ATH must also respect pre-1950 highs (the 1929 peak wasn't regained until 1954)
    ath_1929 = sh_all[sh_all.Date < "1950-01-01"].SP500.max()
    dd_d["value"] = [min(v, (c / max(a, ath_1929) - 1) * 100) for v, c, a in
                     zip(dd_d.value, dm.close.values, dm.ath.values)]
    dd = splice(dd_d, dd_m)
    return level, vol, dd


# ----------------------------------------------------------------------------------- NBER

# Recession dates and the day NBER ANNOUNCED them. Before 1980 NBER did not announce in real time,
# so for point-in-time purposes those recessions were not "officially known" when they happened.
NBER = [  # peak, trough, peak_announced, trough_announced
    ("1926-10", "1927-11", None, None), ("1929-08", "1933-03", None, None),
    ("1937-05", "1938-06", None, None), ("1945-02", "1945-10", None, None),
    ("1948-11", "1949-10", None, None), ("1953-07", "1954-05", None, None),
    ("1957-08", "1958-04", None, None), ("1960-04", "1961-02", None, None),
    ("1969-12", "1970-11", None, None), ("1973-11", "1975-03", None, None),
    ("1980-01", "1980-07", "1980-06-03", "1981-07-08"), ("1981-07", "1982-11", "1982-01-06", "1983-07-08"),
    ("1990-07", "1991-03", "1991-04-25", "1992-12-22"), ("2001-03", "2001-11", "2001-11-26", "2003-07-17"),
    ("2007-12", "2009-06", "2008-12-01", "2010-09-20"), ("2020-02", "2020-04", "2020-06-08", "2021-07-19"),
]


def nber_tables(months):
    """recession_known: 1 if, as of that month-end, NBER had announced we were in a recession
    (peak announced, trough not yet announced). in_recession_hindsight: actual dates (for outcomes only)."""
    known, hind = [], []
    for m in months:
        k = 0
        h = 0
        for pk, tr, pa, ta in NBER:
            p, t = pd.Period(pk), pd.Period(tr)
            if p < m.to_period("M") <= t:
                h = 1
            if pa is not None and pd.Timestamp(pa) <= m and (ta is None or pd.Timestamp(ta) > m):
                k = 1
        known.append(k)
        hind.append(h)
    return known, hind


# ----------------------------------------------------------------------------------- build

def build():
    T = {}  # key -> point-in-time table

    # Rates & bonds (market data)
    ff = pit_market(cur("FEDFUNDS"))
    tb = pit_market(cur("TB3MS"))
    cp = pit_market(cur("M13002US35620M156NNBR"))
    T["fed_funds"] = ff
    T["short_rate"] = splice(ff, tb, cp)
    T["tbill_3m"] = splice(tb, cp)
    sh = pd.read_csv(os.path.join(RAW, "shiller_sp500.csv"), parse_dates=["Date"])
    sh_lr = pit_market(sh[sh["Long Interest Rate"] > 0].rename(columns={"Date": "date", "Long Interest Rate": "value"}))
    T["t10y"] = splice(pit_market(cur("GS10")), sh_lr)
    T["t2y"] = pit_market(cur("GS2"))
    T["mortgage_30y"] = pit_market(cur("MORTGAGE30US"))

    # Credit
    baa, aaa = pit_market(cur("BAA")), pit_market(cur("AAA"))
    T["baa"], T["aaa"] = baa, aaa
    T["hy_spread"] = pit_market(cur("BAMLH0A0HYM2"))
    T["ted_spread"] = pit_market(cur("TEDRATE"))
    lend = cur("DRTSCILM")
    T["lending_standards"] = pd.DataFrame({"obs_date": lend.date, "value": lend.value,
                                           "available": lend.date + pd.Timedelta(days=35), "method": "market"})

    # Inflation & cost of living
    cpi = pit_macro("CPIAUCNS", 47)
    T["cpi_yoy"] = yoy(cpi)
    T["core_cpi_yoy"] = yoy(pit_macro("CPILFENS", 47))
    T["food_home_yoy"] = yoy(pit_macro("CUUR0000SAF11", 47))
    T["gasoline_yoy"] = yoy(pit_macro("CUUR0000SETB01", 47))
    oil = pit_market(cur("WTISPLC"))
    T["oil_price"] = oil
    T["oil_yoy"] = yoy(oil)

    # Housing
    cs = cur("CSUSHPINSA")
    cs_t = pd.DataFrame({"obs_date": cs.date, "value": cs.value, "available": cs.date + pd.Timedelta(days=88),
                         "method": "revised_lagged"})
    T["home_price_yoy"] = yoy(cs_t)
    T["housing_starts"] = pit_macro("HOUST", 48)

    # Economy
    un = pit_macro("UNRATE", 35)
    T["unemployment"] = un
    T["unemployment_chg_12m"] = change(un)
    pay = pit_macro("PAYEMS", 35)
    T["payrolls_yoy"] = yoy(pay)
    gdp = pit_macro("A191RL1Q225SBEA", 120, freq="q")
    T["gdp_growth"] = gdp
    T["indpro_yoy"] = yoy(pit_macro("INDPRO", 46))
    um = cur("UMCSENT")
    T["consumer_sentiment"] = pd.DataFrame({"obs_date": um.date, "value": um.value,
                                            "available": um.date.map(month_end), "method": "market"})

    # Markets
    lvl, vol, dd = sp500_tables()
    T["sp500"] = lvl
    T["sp500_ret_12m"] = yoy(lvl)
    T["sp500_ret_6m"] = yoy(lvl, 6)
    T["realized_vol"] = vol
    T["drawdown_from_ath"] = dd
    vix = splice(pit_market(cur("VIXCLS")), pit_market(cur("VXOCLS")))
    T["vix"] = vix
    cape = pd.read_csv(os.path.join(RAW, "cape_multpl.csv"), parse_dates=["date"])
    cape = cape[cape.date.dt.day == 1]
    T["cape"] = pit_market(cape.rename(columns={"cape": "value"}))

    # Leverage & banks (added): FINRA margin debt (1997+, published ~3 weeks after month-end)
    mp = os.path.join(RAW, "finra_margin_debt.csv")
    if os.path.exists(mp):
        md = pd.read_csv(mp)
        md["date"] = pd.to_datetime(md.month)
        mt = pd.DataFrame({"obs_date": md.date, "value": md.margin_debt_musd / 1000.0,
                           "available": md.date.map(month_end) + pd.Timedelta(days=25), "method": "market"})
        T["margin_debt"] = mt
        T["margin_debt_yoy"] = yoy(mt)
    # FDIC bank failures (1934+): known the day they happen; trailing 12-month count and failed assets
    fp = os.path.join(RAW, "fdic_failures_monthly.csv")
    if os.path.exists(fp):
        fd = pd.read_csv(fp)
        fd["date"] = pd.to_datetime(fd.month)
        allm = pd.DataFrame({"date": pd.date_range("1934-01-01", pd.Timestamp.today(), freq="MS")})
        fd = allm.merge(fd, on="date", how="left").fillna({"n": 0, "assets_bn": 0})
        for key, col in (("bank_failures_12m", "n"), ("bank_failed_assets_12m", "assets_bn")):
            T[key] = pd.DataFrame({"obs_date": fd.date, "value": fd[col].rolling(12, min_periods=1).sum(),
                                   "available": fd.date.map(month_end), "method": "market"})

    # Money & dollar
    T["m2_yoy"] = yoy(pit_macro("M2SL", 43))
    d_new = pit_market(cur("DTWEXBGS"))
    d_old = pit_market(cur("TWEXMMTH"))
    T["dollar_yoy"] = splice(yoy(d_new), yoy(d_old))

    # ------------------------------------------------------------- as-of panel
    last_sp = T["sp500"].obs_date.max()
    months = pd.date_range(START, month_end(last_sp), freq="ME")
    wide = pd.DataFrame(index=months)
    meta = []
    for key, t in T.items():
        t = t.sort_values("available")
        m = pd.merge_asof(pd.DataFrame({"asof": months}), t, left_on="asof", right_on="available",
                          direction="backward")
        # if the most recent available observation is stale (>15 months old), treat as missing
        stale = (m["asof"] - m["obs_date"]).dt.days > 460
        m.loc[stale, ["value", "obs_date", "available", "method"]] = np.nan
        wide[key] = m["value"].values
        mm = m[["asof", "obs_date", "available", "method"]].copy()
        mm["indicator"] = key
        meta.append(mm.dropna(subset=["obs_date"]))

    # Latest partial month: use the most recent daily readings (known as of today)
    lp = os.path.join(RAW, "latest_daily.csv")
    if os.path.exists(lp):
        latest = pd.read_csv(lp, parse_dates=["obs_date"])
        last = wide.index.max()
        rows = []
        for _, r in latest.iterrows():
            if r.indicator in wide.columns and r.obs_date <= last:
                wide.loc[last, r.indicator] = r.value
                rows.append({"asof": last, "obs_date": r.obs_date, "available": r.obs_date,
                             "method": "market_daily_latest", "indicator": r.indicator})
        meta.append(pd.DataFrame(rows))

    # derived columns (computed from point-in-time inputs, so still no hindsight)
    wide["curve_10y_3m"] = wide.t10y - wide.tbill_3m
    wide["curve_10y_2y"] = wide.t10y - wide.t2y
    wide["baa_aaa_spread"] = wide.baa - wide.aaa
    wide["baa_10y_spread"] = wide.baa - wide.t10y
    wide["real_short_rate"] = wide.short_rate - wide.cpi_yoy
    wide["short_rate_chg_12m"] = wide.short_rate - wide.short_rate.shift(12)
    wide["t10y_chg_12m"] = wide.t10y - wide.t10y.shift(12)
    wide["baa_aaa_chg_12m"] = wide.baa_aaa_spread - wide.baa_aaa_spread.shift(12)
    wide["t10y_above_4"] = (wide.t10y > 4).astype(float).where(wide.t10y.notna())
    wide["t10y_above_5"] = (wide.t10y > 5).astype(float).where(wide.t10y.notna())
    wide["curve_inverted"] = (wide.curve_10y_3m < 0).astype(float).where(wide.curve_10y_3m.notna())
    known, hind = nber_tables(wide.index)
    wide["recession_announced"] = known
    wide["recession_hindsight"] = hind   # outcome only - never used as an input

    wide.index.name = "asof"
    wide.round(4).to_csv(os.path.join(OUT, "timeline.csv"))
    meta = pd.concat(meta)
    meta.to_csv(os.path.join(OUT, "timeline_meta.csv"), index=False)
    with open(os.path.join(OUT, "indicators.json"), "w") as f:
        json.dump(INDICATORS, f, indent=1)
    return wide, meta


# label, group, unit, plain-English description, "higher is riskier?" (for colouring)
INDICATORS = {
    "sp500": ["S&P 500", "Markets", "index", "Month-end close of the S&P 500 (monthly averages before 1950).", None],
    "sp500_ret_12m": ["S&P 500, 12-month return", "Markets", "%", "Price change over the last year.", None],
    "sp500_ret_6m": ["S&P 500, 6-month return", "Markets", "%", "Price change over the last 6 months.", None],
    "drawdown_from_ath": ["Drop from all-time high", "Markets", "%", "How far below its record high the S&P is.", None],
    "realized_vol": ["Realized volatility", "Markets", "%", "How much the S&P has actually been swinging (annualized).", True],
    "vix": ["VIX (fear gauge)", "Markets", "index", "Options-implied volatility; VXO before 1990. Starts 1986.", True],
    "cape": ["Shiller CAPE", "Markets", "x", "Price / 10-year average inflation-adjusted earnings. ~17 is the long-run average.", True],
    "short_rate": ["Short-term rate", "Rates & bonds", "%", "Fed funds rate (T-bill before 1954, commercial paper before 1934).", None],
    "fed_funds": ["Fed funds rate", "Rates & bonds", "%", "The Fed's main policy rate (from 1954).", None],
    "short_rate_chg_12m": ["Fed tightening (12m change)", "Rates & bonds", "pp", "How much short rates moved in the last year. Big positive = Fed hiking.", True],
    "real_short_rate": ["Real short rate", "Rates & bonds", "%", "Short rate minus inflation. Positive = money is genuinely tight.", True],
    "tbill_3m": ["3-month T-bill", "Rates & bonds", "%", "Yield on 3-month Treasury bills.", None],
    "t2y": ["2-year Treasury", "Rates & bonds", "%", "From 1976.", None],
    "t10y": ["10-year Treasury", "Rates & bonds", "%", "Long-term government borrowing cost (Shiller long rate before 1953).", None],
    "t10y_chg_12m": ["10-year change (12m)", "Rates & bonds", "pp", "How much long rates rose in a year.", True],
    "t10y_above_4": ["10-year above 4%?", "Rates & bonds", "flag", "1 = yes.", None],
    "t10y_above_5": ["10-year above 5%?", "Rates & bonds", "flag", "1 = yes.", None],
    "curve_10y_3m": ["Yield curve (10y - 3m)", "Rates & bonds", "pp", "Negative = inverted, a classic recession warning.", False],
    "curve_10y_2y": ["Yield curve (10y - 2y)", "Rates & bonds", "pp", "From 1976. Negative = inverted.", False],
    "curve_inverted": ["Curve inverted?", "Rates & bonds", "flag", "1 = 10y below 3m.", None],
    "mortgage_30y": ["30-year mortgage rate", "Rates & bonds", "%", "From 1971.", None],
    "baa_aaa_spread": ["Credit spread (BAA - AAA)", "Credit", "pp", "Extra yield on riskier corporate bonds. Widening = lenders getting nervous. Back to 1919.", True],
    "baa_aaa_chg_12m": ["Credit spread change (12m)", "Credit", "pp", "Widening over the last year.", True],
    "baa_10y_spread": ["BAA - 10y Treasury", "Credit", "pp", "Corporate borrowing premium over Treasuries.", True],
    "baa": ["BAA corporate yield", "Credit", "%", "What medium-quality companies pay to borrow.", None],
    "aaa": ["AAA corporate yield", "Credit", "%", "What top-quality companies pay to borrow.", None],
    "hy_spread": ["High-yield spread", "Credit", "pp", "Junk-bond premium (FRED only provides the last ~3 years).", True],
    "ted_spread": ["TED spread", "Credit", "pp", "Bank funding stress, 1986-2021.", True],
    "lending_standards": ["Banks tightening loans", "Credit", "% net", "Net % of banks tightening business-loan standards (from 1990).", True],
    "cpi_yoy": ["Inflation (CPI)", "Inflation & prices", "%", "Consumer prices vs a year ago.", True],
    "core_cpi_yoy": ["Core inflation", "Inflation & prices", "%", "CPI excluding food and energy (from 1958).", True],
    "food_home_yoy": ["Grocery inflation", "Inflation & prices", "%", "Food-at-home CPI vs a year ago.", True],
    "gasoline_yoy": ["Gasoline price change", "Inflation & prices", "%", "Gasoline CPI vs a year ago.", True],
    "oil_price": ["Oil price (WTI)", "Inflation & prices", "$/bbl", "From 1946.", None],
    "oil_yoy": ["Oil price change", "Inflation & prices", "%", "Oil vs a year ago.", True],
    "home_price_yoy": ["Home prices", "Housing", "%", "Case-Shiller national index vs a year ago (from 1988).", None],
    "housing_starts": ["Housing starts", "Housing", "k/yr", "New homes started, annualized (from 1959).", None],
    "mortgage_30y_dup": None,
    "unemployment": ["Unemployment rate", "Economy", "%", "From 1948.", True],
    "unemployment_chg_12m": ["Unemployment change (12m)", "Economy", "pp", "Rising unemployment is how recessions show up.", True],
    "payrolls_yoy": ["Job growth", "Economy", "%", "Payroll jobs vs a year ago.", False],
    "gdp_growth": ["Real GDP growth", "Economy", "% ann.", "Latest quarter, annualized (from 1947).", False],
    "indpro_yoy": ["Industrial production", "Economy", "%", "Factory/mine/utility output vs a year ago (from 1920).", False],
    "consumer_sentiment": ["Consumer sentiment", "Economy", "index", "University of Michigan survey (from 1952).", False],
    "recession_announced": ["Recession officially announced?", "Economy", "flag", "NBER had declared a recession by this date (real-time announcements began in 1980).", True],
    "margin_debt": ["Margin debt", "Leverage & banks", "$bn", "Money investors borrowed against stocks (FINRA, from 1997).", None],
    "margin_debt_yoy": ["Margin debt growth", "Leverage & banks", "%", "Margin debt vs a year ago. Fast growth = leveraged speculation (FINRA, from 1998).", True],
    "bank_failures_12m": ["Bank failures (12m)", "Leverage & banks", "count", "FDIC-insured banks failed or assisted in the last 12 months (from 1934).", True],
    "bank_failed_assets_12m": ["Failed bank assets (12m)", "Leverage & banks", "$bn", "Total assets of banks that failed in the last 12 months (nominal dollars, from 1934).", True],
    "m2_yoy": ["Money supply growth (M2)", "Money & dollar", "%", "From 1960.", None],
    "dollar_yoy": ["US dollar change", "Money & dollar", "%", "Trade-weighted dollar vs a year ago (from 1974).", None],
}
INDICATORS = {k: v for k, v in INDICATORS.items() if v}


if __name__ == "__main__":
    w, m = build()
    print(w.shape, w.index.min().date(), w.index.max().date())
    cov = w.notna().mean().sort_values()
    print(cov.round(2).to_string())
