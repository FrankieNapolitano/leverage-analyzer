"""
update_data.py - refreshes the Time Machine's raw data from your own laptop, then rebuilds everything.

Run it with update_timemachine.bat (Windows) or update_timemachine_mac.command (Mac).

What it does
  1. FRED current values for every series (free, no key needed).
  2. FRED first-release ("as first published") values, IF you set a free FRED API key:
       get one at https://fredaccount.stlouisfed.org/apikeys  then put it in a file called
       fred_key.txt next to this script (one line). Without a key, months newer than the
       saved first-release files fall back to "revised data, delayed to its release date" and
       are flagged that way in the app.
  3. Latest daily readings (10-yr, 2-yr, 3-month, fed funds, BAA, AAA, mortgage, oil).
  4. S&P 500 daily closes (Yahoo via yfinance), sector ETFs, and Shiller CAPE (multpl.com).
  5. EXTRA data the cloud build couldn't reach:
       - S&P 500 DAILY closes for 1928-1949 (saved separately; see crash_library_prewar_daily.csv)
       - Ken French 49-industry returns back to 1926 -> the full-history version of the thesis test
  6. Reruns the pipeline (run_all.py).

Every download is wrapped so one failure doesn't stop the rest; a summary prints at the end.
"""

import io
import os
import re
import sys
import time
import zipfile

import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data", "raw")
os.makedirs(RAW, exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (LeverageAnalyzer TimeMachine updater)"}
LOG = []

MONTHLY = ["CPIAUCNS", "CPILFENS", "CUUR0000SAF11", "CUUR0000SETB01", "UNRATE", "PAYEMS", "A191RL1Q225SBEA",
           "INDPRO", "HOUST", "M2SL", "FEDFUNDS", "TB3MS", "GS10", "GS2", "GS1", "MORTGAGE30US", "BAA", "AAA",
           "BAMLH0A0HYM2", "TEDRATE", "DRTSCILM", "VIXCLS", "VXOCLS", "WTISPLC", "CSUSHPINSA", "UMCSENT",
           "DTWEXBGS", "TWEXMMTH", "SP500"]
FIRST_RELEASE = ["CPIAUCNS", "CPILFENS", "CUUR0000SAF11", "CUUR0000SETB01", "UNRATE", "PAYEMS",
                 "A191RL1Q225SBEA", "INDPRO", "HOUST", "M2SL"]
DAILY_LATEST = {"DGS10": ["t10y"], "DGS2": ["t2y"], "DTB3": ["tbill_3m"], "DFF": ["fed_funds", "short_rate"],
                "DBAA": ["baa"], "DAAA": ["aaa"], "MORTGAGE30US": ["mortgage_30y"], "DCOILWTICO": ["oil_price"]}
ETFS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]


def step(name, fn):
    t = time.time()
    try:
        msg = fn()
        LOG.append(f"OK    {name}: {msg or ''} ({time.time() - t:.0f}s)")
    except Exception as e:  # keep going
        LOG.append(f"FAIL  {name}: {type(e).__name__}: {str(e)[:160]}")
    print(LOG[-1], flush=True)


def get(url, **kw):
    r = requests.get(url, headers=UA, timeout=60, **kw)
    r.raise_for_status()
    return r


def fred_csv(sid):
    txt = get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}").text
    df = pd.read_csv(io.StringIO(txt))
    df.columns = ["date", "value"]
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna()


def fred_current():
    ok = 0
    for sid in MONTHLY:
        try:
            fred_csv(sid).to_csv(os.path.join(RAW, f"{sid}.csv"), index=False)
            ok += 1
        except Exception as e:
            print(f"   {sid}: {e}")
    if ok == 0:
        raise ConnectionError("could not reach FRED; kept the old files")
    return f"{ok}/{len(MONTHLY)} series"


def fred_key():
    p = os.path.join(HERE, "fred_key.txt")
    if os.environ.get("FRED_API_KEY"):
        return os.environ["FRED_API_KEY"].strip()
    return open(p).read().strip() if os.path.exists(p) else None


def fred_first():
    key = fred_key()
    if not key:
        return "skipped (no fred_key.txt) - first-release files kept as they were"
    ok = 0
    for sid in FIRST_RELEASE:
        url = ("https://api.stlouisfed.org/fred/series/observations"
               f"?series_id={sid}&api_key={key}&file_type=json&output_type=4"
               "&realtime_start=1776-07-04&realtime_end=9999-12-31")
        obs = get(url).json()["observations"]
        df = pd.DataFrame(obs)[["date", "value", "realtime_start"]].rename(columns={"realtime_start": "release_date"})
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df.dropna().to_csv(os.path.join(RAW, f"{sid}_first.csv"), index=False)
        ok += 1
        time.sleep(0.6)
    return f"{ok}/{len(FIRST_RELEASE)} series"


def latest_daily():
    rows = []
    for sid, keys in DAILY_LATEST.items():
        df = fred_csv(sid)
        last = df.iloc[-1]
        for k in keys:
            rows.append({"indicator": k, "value": last.value, "obs_date": last.date, "source": f"FRED {sid}"})
    pd.DataFrame(rows).to_csv(os.path.join(RAW, "latest_daily.csv"), index=False)
    return ", ".join(f"{r['indicator']}={r['value']}" for r in rows)


def yahoo():
    import yfinance as yf
    sp = yf.download("^GSPC", start="1927-12-01", auto_adjust=False, progress=False)
    if isinstance(sp.columns, pd.MultiIndex):
        sp.columns = sp.columns.get_level_values(0)
    sp = sp[["Close", "High", "Low"]].dropna().rename(columns=str.lower)
    if len(sp) < 5000:
        raise ValueError(f"S&P download looked incomplete ({len(sp)} rows); kept the old file")
    sp.index.name = "date"
    old = pd.read_csv(os.path.join(RAW, "gspc_daily.csv"), parse_dates=["date"]).set_index("date")
    new = sp[sp.index >= "1950-01-01"]
    merged = pd.concat([old[old.index < new.index.min()], new]) if len(new) > 1000 else old
    merged.round(4).to_csv(os.path.join(RAW, "gspc_daily.csv"))
    pre = sp[sp.index < "1950-01-01"]
    if len(pre):
        pre.round(4).to_csv(os.path.join(RAW, "gspc_daily_1928_1949.csv"))
    et = yf.download(ETFS, start="1998-12-01", interval="1wk", auto_adjust=True, progress=False)["Close"]
    et.index.name = "date"
    if len(et.dropna(how="all")) < 500:
        raise ValueError("sector ETF download looked incomplete; kept the old file")
    et.dropna(how="all").to_csv(os.path.join(RAW, "sector_etfs_weekly.csv"))
    return f"S&P daily through {merged.index.max():%Y-%m-%d}; pre-1950 daily rows: {len(pre)}"


def cape():
    html = get("https://www.multpl.com/shiller-pe/table/by-month").text
    rows = re.findall(r"<td>([A-Z][a-z]{2} \d{1,2}, \d{4})</td>\s*<td>\s*(?:&#x2002;)?\s*([\d.]+)", html)
    if len(rows) < 1000:
        raise ValueError(f"only parsed {len(rows)} rows")
    df = pd.DataFrame(rows, columns=["date", "cape"])
    df["date"] = pd.to_datetime(df.date)
    df["cape"] = df.cape.astype(float)
    df.sort_values("date").to_csv(os.path.join(RAW, "cape_multpl.csv"), index=False)
    return f"{len(df)} months"


def french_block(url):
    z = zipfile.ZipFile(io.BytesIO(get(url).content))
    lines = z.read(z.namelist()[0]).decode("latin-1").splitlines()
    start = next(i for i, l in enumerate(lines) if re.match(r"^\s*,", l))
    block = []
    for l in lines[start:]:
        if not l.strip():
            break
        block.append(l)
    df = pd.read_csv(io.StringIO("\n".join(block)))
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={df.columns[0]: "month"})
    df["month"] = pd.to_datetime(df.month.astype(str).str.strip(), format="%Y%m") + pd.offsets.MonthEnd(0)
    return df.set_index("month").apply(pd.to_numeric, errors="coerce").where(lambda x: x > -99)


def french_inv():
    df = french_block("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Portfolios_Formed_on_INV_CSV.zip")
    df.to_csv(os.path.join(RAW, "french_inv_monthly.csv"))
    return f"asset-growth portfolios {df.index.min():%Y-%m} to {df.index.max():%Y-%m}"


def french():
    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/49_Industry_Portfolios_CSV.zip"
    z = zipfile.ZipFile(io.BytesIO(get(url).content))
    txt = z.read(z.namelist()[0]).decode("latin-1")
    lines = txt.splitlines()
    # first block = monthly value-weighted returns
    start = next(i for i, l in enumerate(lines) if re.match(r"^\s*,", l))
    block = []
    for l in lines[start:]:
        if not l.strip():
            break
        block.append(l)
    df = pd.read_csv(io.StringIO("\n".join(block)))
    df = df.rename(columns={df.columns[0]: "month"})
    df.columns = [c.strip() for c in df.columns]
    df["month"] = pd.to_datetime(df.month.astype(str).str.strip(), format="%Y%m") + pd.offsets.MonthEnd(0)
    df = df.set_index("month").apply(pd.to_numeric, errors="coerce").where(lambda x: x > -99)
    df.to_csv(os.path.join(RAW, "french_49_industries_monthly.csv"))
    return f"{df.shape[1]} industries, {df.index.min():%Y-%m} to {df.index.max():%Y-%m}"


def fdic():
    js = get("https://api.fdic.gov/banks/failures?limit=10000&fields=FAILDATE,QBFASSET&format=json").json()
    f = pd.DataFrame([x["data"] for x in js["data"]])
    f["date"] = pd.to_datetime(f.FAILDATE)
    m = f.groupby(f.date.dt.to_period("M")).agg(n=("date", "size"), assets_bn=("QBFASSET", "sum")).reset_index()
    m["month"] = m.date.dt.strftime("%Y-%m")
    m["assets_bn"] = (m.assets_bn / 1e6).round(3)
    m[["month", "n", "assets_bn"]].to_csv(os.path.join(RAW, "fdic_failures_monthly.csv"), index=False)
    return f"{len(f)} failures through {f.date.max():%Y-%m}"


def finra_margin():
    html = get("https://www.finra.org/rules-guidance/key-topics/margin-accounts/margin-statistics").text
    link = re.findall(r'href="([^"]+margin-statistics[^"]*\.xlsx)"', html)[0]
    x = pd.read_excel(io.BytesIO(get("https://www.finra.org" + link if link.startswith("/") else link).content))
    x = x.iloc[:, :2]
    x.columns = ["month", "margin_debt_musd"]
    x.sort_values("month").to_csv(os.path.join(RAW, "finra_margin_debt.csv"), index=False)
    return f"{len(x)} months through {x.month.max()}"


def rebuild():
    sys.path.insert(0, os.path.dirname(HERE))
    from timemachine import run_all
    run_all.main()
    return "pipeline rebuilt"


if __name__ == "__main__":
    print("Updating Time Machine data. This takes about 5-15 minutes.\n")
    step("FRED current values", fred_current)
    step("FRED first-release values", fred_first)
    step("Latest daily readings", latest_daily)
    step("Yahoo S&P 500 + sector ETFs", yahoo)
    step("Shiller CAPE (multpl)", cape)
    step("Ken French 49 industries", french)
    step("Ken French asset-growth portfolios", french_inv)
    step("FDIC bank failures", fdic)
    step("FINRA margin debt", finra_margin)
    step("Rebuild database, crash library, analysis, Monte Carlo", rebuild)
    print("\nSummary\n" + "\n".join(LOG))
    with open(os.path.join(HERE, "data", "update_log.txt"), "w") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M") + "\n" + "\n".join(LOG) + "\n")
