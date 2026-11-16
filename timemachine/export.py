"""
export.py - writes the whole database to one Excel workbook: data/TimeMachine_database.xlsx
(the CSVs in data/ stay the source of truth; this is the human-friendly copy).
"""

import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

README = [
    ["Market Time Machine database", ""],
    ["Built by", "timemachine/run_all.py (rerun any time; update_data.py refreshes the raw downloads first)"],
    ["Core rule", "Every value in Timeline is what was KNOWABLE at that month-end: first-published figures where the "
                  "archive exists (ALFRED), otherwise revised data held back until its normal release date."],
    ["Timeline", "One row per month-end since Jan 1928. Columns are indicators (see Indicators sheet)."],
    ["Sources_by_month", "For every month and indicator: which observation was used, when it was published, and how "
                         "(first_release / revised_lagged / market / market_monthly_avg / market_daily_latest)."],
    ["Indicators", "Label, group, unit and description of each column."],
    ["Crash_library", "Every S&P 500 decline of 10%+ since 1928 with its type (cluster) and percentile fingerprint."],
    ["Fingerprints", "The point-in-time readings 0, 6 and 12 months before each peak."],
    ["Narratives", "What people knew before each peak, what set it off, and (hindsight) what happened after."],
    ["Analogs_crashes / Analogs_months", "Past crashes and past months most similar to today."],
    ["Sector_test", "Thesis test: capital-heavy vs asset-light sectors during each decline since 1999."],
    ["MC_today / MC_validation", "Monte Carlo probabilities of a 10/20/30% drop within 12 months, and the "
                                 "1965-2025 walk-forward scorecard."],
    ["Early_warning", "Yes/no warning lights since 1955: how often a 20%+ drop followed when each was on, split-sample lifts, p-values."],
    ["Sector_test_investment", "Thesis test with companies sorted on asset growth (aggressive vs conservative investors), 1963+."],
    ["Prewar_daily_declines", "10%+ declines 1928-1949 measured on DAILY closes (the main library uses monthly averages there)."],
    ["recession_hindsight column", "Official NBER dates known only later. It is an OUTCOME, never used as an input."],
]


def build():
    out = os.path.join(DATA, "TimeMachine_database.xlsx")
    ind = json.load(open(os.path.join(DATA, "indicators.json")))
    nar = json.load(open(os.path.join(DATA, "narratives.json")))
    mc = json.load(open(os.path.join(DATA, "mc_today.json")))
    sheets = {
        "README": pd.DataFrame(README, columns=["Item", "Description"]),
        "Timeline": pd.read_csv(os.path.join(DATA, "timeline.csv")),
        "Sources_by_month": pd.read_csv(os.path.join(DATA, "timeline_meta.csv")),
        "Indicators": pd.DataFrame([[k, *v] for k, v in ind.items()],
                                   columns=["column", "label", "group", "unit", "description", "higher_is_riskier"]),
        "Crash_library": pd.read_csv(os.path.join(DATA, "crash_types.csv")),
        "Fingerprints": pd.read_csv(os.path.join(DATA, "crash_fingerprints.csv")),
        "Narratives": pd.DataFrame([{"id": int(k), "title": v["title"], "before_the_peak": v["before"],
                                     "trigger": v["trigger"], "what_happened_next_HINDSIGHT": v["after"],
                                     "sources": " | ".join(v["sources"])} for k, v in nar.items()]),
        "Analogs_crashes": pd.read_csv(os.path.join(DATA, "analogs_crashes.csv")),
        "Analogs_months": pd.read_csv(os.path.join(DATA, "analogs_months.csv")),
        "Sector_test": pd.read_csv(os.path.join(DATA, "sector_test.csv")),
        "MC_today": pd.DataFrame(mc["results"]).T.reset_index().rename(columns={"index": "model"}),
    }
    for name, fn in (("Early_warning", "early_warning.csv"), ("Sector_test_investment", "sector_test_investment.csv"),
                     ("Prewar_daily_declines", "crash_library_prewar_daily.csv")):
        if os.path.exists(os.path.join(DATA, fn)):
            sheets[name] = pd.read_csv(os.path.join(DATA, fn))
    if os.path.exists(os.path.join(DATA, "sector_test_long.csv")):
        sheets["Sector_test_1928"] = pd.read_csv(os.path.join(DATA, "sector_test_long.csv"))
    vp = os.path.join(DATA, "mc_validation.json")
    if os.path.exists(vp):
        v = json.load(open(vp))
        rows = []
        for per in ("1965-2025", "1965-1994", "1995-2025"):
            for ev, d in v[per].items():
                for m in ("normal", "bootstrap", "regime", "analog", "base"):
                    rows.append({"period": per, "event": ev, "model": m, **d[m], "actual_rate": d["actual_rate"]})
        sheets["MC_validation"] = pd.DataFrame(rows)
        sheets["MC_walkforward"] = pd.read_csv(os.path.join(DATA, "mc_walkforward.csv"))
    with pd.ExcelWriter(out, engine="openpyxl") as xw:
        for name, df in sheets.items():
            df.to_excel(xw, sheet_name=name[:31], index=False)
            ws = xw.sheets[name[:31]]
            ws.freeze_panes = "B2" if name not in ("README", "Narratives") else "A2"
            for i, col in enumerate(df.columns, 1):
                width = min(60, max(10, len(str(col)) + 2))
                if name in ("README", "Narratives"):
                    width = 18 if i == 1 else 70
                ws.column_dimensions[ws.cell(1, i).column_letter].width = width
    return out


if __name__ == "__main__":
    print(build())
