"""
compile_research.py - merges the per-company research files (research/companies/<TICKER>.json)
into the tables the app and spreadsheet use:

  research/hidden_leverage_summary.csv   one row per company: rating, add-on points, totals, biggest risk
  research/off_balance_items.csv         every off-balance-sheet obligation, with source links
  research/news_context.csv              recent news that matters for each company's financing
  research/credit_signals.csv            ratings, CDS, bond deals
  research/Hidden_Leverage_Research.xlsx all of the above in one workbook with a README tab

Run: python research/compile_research.py
"""

import glob
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
COMP = os.path.join(HERE, "companies")

# How much the hidden-leverage rating adds to the balance-sheet risk score (0-100 scale)
ADDON = {"Low": 0, "Moderate": 8, "Elevated": 15, "High": 22, "Severe": 30}


def load():
    out = []
    for p in sorted(glob.glob(os.path.join(COMP, "*.json"))):
        with open(p, encoding="utf-8") as f:
            out.append(json.load(f))
    return out


def build():
    data = load()
    summ, items, news, credit = [], [], [], []
    for d in data:
        tk = d["ticker"]
        a = d.get("hidden_leverage_assessment", {}) or {}
        rating = a.get("rating", "Moderate")
        summ.append({"Ticker": tk, "Company": d.get("company"), "As of": d.get("as_of"),
                     "Hidden leverage rating": rating, "Score add-on": ADDON.get(rating, 8),
                     "Off-balance total ($B)": d.get("total_off_balance_usd_bn"),
                     "How the total was counted": d.get("total_note"),
                     "Biggest hidden risk": a.get("biggest_hidden_risk"),
                     "Why": a.get("rationale"), "What would change my mind": a.get("what_would_change_my_mind")})
        for it in d.get("off_balance_items", []) or []:
            items.append({"Ticker": tk, "Category": it.get("category"), "Description": it.get("description"),
                          "Amount ($B)": it.get("amount_usd_bn"), "Counterparty": it.get("counterparty"),
                          "As of": it.get("as_of_date"), "Confidence": it.get("confidence"),
                          "Source": it.get("source_title"), "Link": it.get("source_url")})
        for n in d.get("news_context", []) or []:
            news.append({"Ticker": tk, "Date": n.get("date"), "What happened": n.get("headline_summary"),
                         "Why it matters": n.get("why_it_matters"), "Link": n.get("source_url")})
        cs = d.get("credit_signals", []) or []
        if isinstance(cs, dict):
            cs = [{"signal": k, "detail": v if not isinstance(v, (dict, list)) else json.dumps(v)} for k, v in cs.items()]
        for c in cs:
            if isinstance(c, dict):
                credit.append({"Ticker": tk, **{str(k): (json.dumps(v) if isinstance(v, (dict, list)) else v)
                                               for k, v in c.items()}})
            else:
                credit.append({"Ticker": tk, "detail": str(c)})
    S, I, N, C = (pd.DataFrame(x) for x in (summ, items, news, credit))
    S.to_csv(os.path.join(HERE, "hidden_leverage_summary.csv"), index=False)
    I.to_csv(os.path.join(HERE, "off_balance_items.csv"), index=False)
    N.to_csv(os.path.join(HERE, "news_context.csv"), index=False)
    C.to_csv(os.path.join(HERE, "credit_signals.csv"), index=False)
    readme = pd.DataFrame([
        ["What this is", "Research on obligations that don't show up as debt on the balance sheet (purchase and compute "
                         "commitments, signed-but-not-started leases, guarantees, vendor financing, customer concentration) "
                         "for the 10 AI-chain companies, plus recent news and credit-market signals."],
        ["Why", "The balance-sheet leverage score misses these. Example: Nvidia has little debt but has funded its own "
                "customers and guaranteed capacity, so its real exposure is larger than its books show."],
        ["Hidden leverage rating", "Low / Moderate / Elevated / High / Severe, judged by comparing hidden obligations and "
                                   "dependence on vendor-financed customers to the company's cash generation."],
        ["Full-picture score", "Balance-sheet risk score (0-100) + add-on: Low 0, Moderate +8, Elevated +15, High +22, "
                               "Severe +30 (capped at 100). Shown on the AI Chain Scan page."],
        ["Sources", "Every number links to its source (SEC filings, company releases, Reuters/Bloomberg/WSJ/CNBC, etc.). "
                    "Confidence = high (filing), medium (reputable press), low (estimate or single report)."],
        ["Snapshots", "company_snapshots.csv records the balance-sheet numbers the app saw each day."],
    ], columns=["Item", "Description"])
    with pd.ExcelWriter(os.path.join(HERE, "Hidden_Leverage_Research.xlsx"), engine="openpyxl") as xw:
        for name, df in (("README", readme), ("Summary", S), ("Off-balance items", I), ("News", N), ("Credit signals", C)):
            df.to_excel(xw, sheet_name=name, index=False)
            ws = xw.sheets[name]
            for i, col in enumerate(df.columns, 1):
                ws.column_dimensions[ws.cell(1, i).column_letter].width = 60 if col in (
                    "Description", "Why", "What happened", "Why it matters", "Biggest hidden risk",
                    "How the total was counted", "What would change my mind") else 18
            ws.freeze_panes = "B2"
    return S, I, N, C


if __name__ == "__main__":
    S, I, N, C = build()
    print(S[["Ticker", "Hidden leverage rating", "Off-balance total ($B)"]].to_string(index=False))
    print(len(I), "items,", len(N), "news,", len(C), "credit rows")
