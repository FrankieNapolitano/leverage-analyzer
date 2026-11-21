"""
snapshot_companies.py - saves a dated record of exactly what the AI Chain Scan saw.

Every run appends today's numbers for the AI-chain companies to research/company_snapshots.csv,
so anyone can check (and re-use) the inputs behind each leverage score on any past date.
Run by the daily GitHub Action; can also be run by hand: python snapshot_companies.py
"""

import datetime as dt
import os

import pandas as pd

import engine as E

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "research", "company_snapshots.csv")
TICKERS = ["ORCL", "CRWV", "NBIS", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "AMD", "AVGO"]


def main():
    df = E.scan(TICKERS)
    df.insert(0, "Date", dt.date.today().isoformat())
    if "Risk score" in df:
        df = df[df["Risk score"].notna()]
    if df.empty:
        print("No data fetched; nothing saved.")
        return
    if os.path.exists(OUT):
        old = pd.read_csv(OUT)
        old = old[old["Date"] != df["Date"].iloc[0]]          # re-running the same day replaces that day
        df = pd.concat([old, df], ignore_index=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.round(4).to_csv(OUT, index=False)
    print(f"Saved {len(df)} rows to {OUT}")


if __name__ == "__main__":
    main()
