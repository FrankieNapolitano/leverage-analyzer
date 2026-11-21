"""
AI Chain Scan - ranks a watchlist by leverage risk.
Default list = the players from the AI Buildout Financing Thesis: leveraged builders, hyperscalers, chipmakers.
"""

import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import engine as E

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "research")
RATING_COLOR = {"Low": "#0ca30c", "Moderate": "#7cbf3a", "Elevated": "#fab219", "High": "#ec835a", "Severe": "#d03b3b"}


def tier_of(score):
    return "Fortress" if score < 25 else "Healthy" if score < 45 else "Stretched" if score < 65 else "Fragile"


@st.cache_data(show_spinner=False)
def load_research():
    def rd(n):
        p = os.path.join(RES, n)
        return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()
    return rd("hidden_leverage_summary.csv"), rd("off_balance_items.csv"), rd("news_context.csv"), \
        rd("credit_signals.csv"), rd("company_snapshots.csv")

st.set_page_config(page_title="AI Chain Scan", page_icon="🔗", layout="wide")

STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
TIER_STATUS = {"Fortress": "good", "Healthy": "good", "Stretched": "warning", "Fragile": "critical"}
GROUPS = {
    "Leveraged builders": "ORCL, CRWV, NBIS",
    "Hyperscalers": "MSFT, GOOGL, AMZN, META",
    "Chipmakers": "NVDA, AMD, AVGO",
}

st.title("🔗 AI Chain Leverage Scan")
st.write("Ranks every company in the AI buildout by how fragile its balance sheet is. The thesis says the "
         "**most leveraged links** get hurt first if financing tightens, so this shows where they are.")

default = ", ".join(GROUPS.values())
tickers_txt = st.text_area("Tickers (comma separated)", value=default, height=70)
tickers = [t.strip().upper() for t in tickers_txt.split(",") if t.strip()]


@st.cache_data(ttl=3600, show_spinner=False)
def run_scan(tks):
    return E.scan(list(tks))


if st.button("Run scan", type="primary"):
    with st.spinner(f"Scanning {len(tickers)} companies (about 5 seconds each)..."):
        st.session_state["scan"] = run_scan(tuple(tickers))

HS, ITEMS, NEWS, CREDIT, SNAPS = load_research()

df = st.session_state.get("scan")
if df is None:
    st.info("Press **Run scan** for live balance-sheet scores. The hidden-leverage research below is always available.")

ok = pd.DataFrame() if df is None else (df[df["Risk score"].notna()] if "Risk score" in df else df.iloc[0:0])
bad = pd.DataFrame() if df is None else (df[df["Risk score"].isna()] if "Risk score" in df else df)
for _, r in bad.iterrows():
    st.warning(f"{r['Ticker']}: {r['Name']}")

if not ok.empty:
    ok = ok.sort_values("Risk score", ascending=True)
    fig = go.Figure(go.Bar(
        x=ok["Risk score"], y=ok["Ticker"], orientation="h",
        marker=dict(color=[STATUS[TIER_STATUS[t]] for t in ok["Tier"]], cornerradius=4),
        text=[f"{s:.0f} · {t}" for s, t in zip(ok["Risk score"], ok["Tier"])], textposition="outside",
        customdata=ok[["Name", "Bias", "Stock view"]].values,
        hovertemplate="<b>%{customdata[0]}</b><br>Risk %{x:.0f}/100<br>Bias: %{customdata[1]}<br>%{customdata[2]}<extra></extra>"))
    fig.update_layout(title="Leverage risk score (higher = more fragile)", height=60 + 38 * len(ok),
                      margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(range=[0, 115], showgrid=False)
    st.plotly_chart(fig, width="stretch")

    show = ok.sort_values("Risk score", ascending=False).copy()
    st.dataframe(
        show.style.format({
            "Risk score": "{:.0f}", "Net debt/EBITDA": "{:.2f}x", "Interest cover": "{:.1f}x", "FCF margin": "{:.1%}",
            "Capex/OCF": "{:.2f}x", "Debt growth 1y": "{:+.0%}", "Rate sens /25bp": "{:+.2%}", "6m return": "{:+.1%}"},
            na_rep="n/a"),
        hide_index=True, width="stretch")
    st.caption("Interest cover shows 'inf' when a company has essentially no interest expense. "
               "Open a ticker on the main page for the full profile and an options idea. Not financial advice.")

# ------------------------------------------------------------------------------------------ hidden leverage
st.divider()
st.header("🕳️ Hidden leverage: what the balance sheet doesn't show")
st.write("The score above only reads each company's **reported** balance sheet. Some of the biggest risks in the AI "
         "buildout never show up as debt: data-center leases that are signed but haven't started, multi-year purchase "
         "and compute commitments, guarantees of other companies' debt, money a supplier has put into its own customers "
         "(vendor financing), and revenue that depends on a few AI labs that are themselves funded this way. "
         "This section adds that research, with a source for every number.")

if HS.empty:
    st.info("Research files not found in research/.")
else:
    order = ["Severe", "High", "Elevated", "Moderate", "Low"]
    hs = HS.copy()
    hs["_o"] = hs["Hidden leverage rating"].map({r: i for i, r in enumerate(order)})
    hs = hs.sort_values(["_o", "Off-balance total ($B)"], ascending=[True, False])

    if not ok.empty:
        m = ok[["Ticker", "Risk score", "Tier"]].merge(hs[["Ticker", "Hidden leverage rating", "Score add-on"]],
                                                       on="Ticker", how="left")
        m["Score add-on"] = m["Score add-on"].fillna(0)
        m["Full-picture score"] = (m["Risk score"] + m["Score add-on"]).clip(upper=100)
        m["Full-picture tier"] = m["Full-picture score"].map(tier_of)
        m = m.sort_values("Full-picture score")
        fig2 = go.Figure()
        fig2.add_bar(y=m["Ticker"], x=m["Risk score"], orientation="h", name="Balance sheet only",
                     marker=dict(color="#9db4d3", cornerradius=3))
        fig2.add_bar(y=m["Ticker"], x=m["Full-picture score"], orientation="h", name="Full picture (incl. hidden)",
                     marker=dict(color=[STATUS[TIER_STATUS[t]] for t in m["Full-picture tier"]], cornerradius=3),
                     text=[f"{s:.0f} · {t}" for s, t in zip(m["Full-picture score"], m["Full-picture tier"])],
                     textposition="outside")
        fig2.update_layout(barmode="group", title="Balance-sheet score vs. full-picture score (higher = more fragile)",
                           height=80 + 52 * len(m), margin=dict(l=10, r=10, t=40, b=10),
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           legend=dict(orientation="h", y=1.08, x=0))
        fig2.update_xaxes(range=[0, 115], showgrid=False)
        st.plotly_chart(fig2, width="stretch")
        st.caption("Full-picture score = balance-sheet score + hidden-leverage add-on (Low 0, Moderate +8, "
                   "Elevated +15, High +22, Severe +30), capped at 100. The add-on is a research judgment, explained "
                   "for each company below.")
    else:
        st.caption("Run the scan above to see balance-sheet vs. full-picture scores side by side.")

    st.subheader("Hidden-leverage ratings")
    show_hs = hs[["Ticker", "Company", "Hidden leverage rating", "Off-balance total ($B)", "Biggest hidden risk"]]
    st.dataframe(show_hs.style.format({"Off-balance total ($B)": "${:,.0f}B"}, na_rep="n/a").map(
        lambda v: f"color: {RATING_COLOR.get(v, 'inherit')}; font-weight: 600", subset=["Hidden leverage rating"]),
        hide_index=True, width="stretch")
    st.caption("Off-balance totals aren't directly comparable: some include ordinary supplier purchases (Alphabet's is "
               "mostly normal multi-year purchasing), others are mostly leases and guarantees. The rating weighs them "
               "against each company's cash generation and how concentrated the risk is.")

    st.subheader("Company deep dives")
    for _, r in hs.iterrows():
        tk = r["Ticker"]
        with st.expander(f"{tk} · {r['Company']} · {r['Hidden leverage rating']}"):
            st.markdown(f"**Why:** {r['Why']}")
            st.markdown(f"**Biggest hidden risk:** {r['Biggest hidden risk']}")
            st.markdown(f"**What would change my mind:** {r['What would change my mind']}")
            it = ITEMS[ITEMS["Ticker"] == tk].drop(columns="Ticker")
            if not it.empty:
                st.markdown("**Obligations and exposures not on the balance sheet**")
                st.dataframe(it, hide_index=True, width="stretch",
                             column_config={"Link": st.column_config.LinkColumn("Link", display_text="source"),
                                            "Amount ($B)": st.column_config.NumberColumn(format="$%.1fB")})
            if isinstance(r.get("How the total was counted"), str):
                st.caption("How the total was counted: " + r["How the total was counted"])
            nw = NEWS[NEWS["Ticker"] == tk].drop(columns="Ticker").sort_values("Date", ascending=False)
            if not nw.empty:
                st.markdown("**Recent news that matters**")
                st.dataframe(nw, hide_index=True, width="stretch",
                             column_config={"Link": st.column_config.LinkColumn("Link", display_text="source")})
            cr = CREDIT[CREDIT["Ticker"] == tk].drop(columns="Ticker").dropna(axis=1, how="all")
            if not cr.empty:
                st.markdown("**Credit-market signals**")
                st.dataframe(cr, hide_index=True, width="stretch")

    st.subheader("Download the data")
    st.write("Everything behind these ratings, so anyone can check or re-use it.")
    d1, d2, d3 = st.columns(3)
    xl = os.path.join(RES, "Hidden_Leverage_Research.xlsx")
    if os.path.exists(xl):
        d1.download_button("📊 Full research workbook (Excel)", open(xl, "rb").read(),
                           file_name="Hidden_Leverage_Research.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    d2.download_button("🧾 Off-balance items (CSV)", ITEMS.to_csv(index=False).encode(),
                       file_name="off_balance_items.csv", mime="text/csv")
    if not SNAPS.empty:
        d3.download_button("📅 Daily balance-sheet snapshots (CSV)", SNAPS.to_csv(index=False).encode(),
                           file_name="company_snapshots.csv", mime="text/csv")
    else:
        d3.caption("Daily balance-sheet snapshots start with the next automatic refresh.")
    asof = HS["As of"].dropna().max() if "As of" in HS else ""
    st.caption(f"Research as of {asof}. Sources: SEC filings (10-Q/10-K/8-K/S-1), company releases, Reuters, Bloomberg, "
               "WSJ, CNBC and others; each row is marked high/medium/low confidence. Also on GitHub under research/. "
               "Not financial advice.")
