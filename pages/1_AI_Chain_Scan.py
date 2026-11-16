"""
AI Chain Scan - ranks a watchlist by leverage risk.
Default list = the players from the AI Buildout Financing Thesis: leveraged builders, hyperscalers, chipmakers.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import engine as E

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

df = st.session_state.get("scan")
if df is None:
    st.info("Press **Run scan** to start.")
    st.stop()

ok = df[df["Risk score"].notna()] if "Risk score" in df else df.iloc[0:0]
bad = df[df["Risk score"].isna()] if "Risk score" in df else df
for _, r in bad.iterrows():
    st.warning(f"{r['Ticker']}: {r['Name']}")
if ok.empty:
    st.stop()

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
