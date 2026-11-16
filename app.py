"""
app.py - Leverage Analyzer dashboard (Streamlit).
Run with run.bat, or:  .venv\\Scripts\\python -m streamlit run app.py
"""

import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import engine as E
import ai_writer as AI

st.set_page_config(page_title="Leverage Analyzer", page_icon="📊", layout="wide")

# Colors (validated categorical + status palettes)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
TIER_STATUS = {"Fortress": "good", "Healthy": "good", "Stretched": "warning", "Fragile": "critical"}
TIER_ICON = {"Fortress": "🟢", "Healthy": "🟢", "Stretched": "🟠", "Fragile": "🔴"}
BIAS_ICON = {"Bullish": "📈", "Neutral": "➖", "Bearish": "📉"}


def score_status(v):
    return "good" if v < 25 else "warning" if v < 50 else "serious" if v < 75 else "critical"


def fmt(x, kind="num"):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    if x == float("inf"):
        return "∞ (no real debt cost)" if kind == "x" else "n/a"
    return {"pct": f"{x:.1%}", "bn": f"{'-' if x < 0 else ''}${abs(x) / 1e9:,.1f}B", "x": f"{x:.2f}x", "usd": f"${x:,.2f}",
            "usd0": f"${x:,.0f}"}.get(kind, f"{x:,.2f}")


def md(text):
    """Escape $ so Streamlit doesn't render dollar amounts as LaTeX math."""
    return text.replace("$", "\\$")


def style(fig, h=360):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", hovermode="x unified",
                      legend=dict(orientation="h", y=1.08, x=0))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.18)", zerolinecolor="rgba(128,128,128,0.4)")
    return fig


@st.cache_data(ttl=3600, show_spinner=False)
def load(ticker, dte):
    return E.analyze(E.fetch_raw(ticker, dte))


# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Settings")
    ticker = st.text_input("Ticker", value="ORCL").strip().upper()
    account = st.number_input("Account size ($)", min_value=500, value=10000, step=500)
    risk_pct = st.slider("Max risk per trade (% of account)", 0.5, 5.0, 2.0, 0.5) / 100
    dte = st.slider("Target days to expiration", 21, 90, 45, 1,
                    help="The tool picks the option expiry closest to this.")
    st.divider()
    model_label = st.selectbox("AI model for the write-up", list(AI.MODELS))
    # Web version: the owner's key lives in Streamlit secrets and is unlocked with a password,
    # so visitors can't spend it. Visitors can always paste their own key instead.
    def _secret(name):
        try:
            return st.secrets.get(name)
        except Exception:
            return None
    owner_key, owner_pw = _secret("OWNER_ANTHROPIC_KEY"), _secret("APP_PASSWORD")
    key_in_env = bool(os.environ.get("ANTHROPIC_API_KEY")) and not owner_key
    api_key = None
    if key_in_env:
        st.caption("✅ Anthropic API key found")
    elif owner_key and owner_pw and st.text_input("Owner password (for the AI write-up)",
                                                   type="password") == owner_pw:
        api_key = owner_key
        st.caption("✅ Unlocked")
    else:
        api_key = st.text_input("Or paste your own Anthropic API key", type="password",
                                help="Only used for this session; never saved.") or None
    if st.button("🔄 Refresh data"):
        st.cache_data.clear()
    st.divider()
    st.caption("Educational tool, not financial advice. Options can lose 100% of what you pay; "
               "check every number with your broker before trading.")

if not ticker:
    st.stop()

try:
    with st.spinner(f"Pulling {ticker} financials, prices, rates and options..."):
        a = load(ticker, dte)
except Exception as e:
    st.error(f"Couldn't analyze **{ticker}**: {e}")
    st.stop()

L, M, O, S, B = a["lev"], a["mkt"], a["options"], a["strategy"], a["bias"]

# ----------------------------------------------------------------------------- header
st.title(f"{a['name']} ({a['ticker']})")
st.caption(f"{a['sector']} · {a['industry']} · Price {fmt(M['price'], 'usd')} · Data pulled {a['fetched']}")
for n in a["notes"]:
    st.warning(n)

c1, c2, c3, c4 = st.columns(4)
if a["tier"]:
    c1.metric("Leverage tier", f"{TIER_ICON[a['tier']]} {a['tier']}", f"Risk score {a['risk_score']:.0f}/100",
              delta_color="off")
c2.metric("Directional bias", f"{BIAS_ICON[B['bias']]} {B['bias']}", f"{B['conviction']} conviction", delta_color="off")
c3.metric("Stock position", a["stock_view"].split(" - ")[0].split(";")[0].split(",")[0])
c4.metric("Options idea", S["name"].split(" (")[0] if S else "No options data")
st.caption(f"**Stock view:** {a['stock_view']}")

DD = a.get("exp_drawdown")
if DD:
    cap = E.stock_position_cap(account, risk_pct, DD)
    d1, d2 = st.columns([1, 2])
    d1.metric("Typical worst drop, next 12 months", f"{DD['mid']:.0%}",
              f"range {DD['bad']:.0%} to {DD['mild']:.0%}", delta_color="off",
              help="From the backtest: driven mainly by the stock's volatility, which was the single best "
                   "predictor of future drawdowns (tested on 2024-25 data it never saw).")
    if cap:
        d2.info(md(f"**Stock sizing:** to keep a typical drawdown under {risk_pct:.1%} of your account "
                f"({fmt(account * risk_pct, 'usd0')}), hold at most **{fmt(cap, 'usd0')}** of {a['ticker']} "
                f"({cap / account:.0%} of the account)."))

with st.expander("📊 What the backtest showed (199 stocks, 2022-2026)"):
    st.markdown(
        "- **Volatility predicts drawdowns.** A stock's past 1-year volatility was the strongest predictor of how far "
        "it fell over the next year, and it held up on years the model never saw. That's what the drawdown estimate above uses.\n"
        "- **Trend predicts returns (a little).** Stocks in strong uptrends beat strong downtrends by about 11 points a year.\n"
        "- **Leverage alone did NOT predict underperformance.** Fragile companies actually outperformed in 2024 "
        "(the AI-power trade: VRT, VST, APLD). Debt growth was the one leverage metric that still flagged bigger drops "
        "after accounting for volatility.\n"
        "- **So:** use the leverage score to understand *why* a stock is risky and to size and hedge, not as a "
        "short signal by itself.\n\n"
        "Limits: only today's companies (survivorship bias), about 4 years of data, and options trades weren't tested.")

tabs = st.tabs(["🏦 Leverage profile", "📉 Price & rates", "🎯 Options trade", "🤖 AI write-up", "🔢 All numbers"])

# ----------------------------------------------------------------------------- tab 1: leverage
with tabs[0]:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Net debt / EBITDA", fmt(L["net_debt_to_ebitda"], "x"), help="Years of earnings to pay off net debt. <2x comfy, >4x heavy. Negative = net cash.")
    m2.metric("Interest coverage", fmt(L["interest_coverage"], "x"), help="EBIT ÷ interest expense. >8x strong, <2x danger.")
    m3.metric("FCF margin", fmt(L["fcf_margin"], "pct"), help="Free cash flow ÷ revenue (trailing 12 months).")
    m4.metric("Capex / operating cash", fmt(L["capex_to_ocf"], "x"), help=">1.0x means building with borrowed or raised money.")
    m5, m6, m7, m8 = st.columns(4)
    m5.metric("Total debt", fmt(L["total_debt"], "bn"))
    m6.metric("Cash", fmt(L["cash"], "bn"))
    m7.metric("Debt growth (1y)", fmt(L["debt_growth_1y"], "pct"))
    m8.metric("Altman Z-score", fmt(L["altman_z"]), help=">3 safe, 1.8-3 grey zone, <1.8 distress.")

    left, right = st.columns([1.1, 1])
    with left:
        rows = [(lbl, a["scores"].get(k), desc, w) for k, lbl, w, desc in E.COMPONENTS]
        rows = [r for r in rows if r[1] is not None][::-1]
        fig = go.Figure(go.Bar(
            x=[r[1] for r in rows], y=[r[0] for r in rows], orientation="h",
            marker=dict(color=[STATUS[score_status(r[1])] for r in rows], cornerradius=4),
            text=[f"{r[1]:.0f} · {score_status(r[1])}" for r in rows], textposition="outside",
            customdata=[[r[2], r[3]] for r in rows],
            hovertemplate="<b>%{y}</b><br>Risk %{x:.0f}/100<br>%{customdata[0]}<br>Weight %{customdata[1]:.0%}<extra></extra>"))
        fig.update_layout(title="What drives the risk score (0 = safe, 100 = dangerous)", hovermode="closest")
        fig.update_xaxes(range=[0, 118])
        st.plotly_chart(style(fig, 340), width="stretch")
    with right:
        h = a["history"]
        if not h.empty:
            fig = go.Figure()
            for col, color in (("Total debt", BLUE), ("Free cash flow", ORANGE)):
                if col in h:
                    fig.add_bar(x=h.index.astype(str), y=h[col], name=col, marker=dict(color=color, cornerradius=4),
                                hovertemplate=f"{col}: $%{{y:,.1f}}B<extra></extra>")
            fig.update_layout(title="Debt vs. free cash flow by year ($B)", bargap=0.3, bargroupgap=0.08)
            st.plotly_chart(style(fig, 340), width="stretch")
    if L["cash_runway_yrs"]:
        st.info(f"Free cash flow is negative. At the current burn, cash on hand lasts about "
                f"**{L['cash_runway_yrs']:.1f} years** before needing more borrowing or stock sales.")

# ----------------------------------------------------------------------------- tab 2: price & rates
with tabs[1]:
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Beta vs S&P 500", fmt(M["beta"]))
    rs_note = ("statistically real" if (M["rate_tstat"] or 0) <= -2 else "not statistically reliable")
    k2.metric("Move per +25 bps in 10-yr", fmt(M["rate_sens_25bp"], "pct"), rs_note, delta_color="off",
              help="From a regression of weekly returns on the market and on changes in the 10-yr yield. "
                   "Only counts toward the score if t-stat ≤ -2.")
    k3.metric("Volatility (1y)", fmt(M["vol_1y"], "pct"))
    k4.metric("6-month return", fmt(M["ret_6m"], "pct"))
    k5.metric("vs 52-week high", fmt(M["off_52w_high"], "pct"))

    px = a["px"].iloc[-252:]
    full = a["px"]
    fig = go.Figure()
    fig.add_scatter(x=px.index, y=px, name="Price", line=dict(color=BLUE, width=2))
    fig.add_scatter(x=px.index, y=full.rolling(50).mean().loc[px.index], name="50-day avg",
                    line=dict(color=ORANGE, width=2))
    fig.add_scatter(x=px.index, y=full.rolling(200).mean().loc[px.index], name="200-day avg",
                    line=dict(color=AQUA, width=2))
    fig.update_layout(title=f"{a['ticker']} price, last 12 months")
    fig.update_yaxes(tickprefix="$")
    st.plotly_chart(style(fig, 380), width="stretch")

    y = a["tnx"].iloc[-252:]
    fig = go.Figure(go.Scatter(x=y.index, y=y, name="10-yr yield", line=dict(color=BLUE, width=2),
                               hovertemplate="%{y:.2f}%<extra></extra>"))
    fig.update_layout(title=f"10-year Treasury yield: {M['tnx_now']:.2f}% ({M['tnx_chg_3m']:+.2f} pp in 3 months)")
    fig.update_yaxes(ticksuffix="%")
    st.plotly_chart(style(fig, 260), width="stretch")

    st.markdown("**Why the bias came out " + B["bias"].lower() + ":** " + " · ".join(B["reasons"]))

# ----------------------------------------------------------------------------- tab 3: options
contracts = None
with tabs[2]:
    if not O:
        st.warning("No options data available for this ticker.")
    else:
        o1, o2, o3, o4, o5 = st.columns(5)
        o1.metric("Expiry", O["expiry"], f"{O['dte']} days", delta_color="off")
        o2.metric("Implied vol (ATM)", fmt(O["atm_iv"], "pct"))
        o3.metric("Realized vol (3m)", fmt(O["rv"], "pct"))
        o4.metric("IV ÷ RV", fmt(O["iv_rv"], "x"), f"options look {O['regime']}", delta_color="off")
        o5.metric("Expected move", f"±${O['exp_move']:.2f}", f"±{O['exp_move'] / M['price']:.1%}", delta_color="off")

    if S:
        st.subheader(S["name"])
        st.write(S["why"])
        if S["legs"]:
            legs = pd.DataFrame([{
                "Action": l["action"], "Type": l["kind"].upper(), "Strike": l["strike"], "Expiry": O["expiry"],
                "Price (mid)": l["price"], "Delta": l["delta"], "Implied vol": l["iv"], "Open interest": l["oi"]}
                for l in S["legs"]])
            st.dataframe(legs.style.format({"Strike": "${:,.2f}", "Price (mid)": "${:,.2f}", "Delta": "{:+.2f}",
                                            "Implied vol": "{:.0%}", "Open interest": "{:,.0f}"}, na_rep="n/a"),
                         hide_index=True, width="stretch")

            net = S["net"]
            contracts = E.size_position(S, account, risk_pct)
            p1, p2, p3, p4, p5 = st.columns(5)
            p1.metric("Net credit" if net > 0 else "Net cost", fmt(abs(net), "usd0"), "per contract set", delta_color="off")
            p2.metric("Max profit", "Unlimited" if S["unbounded_up"] else fmt(S["max_profit"], "usd0"))
            p3.metric("Max loss", fmt(abs(S["max_loss"]), "usd0"),
                      "incl. your 100 shares" if S["stock_shares"] else None, delta_color="off")
            p4.metric("Breakeven", " / ".join(f"${b:.2f}" for b in S["breakevens"]) or "n/a")
            p5.metric("Prob. of profit", fmt(S["pop"], "pct"), help="Model estimate from implied volatility. Not a guarantee.")

            if S["stock_shares"]:
                st.info("This is a hedge: buy 1 put for every 100 shares you own.")
            elif contracts is not None:
                if contracts >= 1:
                    st.success(md(f"**Position size:** {contracts} contract set(s). Max loss "
                               f"{fmt(contracts * abs(S['max_loss']), 'usd0')} = "
                               f"{contracts * abs(S['max_loss']) / account:.1%} of your account."))
                else:
                    st.warning(md(f"One contract set risks {fmt(abs(S['max_loss']), 'usd0')}, more than your "
                               f"{risk_pct:.1%} limit ({fmt(account * risk_pct, 'usd0')}). Skip it, or raise the "
                               f"limit only if you accept that risk."))
            for w in S["warnings"]:
                st.warning(w)

            fig = go.Figure()
            g, pl = S["grid"], S["pl"]
            fig.add_scatter(x=g, y=np.where(pl >= 0, pl, np.nan), name="Profit", fill="tozeroy",
                            line=dict(color=STATUS["good"], width=2), hovertemplate="At $%{x:.2f}: +$%{y:,.0f}<extra></extra>")
            fig.add_scatter(x=g, y=np.where(pl < 0, pl, np.nan), name="Loss", fill="tozeroy",
                            line=dict(color=STATUS["critical"], width=2), hovertemplate="At $%{x:.2f}: -$%{customdata:,.0f}<extra></extra>",
                            customdata=-pl)
            fig.add_vline(x=M["price"], line=dict(color="gray", dash="dot"), annotation_text=f"Now ${M['price']:.2f}")
            for b in S["breakevens"]:
                fig.add_vline(x=b, line=dict(color="gray", dash="dash", width=1), annotation_text=f"BE ${b:.2f}",
                              annotation_position="bottom right")
            fig.update_layout(title=f"Profit/loss at expiry ({O['expiry']}), 1 contract set", hovermode="closest")
            fig.update_xaxes(title="Stock price at expiry", tickprefix="$")
            fig.update_yaxes(tickprefix="$")
            st.plotly_chart(style(fig, 380), width="stretch")

            with st.expander("Common management rules of thumb"):
                if net > 0:
                    st.markdown(f"- Take profit when you can buy it back for about half the credit (≈{fmt(net / 2, 'usd0')} gain).\n"
                                f"- Cut it if the loss reaches about 2x the credit you collected.\n"
                                "- Close or roll with ~21 days left; risk speeds up near expiry.")
                else:
                    st.markdown("- Take profit around 50-75% of max profit; the last part is slow and risky to wait for.\n"
                                "- Cut it if it loses about half its value, or if the reason for the trade breaks "
                                "(e.g., the trend flips).\n"
                                "- Don't hold debit spreads into the final week; time decay accelerates.")

# ----------------------------------------------------------------------------- tab 4: AI
with tabs[3]:
    st.write("Claude reads every number above and explains it. Python did all the math; the AI only interprets.")
    cache_key = f"ai_{a['ticker']}_{model_label}"
    if st.button("✍️ Write the analysis", type="primary"):
        try:
            with st.spinner("Claude is writing..."):
                st.session_state[cache_key] = AI.write_summary(a, model_label, api_key, account, risk_pct, contracts)
        except Exception as e:
            st.error(f"AI write-up failed: {e}")
    if cache_key in st.session_state:
        st.markdown(md(st.session_state[cache_key]))
        st.caption("AI-generated. Check it against the numbers before acting on it.")
    with st.expander("See the exact prompt sent to Claude"):
        st.code(AI.build_prompt(a, account, risk_pct, contracts), language=None)

# ----------------------------------------------------------------------------- tab 5: raw
with tabs[4]:
    nice = {**{f"Leverage · {k}": v for k, v in L.items()}, **{f"Market · {k}": v for k, v in M.items()}}
    raw = pd.DataFrame({"Metric": list(nice), "Value": [str(round(v, 4)) if isinstance(v, float) else str(v)
                                                        for v in nice.values()]})
    st.dataframe(raw, hide_index=True, width="stretch", height=600)
