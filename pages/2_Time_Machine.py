"""
Market Time Machine - relive any month since 1928 using only what was knowable then,
browse every 10%+ S&P 500 decline, compare any moment to today, and run Monte Carlo models.
Data are built by the scripts in /timemachine (run timemachine/update_all.py to refresh).
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from timemachine import montecarlo as MC  # noqa: E402

DATA = os.path.join(ROOT, "timemachine", "data")
st.set_page_config(page_title="Market Time Machine", page_icon="🕰️", layout="wide")

BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
METHOD_TXT = {"first_release": "first published figure", "revised_lagged": "revised data, delayed to its normal release date",
              "market": "market data", "market_monthly_avg": "monthly average", "market_daily_latest": "latest daily reading"}


def md(t):
    return str(t).replace("$", "\\$")


def style(fig, h=320):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", hovermode="x unified", legend=dict(orientation="h", y=1.1, x=0))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.18)", zerolinecolor="rgba(128,128,128,0.4)")
    return fig


def fmt(v, unit):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    if unit == "flag":
        return "Yes" if v >= 0.5 else "No"
    if unit in ("%", "% ann."):
        return f"{v:.1f}%"
    if unit == "pp":
        return f"{v:+.2f} pp"
    if unit == "x":
        return f"{v:.1f}x"
    if unit == "$/bbl":
        return f"${v:,.0f}"
    if unit == "$bn":
        return f"${v:,.0f}B" if v >= 10 else f"${v:,.2f}B"
    if unit == "count":
        return f"{v:,.0f}"
    if unit == "index":
        return f"{v:,.0f}" if v > 500 else f"{v:,.1f}"
    return f"{v:,.1f}"


# ------------------------------------------------------------------------------------------ data
@st.cache_data(show_spinner=False)
def load():
    w = pd.read_csv(os.path.join(DATA, "timeline.csv"), parse_dates=["asof"], index_col="asof")
    meta = pd.read_csv(os.path.join(DATA, "timeline_meta.csv"), parse_dates=["asof", "obs_date", "available"])
    ind = json.load(open(os.path.join(DATA, "indicators.json")))
    lib = pd.read_csv(os.path.join(DATA, "crash_types.csv"))
    nar = json.load(open(os.path.join(DATA, "narratives.json")))
    summ = json.load(open(os.path.join(DATA, "analysis_summary.json")))
    mc = json.load(open(os.path.join(DATA, "mc_today.json")))
    val = json.load(open(os.path.join(DATA, "mc_validation.json"))) if os.path.exists(os.path.join(DATA, "mc_validation.json")) else None
    sector = pd.read_csv(os.path.join(DATA, "sector_test.csv"))
    daily = pd.read_csv(os.path.join(DATA, "raw", "gspc_daily.csv"), parse_dates=["date"]).set_index("date")["close"]
    return w, meta, ind, lib, nar, summ, mc, val, sector, daily


w, meta, IND, lib, NAR, SUMM, MCT, VAL, SECTOR, DAILY = load()
months = list(w.index)
LAST = w.index.max()
LAST_LABEL = f"Today ({DAILY.index.max():%b %d, %Y})"


def label(m):
    return LAST_LABEL if m == LAST else m.strftime("%b %Y")


st.title("🕰️ Market Time Machine")
st.write("Pick any month since 1928 and see the economy **exactly as people saw it then**: only numbers that had "
         "actually been published by that date, with their first-reported values where the archive exists. "
         "No hindsight. Then compare it to today, browse every 10%+ decline in the S&P 500, and run the crash simulators.")

tabs = st.tabs(["🗓️ Time Machine", "⚖️ Then vs. today", "📚 Crash library", "🎲 Monte Carlo", "🧪 Thesis test", "🚦 Warning lights", "ℹ️ Method & limits"])

# ------------------------------------------------------------------------------------------ 1. time machine
with tabs[0]:
    c1, c2 = st.columns([3, 1])
    jump_opts = ["(pick a date with the slider)"] + [f"{r.peak_date[:7]} - month before the peak: {NAR[str(r.id)]['title']}"
                                                      for r in lib.itertuples()]
    jump = c2.selectbox("Jump to the eve of a crash", jump_opts)
    default = LAST
    if jump != jump_opts[0]:
        pk = pd.Timestamp(jump[:7] + "-01")
        default = pk - pd.offsets.MonthEnd(1)
    asof = c1.select_slider("Travel to", options=months, value=default, format_func=label)
    st.session_state["tm_asof"] = asof

    row = w.loc[asof]
    m_now = meta[meta["asof"] == asof].set_index("indicator")
    st.subheader(f"The world as of {label(asof)}")
    dd_now = row.get("drawdown_from_ath")
    st.caption(f"S&P 500 {fmt(row.sp500, 'index')} · {fmt(dd_now, '%')} from its record high · "
               f"10-yr Treasury {fmt(row.t10y, '%')} · inflation {fmt(row.cpi_yoy, '%')}")

    groups = {}
    for k, (lab, grp, unit, desc, risky) in IND.items():
        groups.setdefault(grp, []).append(k)
    for grp, keys in groups.items():
        keys = [k for k in keys if k in w.columns and not pd.isna(row.get(k))]
        if not keys:
            continue
        st.markdown(f"**{grp}**")
        cols = st.columns(4)
        for i, k in enumerate(keys):
            lab, _, unit, desc, risky = IND[k]
            prev_idx = months.index(asof) - 12
            prev = w.iloc[prev_idx][k] if prev_idx >= 0 else np.nan
            delta = None
            if unit not in ("flag",) and not pd.isna(prev):
                dv = row[k] - prev
                delta = f"{dv:+.1f} vs 1 yr ago" if unit not in ("index", "$/bbl") else f"{dv / prev * 100:+.0f}% vs 1 yr ago"
            info = desc
            if k in m_now.index:
                mm = m_now.loc[k]
                if isinstance(mm, pd.DataFrame):
                    mm = mm.iloc[-1]
                info += f"\n\nThis is the {METHOD_TXT.get(mm.method, mm.method)} for {mm.obs_date:%b %Y}, available {mm.available:%b %d, %Y}."
            cols[i % 4].metric(lab, fmt(row[k], unit), delta, delta_color="off", help=info)

    st.divider()
    st.markdown("**Charts, frozen at this date**")
    lc1, lc2 = st.columns([3, 1])
    pick = lc1.multiselect("Indicators", [k for k in IND if k in w.columns],
                           default=["t10y", "short_rate", "baa_aaa_spread", "cpi_yoy"],
                           format_func=lambda k: IND[k][0])
    years = lc2.slider("Years of history", 2, 30, 10)
    start = asof - pd.DateOffset(years=years)
    sp = w.loc[start:asof, "sp500"]
    fig = go.Figure(go.Scatter(x=sp.index, y=sp, line=dict(color=BLUE, width=2), name="S&P 500",
                               hovertemplate="%{y:,.0f}<extra></extra>"))
    fig.update_layout(title=f"S&P 500 up to {label(asof)}")
    fig.update_yaxes(type="log")
    st.plotly_chart(style(fig, 300), width="stretch")
    palette = [BLUE, ORANGE, AQUA, VIOLET, "#e87ba4", "#008300"]
    ch = st.columns(2)
    for i, k in enumerate(pick):
        s = w.loc[start:asof, k].dropna()
        if s.empty:
            continue
        f2 = go.Figure(go.Scatter(x=s.index, y=s, line=dict(color=palette[i % len(palette)], width=2), name=IND[k][0]))
        f2.update_layout(title=f"{IND[k][0]} ({IND[k][2]})")
        ch[i % 2].plotly_chart(style(f2, 240), width="stretch")

# ------------------------------------------------------------------------------------------ 2. then vs today
COMPARE = ["cape", "sp500_ret_12m", "t10y", "t10y_chg_12m", "short_rate", "short_rate_chg_12m", "real_short_rate",
           "curve_10y_3m", "baa_aaa_spread", "baa_aaa_chg_12m", "cpi_yoy", "unemployment", "unemployment_chg_12m",
           "oil_yoy", "realized_vol", "indpro_yoy"]
with tabs[1]:
    asof = st.session_state.get("tm_asof", LAST)
    st.write(f"Comparing **{label(asof)}** with **{label(LAST)}**. The percentile shows where each reading ranks "
             "against every month since 1928 (0 = lowest ever, 100 = highest ever).")
    pct = w[COMPARE].rank(pct=True)
    tbl = pd.DataFrame({
        "Indicator": [IND[k][0] for k in COMPARE],
        label(asof): [fmt(w.loc[asof, k], IND[k][2]) for k in COMPARE],
        "Then (percentile)": [pct.loc[asof, k] * 100 for k in COMPARE],
        LAST_LABEL: [fmt(w.loc[LAST, k], IND[k][2]) for k in COMPARE],
        "Today (percentile)": [pct.loc[LAST, k] * 100 for k in COMPARE],
    })
    st.dataframe(tbl.style.format({"Then (percentile)": "{:.0f}", "Today (percentile)": "{:.0f}"}, na_rep="n/a"),
                 hide_index=True, width="stretch")

    st.markdown("**Which past crashes did today's setup look most like?**")
    ac = pd.read_csv(os.path.join(DATA, "analogs_crashes.csv")).head(8)
    ac["Episode"] = [NAR[str(i)]["title"] for i in ac.id]
    st.dataframe(ac[["Episode", "peak_date", "depth_pct", "crash_type", "similarity", "most_similar_on", "most_different_on"]]
                 .rename(columns={"peak_date": "Peak", "depth_pct": "Fall %", "crash_type": "Type",
                                  "similarity": "Similarity (0-100)", "most_similar_on": "Most alike on",
                                  "most_different_on": "Most different on"}),
                 hide_index=True, width="stretch")
    am = SUMM["analog_months_stats"]
    tt = SUMM["clusters"]["today"]
    st.info(md(f"Today's conditions sit closest to the **{tt['nearest_type']}** crash type. "
               f"Of the {am['n']} historical months most similar to today, **{am['share_with_10pct_drawdown']:.0%}** saw a "
               f"10%+ drop within the next year and **{am['share_with_20pct_drawdown']:.0%}** saw a 20%+ drop "
               f"(since 1950 the base rate for any month is about {MCT['base_rate_since_1950']['p_dd_10']:.0%} and "
               f"{MCT['base_rate_since_1950']['p_dd_20']:.0%}). Small sample: treat as a tilt, not a forecast."))

# ------------------------------------------------------------------------------------------ 3. crash library
with tabs[2]:
    types = ["All"] + sorted(lib.crash_type.unique())
    f1, f2 = st.columns(2)
    tsel = f1.selectbox("Crash type", types)
    ssel = f2.selectbox("Severity", ["All"] + sorted(lib.severity.unique()))
    view = lib.copy()
    if tsel != "All":
        view = view[view.crash_type == tsel]
    if ssel != "All":
        view = view[view.severity == ssel]
    view["Episode"] = [NAR[str(i)]["title"] for i in view.id]
    st.dataframe(view[["Episode", "peak_date", "trough_date", "depth_pct", "days_to_trough", "days_to_recover",
                       "crash_type", "recession_followed", "data"]]
                 .rename(columns={"peak_date": "Peak", "trough_date": "Bottom", "depth_pct": "Fall %",
                                  "days_to_trough": "Days to bottom", "days_to_recover": "Days to recover",
                                  "crash_type": "Type", "recession_followed": "Recession?", "data": "Price data"}),
                 hide_index=True, width="stretch", height=320)
    if tsel != "All":
        st.caption(SUMM.get("type_blurbs", {}).get(tsel, ""))

    pwp = os.path.join(DATA, "crash_library_prewar_daily.csv")
    if os.path.exists(pwp):
        with st.expander("1928-1949 on daily prices (more, sharper drops than the monthly view)"):
            st.caption("Before 1950 the library uses monthly averages, which smooth out the worst days. Measured on "
                       "daily closes, the same years contain 22 declines of 10%+, and the 1929-32 collapse shows up as "
                       "a series of crashes separated by 25%+ bear-market rallies.")
            st.dataframe(pd.read_csv(pwp), hide_index=True, width="stretch")
    pick_ep = st.selectbox("Open an episode", view.id.tolist(),
                           format_func=lambda i: f"{lib.set_index('id').loc[i, 'peak_date'][:7]} - {NAR[str(i)]['title']}")
    if pick_ep:
        e = lib.set_index("id").loc[pick_ep]
        n = NAR[str(pick_ep)]
        st.subheader(n["title"])
        a, b = st.columns(2)
        a.markdown("**Before the peak (what people knew)**")
        a.write(md(n["before"]))
        b.markdown("**What set it off**")
        b.write(md(n["trigger"]))
        key = f"reveal_{pick_ep}"
        if st.button("🔓 Reveal what happened next", key="btn_" + key):
            st.session_state[key] = True
        if st.session_state.get(key):
            st.success(md(f"**Outcome:** the S&P fell {abs(e.depth_pct):.1f}% over {e.days_to_trough} days "
                          f"(peak {e.peak_date}, bottom {e.trough_date})" +
                          (f" and took {int(e.days_to_recover)} days to get back to the old high." if not pd.isna(e.days_to_recover) else ".")))
            st.write(md(n["after"]))
            pk, tr = pd.Timestamp(e.peak_date), pd.Timestamp(e.trough_date)
            rng = (pk - pd.DateOffset(months=12), tr + pd.DateOffset(months=18))
            if pk.year >= 1950:
                s = DAILY.loc[rng[0]:rng[1]]
            else:
                s = w.loc[rng[0]:rng[1], "sp500"]
            fig = go.Figure(go.Scatter(x=s.index, y=s, line=dict(color=BLUE, width=2), name="S&P 500"))
            fig.add_vline(x=pk, line=dict(color="gray", dash="dot"), annotation_text="Peak")
            fig.add_vline(x=tr, line=dict(color="gray", dash="dot"), annotation_text="Bottom")
            fig.update_layout(title="S&P 500 around the episode")
            st.plotly_chart(style(fig, 300), width="stretch")
        st.caption("Sources: " + " · ".join(f"[{u.split('/')[-1].replace('_', ' ')}]({u})" for u in n["sources"]))

# ------------------------------------------------------------------------------------------ 4. monte carlo
with tabs[3]:
    st.write("Four ways to simulate the next year of the S&P 500. Each produces thousands of possible paths; "
             "we count how often the market falls 10%, 20% or 30% from a high along the way.")
    st.markdown(
        "- **Normal (classroom):** every day is an independent draw from one bell curve. Simple, but it treats calm "
        "and panicky periods the same and badly underestimates crashes.\n"
        "- **Block bootstrap:** replays real chunks of history (about 3 months at a time), so fat tails and "
        "streaks of bad days are kept.\n"
        "- **Regime-switching:** the market flips between a *calm* and a *stress* mode at historically measured rates, "
        "starting from whichever mode it's most likely in on the chosen date.\n"
        "- **Analog:** only replays what happened after the past months whose conditions most resembled the chosen date.")
    asof = st.session_state.get("tm_asof", LAST)
    o1, o2, o3, o4 = st.columns(4)
    mc_dates = [LAST] + ([asof] if asof != LAST and asof >= pd.Timestamp("1951-01-31") else [])
    run_date = o1.selectbox("Run as of", mc_dates, format_func=label)
    if asof < pd.Timestamp("1951-01-31"):
        o1.caption("Simulations need daily prices, which start in 1950, so pre-1951 dates run as of today.")
    horizon_m = o2.slider("Horizon (months)", 3, 24, 12)
    n_paths = o3.select_slider("Simulations", [1000, 2000, 5000, 10000], value=5000)
    k_an = o4.slider("Analog months", 10, 60, 25)
    block = st.slider("Bootstrap block length (trading days)", 5, 126, 63)

    @st.cache_data(show_spinner=False)
    def run_mc(run_date, horizon_m, n_paths, k_an, block):
        r = MC.daily_returns()
        hist = r[r.index <= run_date] if run_date != LAST else r
        H = int(round(horizon_m * 21))
        rng = np.random.default_rng(42)
        r_reg = hist.values[-252 * 40:]          # up to 40 years before the date, same as the validation test
        hmm = MC.fit_hmm(r_reg, n_iter=40)
        start = MC.filter_last(hist.values[-252:], hmm)
        months_, _ = MC.analog_windows(w, MC.ANALOG_FEATS, run_date, k=k_an)
        sims = {"Normal": MC.sim_normal(hist.values, n_paths, H, rng),
                "Block bootstrap": MC.sim_bootstrap(hist.values, n_paths, H, block, rng),
                "Regime-switching": MC.sim_regime(r_reg, hmm, start, n_paths, H, rng),
                "Analog": MC.sim_analog(hist, months_, n_paths, H, rng=rng)}
        out = {k: MC.max_drawdown_paths(v) for k, v in sims.items()}
        actual = MC.actual_forward_dd(r, run_date, H) if run_date != LAST else None
        return out, float(start[1]), [str(m.date()) for m in months_], actual

    with st.spinner("Simulating..."):
        dds, p_stress, an_months, actual = run_mc(run_date, horizon_m, n_paths, k_an, block)

    res = pd.DataFrame({k: {f"{int(t*100)}%+ drop": float((v >= t).mean()) for t in (0.1, 0.2, 0.3)} for k, v in dds.items()}).T
    base = MCT["base_rate_since_1950"]
    res.loc["History (any month since 1950, 12m)"] = [base["p_dd_10"], base["p_dd_20"], base["p_dd_30"]]
    st.dataframe(res.style.format("{:.0%}"), width="stretch")
    st.caption(f"Regime model: probability the market is in its stress mode on this date = {p_stress:.0%}. "
               f"Analog months used: {', '.join(m[:7] for m in an_months[:10])}{' ...' if len(an_months) > 10 else ''}")
    if actual is not None:
        st.success(md(f"**What actually happened:** the deepest peak-to-trough drop over the next {horizon_m} months "
                      f"was {actual:.1%}."))

    fig = go.Figure()
    cols_ = [BLUE, ORANGE, AQUA, VIOLET]
    for (k, v), c in zip(dds.items(), cols_):
        fig.add_histogram(x=v * 100, name=k, marker_color=c, opacity=0.55, nbinsx=60,
                          hovertemplate=f"{k}<br>%{{x:.0f}}%% drop: %{{y}} paths<extra></extra>")
    fig.update_layout(barmode="overlay", title="Distribution of the worst drop along each simulated path")
    fig.update_xaxes(title="Worst drop from a high (%)")
    st.plotly_chart(style(fig, 320), width="stretch")

    if VAL:
        st.markdown("**Did the models actually work? (walk-forward test, 1965-2025)**")
        st.write("At every month-end, each model was fitted only on data available then, made a forecast, and was "
                 "scored against what happened next. Brier score: lower is better. Skill: % better than just guessing "
                 "the historical average (positive = useful).")
        rows = []
        for per in ("1965-2025", "1965-1994", "1995-2025"):
            for t in ("p_dd_10", "p_dd_20"):
                d = VAL[per][t]
                for m in ("normal", "bootstrap", "regime", "analog", "base"):
                    rows.append({"Period": per, "Event": t.replace("p_dd_", "") + "%+ drop", "Model": m,
                                 "Brier": d[m]["brier"], "Skill vs base rate": d[m]["skill_vs_base"],
                                 "Avg forecast": d[m]["avg_forecast"], "Actual rate": d["actual_rate"]})
        vt = pd.DataFrame(rows)
        best = VAL["1965-2025"]["p_dd_10"]
        sk = {m: best[m]["skill_vs_base"] for m in ("normal", "bootstrap", "regime", "analog")}
        st.warning(md(f"**Verdict:** over 60 years none of the models meaningfully beat simply using the historical average "
                      f"(skill for the 10% event: " + ", ".join(f"{k} {v:+.0%}" for k, v in sk.items()) +
                      "; anything within a couple of percent of zero is noise). The classroom bell curve is the worst: it badly underestimates 20-30% crashes. Use these to "
                      "size how bad things *can* get, not to time when."))
        st.dataframe(vt.style.format({"Brier": "{:.3f}", "Skill vs base rate": "{:+.0%}", "Avg forecast": "{:.0%}",
                                      "Actual rate": "{:.0%}"}), hide_index=True, width="stretch", height=380)

# ------------------------------------------------------------------------------------------ 5. thesis test
def gap_table(t):
    t = t.copy()
    t["Episode"] = [NAR[str(int(i))]["title"] for i in t.id]
    t["Credit tightened?"] = np.where(t.credit_tightened.astype(str).str.lower() == "true", "Yes", "No")
    return t[["Episode", "peak_date", "sp500_depth", "heavy_avg", "light_avg", "gap_heavy_minus_light",
              "credit_spread_change_during", "Credit tightened?"]].rename(
        columns={"peak_date": "Peak", "sp500_depth": "S&P fall %", "heavy_avg": "Heavy %", "light_avg": "Light %",
                 "gap_heavy_minus_light": "Gap (heavy - light)", "credit_spread_change_during": "Credit spread change"})


def gap_bar(stats, title):
    fig = go.Figure(go.Bar(x=["Credit spreads widened", "Credit spreads stayed calm"],
                           y=[stats["gap_when_tightened"], stats["gap_when_not"]],
                           marker=dict(color=[STATUS["critical"], STATUS["good"]], cornerradius=4),
                           text=[f"{stats['gap_when_tightened']:+.1f} pts", f"{stats['gap_when_not']:+.1f} pts"],
                           textposition="outside"))
    fig.update_layout(title=title, hovermode="closest")
    return style(fig, 300)


with tabs[4]:
    st.write("Your thesis: when credit tightens, the leveraged, capital-hungry parts of the market get hit hardest. "
             "For every decline we compare **capital-heavy** industries (utilities, oil, steel, chemicals, machinery, "
             "transport, banks, real estate, telecom, autos, construction...) with **asset-light** ones (software, "
             "hardware, chips, drugs, food, retail, services...). The gap is how many points worse (negative) the heavy "
             "group did from the S&P's peak to its bottom.")
    SS = json.load(open(os.path.join(DATA, "sector_stats.json"))) if os.path.exists(os.path.join(DATA, "sector_stats.json")) else {}
    lp = os.path.join(DATA, "sector_test_long.csv")
    if "french_1928_plus" in SS and os.path.exists(lp):
        a, b = SS["french_1928_plus"], SS["french_1950_plus"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Heavy did worse", f"{a['share_heavy_worse']:.0%} of declines", help="Share of all 41 declines since 1928")
        c2.metric("Extra underperformance when credit tightened", f"{a['difference']:+.1f} pts",
                  f"p = {a['perm_p_value']:.2f} (1928+) · {b['difference']:+.1f} pts, p = {b['perm_p_value']:.2f} (1950+)",
                  delta_color="off")
        rs = SS.get("rate_shock_only", {})
        c3.metric("Gap after a Fed hiking year", f"{rs.get('gap_after_rate_rise', np.nan):+.1f} pts",
                  f"vs {rs.get('gap_otherwise', np.nan):+.1f} otherwise (no difference)", delta_color="off")
        st.plotly_chart(gap_bar(a, "1928-2025, Ken French industries: average heavy-minus-light gap"), width="stretch")
        st.info("**How to read this:** the direction matches the thesis: heavy industries lag more when credit spreads "
                "widen during the fall, and rising short rates by themselves don't produce the gap. But the p-values "
                "(chance of seeing a gap this big by luck) are 0.09-0.17, so it's suggestive, not proven. "
                "Big exceptions: 1973 and 2022, when oil soared and dragged the 'heavy' group up.")
        with st.expander("All 41 declines (industry data)"):
            st.dataframe(gap_table(pd.read_csv(lp)), hide_index=True, width="stretch")
    if "investment_1963_plus" in SS:
        iv = SS["investment_1963_plus"]
        st.markdown("**Sharper version: sorted by how aggressively companies invest (1963+)**")
        st.write("Instead of guessing by industry, this uses Ken French's portfolios of companies sorted on asset "
                 "growth (how fast they're expanding their balance sheet, i.e. building and buying with new money). "
                 "Top 20% = aggressive builders, bottom 20% = conservative.")
        st.plotly_chart(gap_bar(iv, "Aggressive minus conservative investors, average gap during each decline"),
                        width="stretch")
        st.caption(f"Aggressive builders did worse in {iv['share_heavy_worse']:.0%} of {iv['n']} declines. Median gap: "
                   f"{iv['median_when_tightened']:+.1f} pts when credit widened vs {iv['median_when_not']:+.1f} when calm; "
                   f"without 2000 the widened-credit average is still {iv['gap_when_tightened_ex2000']:+.1f}. "
                   f"Permutation p = {iv['perm_p_value']:.2f}.")
        ivp = os.path.join(DATA, "sector_test_investment.csv")
        if os.path.exists(ivp):
            with st.expander("All 28 declines (asset-growth portfolios)"):
                st.dataframe(gap_table(pd.read_csv(ivp)).rename(columns={"Heavy %": "Aggressive %", "Light %": "Conservative %"}),
                             hide_index=True, width="stretch")
    st.markdown("**Sector-fund version (1999+)**")
    if "etf_1999_plus" in SS:
        st.plotly_chart(gap_bar(SS["etf_1999_plus"], "Since 1999, sector ETFs: average heavy-minus-light gap"),
                        width="stretch")
    st.dataframe(gap_table(SECTOR), hide_index=True, width="stretch")
    st.caption("Heavy ETFs = XLU, XLE, XLI, XLB, XLF. Light = XLK, XLP, XLV, XLY. 'Credit tightened' = the BAA-AAA "
               "corporate bond spread widened by more than 0.15 points between the month before the peak and the bottom.")

# ------------------------------------------------------------------------------------------ 6. warning lights
with tabs[5]:
    ewp = os.path.join(DATA, "early_warning.csv")
    if os.path.exists(ewp):
        EW = pd.read_csv(ewp)
        EWT = json.load(open(os.path.join(DATA, "early_warning_today.json")))
        oos = EWT["oos_test"]
        st.write("Simple yes/no warning lights, each built only from data public at the time. For every month since "
                 "1955 we check whether the light was on, then whether the S&P fell **20% or more** (from that "
                 "month's level or a later high) within the next 12 months. That happened after "
                 f"**{EWT['base_rate_pct']:.0f}%** of all months.")
        c1, c2 = st.columns([1, 2])
        c1.metric("Stable lights on today", f"{oos['today_count']} of {len(oos['lights_picked_on_1955_1990'])}")
        on_core = [k for k in EWT["lights_on"] if k in oos["lights_picked_on_1955_1990"]]
        on_other = [k for k in EWT["lights_on"] if k not in oos["lights_picked_on_1955_1990"]]
        c1.caption("Counted lights on: " + ", ".join(on_core) +
                   (". Also on (not in the counter): " + ", ".join(on_other) if on_other else ""))
        o = pd.DataFrame(oos["out_of_sample_1991_2025"])
        fig = go.Figure(go.Bar(x=[f"{r} lights on" for r in o.lights_on], y=o.hit_rate_pct,
                               marker=dict(color=[STATUS["good"], STATUS["warning"], STATUS["critical"]], cornerradius=4),
                               text=[f"{v:.0f}%" for v in o.hit_rate_pct], textposition="outside"))
        fig.add_hline(y=EWT["base_rate_pct"], line=dict(color="gray", dash="dot"), annotation_text="all months")
        fig.update_layout(title="Out-of-sample test, 1991-2025: chance of a 20%+ drop within a year",
                          hovermode="closest")
        c2.plotly_chart(style(fig, 300), width="stretch")
        st.info("**How this was kept honest:** the lights in the counter were picked using 1955-1990 only (those that "
                "raised the odds back then: " + ", ".join(oos["lights_picked_on_1955_1990"]) + "), then tested on "
                "1991-2025, which they never saw. They still worked. It's a tilt in the odds, not a timer: plenty of "
                "2-light months were followed by nothing.")
        st.markdown("**Every light, one by one**")
        st.dataframe(EW.rename(columns={"light": "Warning light", "lift_1955_1990": "Lift 1955-90",
                                        "lift_1991_2025": "Lift 1991-2025", "months_on_pct": "% of months on",
                                        "hit_rate_pct": "20%+ drop followed (%)", "base_rate_pct": "Base rate (%)",
                                        "lift": "Lift", "p_value": "p (luck)", "crashes_caught": "Big drops caught",
                                        "since": "Data since", "on_today": "On today?"}),
                     hide_index=True, width="stretch")
        st.caption("Lift = how much the light raised the odds vs. a normal month (1.0 = no information). p (luck) "
                   "shifts the light's on/off history in time 2,000 times to see how often a lift this big appears "
                   "by chance. 'Big drops caught' = of the 20%+ declines since the light's data begin, how many had "
                   "it on in the 12 months before the peak. Note the 10-year at 5% on its own told you nothing.")

# ------------------------------------------------------------------------------------------ 7. method
with tabs[6]:
    st.markdown(md(open(os.path.join(ROOT, "timemachine", "METHOD.md")).read()) if os.path.exists(os.path.join(ROOT, "timemachine", "METHOD.md")) else "See the report.")
