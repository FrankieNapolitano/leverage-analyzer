"""
ai_writer.py - sends the computed numbers to Claude and gets back a plain-English write-up.
Python does all the math (engine.py). Claude only interprets.
The API key is read from the ANTHROPIC_API_KEY environment variable (set once by setup.bat).
"""

import os

MODELS = {
    "Claude Sonnet 5.5 (recommended)": "claude-sonnet-5-5",
    "Claude Opus 5.5 (deepest, costs more)": "claude-opus-5-5",
    "Claude Haiku 5.5 (fastest, cheapest)": "claude-haiku-5-5",
}

SYSTEM = (
    "You are a sharp buy-side analyst explaining a stock's leverage profile and an options trade idea "
    "to a 19-year-old finance student who has some options experience. Write casually but precisely. "
    "Use ONLY the numbers provided; never invent figures, news, or events. If a number is missing, say so. "
    "Be honest about weaknesses in the setup. This is educational analysis, not personalized advice."
)


def _f(x, kind="num"):
    if x is None:
        return "n/a"
    if x == float("inf"):
        return "no meaningful debt/interest" if kind != "pct" else "n/a"
    if kind == "pct":
        return f"{x:.1%}"
    if kind == "bn":
        return f"{'-' if x < 0 else ''}${abs(x) / 1e9:,.1f}B"
    if kind == "x":
        return f"{x:.2f}x"
    return f"{x:,.2f}"


def build_prompt(a, account=None, risk_pct=None, contracts=None):
    L, M, O, S, B = a["lev"], a["mkt"], a["options"], a["strategy"], a["bias"]
    lines = [
        f"Company: {a['name']} ({a['ticker']}), sector: {a['sector']} / {a['industry']}. Data pulled {a['fetched']}.",
        "",
        "LEVERAGE (trailing 12 months / latest balance sheet):",
        f"- Total debt {_f(L['total_debt'], 'bn')}, cash {_f(L['cash'], 'bn')}, net debt {_f(L['net_debt'], 'bn')}",
        f"- Net debt / EBITDA: {_f(L['net_debt_to_ebitda'], 'x')} (negative = more cash than debt)",
        f"- Interest coverage (EBIT / interest): {_f(L['interest_coverage'], 'x')}",
        f"- Debt / equity: {_f(L['debt_to_equity'], 'x')}{' (NEGATIVE EQUITY)' if L['negative_equity'] else ''}",
        f"- Free cash flow: {_f(L['fcf_ttm'], 'bn')}, FCF margin {_f(L['fcf_margin'], 'pct')}",
        f"- Capex / operating cash flow: {_f(L['capex_to_ocf'], 'x')} (above 1.0x = spending more than operations bring in)",
        f"- Debt growth over the last year: {_f(L['debt_growth_1y'], 'pct')}",
        f"- Altman Z-score: {_f(L['altman_z'])} (>3 safe, 1.8-3 grey zone, <1.8 distress)",
        f"- Cash runway if FCF stays negative: {_f(L['cash_runway_yrs'])} years",
        f"- Leverage risk score: {a['risk_score']:.0f}/100 -> tier: {a['tier']}",
        "  Component risk scores (0 good, 100 bad): "
        + ", ".join(f"{k}={v:.0f}" for k, v in a["scores"].items() if v is not None),
        "",
        "MARKET:",
        f"- Price ${M['price']:,.2f}; 50-day avg {_f(M['sma50'])}; 200-day avg {_f(M['sma200'])}; "
        f"6-month return {_f(M['ret_6m'], 'pct')}; {_f(M['off_52w_high'], 'pct')} vs 52-week high",
        f"- Beta vs S&P 500: {_f(M['beta'])}; annualized volatility 1y {_f(M['vol_1y'], 'pct')}, 3m {_f(M['vol_3m'], 'pct')}",
        f"- Rate sensitivity: {_f(M['rate_sens_25bp'], 'pct')} per +25 bps in the 10-year yield beyond the market "
        f"(t-stat {_f(M['rate_tstat'])}; only meaningful if t <= -2)",
        f"- 10-year Treasury yield now {M['tnx_now']:.2f}%, change over 3 months {M['tnx_chg_3m']:+.2f} pp",
        "",
        f"DIRECTIONAL BIAS: {B['bias']} (conviction {B['conviction']}, score {B['score']:+.1f}). Inputs: "
        + "; ".join(B["reasons"]),
        f"STOCK VIEW: {a['stock_view']}",
    ]
    dd = a.get("exp_drawdown")
    if dd:
        lines.append(f"TYPICAL WORST DRAWDOWN NEXT 12 MONTHS (volatility model from backtest): {dd['mid']:.0%} "
                     f"(range {dd['bad']:.0%} to {dd['mild']:.0%})")
    lines.append("BACKTEST CONTEXT (199 stocks, 2022-2026): volatility predicted drawdowns; trend weakly predicted "
                 "returns; the leverage score by itself did NOT predict underperformance (fragile names outperformed in "
                 "2024), though debt growth flagged bigger drops. Do not present leverage as a standalone short signal.")
    if O:
        lines += [
            "",
            f"OPTIONS: expiry {O['expiry']} ({O['dte']} days). ATM implied vol {_f(O['atm_iv'], 'pct')} vs "
            f"3-month realized vol {_f(O['rv'], 'pct')} -> IV/RV {_f(O['iv_rv'])} = options look {O['regime']}. "
            f"Expected move by expiry: +/- ${O['exp_move']:.2f}.",
        ]
    if S:
        lines.append(f"SUGGESTED STRATEGY: {S['name']}. Reasoning from the model: {S['why']}")
        for leg in S["legs"]:
            lines.append(f"  {leg['action']} 1 {leg['strike']:g} {leg['kind']} @ ${leg['price']:.2f}")
        if S["legs"]:
            net = S["net"]
            max_p = "unlimited" if S["unbounded_up"] else f"${S['max_profit']:,.0f}"
            bes = ", ".join(f"${b:.2f}" for b in S["breakevens"])
            lines.append(
                f"  Net {'credit' if net > 0 else 'debit'} ${abs(net):,.0f} per contract set; max profit {max_p}; "
                f"max loss ${abs(S['max_loss']):,.0f}; breakeven(s) {bes}; "
                f"model probability of profit {_f(S['pop'], 'pct')}")
        if account and contracts is not None:
            lines.append(f"  Position size for a ${account:,.0f} account risking {risk_pct:.0%}: {contracts} contract set(s).")
        for w in S.get("warnings", []):
            lines.append(f"  Liquidity warning: {w}")
    for n in a["notes"]:
        lines.append(f"NOTE: {n}")

    lines += [
        "",
        "Context: the user follows a thesis that the AI infrastructure buildout is increasingly funded by debt and "
        "vendor financing while the 10-year yield is above 5%, so the most leveraged links (e.g., Oracle, neoclouds) "
        "are most at risk. Only connect this company to that thesis if its numbers or industry make it relevant.",
        "",
        "Write the analysis with these sections (markdown headers, under 450 words total):",
        "1. **Verdict** - 2 sentences: how strong or fragile the balance sheet is and what that means for the stock.",
        "2. **What the balance sheet says** - the 3-4 numbers that matter most, in plain English.",
        "3. **Rates & the debt cycle** - how rising/falling rates affect this company specifically.",
        "4. **The trade** - why the suggested stock view and options strategy fit (or where they don't), "
        "plus simple management rules (when to take profit, when to cut).",
        "5. **What would change the view** - 2-3 specific signals to watch.",
    ]
    return "\n".join(lines)


def write_summary(a, model_label, api_key=None, account=None, risk_pct=None, contracts=None):
    import anthropic

    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("No Anthropic API key found. Run setup.bat again or paste a key in the sidebar.")
    client = anthropic.Anthropic(api_key=key)
    msg = client.messages.create(
        model=MODELS[model_label],
        max_tokens=1800,
        system=SYSTEM,
        messages=[{"role": "user", "content": build_prompt(a, account, risk_pct, contracts)}],
    )
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
