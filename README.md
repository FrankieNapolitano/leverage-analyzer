## Web version
Live on Streamlit Community Cloud (main file: `app.py`). Data in `timemachine/data/` refreshes automatically
every weekday via GitHub Actions (`.github/workflows/refresh-data.yml`); you can also run it by hand from the
repo's **Actions** tab → *Refresh Time Machine data* → **Run workflow**.

Secrets (Streamlit app → Settings → Secrets), all optional:
```
OWNER_ANTHROPIC_KEY = "sk-ant-..."   # your key, only used after the password below is entered
APP_PASSWORD = "pick-something"
```
Optional GitHub secret `FRED_API_KEY` (repo → Settings → Secrets → Actions) keeps first-release data current.

# Leverage Analyzer

Type any ticker -> get its **leverage profile** (how much debt, how well it can carry it, whether it's
building with borrowed money, how it reacts to interest rates), a **directional bias**, a **stock view**
(long / avoid / short candidate), and a specific **options trade** with real strikes, cost, max loss,
breakeven and position size. Claude writes a plain-English explanation of all of it.

Built on the AI Buildout Financing Thesis: the most leveraged links get hurt first when rates rise.

## Start it
**Windows:** double-click **setup.bat** the first time, then **run.bat** (or the Desktop shortcut).
**Mac:** right-click **setup_mac.command** > Open the first time, then **run_mac.command**.

## Backtest (does the score predict anything?)
Double-click **backtest.bat** (Windows) or **backtest_mac.command** (Mac). Takes 20-45 minutes.
It rebuilds the score for ~230 stocks at past dates using only data public at the time, then checks
what happened over the next 6-12 months. Results: backtest_results.csv + backtest_summary.txt.

## Market Time Machine (new page)
Click **Time Machine** in the sidebar. Relive any month since 1928 using only data that was public then,
browse all 41 S&P 500 declines of 10%+ (with what people knew going in), compare any date to today,
run four Monte Carlo crash models, and see the leveraged-vs-asset-light thesis test back to 1928.
Refresh the data: **update_timemachine.bat** (Windows) or **update_timemachine_mac.command** (Mac).
Whole database in Excel: timemachine/data/TimeMachine_database.xlsx. Methods: timemachine/METHOD.md.

## Files
| File | What it does |
|---|---|
| engine.py | All the math: pulls data from Yahoo Finance, leverage metrics, risk score, bias, options strategy |
| ai_writer.py | Builds the prompt and asks Claude to explain the numbers |
| app.py | The main dashboard page |
| backtest.py | Historical test of whether the risk score predicts returns and drawdowns |
| pages/1_AI_Chain_Scan.py | Ranks the whole AI chain (Oracle, CoreWeave, hyperscalers, chipmakers) by leverage risk |

## How the score works (0 = safe, 100 = fragile)
| Component | Weight | Danger zone |
|---|---|---|
| Interest coverage | 22% | < 2x |
| Debt growth (1 yr) | 22% | > 25% |
| Free cash flow margin | 20% | negative |
| Capex / operating cash flow | 15% | > 1.0x (borrowing to build) |
| Net debt / EBITDA | 8% | > 4x (utilities graded on a curve) |
| Rate sensitivity | 8% | only counted if statistically significant (t <= -2) |
| Altman Z-score | 5% | < 1.8 |

Weights were re-set after the Oct 2026 backtest. Key findings: past volatility predicts drawdowns
(used for the "typical worst drop" estimate and stock sizing); trend weakly predicts returns; the
leverage score alone did not predict underperformance, so it now only nudges the bullish/bearish call.

Tiers: Fortress < 25 <= Healthy < 45 <= Stretched < 65 <= Fragile

## How the trade is picked
Bias (balance sheet + trend + rates) x option prices (implied vs. realized volatility):

| | Options cheap/fair | Options expensive |
|---|---|---|
| Bullish | Bull call debit spread | Bull put credit spread |
| Bearish | Bear put debit spread | Bear call credit spread |
| Neutral | No trade (or protective put if fragile) | Iron condor |

All trades are defined-risk. Position size = contracts that keep max loss under your risk limit.

**Educational tool, not financial advice.** Yahoo data can be wrong or stale; verify with your broker.
