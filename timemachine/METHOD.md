### How the Time Machine works

**No hindsight.** Every value on the Time Machine tab is what had been *published* by that month-end. Where the St. Louis Fed's ALFRED archive has the first-published number (CPI, core CPI, groceries, gasoline, unemployment, payrolls, GDP, industrial production, housing starts, M2), that's what you see. Otherwise the revised number is used but held back until its normal release date (for example CPI about 47 days, GDP about 4 months, Case-Shiller about 3 months). Hover any card to see which method was used. Market prices (rates, spreads, S&P, VIX, oil) are known the same day and use the monthly average, or the latest daily reading for the current month. Recessions count only once the NBER announced them, which it only started doing in real time in 1980.

**Crash library.** A decline starts at an S&P 500 peak and counts once it is 10% or more below it. Its bottom is the lowest close before either a full recovery or a 25% rally off the low. Daily closes are used from 1950; Shiller monthly averages before that, which understates some 1930s drops. That gives 41 declines from 1928 to 2025.

**Crash types.** Eleven readings from the month before each peak (valuation vs. its 20-year norm, prior-year return, rate changes, real rate, curve, credit spread vs. norm and its change, volatility vs. norm, inflation change, industrial production) are ranked within history and clustered with k-means (k = 4). The types overlap (silhouette 0.15), so treat them as tendencies.

**Analogs.** Distance between today and each past setup on standardized readings. Only past months whose next 12 months were already known are used.

**Monte Carlo.** There are four models of the next year of daily S&P returns:

- Normal: an independent bell curve.
- Stationary block bootstrap: real history in chunks of about 3 months.
- Two-state hidden-Markov regime model: fitted on up to 40 years of data, it resamples real calm and stress days.
- Analog bootstrap: replays the year after the 25 most similar past months.

They were validated walk-forward at every month-end from 1965 to 2025, using only data available at the time and scored with Brier scores against an expanding historical average. None reliably beat the average.

**Thesis test.** For each decline, compare the return of 19 capital-heavy industries with 19 asset-light ones (Ken French 49 industries, 1928+; sector ETFs 1999+). Split by whether the BAA-AAA spread widened more than 0.15 points during the fall, then run permutation tests.

### Limits
- Small sample: 41 declines, and fewer within each type.
- Pre-1960s macro data are revised-then-lagged, not truly first-release.
- Junk spreads only go back 3 years, so BAA-AAA is the long-run credit gauge.
- Margin debt and bank failures are not included yet.
- Regimes change: 1937 is not 2026.
- Nothing here is financial advice.
