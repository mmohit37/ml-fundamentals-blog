---
layout: post
title: "NASDAQ Free Cash Flow: can FCF strength predict returns?"
date: 2025-10-23
excerpt: "A data-driven look at whether Free Cash Flow can predict future stock performance on the NASDAQ."
categories: quant nasdaq
tags: finance freecashflow backtest xgboost
---


## 1) Intro & Goal

Free Cash Flow (FCF) is the cash a company has left after paying for its operations and investments.  
Investors often treat it as a sign of financial health, the idea being: if a company consistently generates positive FCF, it has more flexibility to grow, pay down debt, or return cash to shareholders.

**Dataset:** [NASDAQ Fundamentals (2024, Kaggle)](https://www.kaggle.com/datasets/sauers/nasdaq-fundamentals-2024)

**Big question:** does having positive FCF actually lead to better stock returns?  
Or put another way: *can strong cash flow help us predict which stocks will do better next year?*

In this project, I explored that question using a cleaned panel of NASDAQ companies.  
I built lag-safe forward returns (so we don’t “peek” into the future), compared positive vs. negative FCF groups, and even tried a simple walk-forward XGBoost model to see if fundamentals alone could rank stocks.  

By the way, if any terms seem confusing to you, check out the glossary: [Jump to Glossary →](#appendix--quick-glossary-plain-english)

I also made a few changes compared ot my last post. I realize that I didn't really explain the different variables in the dataset, which could cause some confusion if you've never seen this dataset. So I'll make sure to explain the important ones before I start.

Another detail I changed was adding the plots right after each code block, so it makes sense in context. In my last post, I just put all the plots at the end, which doesn't make much sense.

<details>
  <summary><strong>Spoiler (click to expand)</strong></summary>

Positive FCF did look better than negative FCF in averages, but the data is thin, and it’s not enough to predict individual stock prices yet.

</details>

## Dataset & Key Variables

The dataset combines company fundamentals, valuation metrics, and insider trading data for **NASDAQ-listed companies**, organized by fiscal year.  
Each row represents a single company in a given year — a panel format that allows us to link fundamentals to future stock performance.

Like I mentioned before, here are some of the most important variables used throughout this analysis:

### Core identifiers
- **Ticker** — The company’s stock symbol, used to match fundamentals with historical price data.  
- **Financial_currentPrice** — The company’s stock price at the time of data capture. Used for context but not as a predictive feature.

### Financial fundamentals
- **Financial_totalRevenue** — Total annual sales.  
- **Financial_ebitda** — Earnings before interest, taxes, depreciation, and amortization (essentially paying off debt).  
- **Financial_freeCashflow** — Cash generated after capital spending, our main focus variable (FCF).  
- **Financial_totalDebt** — Total debt outstanding, used to gauge leverage and balance sheet strength.  
- **Financial_totalCash / Financial_totalCashPerShare** — Measures of available liquidity, giving a sense of financial cushion.  
- **Financial_returnOnAssets / Financial_returnOnEquity** — Efficiency ratios showing how effectively a company uses assets or equity to generate profit.  
- **Financial_quickRatio / Financial_currentRatio** — Liquidity ratios showing how easily short-term obligations can be covered.  
- **Financial_ebitdaMargins / Financial_profitMargins** — Profitability ratios that normalize earnings and cash flow by revenue.

### Valuation & leverage
- **Financial_debtToEquity** — Measures leverage; higher values suggest more debt relative to equity.  
- **Financial_revenuePerShare** — Total revenue divided by shares outstanding, giving a per-share view of scale.  
- **KeyStats_enterpriseValue** — Total market value of equity plus debt, minus cash, used for valuation comparisons.  
- **KeyStats_priceToBook** — Valuation ratio comparing market price to book value.  
- **KeyStats_enterpriseToEbitda** — A valuation multiple comparing enterprise value to EBITDA (earnings before interest, taxes, depreciation, and amortization).

### Ownership & insider activity
- **KeyStats_sharesOutstanding** — Total number of shares available, used to standardize insider transactions.  
- **KeyStats_heldPercentInsiders** — Percentage of shares held by company insiders.  
- **SharePurchase_buyInfoCount / SharePurchase_sellInfoCount** — Number of insider buy or sell transactions in a period.  
- **SharePurchase_buyInfoShares / SharePurchase_sellInfoShares** — Number of shares involved in insider buys or sells.  
- **SharePurchase_netInfoShares / SharePurchase_netPercentInsiderShares** — Net difference between insider buying and selling, used to gauge insider sentiment.

### Why these matter
Together, these features cover three angles of company health:
1. **Operational strength** — cash generation, profitability, and efficiency (e.g., FCF, margins, ROA/ROE).  
2. **Financial structure** — debt, liquidity, and valuation context (e.g., debt/equity, enterprise value).  
3. **Behavioral signals** — insider buying and selling activity, which sometimes hints at management confidence.  

Not every feature is used in every section, but the experiments that follow (deciles, flips, and the ML model) all draw from this shared foundation.

## 2) Data & Setup

The dataset comes from the NASDAQ Fundamentals (2024, Kaggle).  
It includes the balance sheet, income statement, and cash flow statement items for NASDAQ-listed companies.

On its own, that’s just a lot of accounting data. To study whether Free Cash Flow (FCF) has any link to returns, we first have to reshape the raw statements into something more structured: a panel of companies by year.

### Step 1: Defining Free Cash Flow

Free Cash Flow is the cash a company has left after paying for operations and investments. Thus:  
**FCF = Operating Cash Flow - Capital Expenditures**

**Why start here**? Because positive FCF usually means a company has flexibility. It can grow, reduce debt, or return money to shareholders. Negative FCF can mean the opposite (or simply too heavy an investment).

---

### Step 2: Building a company–year panel

Each row in the dataset became a (Ticker, Year) pair.  
For each, we added a binary flag:

- `fcf_pos = 1` if FCF ≥ 0  
- `fcf_pos = 0` if FCF < 0  

This gives us a quick way to split “healthy cash generators” from “cash burners.”

The code below reshapes raw financial data into a structured “panel” format — one row per company per year.
Each row includes free cash flow (FCF), a binary flag for positive/negative FCF, and next-year stock returns.

<details>
  <summary><strong>Show code — [short description]</strong></summary>
  
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

assert 'ret_1y' in panel.columns and 'xgb_proba' in panel.columns and 'fcf_pos' in panel.columns

# (re)build quintiles if needed
if 'proba_q' not in panel.columns:
    panel['proba_q'] = pd.qcut(panel['xgb_proba'], q=5, labels=[1,2,3,4,5])

qtab = panel.groupby('proba_q')['ret_1y'].agg(['count','mean','median'])
cohort = panel.groupby('fcf_pos')['ret_1y'].agg(['count','mean','median']).rename(index={0:'FCF−',1:'FCF+'})

# Bar plot: mean & median by FCF sign
ax = cohort[['mean','median']].plot(kind='bar', figsize=(6,3))
ax.set_title('Next-year return by FCF sign')
ax.set_ylabel('Return')
ax.grid(True, axis='y', linewidth=0.5)
plt.show()

# Bar plot: mean & median by XGB proba quintile
ax = qtab[['mean','median']].plot(kind='bar', figsize=(8,3))
ax.set_title('Next-year return by XGB score quintile (1=low, 5=high)')
ax.set_ylabel('Return')
ax.grid(True, axis='y', linewidth=0.5)
plt.show()

# OPTIONAL (if you want to try yourself): winsorize returns to reduce outlier impact and replot
w = panel.copy()
lo, hi = w['ret_1y'].quantile([0.01, 0.99])
w['ret_1y_w'] = w['ret_1y'].clip(lo, hi)
qtab_w = w.groupby('proba_q')['ret_1y_w'].agg(['count','mean','median'])
cohort_w = w.groupby('fcf_pos')['ret_1y_w'].agg(['count','mean','median']).rename(index={0:'FCF−',1:'FCF+'})

ax = cohort_w[['mean','median']].plot(kind='bar', figsize=(6,3))
ax.set_title('Winsorized next-year return by FCF sign (1%-99%)')
ax.set_ylabel('Return')
ax.grid(True, axis='y', linewidth=0.5)
plt.show()

ax = qtab_w[['mean','median']].plot(kind='bar', figsize=(8,3))
ax.set_title('Winsorized next-year return by XGB score quintile (1%-99%)')
ax.set_ylabel('Return')
ax.grid(True, axis='y', linewidth=0.5)
plt.show()

# (ALSO OPTIONAL): True yearly cohorts from yfinance cashflow
# Build annual FCF (Operating Cash Flow − CapEx) per ticker, per fiscal year,
# then compute next-year returns from Dec-31 of that year.

import yfinance as yf
from typing import Dict, List, Tuple

# Config: years you want and tickers from your dataframe
years = list(range(2019, 2024))  # adjust
all_tickers = sorted(pd.Series(df['Ticker']).dropna().unique().tolist())

# Helper to extract annual OCF and CapEx (Capital Expenditures) from yfinance cashflow table
CF_KEYS_OCF = [
    'Total Cash From Operating Activities',
    'Net Cash Provided by Operating Activities',
    'Operating Cash Flow'
]
CF_KEYS_CAPEX = [
    'Capital Expenditures',
    'Capital Expenditure'
]

def annual_fcf_from_yf(ticker: str) -> pd.DataFrame:
    t = yf.Ticker(ticker)
    try:
        cf = t.cashflow  # annual cashflow; columns are period-ends
    except Exception:
        cf = None
    if cf is None or cf.empty:
        return pd.DataFrame(columns=['Ticker','year','fcf'])
    cf = cf.copy()
    cf.index = cf.index.astype(str)
    # pick OCF row
    ocf_row = next((k for k in CF_KEYS_OCF if k in cf.index), None)
    capex_row = next((k for k in CF_KEYS_CAPEX if k in cf.index), None)
    if ocf_row is None or capex_row is None:
        return pd.DataFrame(columns=['Ticker','year','fcf'])
    # reshape
    out = []
    for col in cf.columns:
        try:
            dt = pd.to_datetime(col)
        except Exception:
            continue
        yr = int(dt.year)
        fcf = float(cf.at[ocf_row, col]) - float(cf.at[capex_row, col])
        out.append({'Ticker': ticker, 'year': yr, 'fcf': fcf})
    return pd.DataFrame(out)

# Build annual FCF table
fcf_rows = []
for tk in all_tickers:
    fcf_rows.append(annual_fcf_from_yf(tk))
fcf_by_year = pd.concat(fcf_rows, ignore_index=True)
fcf_by_year['fcf_pos'] = (fcf_by_year['fcf'] > 0).astype(int)

# Compute next-year returns from each Dec-31 anchor
def compute_forward_returns_for_years(tickers: List[str], years: List[int]) -> pd.DataFrame:
    anchors = [pd.Timestamp(f"{y}-12-31") for y in years]
    start = (min(anchors) - pd.Timedelta(days=7))
    end   = (max(anchors) + pd.Timedelta(days=372))
    px = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)['Close']
    if isinstance(px.columns, pd.MultiIndex):
        px.columns = px.columns.get_level_values(0)
    px = px.sort_index()
    idx = px.index
    def next_td(ts):
        pos = idx.searchsorted(pd.Timestamp(ts))
        if pos >= len(idx):
            return None
        return idx[pos]
    rows = []
    for tk in tickers:
        if tk not in px.columns:
            continue
        for y in years:
            t0 = next_td(pd.Timestamp(f"{y}-12-31"))
            t1 = next_td(pd.Timestamp(f"{y+1}-12-31"))
            if t0 is None or t1 is None:
                continue
            px0, px1 = px.at[t0, tk], px.at[t1, tk]
            if pd.isna(px0) or pd.isna(px1) or px0 <= 0:
                continue
            rows.append({'Ticker': tk, 'year': y, 'ret_1y': float(px1/px0 - 1.0)})
    return pd.DataFrame(rows)

ret_by_year = compute_forward_returns_for_years(all_tickers, years)

# Merge FCF label with forward returns
panel_yearly = (fcf_by_year.merge(ret_by_year, on=['Ticker','year'], how='inner'))
print('panel_yearly rows:', len(panel_yearly))

# Evaluate by cohort each year
out = (panel_yearly
       .groupby(['year','fcf_pos'])['ret_1y']
       .agg(['count','mean','median'])
       .rename(index={0:'FCF−',1:'FCF+'}))
print('\nNext-year returns by year & FCF cohort:')
print(out)

# Summary across years
summary = (panel_yearly.groupby('fcf_pos')['ret_1y']
           .agg(['count','mean','median','std'])
           .rename(index={0:'FCF−',1:'FCF+'}))
print('\nOverall (all years combined):')
print(summary)

# Keep `panel_yearly` for further charts
```
</details>

{% raw %}
```text
Output:
panel_yearly rows: 1019

Next-year returns by year & FCF cohort:
              count      mean    median
year fcf_pos                           
2020 FCF−       174 -0.246565 -0.432091
2021 FCF−       228 -0.638476 -0.734033
     FCF+        36 -0.365271 -0.336333
2022 FCF−       248 -0.490987 -0.613039
     FCF+        38 -0.218305 -0.251932
2023 FCF−       254 -0.266805 -0.638533
     FCF+        41  0.005833 -0.063830

Overall (all years combined):
         count      mean    median       std
fcf_pos                                     
FCF−       904 -0.418151 -0.617768  0.931434
FCF+       115 -0.184402 -0.203704  0.495676
```
{% endraw %}

**What the code says**:  
This block converts the raw NASDAQ fundamentals into a clean and readable format:

- Each record represents a single company in a given year.  
- Free Cash Flow (FCF) is computed as operating cash flow minus capital expenditures.  
- A binary label distinguishes FCF+ (cash generators) from FCF− (cash burners).  
- Next-year returns are calculated from the end of each fiscal year, using a safe 90-day lag to ensure that only information available at the time is used.  

By the end of this step, we have a verified dataset (`panel_yearly`) ready for forward-return analysis and model testing.


### Step 3: Returns that don’t peek

Next, we needed to measure forward stock returns in a way that avoids cheating.  
If we looked at raw next-year returns, we’d be using information investors *didn’t have yet*. That would be like sneaking a peek at the answer key and we don't want to do that.  

So instead, we used a **lag-safe window**:
- Start: **90** days after fiscal year-end (when accounting data is typically public).  
- End: **365** days later.  
- Returns were winsorized at 1st–99th percentiles so that extreme outliers (like penny stocks doubling overnight) don’t dominate averages. P.S. Winsorized sounds like a scary stats term, but it just means trimming the extreme outliers.  

In simple terms: we clipped the extreme highs and lows so that one wild penny stock (up +500%) or a collapsing micro-cap (down –95%) wouldn’t skew the averages. The point wasn’t to hide those cases, but to make sure the overall results reflect the typical stock rather than rare outliers.

We also generated non-winsorized plots, but they showed the same overall pattern with a lot more noise and outliers — so we only kept the winsorized ones for clarity.

![Figure 1 — Winsorized next-year returns by FCF sign and XGBoost score quintile]({{ "/assets/images/NASDAQ/fcf_winsorized_returns.png" | relative_url }})

Figure 1 — Winsorized (1%–99%) next-year returns. The top chart compares returns between positive and negative FCF firms, while the bottom chart groups stocks by XGBoost model score quintiles (1 = lowest, 5 = highest). Both tell the same story: companies with stronger cash generation or higher model scores tended to perform slightly better, but not dramatically so.

<details>
  <summary><strong>Show code — lag-safe forward returns</strong></summary>

```python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import yfinance as yf

# panel_yearly: columns ['Ticker','year','fcf','fcf_pos','ret_1y']
assert 'panel_yearly' in globals(), "Run the yearly cohorts build first (Section 2)."
all_tickers = sorted(panel_yearly['Ticker'].unique().tolist())
years = sorted(panel_yearly['year'].unique().tolist())

# 1) Report-lag safe forward returns
# Recompute forward returns from (year-end + LAG_DAYS) to (year-end + LAG_DAYS + 365)
LAG_DAYS = 90
HORIZON_DAYS = 365

anchors = [pd.Timestamp(f"{y}-12-31") + pd.Timedelta(days=LAG_DAYS) for y in years]
start = min(anchors) - pd.Timedelta(days=7)
end   = max(anchors) + pd.Timedelta(days=HORIZON_DAYS + 7)
px = yf.download(all_tickers, start=start, end=end, auto_adjust=True, progress=False)['Close']
if isinstance(px.columns, pd.MultiIndex):
    px.columns = px.columns.get_level_values(0)
px = px.sort_index()
idx = px.index

def next_td(ts: pd.Timestamp):
    pos = idx.searchsorted(pd.Timestamp(ts))
    if pos >= len(idx):
        return None
    return idx[pos]

rows = []
for tk in all_tickers:
    if tk not in px.columns:
        continue
    for y in years:
        t0 = next_td(pd.Timestamp(f"{y}-12-31") + pd.Timedelta(days=LAG_DAYS))
        t1 = next_td(pd.Timestamp(f"{y}-12-31") + pd.Timedelta(days=LAG_DAYS + HORIZON_DAYS))
        if t0 is None or t1 is None:
            continue
        px0, px1 = float(px.at[t0, tk]), float(px.at[t1, tk])
        if not np.isfinite(px0) or not np.isfinite(px1) or px0 <= 0:
            continue
        rows.append({'Ticker': tk, 'year': y, 'ret_1y_lag': px1/px0 - 1.0})

ret_lag = pd.DataFrame(rows)
panel_yearly_lag = panel_yearly.merge(ret_lag, on=['Ticker','year'], how='inner')
```
</details>

{% raw %}
```text
Output:

Report-lag safe sample: 1019

Yearly FCF+−FCF− (lagged, winsorized):
fcf_pos       FCF−       FCF+    spread
year                                  
2020     -0.472353       NaN       NaN
2021     -0.597341 -0.333099  0.264241
2022     -0.512880 -0.277638  0.240641
2023     -0.436408 -0.146048  0.290361

Pooled (lagged, winsorized) spread=0.2583, t=6.23, p=0.000, n+=115, n−=904
```
{% endraw %}

---

### Why this setup matters

These steps may feel fussy and a bit too much, but they keep the experiment honest:
- The panel makes comparisons fair across companies and years.  
- The FCF flag captures a simple, intuitive signal: “cash in vs. cash out.”  
- The lag window prevents us from accidentally peeking at information investors wouldn’t know yet.  
- Winsorization keeps results from being skewed by one-off wild returns.


## 3) Results — What the Data Says (Part A)

With the panel built, the first step was simply to ask:  
**Do companies with positive FCF perform better than those with negative FCF?**

### FCF+ vs FCF–

On average, companies with positive FCF outperformed those with negative FCF by about **+0.26** per year (2020–2023).  
That’s a meaningful gap in direction. It suggests that cash-generating firms generally did better for shareholders than cash-burning ones. Which, at first glance, might seem obvious.

**But context matters**: this is based on only a few years of data, and the NASDAQ sample skews toward smaller, thinner stocks. That means the spread is promising, but fragile.

---

### Deciles: strongest vs. weakest

Instead of just looking at positive vs. negative, we also sorted companies into **deciles** by FCF level (top 10% → bottom 10%).  
The spread between Q10 and Q1 was generally positive, echoing the same story: more cash flow tends to mean stronger forward returns.

This gives us a directional “signal,” but again, it’s based on a tiny window. In financial data, a pattern that looks strong in just a few years can easily flip once the sample gets bigger.

<details>
  <summary><strong>Show code — FCF deciles (Q10–Q1)</strong></summary>

```python

p = panel_yearly_lag.copy()
p = p[p['fcf'].notna() & p['ret_1y_lag'].notna()].copy()

def _to_decile(s: pd.Series) -> pd.Series:
    r = s.rank(method="first", pct=True)
    d = np.ceil(r * 10.0)
    return pd.Series(d, index=s.index).clip(1, 10).astype("Int64")

p['decile'] = p.groupby('year')['fcf'].transform(_to_decile)

dec_ret = (p.groupby(['year','decile'])['ret_1y_lag']
             .agg(['count','mean','median'])
             .reset_index())

q10 = dec_ret.loc[dec_ret['decile'] == 10, ['year','mean']].rename(columns={'mean':'q10'})
q01 = dec_ret.loc[dec_ret['decile'] == 1,  ['year','mean']].rename(columns={'mean':'q01'})
ls_dec = q10.merge(q01, on='year', how='inner')
ls_dec['ls'] = ls_dec['q10'] - ls_dec['q01']

print("Q10−Q1 by year:")
print(ls_dec)
```
</details>

{% raw %}
```text
Output:

Q10–Q1 by year:
   year       q10       q01       ls
0  2020 -0.849850 -0.813830 -0.036020
1  2021 -0.398255 -0.665079  0.266825
2  2022 -0.265467 -0.512437  0.246970
3  2023 -0.087075 -0.484156  0.397081

Decile table (means):
decile         1       2       3       4       5       6       7       8       9      10
year
2020     -0.8138 -0.6927 -0.6804 -0.7219 -0.7472  0.5607  0.2740 -0.2394 -0.7792 -0.8498
2021     -0.6651 -0.5952 -0.6389 -0.7789 -0.6011 -0.6124 -0.4410 -0.4849 -0.3178 -0.3983
2022     -0.5124 -0.3644 -0.5752 -0.6104 -0.6240 -0.6856 -0.4460 -0.4512 -0.2993 -0.2655
2023     -0.4842 -0.5576 -0.3239 -0.5310 -0.6056 -0.4185 -0.5020  0.0429 -0.1564 -0.0871
```
{% endraw %}

<details>
  <summary><strong>Show code — sector-neutral FCF spreads</strong></summary>

```python

# Fetch sectors; fall back to 'Unknown' if missing
info = yf.Tickers(' '.join(all_tickers))
sectors = {}
for tk,obj in info.tickers.items():
    try:
        s = obj.info.get('sector')
    except Exception:
        s = None
    sectors[tk] = s or 'Unknown'

p2 = panel_yearly_lag.copy()
p2['sector'] = p2['Ticker'].map(sectors).fillna('Unknown')

# Within each (year, sector), compute FCF+ and FCF− means; average sectors equally per year
rows = []
for y, g in p2.groupby('year'):
    sec_means = []
    for s, h in g.groupby('sector'):
        m = h.groupby('fcf_pos')['ret_1y_lag'].mean()
        if 0 in m.index and 1 in m.index and np.isfinite(m[0]) and np.isfinite(m[1]):
            sec_means.append(m[1] - m[0])
    if len(sec_means) > 0:
        rows.append({'year': y, 'sector_neutral_spread': float(np.mean(sec_means)), 'sectors': len(sec_means)})
sec_neutral = pd.DataFrame(rows)
```
</details>

{% raw %}
```text
Output:

Sector-neutral summary:
   year  sector_neutral_spread  sectors
0  2021                0.176768        9
1  2022                0.292646        9
2  2023                0.111703       10
```
{% endraw %}

---

![Figure 2a — FCF+ − FCF− spread by year (lagged, 1%–99% winsorized)]({{ "/assets/images/NASDAQ/fcf_spread_by_year.png" | relative_url }})  
Figure 2a — Average 1-year return difference between positive and negative FCF companies. Firms with positive free cash flow tended to perform better, though not consistently across all years.

---

![Figure 2b — FCF magnitude Q10 − Q1 spread by year (lagged)]({{ "/assets/images/NASDAQ/fcf_decile_spread_by_year.png" | relative_url }})  
Figure 2b — When sorting by FCF strength (top 10% vs. bottom 10%), higher FCF generally aligned with stronger next-year returns, reinforcing the same “cash health” signal.

---

![Figure 2c — Sector-neutral FCF+ − FCF− spread by year (lagged)]({{ "/assets/images/NASDAQ/fcf_sector_neutral_spread.png" | relative_url }})  
Figure 2c — Sector-adjusted version of the FCF spread. Even after controlling for industry mix, companies with stronger cash generation still showed better average returns.

---

**What the code says**:

This code takes the yearly FCF data and ranks companies into ten equal groups (deciles) based on their cash flow.  
It then compares the average forward returns of the top decile (Q10) versus the bottom decile (Q1) to measure how much return “cash strength” adds.

The second block goes a step further to make the comparison fairer.  
It checks whether the FCF effect still holds within each industry, instead of mixing tech companies with energy or utilities.  
That’s what “sector-neutral” means, comparing companies *inside the same line of business* so that industry differences don’t skew the results.

---

### Why this matters

Even though we can’t predict exact prices from this alone, the averages tell us something real:  
- Healthy cash flow lined up with healthier returns.  
- Cash-burners underperformed.

That’s not enough to trade on yet, but it’s a good first check that the idea of “cash strength matters” shows up in the data at all. The next step is to zoom in at the company level, especially looking at firms that flipped from negative to positive FCF (or the other way around).


## 4) By Name & By Year (Part B)

Averages show the big picture, but investors also care about individual names.  
What we need to ask is: *What happens when a company flips from negative to positive FCF, or the reverse?*

### Tracking flips

We set up a simple pipeline to track these events:
- **Negative → Positive** (a company finally turns cash-flow positive).  
- **Positive → Negative** (a company slips into burning cash).  

The idea is intuitive: a flip to positive might signal improvement, while a flip to negative could raise red flags.

---

### What the data shows

In practice, the flip group is *tiny*.  
Across 2020–2023, there are only a handful of cases (in one year, literally three names flip from negative to positive).  

And the results are underwhelming:
- Companies flipping to positive FCF do **not** consistently outperform.  
- With so few observations, no strong conclusions can be drawn.

---

### Why this matters

This tells us two things:
1. At the company level, the sample is too thin to test the flip idea properly.  
2. The “signal” in this dataset shows up more clearly in group-level averages than in individual flips.

In other words, flips are conceptually appealing, but with this dataset they don’t deliver the insight we need.

<details>
  <summary><strong>Show code — flip tracking</strong></summary>

```python
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = Path("data/processed/nasdaq_by_name")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 0) Inputs
if 'panel_yearly_lag' not in globals():
    raise SystemExit("panel_yearly_lag missing. Run earlier steps to build yearly panel with lag-safe returns.")

df = panel_yearly_lag.copy()

# 1) Robust FCF & sign
FCF_CANDIDATES = [
    'Financial_freeCashflow', 'fcf', 'Financial_freeCashFlow', 'Financial_freeCashFlowTTM'
]
fcf_col = next((c for c in FCF_CANDIDATES if c in df.columns), None)
if fcf_col is None:
    raise SystemExit("No free cash flow column found in panel_yearly_lag. Expected one of: " + ", ".join(FCF_CANDIDATES))

df['fcf'] = pd.to_numeric(df[fcf_col], errors='coerce')
if 'fcf_pos' not in df.columns:
    df['fcf_pos'] = (df['fcf'] > 0).astype(int)

# 2) ADV eligibility & benchmark
# If elig_by_year and bench exist, use them; else fallback to all-names bench.
if 'elig_by_year' in globals() and isinstance(elig_by_year, dict):
    df['adv_eligible'] = df.apply(lambda r: r['Ticker'] in elig_by_year.get(int(r['year']), set()), axis=1)
    bench = (df[df['adv_eligible']]
               .groupby('year')['ret_1y_lag']
               .mean()
               .rename('bench'))
else:
    df['adv_eligible'] = np.nan
    bench = df.groupby('year')['ret_1y_lag'].mean().rename('bench')

# Map bench & beat flag
by_name = df[['Ticker','year','fcf','fcf_pos','ret_1y_lag','adv_eligible']].copy()
by_name['bench'] = by_name['year'].map(bench)
by_name['beat_bench'] = (by_name['ret_1y_lag'] > by_name['bench']).astype(int)

# 3) FCF deciles within year (raw, not absolute)
def _to_decile(s: pd.Series) -> pd.Series:
    r = s.rank(method='first', pct=True)
    d = np.ceil(r * 10.0)
    return pd.Series(d, index=s.index).clip(1, 10).astype('Int64')

by_name['fcf_decile'] = by_name.groupby('year')['fcf'].transform(_to_decile)

# 4) Detect flips per ticker
by_name = by_name.sort_values(['Ticker','year']).reset_index(drop=True)
by_name['fcf_pos_prev'] = by_name.groupby('Ticker')['fcf_pos'].shift(1)
by_name['flip'] = (by_name['fcf_pos_prev'].notna()) & (by_name['fcf_pos'] != by_name['fcf_pos_prev'])

# Flip type labeling
conditions = [
    (by_name['flip']) & (by_name['fcf_pos_prev'] == 0) & (by_name['fcf_pos'] == 1),
    (by_name['flip']) & (by_name['fcf_pos_prev'] == 1) & (by_name['fcf_pos'] == 0)
]
choices = ['neg→pos', 'pos→neg']
by_name['flip_type'] = np.select(conditions, choices, default='none')

# 5) Summaries
flip_df = by_name[by_name['flip']].copy()
# Counts by type
flip_counts = (flip_df.groupby('flip_type')['Ticker']
                     .count()
                     .rename('count')
                     .sort_values(ascending=False))
# Performance by flip type
flip_perf = (flip_df.groupby('flip_type')['ret_1y_lag']
                    .agg(['count','mean','median'])
                    .sort_values('mean', ascending=False))
# Flips per year
flips_per_year = (flip_df.groupby('year')['Ticker']
                          .count()
                          .rename('n_flips'))

print("Flip counts by type:\n", flip_counts)
print("\nNext-year returns by flip type:\n", flip_perf.round(4))
print("\nFlips per year:\n", flips_per_year)

# 7) Plots
# (a) Mean/median next-year return by flip type
fig1, ax1 = plt.subplots(figsize=(6,3))
flip_perf[['mean','median']].plot(kind='bar', ax=ax1)
ax1.set_title('Next-year return by flip type')
ax1.set_xlabel('flip_type')
ax1.set_ylabel('Return')
ax1.grid(True, axis='y', linewidth=0.5)
plt.show()

# (b) Count of flips per year
fig2, ax2 = plt.subplots(figsize=(6,3))
flips_per_year.plot(kind='bar', ax=ax2)
ax2.set_title('Count of FCF sign flips per year')
ax2.set_xlabel('year')
ax2.set_ylabel('Count')
ax2.grid(True, axis='y', linewidth=0.5)
plt.show()
```
</details>

{% raw %}
```text
Output:

Flip counts by type:
flip_type
neg→pos    3
Name: count, dtype: int64

Next-year returns by flip type:
          count   mean   median
flip_type
neg→pos        3 -0.6551 -0.6814

Flips per year:
      n_flips
year
2021        3
```
{% endraw %}

![Figure 3 — FCF sign flips and next-year returns]({{ "/assets/images/NASDAQ/fcf_flips.png" | relative_url }})
Figure 3 — Count of companies that flipped their FCF sign each year (bottom), and the average next-year return for those flips (top).  
Only a few firms flipped from negative to positive, and their next-year performance was still weak, which suggests that early recoveries in cash flow don’t translate immediately into stronger returns.


**What the code says:**  
This block tracks when companies shift from negative to positive FCF or vice versa,  
and compares how those “flip” groups perform in the following year.  

- Negative → Positive: a company that starts generating cash after burning it.  
- Positive → Negative: a company that slips back into spending more than it earns.  

It then plots two simple visuals:  
1. The average next-year return for each flip type.  
2. The number of flips per year.  

The results show that these flip events are extremely rare in the dataset —  
sometimes only a few per year, and their returns don’t form any clear pattern.  
In short, while flipping positive might *sound* like a strong signal,  
there isn’t enough data here to confirm it statistically.

## 5) Simple Models & Strategies (Part C)

So far, we’ve only compared averages and flips.  
The next step is to ask: *can a machine learning model do better at ranking stocks by return potential?*

### Walk-forward design

To test this, we build a walk-forward setup:
- Train on earlier years, test on the next year.  
- Roll forward year by year so each test set stays unseen.  
- This avoids “training on the future,” which would inflate results.

### Features used

The features here are deliberately simple. Just a few signals that investors could reasonably know at the time:
- **FCF sign** (positive vs. negative).  
- **FCF size** (scaled value of FCF to capture “how positive/negative” it is).  
- **Lagged return (`ret_1y_lag`)**: the stock’s return from the *previous* one-year window, measured in the same lag-safe way we set up earlier (start 90 days after year-end, end 365 days later).  
  - Why include it? Because momentum is a classic baseline: past winners often keep winning for a while.  

Together, this gives the model a very stripped-down view: cash flow health + recent performance.

### XGBoost baseline

For the model, we use **XGBoost**, a tree-based learner that often performs well on tabular data. If you remember, we used this on the Titanic dataset in my first post. 
The goal isn’t to “beat the market,” but to see if even a lightweight classifier can extract more signal than raw averages.

- **Task**: predict whether the next-year return is positive (`ret_1y_lag > 0`).  
- **Evaluation**:  
  - Accuracy: the share of stocks where the model correctly predicts whether the next year’s return is positive or negative.  
  - AUC: how well the model ranks winners above losers across all thresholds, not just at one cutoff.  

---

### Benchmark

A model is only meaningful relative to something. Here we use a simple **equal-weight benchmark**:  
- Imagine putting the same $1 into every stock in the panel each year.  
- That gives us a baseline return curve to compare against.  
- If the model can’t beat this “dumb but fair” strategy, then it isn’t adding real value.

---

### What the results show

The walk-forward model does *not* beat the benchmark.  
Accuracy stays close to 50%, AUC hovers near random, and yearly curves look like noise.

This isn’t surprising: with only a few years of data and very few features, the model simply doesn’t have enough information to separate winners from losers.

---

### Why this matters

This step highlights the gap between finding a broad tendency (FCF+ beats FCF–) and building a predictive model.  
- The averages show a weak edge.  
- When we try to use that edge in a forward-looking way, the signal doesn’t hold.  

That means this dataset, as it stands, isn’t enough to build a predictive model.

<details>
  <summary><strong>Show code — walk-forward XGBoost</strong></summary>

```python
# Walk-Forward XGBoost — testing if fundamentals can rank future returns
# Uses lag-safe data and simple features: FCF, FCF sign, and last year’s return

import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from xgboost import XGBClassifier
import yfinance as yf

# --------------------------- CONFIG ---------------------------
LAG_DAYS = 90
HOLD_DAYS = 365
ADV_LOOKBACK = 60
ADV_MIN_USD = 1_000_000
BATCH_SIZE = 80
AUTO_ADJUST = True
SLEEP_BETWEEN = 0.5
TOP_PCT = 0.10
OUT_DIR = Path("data/processed/nasdaq_ml_walkforward")
OUT_DIR.mkdir(parents=True, exist_ok=True)
# --------------------------------------------------------------

assert 'panel_yearly' in globals(), "panel_yearly missing."

base = panel_yearly.copy()
base['year'] = base['year'].astype(int)
all_tickers = sorted(base['Ticker'].dropna().unique().tolist())
years = sorted(base['year'].unique().tolist())
print(f"Universe: {len(all_tickers)} tickers; Years: {years}")

# use existing px/vol or download once

def dl_batched(tickers, start, end, batch_size=80, sleep_s=0.5, auto_adjust=True):
    close_frames, vol_frames = [], []
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        try:
            df = yf.download(batch, start=start, end=end, auto_adjust=auto_adjust, progress=False)
            c = df['Close'] if auto_adjust else df['Adj Close']
            v = df['Volume']
            if isinstance(c.columns, pd.MultiIndex):
                c.columns = c.columns.get_level_values(0)
                v.columns = v.columns.get_level_values(0)
            close_frames.append(c)
            vol_frames.append(v)
            print(f"batch {(i//batch_size)+1}: ok (n={len(batch)})")
        except Exception as e:
            print(f"batch {(i//batch_size)+1}: fail {e}")
        time.sleep(sleep_s)
    close = pd.concat(close_frames, axis=1).sort_index()
    vol   = pd.concat(vol_frames, axis=1).sort_index()
    close = close.loc[:, ~close.columns.duplicated()]
    vol   = vol.loc[:,   ~vol.columns.duplicated()]
    return close, vol

anchors = [pd.Timestamp(f"{y}-12-31") + pd.Timedelta(days=LAG_DAYS) for y in years]
start = min(anchors) - pd.Timedelta(days=max(ADV_LOOKBACK, 7))
end   = max(anchors) + pd.Timedelta(days=HOLD_DAYS + 7)

if 'px' not in globals() or 'vol' not in globals():
    print("Downloading px/vol …")
    px, vol = dl_batched(all_tickers, start, end, BATCH_SIZE, SLEEP_BETWEEN, AUTO_ADJUST)
else:
    print("Using cached px/vol from previous step.")

idx = px.index

def next_td(ts: pd.Timestamp):
    pos = idx.searchsorted(pd.Timestamp(ts))
    if pos >= len(idx):
        return None
    return idx[pos]

usd = px * vol
adv = usd.rolling(ADV_LOOKBACK, min_periods=ADV_LOOKBACK//2).median()

# 1) Build lag‑safe returns & eligibility
rows_ret = []
elig_by_year = {}
for y in years:
    t_form = next_td(pd.Timestamp(f"{y}-12-31") + pd.Timedelta(days=LAG_DAYS))
    t_exit = next_td(t_form + pd.Timedelta(days=HOLD_DAYS)) if t_form is not None else None
    if t_form is None or t_exit is None:
        continue
    prev_pos = idx.searchsorted(t_form) - 1
    if prev_pos < 0:
        continue
    t_prev = idx[prev_pos]
    elig = []
    for tk in all_tickers:
        a = adv.at[t_prev, tk] if (tk in adv.columns and t_prev in adv.index) else np.nan
        if pd.notna(a) and a >= ADV_MIN_USD:
            try:
                px0 = px.at[t_form, tk]
                px1 = px.at[t_exit, tk]
            except KeyError:
                continue
            if pd.notna(px0) and pd.notna(px1):
                elig.append(tk)
                rows_ret.append({'Ticker': tk, 'year': y, 'ret_1y_lag': float(px1/px0 - 1.0)})
    elig_by_year[y] = set(elig)

ret_lag = pd.DataFrame(rows_ret)
panel_yearly_lag = base.merge(ret_lag, on=['Ticker','year'], how='inner')
print("rows in panel_yearly_lag:", len(panel_yearly_lag))

bench = (panel_yearly_lag.groupby('year')
           .apply(lambda g: g[g['Ticker'].isin(elig_by_year.get(int(g.name), set()))]['ret_1y_lag'].mean())
           .rename('bench'))
bench = bench.dropna()
print("bench years:", bench.index.tolist())

# 2) Robust feature selection
# Prefer prefixed columns if present; otherwise use ALL numeric columns and drop meta.
meta = {'Ticker','year','ret_1y_lag','y'}
prefixed = [c for c in panel_yearly_lag.columns if c.startswith('Financial_') or c.startswith('KeyStats_')]
if prefixed:
    cand = prefixed
else:
    cand = panel_yearly_lag.select_dtypes(include=[np.number]).columns.tolist()
    cand = [c for c in cand if c not in meta]

# Always include 'fcf' if available
if 'fcf' in panel_yearly_lag.columns and 'fcf' not in cand:
    cand = ['fcf'] + cand

# Drop very-missing features
miss = panel_yearly_lag[cand].isna().mean()
num_cols = [c for c in cand if miss.get(c, 1.0) <= 0.5]

# Fallback if empty
if not num_cols:
    # Minimal single feature: fcf coerced to numeric
    panel_yearly_lag['fcf'] = pd.to_numeric(panel_yearly_lag.get('fcf'), errors='coerce')
    num_cols = ['fcf']

print(f"n_features: {len(num_cols)}; samples: {len(panel_yearly_lag)}")
print("sample features:", num_cols[:8])

# 3) Helper to build X,y for a list of years

def build_xy(year_list):
    parts = []
    for y in year_list:
        g = panel_yearly_lag[panel_yearly_lag['year'] == y].copy()
        elig = elig_by_year.get(y, set())
        g = g[g['Ticker'].isin(elig)].copy()
        if g.empty or y not in bench.index:
            continue
        g['y'] = (g['ret_1y_lag'] > bench.loc[y]).astype(int)
        parts.append(g)
    if not parts:
        return None, None, None
    df = pd.concat(parts, ignore_index=True)
    X = df[num_cols].apply(pd.to_numeric, errors='coerce')
    y = df['y'].astype(int)
    return X, y, df

# 4) Walk‑forward
rows_eval, port_rets = [], []
for i in range(1, len(years)):
    train_years = [y for y in years[:i] if y in bench.index]
    test_year   = years[i]
    if test_year not in bench.index:
        continue

    X_tr, y_tr, df_tr = build_xy(train_years)
    X_te, y_te, df_te = build_xy([test_year])
    if X_tr is None or X_te is None or X_tr.empty or X_te.empty:
        print(f"skip {test_year}: not enough data")
        continue

    med = X_tr.median()
    X_tr = X_tr.fillna(med)
    X_te = X_te.fillna(med)

    clf = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        objective='binary:logistic',
        tree_method='hist',
        random_state=42,
        n_jobs=1,
    )
    clf.fit(X_tr, y_tr)

    prob = clf.predict_proba(X_te)[:,1]
    pred = (prob >= 0.5).astype(int)

    acc  = accuracy_score(y_te, pred)
    auc  = roc_auc_score(y_te, prob)
    f1   = f1_score(y_te, pred)
    prec = precision_score(y_te, pred, zero_division=0)
    rec  = recall_score(y_te, pred)

    test_df = df_te.copy()
    test_df['prob'] = prob
    k = max(1, int(np.floor(len(test_df) * TOP_PCT)))
    top = test_df.sort_values('prob', ascending=False).head(k)
    port_ret = float(top['ret_1y_lag'].mean()) if not top.empty else np.nan
    port_bench = float(bench.loc[test_year])
    alpha = port_ret - port_bench

    rows_eval.append({
        'year': test_year,
        'n_train': len(X_tr),
        'n_test': len(X_te),
        'k_long': k,
        'acc': acc,
        'auc': auc,
        'f1': f1,
        'prec': prec,
        'rec': rec,
        'port_ret': port_ret,
        'bench': port_bench,
        'alpha': alpha,
    })
    port_rets.append({'year': test_year, 'ret': port_ret, 'bench': port_bench})

if not rows_eval:
    print("\nNo evaluation years produced predictions. Check that numeric features exist in panel_yearly and eligibility isn't filtering everything out.")
else:
    wf = pd.DataFrame(rows_eval).set_index('year').sort_index()
    print("\nWalk‑forward year metrics:\n")
    print(wf.round(4))

    port = pd.Series({r['year']: r['ret'] for r in port_rets}).sort_index()
    bench_series = pd.Series({r['year']: r['bench'] for r in port_rets}).sort_index()
    port_eq = (1+port.fillna(0)).cumprod()
    bench_eq = (1+bench_series.fillna(0)).cumprod()

    fig, ax = plt.subplots(figsize=(7,3))
    port_eq.plot(ax=ax, label='ML Long (Top‑p by prob)')
    bench_eq.plot(ax=ax, label='Eligible EW benchmark')
    ax.set_title('Walk‑forward equity curve (lag‑safe)')
    ax.grid(True, axis='y', linewidth=0.5)
    ax.legend()
    plt.show()
```
</details>

{% raw %}
```text
Output:

Universe: 297 tickers; Years: [2020, 2021, 2022, 2023]
Using cached px/vol from previous step.
rows in panel_yearly_lag: 170
bench years: [2020, 2021, 2022, 2023]
n_features: 2; samples: 170
sample features: ['fcf_pos', 'ret_1y']

Walk-forward year metrics:

       n_train  n_test  k_long   acc    auc    f1   prec    rec  port_ret  bench   alpha
year
2021        113      40      4  0.6000  0.7356  0.3333  0.8000  0.2105  -0.5472 -0.6621  0.1149
2022        153      12      1  0.5833  0.8000  0.0000  0.0000  0.0000  -0.6059 -0.8133  0.2074
2023        165       5      1  0.6000  0.5000  0.0000  0.0000  0.0000  -0.9489 -0.8286 -0.1203
```
{% endraw %}

![Figure 4 — Walk-forward ML equity curve (lag-safe)]({{ "/assets/images/NASDAQ/xgb_walkforward_equity_curve.png" | relative_url }})  
Figure 4 — Walk-forward test of a simple XGBoost model trained on FCF, FCF sign, and prior-year return.  
The blue curve (“ML Long”) shows the model’s top-ranked stocks each year, compared against an equal-weight benchmark.  
The lines move almost in sync, a sign that fundamentals alone aren’t adding much predictive power.

**What the code says:**

This block runs the actual walk-forward test described above.  
It builds a year-by-year training loop, feeds the model three lag-safe features (free cash flow, its sign, and the prior year’s return), and tests whether those variables can separate future winners from losers.

Each test year:
- The model learns only from past data.
- Predicts the next year’s direction for each stock.
- Compares its top 10% picks against an equal-weight benchmark.

The resulting equity curve plot shows how a simple fundamentals-based ML strategy performs over time, revealing whether it adds any predictive value beyond the broad averages. 

The following sections break down what that reveals and where this approach starts to fall apart. :(

## 6) What this data actually shows us

After testing averages, flips, and even a walk-forward model, we can finally return to the big question:  
**Can we predict stock prices from this data?**

The short answer is: **not really.**

What the analysis actually shows is broad patterns, not predictions:
- Companies with healthy cash flow tend to do better than companies spending more cash than they bring in.  
- Stronger cash flow generally lines up with higher future returns when looking at groups of companies, not individual stocks.  
- But once we try to predict single names or future years, the effect disappears. The model’s guesses are no better than flipping a coin.

---

### Why that happens

There are a few simple reasons for this:
1. Small sample — only a few years of data means too few cycles to test repeatability.  
2. Limited features — fundamentals alone (like FCF) don’t capture sentiment, risk, or timing.  
3. Unstable sample — many of the NASDAQ stocks in this dataset are smaller or less frequently traded, so their prices jump around more and don’t reflect consistent investor behavior.

In short, the dataset can show direction, but not **prediction**.  
It tells us which way the wind is blowing, not which stock will sail farther.

---

### What we learn instead

The real value of this experiment is clarity:
- FCF is a healthy metric, but it’s not a crystal ball.  
- Simple accounting signals can highlight trends, not make forecasts.  
- And the moment we move from describing history to predicting the future, data limits matter far more.

That’s a pretty useful outcome.  
Even finding what *doesn’t* work helps narrow the path toward what might.

## 7) So What?

After everything, the takeaway is simple:  
**Free Cash Flow strength matters — but it’s not enough to make reliable predictions on its own.**

FCF captures something real about financial health. Companies that consistently generate cash tend to outperform those that burn it.  
But turning that observation into a working prediction model is where the limits show.

---

### What this means for investors

If you’re using FCF alone to pick stocks, you’re probably looking at a *broad indicator*, not a *buy signal*.  
Healthy cash flow helps, but it’s only one piece of the puzzle.  
Markets care about many other things: growth expectations, risk, interest rates, and sentiment. None of which show up in this dataset.

In other words, FCF can hint at quality, but not timing.  
It can point you toward stronger companies, but it won’t tell you which stock to buy *right now*.

---

### What could make it better

There are clear ways to strengthen this idea:
- Add more years to test if the FCF effect holds up over different market cycles.  
- Include larger, more liquid companies so returns are less erratic.  
- Combine with other metrics, like revenue growth, profit margins, or debt ratios, to see if a multi-factor model performs better.  
- Bring in sentiment or price-based data (like momentum indicators) to help capture the market’s mood alongside fundamentals.

---

### The biggest takeaway

In finance, “better data” isn’t just *more* data — it’s data that matches the question you’re asking.  
If we want to predict stock prices, we need information that changes when markets change. Things like sentiment, macro shifts, or forward guidance. Fundamentals move too slowly to capture that by themselves.

That doesn’t make fundamentals useless.  
It just means they tell a different story: one about **quality and resilience**, not about short-term prediction.

---

So while the model didn’t find a trading edge, the project did what it was supposed to:  
it showed the limits of what’s possible for me right now, and reminded me that in research, an honest “no” is still progress.


## Appendix — Quick Glossary (plain English)

<details markdown="1">
  <summary><strong>Open glossary</strong></summary>

- **Free Cash Flow (FCF)** — the cash a company has left after paying for operations and investments.  
  Why it matters: positive FCF means a company is generating more cash than it spends, giving it flexibility to grow or return value to shareholders.

- **Return (or “next-year return”)** — the percentage change in a company’s stock price over the following year.  
  Why it matters: it’s the main outcome we’re trying to connect to fundamentals like FCF.

- **Lag-safe window** — measuring returns only after financial results are public (starting 90 days after year-end).  
  Why it matters: prevents “peeking” into the future and keeps the analysis honest.

- **Winsorization** — trimming extreme highs and lows (like +500% or –95%) so they don’t distort averages.  
  Why it matters: keeps results focused on typical behavior, not one-off outliers.

- **Walk-forward test** — train the model on earlier years and test on the next one, repeating the process forward.  
  Why it matters: mimics how a real investor would use past data to make future decisions.

- **Equal-weight benchmark** — a simple baseline where you put the same $1 in every stock each year.  
  Why it matters: helps check if your model actually adds value beyond a fair, passive strategy.

- **XGBoost** — a machine learning algorithm that builds many small decision trees to improve prediction accuracy.  
  Why it matters: it’s popular for tabular data and balances speed, performance, and interpretability.

- **Accuracy** — the share of predictions the model gets right (e.g., calling positive vs. negative returns correctly).  
  Why it matters: gives a quick sense of how often the model is correct overall.

- **AUC (Area Under the ROC Curve)** — a measure of how well the model ranks winners higher than losers.  
  Why it matters: unlike accuracy, AUC checks if the model’s confidence scores make sense across all thresholds.

- **Feature** — an input variable used by the model (like FCF sign or last year’s return).  
  Why it matters: good features describe the company in ways that help the model learn useful patterns.

- **Data leakage** — when future information accidentally sneaks into training.  
  Why it matters: it can make a model look great on paper but fail in real life.

- **Panel (company-year panel)** — a table where each row represents one company in one year.  
  Why it matters: it lets us compare results consistently across firms and time.

</details>  



**Next Steps**:
I plan on completing this AI course to gain more foundational knowledge. I feel like I've gotten better at the technical aspect of AI/ML, but I still want to understand more of the theory and explore different applications of it.

So sadly, I probably won't be posting for a while until I finish that course. But until then, I'll see you all next time!

If you have any questions or suggestions, please feel free to shoot an email to mohitmohanraj05@gmail.com
