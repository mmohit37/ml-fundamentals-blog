---
layout: post
title: "NASDAQ Free Cash Flow (Part 1): Do cash-generating companies outperform?"
date: 2025-10-23
excerpt: "Part 1 of a three-part series exploring whether Free Cash Flow can predict future stock returns. This first post compares companies with positive vs. negative cash flow."
categories: quant nasdaq
tags: finance freecashflow fundamentals returns
series: "NASDAQ Free Cash Flow Series"
---

**This post is Part 1 of a 3-part series** analyzing whether Free Cash Flow (FCF) can predict future stock returns on the NASDAQ.  
Here, we start simple, testing whether companies that generate positive cash flow actually outperform those that burn cash.  

You’ll see how we define FCF, build a lag-safe dataset, and compare average future returns between the two groups.  
The goal is to find out: *does financial health, measured through FCF, really show up in next year’s performance?*

[Next → Part 2 — Quartile & Sector-Neutral Analysis](2025-10-31-nasdaq-fcf-part2)


## 1) Intro & Goal

Free Cash Flow (FCF) is the cash a company has left after paying for its operations and investments.  
Investors often treat it as a sign of financial health, the idea being: if a company consistently generates positive FCF, it has more flexibility to grow, pay down debt, or return cash to shareholders.

**Dataset:** [NASDAQ Fundamentals (2024, Kaggle)](https://www.kaggle.com/datasets/sauers/nasdaq-fundamentals-2024)

**Big question:** does having positive FCF actually lead to better stock returns?  
Or put another way: *can strong cash flow help us predict which stocks will do better next year?*

In this project, I explored that question using a cleaned panel of NASDAQ companies.  
I built lag-safe forward returns (so we don’t “peek” into the future), compared positive vs. negative FCF groups, and even tried a simple walk-forward XGBoost model to see if fundamentals alone could rank stocks.  

By the way, if any terms seem confusing to you, check out the glossary: [Jump to Glossary →](#appendix--quick-glossary-plain-english)

I also made a few changes compared to my last post. I realize that I didn't really explain the different variables in the dataset, which could cause some confusion if you've never seen this dataset. So I'll make sure to explain the important ones before I start.

Another detail I changed was adding the plots right after each code block, so it makes sense in context. In my last post, I just put all the plots at the end, which doesn't make much sense.

<details class="code-alt" markdown="1">
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

<details class="code-alt" markdown="1">
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

<details class="code-alt" markdown="1">
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

<details class="code-alt" markdown="1">
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

<details class="code-alt" markdown="1">
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



## Appendix — Quick Glossary (plain English)

<details markdown="1">
  <summary><strong>Open glossary</strong></summary>

- **Free Cash Flow (FCF)** — the cash a company has left after paying for operations and investments.  
  *Why it matters:* positive FCF means a company is generating more cash than it spends, giving it flexibility to grow or return value to shareholders.

- **Return (or “next-year return”)** — the percentage change in a company’s stock price over the following year.  
  *Why it matters:* it’s the main outcome we’re trying to connect to fundamentals like FCF.

- **Lag-safe window** — measuring returns only after financial results are public (starting 90 days after year-end).  
  *Why it matters:* prevents “peeking” into the future and keeps the analysis honest.

- **Winsorization** — trimming extreme highs and lows (like +500% or –95%) so they don’t distort averages.  
  *Why it matters:* keeps results focused on typical behavior, not one-off outliers.

- **Panel (company-year panel)** — a table where each row represents one company in one year.  
  *Why it matters:* lets us compare results consistently across firms and time.

- **Sector-neutral** — comparing companies within the same industry to control for sector effects.  
  *Why it matters:* ensures differences aren’t just because one industry was hot that year.

- **Decile analysis** — sorting companies into ten equal groups by a metric (like FCF).  
  *Why it matters:* helps show whether stronger cash flow consistently links to better returns across the distribution.

</details>

---

## Next Steps

In the next part of this series, we’ll go beyond the simple positive-vs-negative split.  
We’ll look at **quartiles, sector-neutral spreads, and “flip” events**. These are cases where companies move from negative to positive FCF (or vice versa) and to see whether those transitions carry stronger signals.  

If you have any questions or suggestions, please feel free to tell me at mohitmohanraj05@gmail.com. I'll see you all in the next post!

[Next → Part 2 — Quartile & Sector-Neutral Analysis](2025-10-31-nasdaq-fcf-part2)
