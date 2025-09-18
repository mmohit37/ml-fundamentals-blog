---
layout: post
title: "Titanic: a quick ML walkthrough (EDA → RandomForest → XGBoost)"
date: 2025-09-15
categories: kaggle titanic
tags: tutorial sklearn randomforest xgboost dtreeviz
---

<style>
/* Dark “Nord-ish” palette for code blocks (per-post) */
pre.highlight, .highlighter-rouge pre {
  background: #0b1221; 
  color: #e6edf3;
  border-radius: 12px;
  padding: 1rem 1.25rem;
  overflow: auto;
}
.highlighter-rouge .na, .highlight .na { color:#7ee787; }
.highlighter-rouge .nb, .highlight .nb { color:#79c0ff; } 
.highlighter-rouge .kd, .highlight .kd,
.highlighter-rouge .k,  .highlight .k  { color:#ff7b72; }
.highlighter-rouge .s,  .highlight .s  { color:#a5d6ff; }
.highlighter-rouge .mi, .highlight .mi { color:#ffa657; }
.highlighter-rouge .c,  .highlight .c  { color:#8b949e; font-style:italic; }

details.code-alt pre {
  background: #151a2d;
  color: #e6edf3;
}
details > summary {
  cursor: pointer;
  list-style: none;
  font-weight: 700;
  margin: .6rem 0 .3rem;
  padding: .45rem .7rem;
  border-left: 4px solid #6ea8fe;      /* accent bar */
  background: #0f1a33;                 /* darker chip */
  color: #cfe4ff;                       /* text color */
  border-radius: 8px;
}
details[open] > summary {
  background: #162544;
  color: #ffffff;
}
/* optional: small arrow */
details > summary::after {
  content: " ⌄";
  float: right;
  opacity: .8;
}
details[open] > summary::after { content: " ⌃"; }
</style>

I used the classic Kaggle **Titanic** dataset to warm up on feature engineering and tree-based models.  
This post walks through what I did and why I did it, so people can replicate the experiment and understand my thought process.

> **Heads-up:** If you are new to ML or unfamiliar with certain vocabulary, here’s a short **glossary** at the end for any jargon — jump to [Appendix: quick vocab](#vocab).
---

## 1) Goal & data

**Goal:** predict whether a passenger survived the Titanic disaster.  
**Data:** the classic Kaggle competition — [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic)

**Why:** it’s a small tabular dataset that’s perfect for practicing the full exploratory data analysis (EDA) → features → modeling → interpretation without getting buried in scale or noise.

---

## 2) Fast features that carry meaning

I built a few simple features from the existing variables. Since predicting an outcome isn't as simple as putting everything into the algorithm and hoping for a good output, I chose to help the model by creating variables that tell a better story:

- `fam_size = SibSp + Parch + 1`  
  **Why:** Family presence matters socially (someone to help you get to lifeboats), and this is an easy way to encode it. It also makes sense to group siblings and spouses (SibSp) and parents and children (Parch) instead of looking at them individually.

- `alone = (fam_size == 1)`  
  **Why:** Lots of Kaggle notebooks show that being alone correlates with survival odds. The model can learn that from `fam_size`, but as an explicit boolean value that classifies someone as alone (1) or not (0).

- **Title** from `Name` (Mr, Mrs, Miss, Master, Rare)  
  **Why:** Titles represent `age`, `gender`, and sometimes `status`. They’re surprisingly strong signals and more stable than raw name text. It also makes sense that women and children would be prioritized for safety first, so creating this variable can help incorporate that into the model.

- **Grouped imputations**  
  - `Age`: filled by **median** within title  
    **Why:** Kids and “Master” shouldn’t receive adult medians; imputing within groups preserves a strong structure for the model.  
  - `Fare`: median within `(Pclass, Embarked)` 
    **Why:** Ticket cost varies a lot by class and port. Thus, grouped medians give more sensible values.  
  - `Embarked`: filled with **mode**  
    **Why:** It’s categorical and only a few values are missing. Which means the simplest fix wins.

- **One-hot encoding** for `Sex`, `Embarked`, and `title` with `drop_first=True`  
  **Why:** tree models don’t need scaling, but they still need categorical variables turned into columns. `drop_first` avoids a redundant column (a little overfitting reduction for free).

Feature set I used:  
`["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked","title","fam_size","alone"]`

---

## 3) Class balance check

I plotted the class counts:

![Survivor counts]({{ '/assets/images/titanic/survivor_counts_v2.png' | relative_url }})

**Why:** Before we judge any model, we need to know how many people survived vs. didn’t survive in the training data. If one group is much bigger than the other, a lazy model can look “good” just by always guessing the bigger group.  
Checking the bar chart tells us how skewed things are and which scores make sense. In Titanic, non-survivors are more than survivors, but it’s not crazy unbalanced, so accuracy/AUC/F1 are still meaningful without fancy tricks. If the gap were huge, we’d consider things like class weights or different evaluation metrics.

---

## 4) Train/valid/test split

**Why:** We keep a final test set untouched for an honest evaluation, and carve a validation set out of the training data to tune/stop models without peeking at the test set.

<details class="code-alt">
  <summary><strong>Show code — split, drop dupes, freeze columns</strong></summary>

```python
# One-hot encode categoricals (drop_first avoids a redundant column)
df = pd.get_dummies(df, columns=["Sex", "Embarked", "title"], drop_first=True)

# Define X/y (keeps things readable + leak-proof)
X = df.drop(columns=["Survived"])         # features — includes one-hot columns
y = df["Survived"].astype(int)            # target

# Hold-out test set (20%), stratified (preserving proportions so dataframes don’t get mismatched)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Then carve validation set (20% of train = 64/16/20)
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.20, random_state=42, stratify=y_train
)

# Drop any duplicate columns (can happen after merges/dummies)
def _drop_dupes(df_: pd.DataFrame) -> pd.DataFrame:
    return df_.loc[:, ~df_.columns.duplicated()].copy()

X_tr  = _drop_dupes(X_tr)
X_val = _drop_dupes(X_val)
X_test= _drop_dupes(X_test)

# Freeze the exact feature list/order for consistency in future exports
COLS  = list(X_tr.columns)
X_tr   = X_tr[COLS].copy()
X_val  = X_val[COLS].copy()
X_test = X_test[COLS].copy()

print("train:", X_tr.shape, " valid:", X_val.shape, " test:", X_test.shape)
print("pos rate -> train:", float(y_tr.mean()), " valid:", float(y_val.mean()), " test:", float(y_test.mean()))
```

</details>

### Why these little details matter

- **One-hot with `drop_first=True`.** Columns like `Sex` or `Embarked` are words. Models need numbers. One-hot makes a 0/1 column for each choice (e.g., `Sex_male`). Dropping the first level avoids sending the same information twice.
- **Freeze `COLS`.** After splitting the data, I lock the final feature list and its order so every model and plot uses the exact same columns. This prevents the annoying “X has N features, model expects M” error.
- **Drop duplicate columns.** Encoding/merges can accidentally create duplicate columns. I keep the first and drop the rest so the model doesn’t double-count anything.
- **pos rate** (Probability of Success). This is the share of rows with `Survived == 1`. Because I used `stratify=...`, the train/valid/test sets have similar pos rates. If one split was way off, accuracy comparisons wouldn’t be fair or accurate.


## 5) Baseline model: Random Forest

**Why:** Random Forests (RF) are strong, quick baselines for tabular data. They handle mixed features (after one-hot encoding), need little preprocessing, and give valuable feature importances, which can reveal more details about the data.

<details>
  <summary><strong>Show code — RandomForest (train & evaluate)</strong></summary>

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=9,
    max_features='sqrt',
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    oob_score=True,
    n_jobs=-1,
)
rf.fit(X_tr, y_tr)

# Validation metrics
rf_val_pred  = rf.predict(X_val)
rf_val_proba = rf.predict_proba(X_val)[:, 1]
rf_val = dict(
    model="RandomForest",
    split="valid",
    acc=accuracy_score(y_val, rf_val_pred),
    f1=f1_score(y_val, rf_val_pred),
    auc=roc_auc_score(y_val, rf_val_proba),
)

# Test metrics
rf_test_pred  = rf.predict(X_test)
rf_test_proba = rf.predict_proba(X_test)[:, 1]
rf_test = dict(
    model="RandomForest",
    split="test",
    acc=accuracy_score(y_test, rf_test_pred),
    f1=f1_score(y_test, rf_test_pred),
    auc=roc_auc_score(y_test, rf_test_proba),
)

print("RF oob_score (train):", getattr(rf, "oob_score_", None))
print("RF valid:", rf_val)   # e.g. {'acc': 0.825, 'f1': 0.786, 'auc': 0.868}
print("RF test :", rf_test)  # e.g. {'acc': 0.811, 'f1': 0.738, 'auc': 0.845}
```

</details>

### Interpreting the Random Forest results

**Why AUC matters:**  
Accuracy depends on a single cutoff (usually 0.5), but **AUC** measures how well the model ranks survivors above non-survivors across all cutoffs. An AUC of 0.85 means that if you pick one random survivor and one random non-survivor, the model gives the survivor a higher score about 85% of the time. That’s strong.

**What do these numbers mean?**  
- **Accuracy ~0.81** → about 81 out of 100 passengers are classified correctly on unseen data. A simple baseline (“always predict didn’t survive”) is only ~62%, so this is clearly better.
- **F1 ~0.74** → balanced view of finding survivors without calling too many false alarms.

**Real-world tie-in:** The top signals match history: `Sex`, `Pclass`, `Age/Fare`, and `Title`. “Women and children first” and better lifeboat access for higher classes show up in the data, and the model learns those patterns.

**Is the model overfitting?**  
There’s a small drop from validation (≈0.825 acc) to test (≈0.811 acc). That’s normal and suggests the model generalizes pretty well, with only mild overfit. If we want to push performance, we can tune hyperparameters, adjust the decision threshold* (to boost F1), or use cross-validation to pick better settings more reliably.

- **Threshold:** models give a score (0→1). We choose a cutoff (like 0.5) to turn the score into a Yes/No. Moving this cutoff changes accuracy.

## 6) XGBoost + Early stopping

**Why:** Boosted trees typically beat single trees/forests on tabular data. Early stopping halts training when the validation score stops improving, and it also helps avoid overfitting.

<details class="code-alt">
  <summary><strong>Show code — XGBoost (train, early stopping, evaluate)</strong></summary>

```python
import xgboost as xgb
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

feat_names = list(X_tr.columns)

dtr = xgb.DMatrix(X_tr.to_numpy(dtype=float),   label=y_tr.to_numpy(),   feature_names=feat_names)
dval= xgb.DMatrix(X_val.to_numpy(dtype=float),  label=y_val.to_numpy(),  feature_names=feat_names)
dte = xgb.DMatrix(X_test.to_numpy(dtype=float), label=y_test.to_numpy(), feature_names=feat_names)

pos = int(y_tr.sum()); neg = len(y_tr) - pos
params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "eta": 0.05,
    "max_depth": 4,
    "subsample": 0.9,
    "colsample_bytree": 0.8,
    "min_child_weight": 1,
    "lambda": 1.0,
    "alpha": 0.0,
    "scale_pos_weight": neg / max(pos, 1),
    "tree_method": "hist",
    "seed": 42,
}

bst = xgb.train(
    params=params,
    dtrain=dtr,
    num_boost_round=2000,
    evals=[(dtr, "train"), (dval, "valid")],
    early_stopping_rounds=50,
    verbose_eval=False,
)

# Predictions at the best iteration
xgb_val_proba  = bst.predict(dval, iteration_range=(0, bst.best_iteration + 1))
xgb_test_proba = bst.predict(dte,  iteration_range=(0, bst.best_iteration + 1))
xgb_val_pred   = (xgb_val_proba  >= 0.5).astype(int)
xgb_test_pred  = (xgb_test_proba >= 0.5).astype(int)

xgb_val = dict(
    model="XGBoost",
    split="valid",
    acc=accuracy_score(y_val, xgb_val_pred),
    f1=f1_score(y_val, xgb_val_pred),
    auc=roc_auc_score(y_val, xgb_val_proba),
)
xgb_test = dict(
    model="XGBoost",
    split="test",
    acc=accuracy_score(y_test, xgb_test_pred),
    f1=f1_score(y_test, xgb_test_pred),
    auc=roc_auc_score(y_test, xgb_test_proba),
)

print("XGB best_iteration:", bst.best_iteration, "best_valid_auc:", bst.best_score)  #output: ("XGB best_iteration: 0", best_valid_auc: 0.9138576779026217
print("XGB valid:", xgb_val) #output: {'acc': 0.8391608391608392, 'auc': 0.8986683312526009, 'f1': 0.7850467289719626}
print("XGB test :", xgb_test) #output: {'acc': 0.8044692737430168, 'auc': 0.8978764478764477, 'f1': 0.7586206896551724}
```

</details>

**Fresh run (corrected build):**
- **best_iteration ≈ 71** (picked by validation AUC)  
- **Validation:** ACC ≈ 0.825, AUC ≈ 0.810, F1 ≈ 0.775  
- **Test:** ACC ≈ 0.827, AUC ≈ 0.851, F1 ≈ 0.786

**Why this matters:**
- XGBoost slightly edges Random Forest on test accuracy and AUC. Boosting focuses on tough cases and usually squeezes out a bit more signal on tabular data.
- Early stopping keeps things honest: I stop when validation AUC stops improving and then predict using the best iteration. The test set stays untouched until the very end.

**Real-world tie-in:** Same story as the RF (Sex, Pclass, Title, Age/Fare) but a bit better at ranking the close calls, which nudges the AUC a bit higher.

## 7) Results + Takeaways

### A previous mistake

I first trained an XGBoost model and saw an accuracy close to **0.9**. That felt way too high for this dataset, especially since it was my first time building something like this. So before posting anything, I stopped and double-checked. I looked up common pitfalls (like data leakage, tweaking the classification threshold on the same data I was measuring, or accidentally peeking at the test set) and asked a couple of colleagues with more ML experience. I also discovered a couple of mismatched variable/column names, which meant some cells were training/evaluating on the wrong columns (or in the wrong order), quietly boosting the score. On top of that, there was data leakage: information from the answers (or test-like data) slipped into training/threshold-tuning so the model looked smarter than it really was on truly unseen data.   
The feedback was clear: numbers that high on Titanic with these features are a **red flag** unless the validation is airtight.

Since being safe and trustworthy is better to me than being flashy, I decided to start from scratch (the models you saw above).

**What I fixed:**
- I used a clean 64/16/20 split (train/valid/test), stratified to reduce mismatching.
- I froze the feature columns after one-hot encoding, so the matrices stay consistent.
- For XGBoost I used early stopping on the validation set and predicted with the best iteration it found.
- I report accuracy at a fixed 0.5 threshold. If I end up changing the threshold, I tune it on validation and apply it only **once** to test.
- I reran everything. The new numbers are lower, which was a bit sad, but they’re the kind you can trust, so I still felt satisfied with the results.

> TL;DR: I didn’t publish the first result because it looked too good to be true. I rebuilt the pipeline carefully, and now the results are honest and reproducible.

### Metrics table

| model        | split      | threshold | acc    | f1     | auc    |
|--------------|------------|-----------|--------|--------|--------|
| RandomForest | valid      | 0.50      | 0.8252 | 0.7664 | 0.8679 |
| RandomForest | test       | 0.50      | 0.8101 | 0.7384 | 0.8451 |
| RandomForest | test@tuned | 0.37      | 0.8156 | 0.7724 | 0.8451 |
| XGBoost      | valid      | 0.50      | 0.8252 | 0.7748 | 0.8099 |
| XGBoost      | test       | 0.50      | 0.8268 | 0.7862 | 0.8507 |
| XGBoost      | test@tuned | 0.74      | 0.7821 | 0.6549 | 0.8507 |

**What these numbers mean (no jargon):**
- **Accuracy (“acc”)** — out of 100 passengers, how many the model gets right.  
  Example: RF at 0.81 ≈ **81 out of 100** correct on unseen data.
- **AUC (“auc”)** — how well the model orders people from “less likely” to “more likely” to survive.  
  AUC ~0.85 means that if you pick one survivor and one non-survivor at random, the model gives the survivor a higher score about 85% of the time.
- **F1** — a balance between “catching survivors” and false alarms  
  Higher F1 means we’re finding more true survivors without too many false alarms.

**Reading the table:**
- Both models land in the mid–high 0.8 AUC range → they rank people reasonably well.
- At the standard 0.5 cutoff, XGBoost is a touch better than Random Forest on test accuracy and AUC.
- **test@tuned** shows what happens if we move the cutoff to one chosen on validation:  
  RF’s tuned threshold raises F1 (we catch more true survivors).  
  XGB’s tuned threshold got stricter (higher cutoff), so the accuracy drops, which is an honest trade-off, not much of an error.

### Plots

- ROC (validation): ![ROC valid]({{ "/assets/images/titanic/titanic_roc_valid.png" | relative_url }})
**What you’re seeing:** For many different cutoffs, how often the model catches survivors (true-positive rate) vs. how often it cries wolf (false-positive rate). The curve up toward the top-left is good; AUC ≈ 0.89 which means a strong ranking on the validation set.  
**Why it matters:** Shows the model’s general skill without picking a single threshold. Great for comparing models fairly.

- ROC (test): ![ROC test]({{ "/assets/images/titanic/titanic_roc_test.png" | relative_url }})
**What you’re seeing:** Same idea, but on unseen test data. AUC ≈ 0.85 is a small drop from validation, which is normal and suggests the model generalizes.  
**Why it matters:** Confirms the ranking quality holds up when we leave the training sandbox.

- Confusion (test, RF @ tuned): ![CM RF]({{ "/assets/images/titanic/titanic_cm_test_rf.png" | relative_url }})
**What you’re seeing:** Counts of correct/incorrect predictions at the chosen cutoff.  
Top-left = true “not survived,” top-right = false alarms, bottom-left = missed survivors, bottom-right = correctly found survivors.  
**Why it matters:** With a lower threshold (~0.37), RF finds more survivors (higher recall) but pays with more false alarms. Useful when missing a survivor is worse than a false alert for the model.

- Confusion (test, XGB @ tuned): ![CM XGB]({{ "/assets/images/titanic/titanic_cm_test_xgb.png" | relative_url }})
**What you’re seeing:** Same grid, stricter cutoff.  
**Why it matters:** A higher threshold (~0.74) means XGB is pickier: fewer false alarms but more missed survivors. Good when false positives are costly, but bad if recall is your priority.

- Feature importance (RF): ![FI RF]({{ "/assets/images/titanic/titanic_importance_rf.png" | relative_url }})
**What you’re seeing:** Which inputs reduced impurity the most across the forest (RF’s notion of “importance”). `Fare, Age, Sex/Title, Pclass` bubble to the top.  
**Why it matters:** Matches the story: higher class and “women/children first” patterns carry a signal. Helps explain *why* the model works (not proof of causation). Also good for checking which features play the biggest roles in the model

- Feature importance (XGB): ![FI XGB]({{ "/assets/images/titanic/titanic_importance_xgb.png" | relative_url }})
**What you’re seeing:** Features that gave the biggest improvement when XGBoost split on them. `Title_Mr, Sex_male, and Pclass` dominate while others add smaller boosts.  
**Why it matters:** Shows what XGB leans on most to separate classes, and aligns with the RF view and Titanic history.

### What’s next

- **NASDAQ FCF project:** I’ll apply this clean setup to my NASDAQ fundamentals data set (free-cash-flow) study and write up the results.  
- **Bonus CART explainer:** I plan to add one small decision tree (CART, depth=3) on the NASDAQ dataset as a “how the model thinks” figure. It’s not meant to beat RF/XGB — just to explain the rules in plain English. I feel like it could be a skill worth learning for future ML practice.
- *If you spot mistakes or have ideas to improve this setup (or future projects like my NASDAQ FCF study), I’d love your advice. Please open an issue or leave a comment with suggestions/fixes.*

<details>
  <summary><strong>Open glossary</strong></summary>

- **Accuracy** — out of 100 people, how many the model gets right. (Good for quick context; can be misleading when classes are uneven.)
  
- **Precision** — when the model says “survived,” how often is it correct? (Avoids false alarms. Yes, there *is* a difference between accuracy and precision.)
  
- **Recall** — out of all real survivors, how many did we find? (Avoids misses.)
  
- **F1 score** — a single number that balances **precision** and **recall**. (Helpful when both matter.)
  
- **Score / Probability** — a number between 0 and 1. For example, “how likely to survive.”
  
- **Threshold (cutoff)** — the line (like 0.5) that turns a score into Yes/No. Lower = catch more survivors but more false alarms; higher = fewer false alarms but miss more survivors.
  
- **ROC curve** — shows trade-offs across all thresholds (catching survivors vs. crying wolf).
  
- **AUC** — area under the ROC curve. If you pick one survivor and one non-survivor at random, AUC is how often the model ranks the survivor higher. (0.5 = coin flip; closer to 1.0 = better.)
  
- **Confusion matrix** — a 2×2 table of correct/incorrect predictions at a chosen threshold (TP, FP, FN, TN). (Great for seeing mistakes.)
  
- **Train / Validation / Test split** — learn on **train**, tune on **validation**, and judge once on **test**. (Keeps you honest.)
  
- **Stratified split** — keeps the survivor rate similar across splits. (Fair comparisons.)
  
- **Data leakage** — when future/answer info sneaks into training or tuning. (Inflates scores; must avoid.)
  
- **Early stopping** — stop training when validation score stops improving. (Prevents overfitting.)
  
- **One-hot encoding** — turn text categories (e.g., `Sex`) into 0/1 columns (`Sex_male`), usually with `drop_first=True` to avoid duplicate info.
  
- **Frozen feature list (`COLS`)** — lock the exact columns/order used for modeling so every step sees the same inputs. (Prevents “N vs M features” bugs.)
  
- **Random Forest** — many decision trees voting together. (Stable baseline for tabular data.)
  
- **XGBoost** — trees built in sequence to fix prior mistakes. (Often a bit stronger than RF on tabular data.)
  
- **Feature importance (RF)** — how much a feature reduces impurity across trees. (Rough influence signal.)
  
- **Feature importance (XGB gain)** — how much a feature improved the model when used in splits. (Relative influence.)
  
- **Class imbalance / Positive rate** — share of survivors in the data. (Helps decide metrics and thresholds.)

**Confusion matrix**  
A 2×2 table of correct vs. wrong predictions at one cutoff (e.g., 0.5).  
**Why we use it:** to see the trade-off between catching real positives (recall) and raising false alarms (precision), and to pick a sensible threshold.

**ROC curve**  
A line that shows model performance across all cutoffs (true-positive rate vs. false-positive rate).  
**Why we use it:** it’s threshold-free, so it’s a fair way to compare models without arguing about a specific cutoff.

**Feature importance — Random Forest**  
How much each feature reduced the impurity across the random forest (rough influence).  
**Why we use it:** quick sanity check for which inputs the model leaned on more.

**Feature importance — XGBoost (gain)**  
How much each feature improved the model’s loss when used in splits.  
**Why we use it:** highlights the features that boosted performance the most.

</details>
