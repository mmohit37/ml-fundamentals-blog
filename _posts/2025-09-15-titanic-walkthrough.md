---
layout: post
title: "Titanic: a quick ML walkthrough (EDA → CART → XGBoost)"
date: 2025-09-15
categories: kaggle titanic
tags: tutorial sklearn xgboost dtreeviz
---

I used the classic Kaggle **Titanic** dataset to warm up on feature engineering and tree-based models.  
This post walks through what I did **and why I did it**, so people can replicate the experiment and understand my thought process.

---

## 1) Goal & data

**Goal:** predict whether a passenger survived the Titanic disaster.  
**Data:** `data/raw/titanic/train.csv` (Kaggle’s Titanic competition).

**Why:** it’s a small tabular dataset that’s perfect for practicing the full exploratory data analysis (EDA) → features → modeling → interpretation without getting buried in scale or noise.

---

## 2) Fast features that carry meaning

I built a few simple features:

- `fam_size = SibSp + Parch + 1`  
  **Why:** family presence matters socially (someone to help you get to lifeboats), and this is an easy way to encode it. It also makes sense to group siblings and spouses (SibSp) and parents and children (Parch) instead of looking at them individually.

- `alone = (fam_size == 1)`  
  **Why:** lots of Kaggle notebooks show that being alone correlates with survival odds. The model can learn it from `fam_size`, but an explicit boolean value that classifies someone as alone (1) or not (0).

- **Title** from `Name` (Mr, Mrs, Miss, Master, Rare)  
  **Why:** titles represent **age**, **gender**, and sometimes **status**. They’re surprisingly strong signals and more stable than raw name text. It also makes sense that women and children would be prioritized for safety first, so creating this variable can help incorporate that into the model.

- **Grouped imputations**  
  - `Age`: filled by **median within title**  
    **Why:** Kids and “Master” shouldn’t receive adult medians; imputing within groups preserves a strong structure for the model.  
  - `Fare`: median within **(Pclass, Embarked)**  
    **Why:** Ticket cost varies a lot by class and port. Thus, grouped medians give more sensible values.  
  - `Embarked`: filled with **mode**  
    **Why:** It’s categorical and only a few values are missing; the simplest fix wins.

- **One-hot encoding** for `Sex`, `Embarked`, and `title` with `drop_first=True`  
  **Why:** tree models don’t need scaling, but they still need categorical variables turned into columns. `drop_first` avoids a redundant column (a little regularization for free).

Feature set I used:  
`["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked","title","fam_size","alone"]`

---

## 3) Class balance check

I plotted the class counts:

![Survivor counts]({{ '/assets/images/titanic/survivor_counts_v2.png' | relative_url }})

**Why:** Before we judge any model, we need to know how many people survived vs. didn’t survive in the training data. If one group is much bigger than the other, a lazy model can look “good” just by always guessing the bigger group. Checking the bar chart tells us how skewed things are and which scores make sense. In Titanic, non-survivors are more than survivors, but it’s not crazy unbalanced, so accuracy/AUC/F1 are still meaningful without fancy tricks. If the gap were huge, we’d consider things like class weights or different evaluation metrics.

---

## 4) Train/valid/test split

```python
from sklearn.model_selection import train_test_split

X = df.drop("Survived", axis=1)
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
