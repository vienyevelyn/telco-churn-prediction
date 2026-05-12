# telco-churn-prediction

Predicting customer churn using Random Forest + SMOTE. ML portfolio project.

# Telco Customer Churn Prediction

## Problem
A telecom company loses ~20% of customers yearly. Predicting churn helps target retention offers, potentially saving millions in lost revenue.

## Dataset
- **Source:** IBM Telco Customer Churn (public)
- **Size:** 7,043 customers, 19 features after cleaning
- **Target:** `Churn` (Yes=1, No=0) – 26.5% churn rate

## Approach

### Data Cleaning
- Removed `customerID` (not predictive)
- Converted `TotalCharges` to numeric, filled missing with median
- Encoded `Churn` as 0/1

### Preprocessing
- **Numeric features** (tenure, MonthlyCharges, TotalCharges): scaled with StandardScaler
- **Categorical features** (Contract, PaymentMethod, etc.): one-hot encoded with drop='first'

### Handling Imbalance
- Used **SMOTE** (Synthetic Minority Over-sampling) on the training set.
- Tuned decision threshold to **0.45** (instead of default 0.5) to improve recall.

### Model
**Random Forest Classifier** (100 trees) trained on SMOTE-resampled data.

## Results

| Metric | Score |
|--------|-------|
| **Recall (Churn)** | **0.62** |
| **ROC-AUC** | **0.824** |
| **Precision (Churn)** | 0.54 |
| **Accuracy** | 0.76 |

**Confusion Matrix:**
[[842 193]
[143 231]]


**Interpretation:**  
- Catches **62% of actual churners** (recall).  
- For every 100 customers targeted, 54 are real churners (precision).  
- Trade-off is acceptable because retaining a customer is far more valuable than the cost of a retention offer.

Top drivers of churn:
- **Tenure** – new customers (<6 months) are highest risk
- **Month-to-month contracts** – much higher churn than annual plans
- **No online security** or **no tech support**

## Recommendations
1. Offer annual contract discounts to month-to-month customers
2. Provide free online security for first 6 months
3. Launch a welcome retention program for customers with tenure <6 months

## How to Reproduce
1. Open `telco_churn_portfolio.ipynb` in Google Colab
2. Run all cells (requires pandas, scikit-learn, imbalanced-learn, matplotlib)
3. Or click the badge:

https://colab.research.google.com/drive/1zMUUPkJCnVyRWl-DLc3qk97-wdN502l3?usp=sharing

## Limitations & Next Steps
- No customer support call logs – could improve recall
- Try XGBoost for comparison
- Build a web app (Gradio) for live predictions

## Tools Used
- Python, pandas, scikit-learn, imbalanced-learn (SMOTE), matplotlib
- Google Colab, GitHub
