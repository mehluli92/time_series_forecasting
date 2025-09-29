# Credit Card Clients Default Prediction

## Overview
This project predicts credit card client defaults using a dataset of 30,000 records with 25 attributes. The goal is to identify high-risk clients for proactive risk management.

## Dataset
- **Source**: `credit_card_clients.xls`
- **Features**: Demographic data, credit limit, payment history, bill amounts, payment amounts
- **Target**: `default_payment_next_month` (0 = no default, 1 = default)

## Methodology
- **Data Cleaning**: Removed `ID`, handled invalid categorical values
- **Feature Engineering**: Added `AVG_BILL_AMT`, `AVG_PAY_AMT`, `PAY_TO_BILL_RATIO`
- **Models**: Logistic Regression, Random Forest, XGBoost
- **Evaluation**: ROC-AUC and accuracy; XGBoost performed best (ROC-AUC: 0.787, Accuracy: 0.821)

## Key Findings
- Payment history (`PAY_0` to `PAY_6`) and credit limit (`LIMIT_BAL`) are key predictors
- Lower payment-to-bill ratios indicate higher default risk

## Requirements
- Python 3.12
- Libraries: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`, `seaborn`, `openpyxl`, `xlrd`

## Usage
Run `2Credit_Card_Clients_Analysis.ipynb` to preprocess data, train models, and evaluate performance.

## Next Steps
- Hyperparameter tuning for XGBoost
- Explore SMOTE for class imbalance
- Use SHAP for interpretability
- Integrate external data for enhanced predictions

---

*Generated from `2Credit_Card_Clients_Analysis.ipynb`*
