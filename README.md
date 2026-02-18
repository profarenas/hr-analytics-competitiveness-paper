# HR Analytics Competitiveness Paper

Code for the paper "What Drives Employee Productivity and Organizational Competitiveness? Evidence from Predictive HR Analytics with External Validation".

## Description
This repository contains scripts for preprocessing, predictive modeling (OLS, CART, RF, XGBoost), evaluation, and interpretability (SHAP) using HR analytics datasets. It uses public data from Kaggle (IBM HR Analytics) and synthetic data for the company (confidential). The fixed seed is 20260214 for full reproducibility.

## Dependencies
- Python 3.10+
- Libraries: numpy, pandas, scipy, scikit-learn, xgboost, shap, joblib (for saving models)
Install: `pip install numpy pandas scipy scikit-learn xgboost shap joblib`

## Files
- **Generic Data/generate_synthetic_data.py**: Generates synthetic data for the company (N=500).
- **evaluate.py**: Evaluation (RMSE/MAE/R² metrics, bootstrap for CIs, SHAP for interpretability).
- **preprocess.py**: Preprocessing (winsorizing, imputation, feature engineering, splits without leakage).
- **train.py**: Model training (OLS, CART, RF, XGBoost with grid search).

## How to Run
1. Download the public Kaggle dataset: [https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) and save it as `hr_data.csv` in the root.
2. Run `python Generic Data/generate_synthetic_data.py` to generate synthetic company data (saves `synthetic_company_data.csv`).
3. Run `python preprocess.py` (generates preprocessed CSVs: public_preprocessed.csv and company_preprocessed.csv).
4. Run `python train.py` (trains and saves model .pkl files for satisfaction/performance).
5. Run `python evaluate.py` (computes metrics, CIs, and generates SHAP plots; saves evaluation_results.csv).

Notes: Adjust paths if necessary. For external validation, use the preprocessed company CSVs in evaluate.py.

## Output Files (Examples)
- synthetic_company_data.csv: Generated synthetic company data.
- evaluation_results.csv: Table of evaluation metrics (RMSE, MAE, R²).

If you have questions or errors when running, open an issue in this repo.
