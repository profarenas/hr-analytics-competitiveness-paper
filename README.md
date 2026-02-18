# HR Analytics Competitiveness Paper
Código para el paper "What Drives Employee Productivity and Organizational Competitiveness? Evidence from Predictive HR Analytics with External Validation".
## Descripción
Este repositorio contiene scripts para preprocesamiento, modelado predictivo (OLS, CART, RF, XGBoost), evaluación y interpretabilidad (SHAP) usando datasets de HR analytics. Utiliza datos públicos de Kaggle (IBM HR Analytics) y datos sintéticos para la compañía (confidencial). El seed fijo es 20260214 para reproducibilidad total.
## Dependencias

Python 3.10+
Librerías: numpy, pandas, scipy, scikit-learn, xgboost, shap, joblib (para guardar modelos)
Instala: pip install numpy pandas scipy scikit-learn xgboost shap joblib

## Archivos

**Generic Data/generate_synthetic_data.py**: Genera datos sintéticos para la compañía (N=500).
**scripts/preprocess.py**: Preprocesamiento (winsorizing, imputation, feature engineering, splits sin leakage).
**scripts/train.py**: Entrenamiento de modelos (OLS, CART, RF, XGBoost con grid search).
**scripts/evaluate.py**: Evaluación (métricas RMSE/MAE/R², bootstrap para CIs, SHAP para interpretabilidad).

## Cómo ejecutar

Descarga el dataset público de Kaggle: https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset y guárdalo como hr_data.csv en la raíz.
Corre python Generic Data/generate_synthetic_data.py para generar datos sintéticos de la compañía (guarda synthetic_company_data.csv).
Corre python scripts/preprocess.py (genera CSVs preprocesados: public_preprocessed.csv y company_preprocessed.csv).
Corre python scripts/train.py (entrena y guarda modelos .pkl para satisfaction/performance).
Corre python scripts/evaluate.py (calcula métricas, CIs y genera SHAP plots; guarda evaluation_results.csv).
Notas: Ajusta paths si es necesario. Para validación externa, usa los CSVs preprocesados de compañía en evaluate.py.

## Archivos de salida (ejemplos)

synthetic_company_data.csv: Datos sintéticos generados (agrega cuando subas).
evaluation_results.csv: Tabla de métricas (RMSE, MAE, R²).
