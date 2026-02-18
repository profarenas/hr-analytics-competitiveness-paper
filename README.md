# hr-analytics-competitiveness-paper
Code for Paper What Drives Employee Productivity and Organizational Competitiveness? Evidence from Predictive HR Analytics with External Validation
# HR Analytics Competitiveness Paper

Código para el paper "What Drives Employee Productivity and Organizational Competitiveness? Evidence from Predictive HR Analytics with External Validation".

## Descripción
Este repo contiene scripts para preprocesamiento, modelado predictivo (OLS, CART, RF, XGBoost), evaluación y interpretabilidad (SHAP) usando datasets de HR. Usa datos públicos de Kaggle y sintéticos para la compañía (confidencial).

## Dependencias
- Python 3.10+
- Librerías: numpy, pandas, scikit-learn, xgboost, shap
Instala: `pip install numpy pandas scikit-learn xgboost shap`

## Archivos
- Generic Data/generate_synthetic_data.py: Genera datos sintéticos para la compañía (N=500).
- (Agrega más: preprocess.py, train.py, etc., cuando los subas)

## Cómo ejecutar
1. Descarga Kaggle dataset: [link a Kaggle].
2. Corre `python generate_synthetic_data.py` para datos de compañía.
3. Usa los modelos en train.py (próximamente).

Seed fijo: 20260214 para reproducibilidad.
