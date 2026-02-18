import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import shap
import joblib
from sklearn.utils import resample  # Para bootstrap CIs

np.random.seed(20260214)

# Cargar test data y modelos (ej. para satisfaction)
X_test = pd.read_csv('public_preprocessed.csv').drop(['satisfaction_level', 'last_evaluation'], axis=1)  # Ajusta para test split
y_test = pd.read_csv('public_preprocessed.csv')['satisfaction_level']
models = {name: joblib.load(f'{name}_sat_model.pkl') for name in ['OLS', 'CART', 'RF', 'XGB']}

# Evaluación básica
results = {}
for name, model in models.items():
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {'RMSE': rmse, 'MAE': mae, 'R2': r2}

# Bootstrap para CIs (1000 resamples)
cis = {}
for name in models:
    bootstraps = []
    for _ in range(1000):
        X_boot, y_boot = resample(X_test, y_test)
        y_pred_boot = models[name].predict(X_boot)
        r2_boot = r2_score(y_boot, y_pred_boot)
        bootstraps.append(r2_boot)
    cis[name] = np.percentile(bootstraps, [2.5, 97.5])

# SHAP para interpretabilidad (ej. para XGB)
explainer = shap.TreeExplainer(models['XGB'])
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, show=False)  # Guarda como fig

# Robustez: Ej. alternative imputation (mean en vez de median)
# ... (agrega checks similares)

# Imprimir resultados
print(results)
print("CIs for R2:", cis)

# Guardar resultados
pd.DataFrame(results).to_csv('evaluation_results.csv')
