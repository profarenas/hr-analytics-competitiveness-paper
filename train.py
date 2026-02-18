import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, StratifiedKFold

np.random.seed(20260214)

# Cargar datos preprocesados
X_train = pd.read_csv('public_preprocessed.csv').drop(['satisfaction_level', 'last_evaluation'], axis=1)  # Ejemplo para público, ajusta
y_sat_train = pd.read_csv('public_preprocessed.csv')['satisfaction_level']

# Modelos
models = {}

# OLS (Linear Regression)
ols = LinearRegression()
ols.fit(X_train, y_sat_train)
models['OLS'] = ols

# CART (Decision Tree)
cart_params = {'max_depth': [3, 5, 7]}
cart_grid = GridSearchCV(DecisionTreeRegressor(random_state=20260214), cart_params, cv=5)
cart_grid.fit(X_train, y_sat_train)
models['CART'] = cart_grid.best_estimator_

# RF (Random Forest, bagging)
rf_params = {'n_estimators': [50, 100], 'max_depth': [5, 10]}
rf_grid = GridSearchCV(RandomForestRegressor(random_state=20260214), rf_params, cv=5)
rf_grid.fit(X_train, y_sat_train)
models['RF'] = rf_grid.best_estimator_

# XGBoost (boosting)
xgb_params = {'n_estimators': [50, 100], 'max_depth': [3, 5], 'learning_rate': [0.01, 0.1]}
xgb_grid = GridSearchCV(XGBRegressor(random_state=20260214, early_stopping_rounds=10), xgb_params, cv=5)
xgb_grid.fit(X_train, y_sat_train)
models['XGB'] = xgb_grid.best_estimator_

# Guardar modelos (usa joblib para persistencia)
import joblib
for name, model in models.items():
    joblib.dump(model, f'{name}_sat_model.pkl')  # Similar para performance

print("Entrenamiento completado y modelos guardados.")
