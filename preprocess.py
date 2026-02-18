import numpy as np
import pandas as pd
from scipy.stats import mstats  # Para winsorizing

np.random.seed(20260214)  # Seed fijo para reproducibilidad

# Carga datos públicos (Kaggle) y sintéticos (compañía)
public_data = pd.read_csv('hr_data.csv')  # Ajusta path si es necesario
company_data = pd.read_csv('synthetic_company_data.csv')  # Generado por generate_synthetic_data.py

# Preprocesamiento común
def preprocess(df):
    # Winsorizing outliers al 1%/99%
    numeric_cols = ['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours', 'time_spend_company']  # Ajusta nombres si difieren
    for col in numeric_cols:
        if col in df.columns:
            df[col] = mstats.winsorize(df[col], limits=[0.01, 0.01])
    
    # Imputación mediana para missing <1%
    df.fillna(df.median(numeric_only=True), inplace=True)
    
    # Feature engineering
    df['hours_per_project'] = df['average_montly_hours'] / df['number_project']
    df['time_spend_company_sq'] = df['time_spend_company'] ** 2  # Polinomial para no-linealidad
    
    # Encoding categóricos
    df = pd.get_dummies(df, columns=['department', 'salary'], drop_first=True)
    
    # No-leakage: Excluir 'left' para predicción de satisfaction/performance
    if 'left' in df.columns:
        df.drop('left', axis=1, inplace=True)
    
    return df

# Aplicar
public_preprocessed = preprocess(public_data)
company_preprocessed = preprocess(company_data)

# Splits estratificados (80/20) para público
from sklearn.model_selection import train_test_split
X_public = public_preprocessed.drop(['satisfaction_level', 'last_evaluation'], axis=1)  # Features
y_satisfaction_public = public_preprocessed['satisfaction_level']
y_performance_public = public_preprocessed['last_evaluation']
X_train_pub, X_test_pub, y_sat_train_pub, y_sat_test_pub = train_test_split(X_public, y_satisfaction_public, test_size=0.2, stratify=public_preprocessed[['department', 'salary']], random_state=20260214)
# Similar para performance...

# Guardar preprocesados
public_preprocessed.to_csv('public_preprocessed.csv', index=False)
company_preprocessed.to_csv('company_preprocessed.csv', index=False)

print("Preprocesamiento completado y archivos guardados.")
