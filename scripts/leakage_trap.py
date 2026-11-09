import sys
sys.path.append('.')
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from configs.lab2_config import Lab2Config

config = Lab2Config()

df = pd.read_csv(config.data_path, header=None, na_values="?")
y = (df.iloc[:, -1] != 1).astype(int).values
X = df.iloc[:, :-1].values

# Правильно: Imputer и Scaler только на train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.test_size, random_state=config.random_state, stratify=y)
imp_ok = SimpleImputer(strategy="median")
sc_ok = StandardScaler()
X_train_ok = sc_ok.fit_transform(imp_ok.fit_transform(X_train))
X_test_ok = sc_ok.transform(imp_ok.transform(X_test))

model_ok = IsolationForest(random_state=config.random_state, n_jobs=-1)
model_ok.fit(X_train_ok)
auc_ok = roc_auc_score(y_test, -model_ok.decision_function(X_test_ok))

# Неправильно (утечка): Imputer и Scaler на всех данных
imp_leak = SimpleImputer(strategy="median")
sc_leak = StandardScaler()
X_all_imp = imp_leak.fit_transform(X)  # Утечка!
X_all_sc = sc_leak.fit_transform(X_all_imp) # Утечка!
X_train_leak, X_test_leak, y_train, y_test = train_test_split(X_all_sc, y, test_size=config.test_size, random_state=config.random_state, stratify=y)

model_leak = IsolationForest(random_state=config.random_state, n_jobs=-1)
model_leak.fit(X_train_leak)
auc_leak = roc_auc_score(y_test, -model_leak.decision_function(X_test_leak))

with open("leakage_trap.log", "w", encoding="utf-8") as f:
    f.write(f"Правильный ROC-AUC (без утечки): {auc_ok:.4f}\n")
    f.write(f"Неправильный ROC-AUC (с утечкой): {auc_leak:.4f}\n")
    f.write(f"Разница: {(auc_leak - auc_ok):+.4f}\n")

print("Лог ловушки утечки создан")
