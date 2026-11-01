import pandas as pd
import numpy as np
import time
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import SGDClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import sys
sys.path.append('.')
from configs.lab2_config import Lab2Config

def load_data(path):
    df = pd.read_csv(path, header=None, na_values="?")
    # Класс 1 - норма (0), остальные - аритмия (1)
    y = (df.iloc[:, -1] != 1).astype(int).values
    X = df.iloc[:, :-1].values
    return X, y

def bootstrap_ci(y_true, y_scores, metric_fn, n_boot=500, alpha=0.05):
    rng = np.random.RandomState(42)
    vals = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        vals.append(metric_fn(y_true[idx], y_scores[idx]))
    return np.percentile(vals, [alpha/2*100, (1-alpha/2)*100])

def run_full(X, y, config):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.test_size, random_state=config.random_state, stratify=y
    )
    
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(imputer.fit_transform(X_train))
    X_test = scaler.transform(imputer.transform(X_test))
    
    model = IsolationForest(
        contamination=config.contamination,
        n_estimators=config.n_estimators,
        random_state=config.random_state,
        n_jobs=-1
    )
    t0 = time.time()
    model.fit(X_train)
    train_time = time.time() - t0
    
    y_pred = (model.predict(X_test) == -1).astype(int)
    y_scores = -model.decision_function(X_test)
    
    metrics = {
        "roc_auc": roc_auc_score(y_test, y_scores),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "train_time_sec": train_time,
    }
    return model, metrics, y_test, y_scores

def run_chunk(X, y, config):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.test_size, random_state=config.random_state, stratify=y
    )
    
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(imputer.fit_transform(X_train))
    X_test = scaler.transform(imputer.transform(X_test))
    
    model = SGDClassifier(
        loss="log_loss", alpha=config.sgd_alpha,
        random_state=config.random_state,
        max_iter=1, warm_start=True
    )
    
    t0 = time.time()
    n = len(X_train)
    for epoch in range(config.epochs):
        idx = np.random.RandomState(config.random_state + epoch).permutation(n)
        for start in range(0, n, config.chunk_size):
            batch = idx[start:start + config.chunk_size]
            model.partial_fit(X_train[batch], y_train[batch], classes=np.array([0, 1]))
    train_time = time.time() - t0
    
    y_pred = model.predict(X_test)
    y_scores = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        "roc_auc": roc_auc_score(y_test, y_scores),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "train_time_sec": train_time,
    }
    return model, metrics, y_test, y_scores

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["full", "chunk"], required=True)
    args = parser.parse_args()
    
    config = Lab2Config()
    X, y = load_data(config.data_path)
    
    print(f"Запуск в режиме: {args.mode}")
    if args.mode == "full":
        model, metrics, y_test, y_scores = run_full(X, y, config)
    else:
        model, metrics, y_test, y_scores = run_chunk(X, y, config)
    
    lo, hi = bootstrap_ci(y_test, y_scores, roc_auc_score)
    print(f"{args.mode}: ROC-AUC={metrics['roc_auc']:.3f} [{lo:.3f}, {hi:.3f}]")
    
    # Сохраняем модель
    joblib.dump(model, f"reports/LAB2/model_{args.mode}.joblib")
    
    # Сохраняем метрики
    res_df = pd.DataFrame([{"mode": args.mode, **metrics, "roc_auc_lower": lo, "roc_auc_upper": hi}])
    res_df.to_csv(f"reports/LAB2/metrics_{args.mode}.csv", index=False)
    print(f"Метрики сохранены в reports/LAB2/metrics_{args.mode}.csv")

    # MLflow
    mlflow.set_experiment("lab2_arrhythmia")
    with mlflow.start_run(run_name=f"{args.mode}_run"):
        mlflow.log_params(config.model_dump())
        mlflow.log_metrics(metrics)
        print(f"Прогон MLflow сохранен. ID: {mlflow.active_run().info.run_id}")
