import pandas as pd
import numpy as np
import time
import joblib
import mlflow
import mlflow.sklearn
import hashlib
import os
import psutil
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import SGDClassifier
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, average_precision_score
import sys
sys.path.append('.')
from configs.lab2_config import Lab2Config

# Функция для замера памяти (в МБ)
def get_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

# Функция для хеша файла
def get_file_hash(filepath):
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

# Функция для Precision@K (профиль P6)
def precision_at_k(y_true, y_scores, k=20):
    # K самых аномальных предсказаний (наибольший y_scores)
    top_k_indices = np.argsort(y_scores)[-k:]
    top_k_true = y_true[top_k_indices]
    return np.sum(top_k_true) / k

def load_data(path):
    df = pd.read_csv(path, header=None, na_values="?")
    y = (df.iloc[:, -1] != 1).astype(int).values  # 0 - норма, 1 - аномалия
    X = df.iloc[:, :-1].values
    return X, y

def bootstrap_ci(y_true, y_scores, metric_fn, n_boot=500, alpha=0.05):
    rng = np.random.RandomState(42)
    vals = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        # обработка крайних случаев, когда в выборке только 1 класс
        if len(np.unique(y_true[idx])) < 2: continue
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
    
    # наивный ориентир
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    dummy_pred = dummy.predict(X_test)
    dummy_f1 = f1_score(y_test, dummy_pred, zero_division=0)
    
    model = IsolationForest(
        contamination=config.contamination,
        n_estimators=config.n_estimators,
        random_state=config.random_state,
        n_jobs=-1
    )
    
    mem_before = get_memory_mb()
    t0 = time.time()
    model.fit(X_train)
    train_time = time.time() - t0
    mem_after = get_memory_mb()
    mem_peak = mem_after - mem_before
    
    y_pred = (model.predict(X_test) == -1).astype(int)
    y_scores = -model.decision_function(X_test)
    
    # новые метрики акцента
    pr_auc = average_precision_score(y_test, y_scores)
    p_at_k = precision_at_k(y_test, y_scores, k=20)
    
    metrics = {
        "roc_auc": roc_auc_score(y_test, y_scores),
        "pr_auc": pr_auc,
        "precision_at_20": p_at_k,
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "dummy_f1": dummy_f1,
        "train_time_sec": train_time,
        "mem_peak_mb": mem_peak
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
    
    # наивный ориентир
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    dummy_f1 = f1_score(y_test, dummy.predict(X_test), zero_division=0)
    
    model = SGDClassifier(
        loss="log_loss", alpha=config.sgd_alpha,
        random_state=config.random_state,
        max_iter=1, warm_start=True
    )
    
    mem_before = get_memory_mb()
    t0 = time.time()
    n = len(X_train)
    for epoch in range(config.epochs):
        idx = np.random.RandomState(config.random_state + epoch).permutation(n)
        for start in range(0, n, config.chunk_size):
            batch = idx[start:start + config.chunk_size]
            model.partial_fit(X_train[batch], y_train[batch], classes=np.array([0, 1]))
    train_time = time.time() - t0
    mem_after = get_memory_mb()
    mem_peak = mem_after - mem_before
    
    y_pred = model.predict(X_test)
    y_scores = model.predict_proba(X_test)[:, 1]
    
    pr_auc = average_precision_score(y_test, y_scores)
    p_at_k = precision_at_k(y_test, y_scores, k=20)
    
    metrics = {
        "roc_auc": roc_auc_score(y_test, y_scores),
        "pr_auc": pr_auc,
        "precision_at_20": p_at_k,
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "dummy_f1": dummy_f1,
        "train_time_sec": train_time,
        "mem_peak_mb": mem_peak
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
    metrics["roc_auc_lower"] = lo
    metrics["roc_auc_upper"] = hi
    
    print(f"{args.mode}: ROC-AUC={metrics['roc_auc']:.3f} [{lo:.3f}, {hi:.3f}] | PR-AUC={metrics['pr_auc']:.3f} | P@20={metrics['precision_at_20']:.3f}")
    
    # хеш и сохр модель
    model_path = f"reports/LAB2/model_{args.mode}.joblib"
    joblib.dump(model, model_path)
    model_hash = get_file_hash(model_path)
    print(f"Хеш модели ({args.mode}): {model_hash}")
    
    # сохр метрики
    res_df = pd.DataFrame([{"mode": args.mode, **metrics, "model_hash": model_hash}])
    res_df.to_csv(f"reports/LAB2/metrics_{args.mode}.csv", index=False)

    # MLflow
    mlflow.set_experiment("lab2_arrhythmia")
    with mlflow.start_run(run_name=f"{args.mode}_run"):
        mlflow.log_params(config.model_dump())
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(model_path)
        print(f"Прогон MLflow сохранен. ID: {mlflow.active_run().info.run_id}")
