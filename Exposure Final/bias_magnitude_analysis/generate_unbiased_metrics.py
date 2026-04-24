"""
Generate unbiased_metrics.csv by evaluating saved models on unbiased evaluation sets.
Run from pranathi3: python bias_magnitude_analysis/generate_unbiased_metrics.py
Or from bias_magnitude_analysis: python generate_unbiased_metrics.py (set PRANATHI3_ROOT).
"""
import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

# Paths: use PRANATHI3_ROOT env var, or assume run from pranathi3 or from bias_magnitude_analysis
PRANATHI3 = os.environ.get("PRANATHI3_ROOT")
if not PRANATHI3 or not os.path.isdir(PRANATHI3):
    if os.path.basename(os.getcwd()) == "bias_magnitude_analysis":
        PRANATHI3 = os.path.dirname(os.getcwd())
    else:
        PRANATHI3 = os.getcwd()
PRANATHI3 = os.path.abspath(PRANATHI3)

DATA = os.path.join(PRANATHI3, "data")
BIASED = os.path.join(PRANATHI3, "biased data")
OUT_CSV = os.path.join(PRANATHI3, "bias_magnitude_analysis", "data", "unbiased_metrics.csv")
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)


class _LightGCNLambdaReplacement(tf.keras.layers.Layer):
    """Replacement for Lambda layers in saved LightGCN models so they load without bytecode.
    Behaves as: 1 input -> l2_normalize (out shape (None,1,32)); 2 inputs -> dot (out (None,1,1))."""
    def call(self, inputs, mask=None, **kwargs):
        if isinstance(inputs, (list, tuple)) and len(inputs) == 2:
            return tf.reduce_sum(inputs[0] * inputs[1], axis=-1, keepdims=True)
        return tf.nn.l2_normalize(inputs, axis=-1)

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, (list, tuple)) and len(input_shape) == 2:
            return (None, 1, 1)
        return (None, 1, 32)

    def get_config(self):
        return super().get_config()

    @classmethod
    def from_config(cls, config):
        return cls(name=config.get("name"), trainable=config.get("trainable", True))


def _load_model(model_path):
    """Load model; for LightGCN, patch Lambda.from_config so layers load without deserializing bytecode."""
    is_lightgcn = "lightgcn" in os.path.basename(model_path).lower()
    if not is_lightgcn:
        return tf.keras.models.load_model(model_path, safe_mode=False)
    # Patch Lambda.from_config so deserialization uses our replacement (avoids marshal.loads)
    lambda_module = getattr(tf.keras.layers, "Lambda", None)
    if lambda_module is None:
        try:
            from keras.src.layers.core import lambda_layer as _lm
            lambda_module = _lm
        except ImportError:
            lambda_module = tf.keras.layers
    Lambda_cls = getattr(lambda_module, "Lambda", None) or tf.keras.layers.Lambda
    _original_from_config = Lambda_cls.from_config

    def _replacement_from_config(cls, config):
        return _LightGCNLambdaReplacement.from_config(config)

    Lambda_cls.from_config = classmethod(_replacement_from_config)
    try:
        return tf.keras.models.load_model(model_path, safe_mode=False)
    finally:
        Lambda_cls.from_config = _original_from_config


def ndcg_at_k(y_true, y_pred, k=10):
    top_k_indices = np.argsort(y_pred)[::-1][:k]
    relevance = y_true[top_k_indices]
    dcg = np.sum(relevance / np.log2(np.arange(2, len(relevance) + 2)))
    ideal_relevance = np.sort(y_true)[::-1][:k]
    idcg = np.sum(ideal_relevance / np.log2(np.arange(2, len(ideal_relevance) + 2)))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def recall_at_k(y_true, y_pred, k=10, threshold=0.5):
    top_k_indices = np.argsort(y_pred)[::-1][:k]
    top_k_relevance = y_true[top_k_indices]
    relevant_in_topk = np.sum(top_k_relevance >= threshold)
    total_relevant = np.sum(y_true >= threshold)
    if total_relevant == 0:
        return 0.0
    return relevant_in_topk / total_relevant


def compute_metrics(y_true, y_pred, df_with_user=None):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    ndcg5 = ndcg10 = recall5 = recall10 = np.nan
    if df_with_user is not None and "user" in df_with_user.columns:
        ndcg5_list, ndcg10_list, r5_list, r10_list = [], [], [], []
        for uid in df_with_user["user"].unique():
            mask = df_with_user["user"] == uid
            yt, yp = y_true[mask], y_pred[mask]
            if len(yt) > 0:
                ndcg5_list.append(ndcg_at_k(yt, yp, k=5))
                ndcg10_list.append(ndcg_at_k(yt, yp, k=10))
                r5_list.append(recall_at_k(yt, yp, k=5))
                r10_list.append(recall_at_k(yt, yp, k=10))
        ndcg5 = np.mean(ndcg5_list) if ndcg5_list else np.nan
        ndcg10 = np.mean(ndcg10_list) if ndcg10_list else np.nan
        recall5 = np.mean(r5_list) if r5_list else np.nan
        recall10 = np.mean(r10_list) if r10_list else np.nan
    return rmse, mae, ndcg5, ndcg10, recall5, recall10


def eval_yahoo(dataset_label, arch, model_path, biased_path, unbiased_path):
    """Yahoo: biased data for encoding; unbiased = random.txt."""
    if not os.path.isfile(biased_path) or not os.path.isfile(unbiased_path):
        return None
    ratings_b = pd.read_csv(biased_path, names=["userId", "itemId", "rating"])
    ratings_b["userId"] = ratings_b["userId"].astype(int)
    ratings_b["itemId"] = ratings_b["itemId"].astype(int)
    scaler = MinMaxScaler()
    ratings_b["rating"] = scaler.fit_transform(ratings_b[["rating"]])

    user_ids = ratings_b["userId"].unique().tolist()
    item_ids = ratings_b["itemId"].unique().tolist()
    user2enc = {x: i for i, x in enumerate(user_ids)}
    item2enc = {x: i for i, x in enumerate(item_ids)}

    df_u = pd.read_csv(unbiased_path, names=["userId", "itemId", "rating"])
    df_u["userId"] = df_u["userId"].astype(int)
    df_u["itemId"] = df_u["itemId"].astype(int)
    df_u = df_u[df_u["userId"].isin(user2enc) & df_u["itemId"].isin(item2enc)]
    if df_u.empty:
        return None
    df_u["user"] = df_u["userId"].map(user2enc)
    df_u["item"] = df_u["itemId"].map(item2enc)
    df_u["rating"] = scaler.transform(df_u[["rating"]])

    try:
        model = _load_model(model_path)
    except Exception as e:
        print(f"  Load model failed {model_path}: {e}")
        return None
    X = [df_u["user"].values, df_u["item"].values]
    y_true = df_u["rating"].values.astype(np.float32)
    y_pred = model.predict(X, verbose=0).flatten()
    rmse, mae, ndcg5, ndcg10, r5, r10 = compute_metrics(y_true, y_pred, df_u)
    return {
        "Dataset": dataset_label, "Architecture": arch, "Model": "Naive ERM",
        "RMSE": rmse, "MAE": mae, "NDCG@5": ndcg5, "NDCG@10": ndcg10,
        "Recall@5": r5, "Recall@10": r10,
    }


def eval_coat(dataset_label, arch, model_path, train_path, test_path):
    """Coat: train for encoding; test.csv = unbiased."""
    if not os.path.isfile(train_path) or not os.path.isfile(test_path):
        return None
    ratings_b = pd.read_csv(train_path, header=0)
    ratings_b["userId"] = ratings_b["userId"].astype(int)
    ratings_b["itemId"] = ratings_b["itemId"].astype(int)
    user_ids = ratings_b["userId"].unique().tolist()
    item_ids = ratings_b["itemId"].unique().tolist()
    user2enc = {x: i for i, x in enumerate(user_ids)}
    item2enc = {x: i for i, x in enumerate(item_ids)}
    ratings_b["user"] = ratings_b["userId"].map(user2enc)
    ratings_b["item"] = ratings_b["itemId"].map(item2enc)

    df_u = pd.read_csv(test_path, header=0)
    df_u["userId"] = df_u["userId"].astype(int)
    df_u["itemId"] = df_u["itemId"].astype(int)
    df_u = df_u[df_u["userId"].isin(user2enc) & df_u["itemId"].isin(item2enc)]
    if df_u.empty:
        return None
    df_u["user"] = df_u["userId"].map(user2enc)
    df_u["item"] = df_u["itemId"].map(item2enc)
    y_true = df_u["rating"].values.astype(np.float32)

    try:
        model = _load_model(model_path)
    except Exception as e:
        print(f"  Load model failed {model_path}: {e}")
        return None
    X = [df_u["user"].values, df_u["item"].values]
    y_pred = model.predict(X, verbose=0).flatten()
    rmse, mae, ndcg5, ndcg10, r5, r10 = compute_metrics(y_true, y_pred, df_u)
    return {
        "Dataset": dataset_label, "Architecture": arch, "Model": "Naive ERM",
        "RMSE": rmse, "MAE": mae, "NDCG@5": ndcg5, "NDCG@10": ndcg10,
        "Recall@5": r5, "Recall@10": r10,
    }


def eval_kuairec(dataset_label, arch, model_path, combined_path):
    """KuaiRec: use 80/20 split; test = unbiased evaluation set."""
    if not os.path.isfile(combined_path):
        return None
    ratings = pd.read_csv(combined_path)
    if "user_id" in ratings.columns:
        ratings = ratings.rename(columns={"user_id": "userId", "video_id": "itemId"})
    if "watch_ratio" in ratings.columns and "rating" not in ratings.columns:
        ratings["rating"] = ratings["watch_ratio"]
    ratings = ratings[["userId", "itemId", "rating"]].dropna()
    user_ids = ratings["userId"].unique()
    item_ids = ratings["itemId"].unique()
    user2idx = {u: i for i, u in enumerate(user_ids)}
    item2idx = {u: i for i, u in enumerate(item_ids)}
    ratings["user"] = ratings["userId"].map(user2idx)
    ratings["item"] = ratings["itemId"].map(item2idx)
    ratings = ratings.dropna(subset=["user", "item"])

    train_rows, test_rows = [], []
    for uid, grp in ratings.groupby("user"):
        n = len(grp)
        grp = grp.sample(frac=1, random_state=42)
        tr_size = max(1, int(0.8 * n))
        train_rows.append(grp.iloc[:tr_size])
        if tr_size < n:
            test_rows.append(grp.iloc[tr_size:])
    test_df = pd.concat(test_rows) if test_rows else None
    if test_df is None or len(test_df) == 0:
        return None
    y_true = test_df["rating"].values.astype(np.float32)
    X = [test_df["user"].values, test_df["item"].values]

    try:
        model = _load_model(model_path)
    except Exception as e:
        print(f"  Load model failed {model_path}: {e}")
        return None
    y_pred = model.predict(X, verbose=0).flatten()
    rmse, mae, ndcg5, ndcg10, r5, r10 = compute_metrics(y_true, y_pred, test_df)
    return {
        "Dataset": dataset_label, "Architecture": arch, "Model": "Naive ERM",
        "RMSE": rmse, "MAE": mae, "NDCG@5": ndcg5, "NDCG@10": ndcg10,
        "Recall@5": r5, "Recall@10": r10,
    }


def main():
    yahoo_data = os.path.join(DATA, "yahoo_data")
    coat_data = os.path.join(DATA, "coat_data", "coat_data", "coat")
    kuairec_data = os.path.join(DATA, "kuairec_data")
    yahoo_biased = os.path.join(BIASED, "yahoo")
    coat_biased = os.path.join(BIASED, "coat shopping")
    kuairec_biased = os.path.join(BIASED, "kuairec")

    configs = [
        ("YAHOO", "MF", os.path.join(yahoo_biased, "mf_pranathi_best_model.keras"),
         os.path.join(yahoo_data, "user.txt"), os.path.join(yahoo_data, "random.txt"), "yahoo"),
        ("YAHOO", "NCF", os.path.join(yahoo_biased, "ncf_pranathi_best_model.keras"),
         os.path.join(yahoo_data, "user.txt"), os.path.join(yahoo_data, "random.txt"), "yahoo"),
        ("YAHOO", "LightGCN", os.path.join(yahoo_biased, "lightgcn_yahoo_best_model.keras"),
         os.path.join(yahoo_data, "user.txt"), os.path.join(yahoo_data, "random.txt"), "yahoo"),
        ("COAT", "MF", os.path.join(coat_biased, "mf_coat_best_model.keras"),
         os.path.join(coat_data, "train.csv"), os.path.join(coat_data, "test.csv"), "coat"),
        ("COAT", "NCF", os.path.join(coat_biased, "ncf_coat_best_model.keras"),
         os.path.join(coat_data, "train.csv"), os.path.join(coat_data, "test.csv"), "coat"),
        ("COAT", "LightGCN", os.path.join(coat_biased, "lightgcn_coat_best_model.keras"),
         os.path.join(coat_data, "train.csv"), os.path.join(coat_data, "test.csv"), "coat"),
        ("KUAIREC", "MF", os.path.join(kuairec_biased, "mf_kuairec_best_model.keras"),
         os.path.join(kuairec_data, "kuairec_combined.csv"), None, "kuairec"),
        ("KUAIREC", "NCF", os.path.join(kuairec_biased, "ncf_kuairec_best_model.keras"),
         os.path.join(kuairec_data, "kuairec_combined.csv"), None, "kuairec"),
        ("KUAIREC", "LightGCN", os.path.join(kuairec_biased, "lightgcn_kuairec_best_model.keras"),
         os.path.join(kuairec_data, "kuairec_combined.csv"), None, "kuairec"),
    ]

    METRIC_COLS = ["RMSE", "MAE", "NDCG@5", "NDCG@10", "Recall@5", "Recall@10"]
    # Canonical 9 rows: (Dataset, Architecture) so downstream analysis always has 9 rows
    canonical = [
        ("YAHOO", "MF"), ("YAHOO", "NCF"), ("YAHOO", "LightGCN"),
        ("COAT", "MF"), ("COAT", "NCF"), ("COAT", "LightGCN"),
        ("KUAIREC", "MF"), ("KUAIREC", "NCF"), ("KUAIREC", "LightGCN"),
    ]
    results_by_key = {}
    for cfg in configs:
        dataset_label, arch, model_path, path_a, path_b, kind = cfg
        print(f"{dataset_label} {arch} ...", end=" ")
        if kind == "yahoo":
            row = eval_yahoo(dataset_label, arch, model_path, path_a, path_b)
        elif kind == "coat":
            row = eval_coat(dataset_label, arch, model_path, path_a, path_b)
        else:
            row = eval_kuairec(dataset_label, arch, model_path, path_a)
        key = (dataset_label, arch)
        if row is not None:
            results_by_key[key] = row
            print("OK")
        else:
            results_by_key[key] = {
                "Dataset": dataset_label, "Architecture": arch, "Model": "Naive ERM",
                **{m: np.nan for m in METRIC_COLS},
            }
            print("skip (missing data or model)")
    rows = [results_by_key[k] for k in canonical]
    out = pd.DataFrame(rows)
    out = out[["Dataset", "Architecture", "Model"] + METRIC_COLS]
    out.to_csv(OUT_CSV, index=False)
    n_ok = sum(1 for r in rows if pd.notna(r.get("RMSE")) and np.isfinite(r.get("RMSE", np.nan)))
    print(f"Wrote {len(out)} rows to {OUT_CSV} ({n_ok} with metrics, {len(out) - n_ok} placeholder)")


if __name__ == "__main__":
    main()
