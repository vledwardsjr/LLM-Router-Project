import json
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

RANDOM_STATE = 42
TEST_SIZE = 0.2

NUMERIC_FEATURES = [
    "query_word_count", "query_avg_word_length", "query_has_question_mark",
    "query_number_density", "query_capitalized_density",
    "reasoning_step_count", "ambiguity_score",
]
CATEGORICAL_FEATURES = ["domain", "question_type", "source_category"]


def load_and_encode(path: Path = DATA_DIR / "features.parquet") -> tuple[pd.DataFrame, pd.Series, list[str]]:
    df = pd.read_parquet(path)
    encoded = pd.get_dummies(df[CATEGORICAL_FEATURES], prefix=CATEGORICAL_FEATURES)
    X = pd.concat([df[NUMERIC_FEATURES], encoded], axis=1)
    y = df["label"]
    return X, y, list(X.columns)


def stratified_split(X: pd.DataFrame, y: pd.Series):
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )


def train_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series) -> tuple[LogisticRegression, StandardScaler]:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)
    model.fit(X_train_scaled, y_train)
    return model, scaler


def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series) -> xgb.XGBClassifier:
    # Regularized: only 54-68 label=0 examples, so XGBoost's default depth/no-subsampling
    # overfits this minority class. Shallow trees + subsampling + L2 fix that.
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    model = xgb.XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        max_depth=3,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        n_estimators=100,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)
    return model


def cross_validate(X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> dict:
    """A single 80/20 split has only ~14 label=0 examples in the test fold -- too few to
    trust. CV averages over 5 folds so no single fold's luck dominates the result."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    lr_aucs, xgb_aucs = [], []
    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        lr_model, scaler = train_logistic_regression(X_tr, y_tr)
        lr_proba = lr_model.predict_proba(scaler.transform(X_te))[:, 1]
        lr_aucs.append(roc_auc_score(y_te, lr_proba))

        xgb_model = train_xgboost(X_tr, y_tr)
        xgb_proba = xgb_model.predict_proba(X_te)[:, 1]
        xgb_aucs.append(roc_auc_score(y_te, xgb_proba))

    return {
        "n_splits": n_splits,
        "logistic_regression_roc_auc_mean": round(float(np.mean(lr_aucs)), 4),
        "logistic_regression_roc_auc_std": round(float(np.std(lr_aucs)), 4),
        "logistic_regression_roc_auc_per_fold": [round(float(a), 4) for a in lr_aucs],
        "xgboost_roc_auc_mean": round(float(np.mean(xgb_aucs)), 4),
        "xgboost_roc_auc_std": round(float(np.std(xgb_aucs)), 4),
        "xgboost_roc_auc_per_fold": [round(float(a), 4) for a in xgb_aucs],
    }


def evaluate(model_name: str, y_true, y_pred, y_proba) -> dict:
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_proba)
    return {
        "model": model_name,
        "roc_auc": round(float(roc_auc), 4),
        "precision_label0": round(report["0"]["precision"], 4),
        "recall_label0": round(report["0"]["recall"], 4),
        "f1_label0": round(report["0"]["f1-score"], 4),
        "precision_label1": round(report["1"]["precision"], 4),
        "recall_label1": round(report["1"]["recall"], 4),
        "f1_label1": round(report["1"]["f1-score"], 4),
        "accuracy": round(report["accuracy"], 4),
        "support_label0": int(report["0"]["support"]),
        "support_label1": int(report["1"]["support"]),
    }


def shap_feature_importance(model: xgb.XGBClassifier, X: pd.DataFrame) -> pd.DataFrame:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    return pd.DataFrame({
        "feature": X.columns,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    X, y, feature_names = load_and_encode()
    X_train, X_test, y_train, y_test = stratified_split(X, y)
    print(f"Train: {len(X_train)} rows ({(y_train == 0).sum()} label=0, {(y_train == 1).sum()} label=1)")
    print(f"Test:  {len(X_test)} rows ({(y_test == 0).sum()} label=0, {(y_test == 1).sum()} label=1)")

    lr_model, scaler = train_logistic_regression(X_train, y_train)
    X_test_scaled = scaler.transform(X_test)
    lr_pred = lr_model.predict(X_test_scaled)
    lr_proba = lr_model.predict_proba(X_test_scaled)[:, 1]
    lr_eval = evaluate("logistic_regression", y_test, lr_pred, lr_proba)
    print(json.dumps(lr_eval, indent=2))

    xgb_model = train_xgboost(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
    xgb_eval = evaluate("xgboost", y_test, xgb_pred, xgb_proba)
    print(json.dumps(xgb_eval, indent=2))

    cv_results = cross_validate(X, y, n_splits=5)
    print(json.dumps(cv_results, indent=2))

    eval_report = {
        "single_split": {"logistic_regression": lr_eval, "xgboost": xgb_eval},
        "cross_validated": cv_results,
    }
    (REPORTS_DIR / "phase3_model_evaluation.json").write_text(json.dumps(eval_report, indent=2))

    shap_report = shap_feature_importance(xgb_model, X_test)
    shap_report.to_csv(REPORTS_DIR / "phase3_shap_feature_importance.csv", index=False)
    print(shap_report.to_string(index=False))

    xgb_model.save_model(MODELS_DIR / "xgboost_router.json")
    import joblib
    joblib.dump({"model": lr_model, "scaler": scaler, "feature_names": feature_names}, MODELS_DIR / "logistic_regression_router.pkl")
    print("Saved models/xgboost_router.json and models/logistic_regression_router.pkl")
