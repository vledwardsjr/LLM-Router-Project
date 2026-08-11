import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_model import (  # noqa: E402
    CATEGORICAL_FEATURES, NUMERIC_FEATURES, RANDOM_STATE,
    load_and_encode, train_logistic_regression, train_xgboost,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"

N_THRESHOLD_STEPS = 100
# Average quality loss stays <= 0 across the whole threshold range on this dataset, so an
# average-based target is trivially satisfied everywhere. Worst single-query (max) loss
# is the metric that actually varies with the threshold, so it's what the operating point
# is chosen against instead.
MAX_QUALITY_LOSS_TARGET = 4.0  # no single routed-small query should lose more than 4 judge points


def generate_oof_probabilities(X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> pd.DataFrame:
    """Out-of-fold predicted probabilities for every row -- each row's prediction comes
    from a model that never saw it during training, so the full 900-row set can be used
    for the Pareto curve without overstating performance the way training-set
    probabilities would.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    lr_proba = np.zeros(len(X))
    xgb_proba = np.zeros(len(X))

    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr = y.iloc[train_idx]

        lr_model, scaler = train_logistic_regression(X_tr, y_tr)
        lr_proba[test_idx] = lr_model.predict_proba(scaler.transform(X_te))[:, 1]

        xgb_model = train_xgboost(X_tr, y_tr)
        xgb_proba[test_idx] = xgb_model.predict_proba(X_te)[:, 1]

    return pd.DataFrame({
        "lr_proba_small_ok": lr_proba,
        "xgb_proba_small_ok": xgb_proba,
    }, index=X.index)


def sweep_thresholds(proba: np.ndarray, quality_delta: np.ndarray, n_steps: int = N_THRESHOLD_STEPS) -> pd.DataFrame:
    """quality_delta = large_model_score - small_model_score, per query. At each threshold,
    route to small when proba >= threshold; report the fraction routed to small (cost
    savings) plus the average, p90, and max quality loss among the routed-small queries."""
    rows = []
    for t in np.linspace(0, 1, n_steps + 1):
        routed_small = proba >= t
        pct_routed_small = routed_small.mean()
        deltas = quality_delta[routed_small]
        avg_quality_loss = deltas.mean() if routed_small.any() else 0.0
        p90_quality_loss = np.percentile(deltas, 90) if routed_small.any() else 0.0
        max_quality_loss = deltas.max() if routed_small.any() else 0.0
        rows.append({
            "threshold": round(float(t), 4),
            "pct_routed_small": round(float(pct_routed_small), 4),
            "avg_quality_loss": round(float(avg_quality_loss), 4),
            "p90_quality_loss": round(float(p90_quality_loss), 4),
            "max_quality_loss": round(float(max_quality_loss), 4),
        })
    return pd.DataFrame(rows)


def recommend_threshold(pareto_df: pd.DataFrame, max_quality_loss_target: float = MAX_QUALITY_LOSS_TARGET) -> dict:
    """Pick the threshold that maximizes cost savings while keeping the worst single-query
    quality loss among routed-small queries at or below the target."""
    ok = pareto_df[pareto_df["max_quality_loss"] <= max_quality_loss_target]
    if ok.empty:
        return {"threshold": 1.0, "pct_routed_small": 0.0, "avg_quality_loss": 0.0, "max_quality_loss": 0.0, "note": "no threshold meets target; default to always-large"}
    best = ok.loc[ok["pct_routed_small"].idxmax()]
    return {
        "threshold": float(best["threshold"]),
        "pct_routed_small": float(best["pct_routed_small"]),
        "avg_quality_loss": float(best["avg_quality_loss"]),
        "p90_quality_loss": float(best["p90_quality_loss"]),
        "max_quality_loss": float(best["max_quality_loss"]),
        "max_quality_loss_target": max_quality_loss_target,
    }


def plot_pareto(pareto_lr: pd.DataFrame, pareto_xgb: pd.DataFrame, out_path: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.plot(pareto_lr["pct_routed_small"], pareto_lr["avg_quality_loss"], marker="o", markersize=3, label="Logistic Regression")
    ax1.plot(pareto_xgb["pct_routed_small"], pareto_xgb["avg_quality_loss"], marker="s", markersize=3, label="XGBoost")
    ax1.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax1.set_xlabel("% of queries routed to small model (cost savings)")
    ax1.set_ylabel("Avg quality loss on routed-small queries\n(large score - small score)")
    ax1.set_title("Average quality loss (stays <= 0 across the whole curve)")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(pareto_lr["pct_routed_small"], pareto_lr["max_quality_loss"], marker="o", markersize=3, label="Logistic Regression")
    ax2.plot(pareto_xgb["pct_routed_small"], pareto_xgb["max_quality_loss"], marker="s", markersize=3, label="XGBoost")
    ax2.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax2.set_xlabel("% of queries routed to small model (cost savings)")
    ax2.set_ylabel("Worst single-query quality loss on routed-small queries")
    ax2.set_title("Tail risk (max quality loss) -- the real decision curve")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.suptitle("Cost Savings vs. Quality Loss Pareto Frontier")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    REPORTS_DIR.mkdir(exist_ok=True)

    features_df = pd.read_parquet(DATA_DIR / "features.parquet")
    dataset_df = pd.read_parquet(DATA_DIR / "query_dataset.parquet")[["query_id", "small_model_score", "large_model_score"]]
    merged = features_df.merge(dataset_df, on="query_id")

    X, y, _ = load_and_encode()
    oof = generate_oof_probabilities(X, y)
    merged = pd.concat([merged.reset_index(drop=True), oof.reset_index(drop=True)], axis=1)
    quality_delta = (merged["large_model_score"] - merged["small_model_score"]).to_numpy()

    pareto_lr = sweep_thresholds(merged["lr_proba_small_ok"].to_numpy(), quality_delta)
    pareto_xgb = sweep_thresholds(merged["xgb_proba_small_ok"].to_numpy(), quality_delta)
    pareto_lr.to_csv(REPORTS_DIR / "phase4_pareto_logistic_regression.csv", index=False)
    pareto_xgb.to_csv(REPORTS_DIR / "phase4_pareto_xgboost.csv", index=False)

    plot_pareto(pareto_lr, pareto_xgb, REPORTS_DIR / "phase4_pareto_frontier.png")

    rec_lr = recommend_threshold(pareto_lr)
    rec_xgb = recommend_threshold(pareto_xgb)
    recommendation = {
        "max_quality_loss_target": MAX_QUALITY_LOSS_TARGET,
        "logistic_regression": rec_lr,
        "xgboost": rec_xgb,
        "always_large_baseline": {"pct_routed_small": 0.0, "avg_quality_loss": 0.0, "max_quality_loss": 0.0},
        "always_small_baseline": {
            "pct_routed_small": 1.0,
            "avg_quality_loss": round(float(quality_delta.mean()), 4),
            "max_quality_loss": round(float(quality_delta.max()), 4),
        },
        "note": (
            "avg_quality_loss stays <= 0 across the ENTIRE threshold range on this dataset "
            "-- even always-route-to-small doesn't hurt average judge-scored quality, and "
            "p90 is flat too (judge scores are integers 1-10, so percentiles snap to "
            "discrete values). This is a real finding tied to the labeling rule's generous "
            "score-gap threshold and judge scores clustering high. The recommendation above "
            "is therefore based on the worst single-query (max) quality loss instead, which "
            "moves meaningfully across the curve -- the router's real value here is cutting "
            "tail-risk, not improving the average."
        ),
    }
    (REPORTS_DIR / "phase4_threshold_recommendation.json").write_text(json.dumps(recommendation, indent=2))
    print(json.dumps(recommendation, indent=2))
