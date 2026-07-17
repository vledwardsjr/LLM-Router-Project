# LLM Cost/Latency Router

Predicts whether a cheap/fast LLM is "good enough" for a given query, or whether it needs
to be routed to a larger, more expensive model — **without calling either model first.**

## The Problem

Evaluating LLM outputs professionally means constantly making one judgment call: *is this
response good enough?* Watching that question play out at scale surfaces an expensive
pattern — most teams answer it by defaulting every query to the largest, most capable
model available. That's a safe answer, but it isn't a cheap one, and it isn't a
*designed* one. There's no principled, upfront decision about which queries actually need
the expensive model and which ones a cheap model would have handled just as well.

That's the gap this project targets: a way to make the cost-vs-quality tradeoff a
deliberate engineering decision instead of a default.

## The Approach

Instead of building another LLM, this builds a classical ML classifier that looks at a
query *before* either model is called and predicts whether the small/cheap model will be
sufficient. If a routing decision requires calling the expensive model to check whether
you needed the expensive model, it isn't saving anything — so every feature used here has
to be derivable from the query alone.

**Baselines it has to beat:**
- Always route to the large model → 100% quality, 0% cost savings.
- Always route to the small model → maximum savings, unknown quality loss.

The deliverable is a cost-savings-vs-quality-loss Pareto frontier — the business-facing
answer to "how much can we save without hurting quality" — plus a real-time endpoint that
acts on it.

## Status

🚧 **Day 1 — Phase 1 (dataset construction)** in progress. Nothing shipped yet; this
README tracks the live plan as it's built out.

## Roadmap by Phase

1. **Dataset construction** — Run a query set through a small (cheap/fast) and large
   (expensive/slow) LLM, score both outputs with an LLM-judge, and derive a "small model
   sufficient" label. The judge itself gets validated first: a manual sample (~50-100
   queries) is hand-scored and checked for agreement before trusting it on the full set.
2. **Feature engineering** — Extract query-level signals only: length, estimated
   reasoning-step count, entity density, domain/task type, ambiguity, question type. No
   response-level features — the model has to decide before either LLM runs. Every new
   feature gets checked against raw query length before being trusted, so it's adding real
   signal and not just re-encoding word count.
3. **Modeling** — Logistic Regression baseline, then XGBoost as the main model. Evaluated
   on precision/recall/ROC-AUC, with feature importance via SHAP — but accuracy alone
   isn't the bar; Phase 4's tradeoff curve is.
4. **Threshold tuning & Pareto analysis** — Sweep the classification threshold and plot
   cost savings (% routed to the cheap model) against quality loss on those queries. This
   chart is the headline deliverable for stakeholders.
5. **Deployment & report** — Wrap the trained model in a Lambda function behind API
   Gateway, with DynamoDB for low-latency decision caching (a key-based lookup at request
   time, not a relational join — deliberately not RDS). Final report covers dataset
   methodology, judge validation results, model comparison, the Pareto curve, and the
   chosen threshold with business justification.

## Tech Stack & Tools by Phase

| Phase | Focus | Tools |
|---|---|---|
| 1 — Dataset construction | Dual-model runs, LLM-judge scoring, manual validation sample | Python, pandas, pyarrow, LLM API (small/cheap + large/expensive model pairing), LLM-as-judge |
| 2 — Feature engineering | Query-level feature extraction | pandas, NumPy, LLM-assisted extraction (reasoning-step count, ambiguity rating) |
| 3 — Modeling | Baseline + main classifier, feature importance | scikit-learn (Logistic Regression), XGBoost, SHAP |
| 4 — Threshold tuning & Pareto analysis | Cost-savings vs. quality-loss frontier | pandas, NumPy, matplotlib |
| 5 — Deployment & reporting | Real-time routing endpoint, stakeholder report | AWS Lambda, API Gateway, DynamoDB, boto3, S3 |

**Environment:** Python 3.11, dedicated conda env, JupyterLab for notebooks.

## Project Structure

```
LLM-Router-Project/
├── data/                              # datasets (gitignored — populated in Phase 1)
├── models/                            # trained model artifacts (gitignored)
├── notebooks/
│   ├── 01_dataset_build.ipynb         # build labeled query dataset
│   ├── 02_feature_engineering.ipynb   # query-level feature extraction
│   └── 03_modeling_evaluation.ipynb   # model training, tuning, Pareto analysis
├── reports/                           # final writeup, Pareto curve, findings (Phase 4-5)
├── src/
│   ├── judge_labeling.py              # LLM-judge scoring + label derivation
│   ├── extract_features.py            # feature engineering functions
│   ├── train_model.py                 # model training/evaluation
│   └── routing_api.py                 # Lambda handler for deployed routing endpoint
├── requirements.txt                   # pinned dependencies
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Engineering Principles

Carried forward from lessons on a prior evaluation project, applied here from day one
rather than discovered the hard way twice:

1. **Length confound check** — every candidate feature gets tested against raw query
   length before it's trusted as signal.
2. **Judge validation before trust** — LLM-generated labels are checked against a
   manually-scored sample for agreement before being used at scale.
3. **Business framing over raw accuracy** — the deliverable stakeholders see is the
   cost/quality Pareto curve, not a bare accuracy number.
4. **Cloud tools chosen for fit, not familiarity** — DynamoDB for latency-critical
   key lookups, S3 for logs, Lambda for the endpoint. Each choice is justified in the
   final report rather than defaulted to.
