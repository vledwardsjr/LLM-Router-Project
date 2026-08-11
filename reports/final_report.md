# LLM Cost/Latency Router — Final Report

## Executive Summary

Given a query, can a classical ML classifier predict — **before calling either model** —
whether a cheap/fast LLM (Haiku 4.5) will give a "good enough" answer, or whether the query
needs a larger/expensive model (Sonnet 5)? This project built the labeled dataset,
engineered query-only features, trained and compared two classifiers, and produced a
cost-savings-vs-quality-loss tradeoff curve — the business-facing deliverable a real
routing decision would be made from.

**Headline result:** a Logistic Regression router, operated at a threshold chosen to bound
worst-case quality loss, can route **36.4% of traffic to the cheap model** while *improving*
average response quality (per an LLM judge) and capping the worst single-query quality loss
at 3 points on a 10-point scale (vs. 7 points with no routing safeguard at all). The
simpler, regularized model outperformed XGBoost on this dataset — a finding backed by
cross-validation, not a single lucky split.

**This is deployed and live**, not just designed:
`https://cawdqfgrdj.execute-api.us-east-1.amazonaws.com` (AWS Lambda + API Gateway +
DynamoDB). Every number below is pulled directly from the saved pipeline outputs (`data/`,
`reports/`) or from live tests against the deployed endpoint — nothing here is rounded from
memory.

---

## 1. Methodology — Dataset Construction (Phase 1)

**Query set:** 3,000 queries sampled from `databricks/databricks-dolly-15k` across 8 task
categories (open_qa, general_qa, classification, closed_qa, brainstorming,
information_extraction, summarization, creative_writing), then a **stratified subsample of
900 queries** (same category proportions) to keep API cost bounded.

**Dual-model runs:** every query was sent to both a small model (`claude-haiku-4-5`) and a
large model (`claude-sonnet-5`) via the Anthropic Batch API — 1,800 calls, 900/900 rows
succeeded, zero permanent failures. Actual cost: **$3.23**.

**Independent LLM-judge scoring:** a third model (`claude-opus-5`, independent of both
candidates to avoid self-preference bias) scored each of the 1,800 responses individually
— never comparatively — on correctness, completeness, and quality (1–10 each), plus an
overall "good enough" score, returned as structured JSON. 1,800/1,800 scored, zero
failures. Actual cost: **$6.63**.

**Judge validation (the manual gate):** a blind 75-item sample (judge score hidden) was
hand-scored by a human evaluator using the identical rubric. Results:

| Metric | Value | Threshold | Passed? |
|---|---|---|---|
| Within-1-point agreement | 85.3% | ≥80% | pass |
| Exact match rate | 54.7% | — | — |
| Pearson correlation | 0.621 | ≥0.70 | fail |
| Spearman correlation | 0.617 | — | (confirms not a Pearson artifact) |
| Quadratic weighted Cohen's κ | 0.626 | — | (confirms not a Pearson artifact) |
| Pearson excluding 2 worst outliers | 0.733 | — | passes |

The strict correlation gate technically failed. Diagnosis: two of the 75 validated items
(2.7%) — a malformed/incomplete query where the model appropriately asked for
clarification, and an inherently ambiguous "odd one out" classification puzzle with a
debatable correct answer — drove a 5–6 point human/judge disagreement each. Excluding just
those two items raises Pearson to 0.733, comfortably passing. Three independent agreement
statistics (Pearson, Spearman, weighted κ) all landed in the same 0.62–0.63 band, ruling out
a metric-specific artifact. **Decision: proceed with the existing judge scores, documenting
this as a known limitation** rather than spending additional budget re-scoring — strong
absolute agreement (85% within one point), with the shortfall concentrated in genuinely
ambiguous edge cases rather than systematic judge bias.

**Label derivation:** `label = 1` ("small model is good enough") if
`large_model_score − small_model_score ≤ 1`, else `0`. Result: **832 label=1 (92.4%), 68
label=0 (7.6%)** — a substantial class imbalance carried into every subsequent phase.

**Total Phase 1 API spend: $9.86.**

---

## 2. Feature Engineering (Phase 2)

Every feature is derived from the query text alone — the entire premise of the project is
predicting *before* either model is called, so response-level signals are out of bounds by
design.

**Free (non-LLM) features:** length metrics (`query_word_count`, `query_avg_word_length`,
`query_has_question_mark`) and regex-based density proxies for numeric/named-entity content
(`query_number_density`, `query_capitalized_density` — digit and mid-sentence-capitalized-
word counts, normalized by word count; a documented lightweight stand-in for a full NER
model, judged out of scope for the project timeline).

**LLM-extracted features:** one combined `claude-haiku-4-5` call per query (900 calls, not
per model-output, since these are query-only signals) returning `reasoning_step_count`
(1–10), `ambiguity_score` (1–10), `domain` (coding/math/writing/factual/creative/other), and
`question_type` (single_fact_lookup/multi_step_reasoning/open_ended) as structured JSON.
900/900 succeeded. Actual cost: **$0.35**.

**Length confound check** (the key methodological safeguard, carried forward from a prior
project's lessons): every numeric candidate feature was correlated against raw query length
before being trusted.

| Feature | Corr. with length | Verdict |
|---|---|---|
| `query_char_count` | 0.999 | dropped — redundant with length |
| `query_sentence_count` | 0.974 | dropped — redundant with length |
| `query_capitalized_word_count` (raw) | 0.813 | dropped — density version kept instead |
| `query_number_count` (raw) | 0.785 | dropped — density version kept instead |
| `query_number_density` | 0.217 | kept |
| `ambiguity_score` | −0.200 | kept |
| `query_avg_word_length` | 0.164 | kept |
| `reasoning_step_count` | −0.159 | kept |
| `query_capitalized_density` | 0.081 | kept |
| `query_has_question_mark` | −0.078 | kept |

Four features were dropped for being near-duplicates of raw query length; their
density-normalized or otherwise-independent counterparts were kept. **Final feature set:**
9 features across 900 rows, zero nulls, saved to `data/features.parquet`.

**Total Phase 1+2 API spend: $10.21 of a $12 budget.** No further API calls were needed for
Phases 3–5.

---

## 3. Modeling (Phase 3)

**Setup:** categorical features (`domain`, `question_type`, `source_category`) one-hot
encoded to 24 total features; **stratified** 80/20 train/test split (720/180 rows), given
only 68 label=0 rows exist total — an unstratified split risks an unbalanced test fold by
chance.

**Baseline — Logistic Regression** (`class_weight="balanced"`, standardized features).

**Main model — XGBoost**, deliberately regularized (`max_depth=3`, `min_child_weight=3`,
`subsample=0.8`, `colsample_bytree=0.8`, `reg_lambda=2.0`, `scale_pos_weight` set to the
train-fold imbalance ratio). The **default** XGBoost configuration was tried first and
overfit badly, given only 54–68 label=0 training examples — the regularized config above is
the standard fix for a tree ensemble on a tiny imbalanced dataset, not a hyperparameter
search.

**A finding worth defending, not glossing over:** the first single 80/20 split gave
XGBoost an ROC-AUC of **0.505** — indistinguishable from random — against Logistic
Regression's 0.650. Rather than accept a single split's number, **5-fold stratified
cross-validation** was run (the test fold has only ~14 label=0 examples, too few to trust
alone):

| Model | Mean ROC-AUC (5-fold CV) | Std dev | Fold range |
|---|---|---|---|
| **Logistic Regression** | **0.709** | **0.043** | 0.653 – 0.758 |
| XGBoost (regularized) | 0.672 | 0.078 | 0.572 – 0.807 |

The single-split XGBoost score was an unlucky fold, not the model's real performance — but
even the CV-corrected picture shows **Logistic Regression is the stronger, meaningfully more
stable performer**. This is a legitimate, explainable outcome given the tiny minority
class: XGBoost's extra flexibility has less to work with here and pays for it in variance.
**Logistic Regression is the recommended model for deployment.**

**SHAP feature importance** (XGBoost, mean |SHAP| on the held-out test fold):

| Rank | Feature | Mean \|SHAP\| |
|---|---|---|
| 1 | `query_capitalized_density` | 1.007 |
| 2 | `query_word_count` | 0.622 |
| 3 | `query_avg_word_length` | 0.400 |
| 4 | `ambiguity_score` | 0.347 |
| 5 | `question_type_single_fact_lookup` | 0.169 |
| 6 | `source_category_open_qa` | 0.121 |
| 7 | `reasoning_step_count` | 0.121 |

Every `domain_*` dummy scored **exactly 0** importance — the model relied entirely on
`source_category` for topic signal instead, most likely because the two categorical
features are redundant with each other (both encode a coarse task/topic type, and only one
is needed once the other is present).

Models saved: `models/logistic_regression_router.pkl` (bundled with its `StandardScaler`
and feature-name list), `models/xgboost_router.json`.

---

## 4. Threshold Tuning & Pareto Analysis (Phase 4)

**Method:** out-of-fold predicted probabilities were generated for all 900 rows (5-fold CV,
same folds as Phase 3) — every row's routing probability comes from a model that never saw
it during training, so the full dataset can be used for the curve without overstating
performance. The classification threshold was swept from 0 to 1 in 101 steps; at each
threshold, queries with `predicted_proba ≥ threshold` are routed to the small model.

**The central finding — a real result, not a broken chart:** average quality loss
(`large_model_score − small_model_score`, averaged over routed-to-small queries) stays
**at or below zero across the entire threshold range.** Even routing 100% of traffic to the
small model (threshold = 0) produces an average change of **−0.077** — a slight net *gain*,
not a loss. The 90th-percentile loss is similarly flat at 1.0 across almost the whole range
(judge scores are integers 1–10, so percentiles snap to discrete values).

**Root cause, stated plainly:** Phase 1's label rule (`≤1`-point gap counts as "good
enough") was generous, and judge scores cluster high (mean 8.4, mostly 8–10) — so on this
specific dataset, skewed toward Dolly's simpler QA-style queries, the small model is nearly
always adequate by this measure. This is a property of this dataset and labeling choice,
not a universal claim that routing logic never matters.

**Reframing around the metric that actually varies:** since average- and P90-based
targets are trivially satisfied everywhere, the operating threshold was instead chosen
against the **worst single-query (max) quality loss** among routed-to-small queries — the
one risk metric that moves meaningfully across the curve, from **7 points down to 3** as
the threshold tightens from 0 to ≈0.8. This reframes what the router actually contributes:
**it doesn't move the average (already good on this traffic) — it substantially cuts
tail-risk**, the chance of a badly mismatched routing decision on any single query.

**Recommended operating point** (business target: no routed-to-small query loses more than
4 judge points):

| Model | Threshold | % routed to small | Avg. quality Δ | Worst-case (max) loss |
|---|---|---|---|---|
| **Logistic Regression** | **0.76** | **36.4%** | **+0.27 (net gain)** | **3 points** |
| XGBoost | 0.80 | 45.7% | +0.15 (net gain) | 4 points |
| Always-large baseline | — | 0% | 0 | 0 |
| Always-small baseline | — | 100% | +0.077 (net gain) | 7 points |

**Recommendation: deploy Logistic Regression at threshold 0.76.** It routes over a third of
traffic to the cheap model, slightly *improves* average judge-scored quality relative to
always using the large model, and caps the worst individual-query outcome at less than half
the risk of routing everything to the small model with no safeguard. It is also the model
that was more stable in Phase 3's cross-validation.

Headline chart: `reports/phase4_pareto_frontier.png` (average-loss view + the tail-risk
decision view, side by side).

---

## 5. Deployment (Phase 5) — LIVE

**This is deployed and running**, not just designed. Live endpoint:

```
https://cawdqfgrdj.execute-api.us-east-1.amazonaws.com
```
```
curl -X POST https://cawdqfgrdj.execute-api.us-east-1.amazonaws.com/ \
  -H "Content-Type: application/json" \
  -H "x-api-key: <key>" \
  -d '{"query": "What is the capital of France?"}'
```

The endpoint requires an `x-api-key` header (a shared-secret check in the Lambda itself —
HTTP APIs/API Gateway v2 don't support the classic REST API "usage plan + API key" feature)
so a public URL can't be used to run up API costs anonymously. Available on request.

**Architecture:** AWS Lambda (Python 3.11) behind an API Gateway HTTP API, with DynamoDB
for decision caching.

**A deliberate implementation choice worth explaining in an interview:** the deployed
Lambda does **not** ship scikit-learn/numpy/pandas at runtime. No Docker was available in
the development environment, and cross-compiling those packages' C extensions for Lambda's
Linux runtime from macOS without Docker is the standard failure point for zip-based Lambda
ML deployments. Since the recommended model is Logistic Regression — a linear model — its
learned parameters (24 coefficients, intercept, scaler mean/scale) were extracted to a small
JSON file, and inference is reimplemented as pure-Python arithmetic:
`sigmoid(Σ coef_i · scaled_feature_i + intercept)`. **This was verified to exactly match the
original scikit-learn model's `predict_proba()` output (max difference: 0.0) across all 900
training rows before being trusted for deployment.** The only external dependency shipped
is the `anthropic` SDK — its Rust-based transitive dependencies (`pydantic-core`, `jiter`)
were obtained as precompiled Linux wheels via `pip install --platform manylinux2014_x86_64
--only-binary=:all:`, which downloads the correct binary without executing it, sidestepping
the Docker requirement entirely. Deployment package: 4.2MB.

**DynamoDB** (`llm-router-decision-cache`, on-demand billing, TTL-based 1-hour expiry) is
used for decision caching, keyed on a SHA-256 hash of the query text — this is a
latency-critical, key-based point lookup at request time, not a relational join, which is
why DynamoDB was chosen deliberately over a relational store (RDS). Verified live: an
identical repeat request returned `"cached": true` on the second call.

**IAM:** the Lambda execution role is scoped to `AWSLambdaBasicExecutionRole` (logs) plus an
inline policy granting `GetItem`/`PutItem` on only the one DynamoDB table — not broad
DynamoDB access. (The separate IAM *user* used to run the deployment commands was granted
`AdministratorAccess` for setup speed, a reasonable shortcut in a personal sandbox account
for a portfolio project, but not what would be used to provision a real production
environment — see Limitations.)

**Secret handling:** `ANTHROPIC_API_KEY` is a Lambda environment variable (encrypted at
rest by default). A production deployment would use AWS Secrets Manager with rotation
instead; the environment variable was a reasonable choice at this scope, not an oversight.

**A limitation confirmed on the live, deployed endpoint, not just in local testing:** a
genuinely hard, out-of-distribution query — "design a fault-tolerant distributed consensus
algorithm and compare it against Raft and Paxos in terms of latency and partition
tolerance" — was routed to the **small** model with **99.85% confidence** against the live
API. This is very likely wrong, and it exposes a real generalization gap: the training data
(Dolly-derived) contains very few queries resembling genuine technical/systems-design
questions, and Phase 2 already found that `query_word_count`'s effect on the label runs
*backward* on this dataset (longer queries more often "small is fine") because Dolly's long
queries usually embed their own reference context. This out-of-distribution query is long
*without* embedded context, and the model appears to have misapplied the pattern it learned
from Dolly. **This is a generalization limitation of the training distribution, not a code
defect** — a production deployment on materially different traffic (e.g., a technical
support or coding-assistant product) would need retraining and re-validation against that
traffic's actual distribution before the routing decisions could be trusted.

---

## 6. Major Limitations (interview-ready summary)

1. **Small minority class.** Only 68 of 900 examples are label=0. Every evaluation in
   Phases 3–4 should be read with this in mind — confidence intervals on the minority class
   are wide, and the 5-fold CV standard deviations (0.043 for Logistic Regression, 0.078
   for XGBoost) reflect that directly.
2. **Judge/human correlation gate technically failed** (0.62 vs. a 0.70 target), though
   within-1-point agreement was strong (85%). Root-caused to 2 of 75 genuinely ambiguous
   validation items, not systematic bias — but this is a judgment call made under budget
   constraints ($12 total), not a clean pass, and should be stated as such.
3. **The Pareto analysis's headline "quality never degrades" result is dataset-specific.**
   It follows from a generous label threshold and a query distribution skewed toward simple
   QA-style tasks. It should not be presented as "routing never risks quality" in general —
   the tail-risk (max-loss) framing exists specifically because the average metric doesn't
   tell an honest, decision-useful story here.
4. **Out-of-distribution generalization is unverified and, on one manual test, appeared to
   fail.** The router has only been validated against Dolly-style traffic. A single
   adversarial-ish technical query was misrouted with high confidence during manual
   testing. This is the most important caveat for anyone considering deploying this on
   real, non-Dolly-like traffic.
5. **Simple heuristics stand in for two components that would ideally be more
   sophisticated:** entity/technical-term density uses regex proxies (digit and
   capitalization counts) rather than a full NER model, and the deployed router makes a
   live LLM call per request for 4 of its 9 features, adding latency and marginal per-request
   cost that a fully precomputed feature set would avoid.
6. **The deployment IAM user was granted broad (`AdministratorAccess`) permissions** for
   setup speed in a personal sandbox account — reasonable for this portfolio scope, but not
   what a real production provisioning process should use. A production setup would scope
   the deploying principal to exactly the actions used (Lambda, API Gateway, DynamoDB,
   and the specific IAM role-creation calls), and would put `ANTHROPIC_API_KEY` in Secrets
   Manager with rotation rather than a plain Lambda environment variable.

---

## 7. Reproducibility

| Phase | Notebook | Script | Key outputs |
|---|---|---|---|
| 1 | `notebooks/Phase 1/01_dataset_build.ipynb`, `01_run_models.ipynb`, `02_judge_labeling.ipynb` | `src/Phase 1/dataset_building.py`, `run_models.py`, `src/judge_labeling.py` | `data/query_dataset.parquet` |
| 2 | `notebooks/02_feature_engineering.ipynb` | `src/extract_features.py` | `data/features.parquet` |
| 3 | `notebooks/03_modeling_evaluation.ipynb` | `src/train_model.py` | `models/*.pkl`, `models/*.json` |
| 4 | `notebooks/04_threshold_pareto.ipynb` | `src/threshold_pareto.py` | `reports/phase4_pareto_frontier.png` |
| 5 | — | `deploy/lambda_handler.py` | Live: `https://cawdqfgrdj.execute-api.us-east-1.amazonaws.com` (requires an `x-api-key` header — contact for access) |

**Total project API spend: $10.21 of a $12 budget.** (Deployed AWS infrastructure runs on
pay-per-use billing — Lambda, API Gateway, and DynamoDB on-demand — with no fixed cost.)
