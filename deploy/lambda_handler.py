"""Lambda handler for the LLM cost/latency router.

Pure-Python Logistic Regression inference (no scikit-learn/numpy/pandas at runtime --
hard to cross-compile for Lambda's Linux runtime without Docker, and unnecessary for a
dozen-line sigmoid computation). Verified to match the trained sklearn model's
predict_proba() exactly -- see reports/final_report.md.

Only external dependency is `anthropic`, used to extract the 4 LLM-derived features that
can't be precomputed for a query nobody has seen yet. boto3 ships with the Lambda runtime.
"""

import hashlib
import json
import math
import os
import re
import time

import anthropic
import boto3

MODEL_PARAMS = json.load(open(os.path.join(os.path.dirname(__file__), "model_params.json")))
FEATURE_MODEL = "claude-haiku-4-5"
FEATURE_MAX_TOKENS = 150
CACHE_TABLE_NAME = os.environ.get("CACHE_TABLE_NAME", "llm-router-decision-cache")
CACHE_TTL_SECONDS = 3600

FEATURE_PROMPT_TEMPLATE = """Analyze this query and classify it along four dimensions. Do not answer the query itself.

Query:
{query}

1. reasoning_step_count: estimate how many distinct reasoning/lookup steps are needed to answer this well (integer, 1-10).
2. ambiguity_score: how underspecified/ambiguous is this query on its own, with no other context (integer 1-10; 1 = fully unambiguous, 10 = severely underspecified).
3. domain: the primary subject domain -- one of: coding, math, writing, factual, creative, other.
4. question_type: the shape of the task -- one of: single_fact_lookup, multi_step_reasoning, open_ended."""

FEATURE_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning_step_count": {"type": "integer"},
        "ambiguity_score": {"type": "integer"},
        "domain": {"type": "string", "enum": ["coding", "math", "writing", "factual", "creative", "other"]},
        "question_type": {"type": "string", "enum": ["single_fact_lookup", "multi_step_reasoning", "open_ended"]},
    },
    "required": ["reasoning_step_count", "ambiguity_score", "domain", "question_type"],
    "additionalProperties": False,
}

_DIGIT_RE = re.compile(r"\d+")
_CAPITALIZED_WORD_RE = re.compile(r"(?<!^)(?<!\. )(?<!\.\n)\b[A-Z][a-z]+")

# Reused across warm Lambda invocations.
_anthropic_client = None
_dynamodb_table = None


def _get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = anthropic.Anthropic()
    return _anthropic_client


def _get_cache_table():
    global _dynamodb_table
    if _dynamodb_table is None:
        _dynamodb_table = boto3.resource("dynamodb").Table(CACHE_TABLE_NAME)
    return _dynamodb_table


def _query_hash(query: str) -> str:
    return hashlib.sha256(query.encode()).hexdigest()


def _extract_free_features(query: str) -> dict:
    words = query.split()
    word_count = max(1, len(words))
    return {
        "query_word_count": len(words),
        "query_avg_word_length": (sum(len(w) for w in words) / word_count) if words else 0.0,
        "query_has_question_mark": int("?" in query),
        "query_number_density": len(_DIGIT_RE.findall(query)) / word_count,
        "query_capitalized_density": len(_CAPITALIZED_WORD_RE.findall(query)) / word_count,
    }


def _extract_llm_features(query: str) -> dict:
    client = _get_anthropic_client()
    prompt = FEATURE_PROMPT_TEMPLATE.format(query=query)
    response = client.messages.create(
        model=FEATURE_MODEL,
        max_tokens=FEATURE_MAX_TOKENS,
        thinking={"type": "disabled"},
        messages=[{"role": "user", "content": prompt}],
        output_config={"format": {"type": "json_schema", "schema": FEATURE_SCHEMA}},
    )
    text = next((b.text for b in response.content if b.type == "text"), None)
    if text is None:
        raise RuntimeError(f"LLM feature extraction returned no text (stop_reason={response.stop_reason})")
    return json.loads(text)


def _predict_proba(feature_dict: dict) -> float:
    """Pure-Python Logistic Regression: sigmoid(sum(coef_i * scaled_feature_i) + intercept).
    Verified to match the trained sklearn model's predict_proba() exactly before deployment.
    """
    z = MODEL_PARAMS["intercept"]
    for i, name in enumerate(MODEL_PARAMS["feature_names"]):
        raw = feature_dict.get(name, 0)
        scaled = (raw - MODEL_PARAMS["scaler_mean"][i]) / MODEL_PARAMS["scaler_scale"][i]
        z += MODEL_PARAMS["coef"][i] * scaled
    return 1 / (1 + math.exp(-z))


def route_query(query: str) -> dict:
    query_id = _query_hash(query)
    table = _get_cache_table()

    cached = table.get_item(Key={"query_hash": query_id}).get("Item")
    if cached and int(cached.get("expires_at", 0)) > time.time():
        return {
            "route_to": cached["route_to"],
            "probability_small_ok": float(cached["probability_small_ok"]),
            "threshold": MODEL_PARAMS["threshold"],
            "cached": True,
        }

    free_feats = _extract_free_features(query)
    llm_feats = _extract_llm_features(query)
    feature_dict = {
        **free_feats,
        "reasoning_step_count": llm_feats["reasoning_step_count"],
        "ambiguity_score": llm_feats["ambiguity_score"],
        f"domain_{llm_feats['domain']}": 1,
        f"question_type_{llm_feats['question_type']}": 1,
    }

    proba_small_ok = _predict_proba(feature_dict)
    route_to = "small" if proba_small_ok >= MODEL_PARAMS["threshold"] else "large"

    table.put_item(Item={
        "query_hash": query_id,
        "route_to": route_to,
        "probability_small_ok": str(round(proba_small_ok, 4)),
        "expires_at": int(time.time()) + CACHE_TTL_SECONDS,
    })

    return {
        "route_to": route_to,
        "probability_small_ok": round(proba_small_ok, 4),
        "threshold": MODEL_PARAMS["threshold"],
        "cached": False,
    }


def handler(event: dict, _context) -> dict:
    """API Gateway (HTTP API, payload format 2.0) Lambda entry point.

    Requires an `x-api-key` header matching ROUTER_API_KEY. HTTP APIs (API Gateway v2)
    don't support the classic REST API "usage plan + API key" feature, so this is a
    shared-secret header check instead -- same practical effect (unauthenticated requests
    can't run up API cost against a public endpoint), less infrastructure.
    """
    try:
        headers = {k.lower(): v for k, v in (event.get("headers") or {}).items()}
        expected_key = os.environ.get("ROUTER_API_KEY")
        if expected_key and headers.get("x-api-key") != expected_key:
            return {"statusCode": 401, "body": json.dumps({"error": "invalid or missing x-api-key header"})}

        body = json.loads(event.get("body") or "{}")
        query = body.get("query")
        if not query:
            return {"statusCode": 400, "body": json.dumps({"error": "missing 'query' field"})}

        result = route_query(query)
        return {"statusCode": 200, "headers": {"Content-Type": "application/json"}, "body": json.dumps(result)}
    except Exception as e:  # noqa: BLE001 -- Lambda boundary: never let this crash uncaught
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}
