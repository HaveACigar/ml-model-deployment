from datetime import datetime, timezone
from typing import List

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

app = FastAPI(title="ML Model Deployment", version="1.0.0")
ARTS = joblib.load("models/artifacts.pkl")
PIPELINE = ARTS["pipeline"]
REFERENCE = ARTS["reference_data"]
FEATURES = ARTS["feature_names"]


class PredictionRequest(BaseModel):
    monthly_spend: float = Field(..., ge=0)
    tenure_months: int = Field(..., ge=0)
    support_tickets: int = Field(..., ge=0)
    logins_last_30d: int = Field(..., ge=0)
    discount_ratio: float = Field(..., ge=0, le=1)
    payment_failures: int = Field(..., ge=0)


class BatchPredictionRequest(BaseModel):
    rows: List[PredictionRequest]


recent_payloads: List[dict] = []


def frame_from_rows(rows):
    return pd.DataFrame(rows)[FEATURES]


def summarize_drift(current: pd.DataFrame):
    result = []
    for feature in FEATURES:
        ref_mean = float(REFERENCE[feature].mean())
        cur_mean = float(current[feature].mean())
        drift_pct = 0.0 if ref_mean == 0 else abs(cur_mean - ref_mean) / abs(ref_mean)
        result.append({
            "feature": feature,
            "reference_mean": round(ref_mean, 3),
            "current_mean": round(cur_mean, 3),
            "relative_shift": round(drift_pct, 3),
        })
    return result


@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/metrics")
def metrics():
    return {
        "model_roc_auc": ARTS["roc_auc"],
        "recent_prediction_requests": len(recent_payloads),
        "features": FEATURES,
    }


@app.get("/drift")
def drift():
    if not recent_payloads:
        return {"message": "No live payloads yet", "reference_rows": len(REFERENCE)}
    current = frame_from_rows(recent_payloads)
    return {"reference_rows": len(REFERENCE), "live_rows": len(current), "feature_drift": summarize_drift(current)}


@app.post("/predict")
def predict(payload: PredictionRequest):
    row = payload.model_dump()
    recent_payloads.append(row)
    df = frame_from_rows([row])
    prob = float(PIPELINE.predict_proba(df)[0, 1])
    label = int(prob >= 0.5)
    return {"prediction": label, "probability": round(prob, 4)}


@app.post("/batch-predict")
def batch_predict(payload: BatchPredictionRequest):
    rows = [row.model_dump() for row in payload.rows]
    recent_payloads.extend(rows)
    df = frame_from_rows(rows)
    probs = PIPELINE.predict_proba(df)[:, 1]
    return {"predictions": [{"prediction": int(prob >= 0.5), "probability": round(float(prob), 4)} for prob in probs]}


@app.get("/", response_class=HTMLResponse)
def index():
    return f"""
    <html>
      <head>
        <title>ML Model Deployment</title>
        <style>
          body {{ font-family: Arial, sans-serif; margin: 40px; background: #0f172a; color: #e2e8f0; }}
          code, pre {{ background: #111827; padding: 6px 8px; border-radius: 6px; color: #93c5fd; }}
          .card {{ background: #111827; border: 1px solid #334155; border-radius: 12px; padding: 20px; margin-bottom: 18px; }}
          a {{ color: #93c5fd; }}
        </style>
      </head>
      <body>
        <h1>ML Model Deployment (MLOps)</h1>
        <div class="card">
          <p>FastAPI service serving a binary classification model with health checks, live prediction endpoints, batch scoring, and lightweight drift summaries.</p>
          <p><strong>Reference ROC-AUC:</strong> {ARTS['roc_auc']:.3f}</p>
        </div>
        <div class="card">
          <h2>Endpoints</h2>
          <ul>
            <li><a href="/docs">/docs</a> - interactive OpenAPI docs</li>
            <li><a href="/health">/health</a> - health check</li>
            <li><a href="/metrics">/metrics</a> - basic service metrics</li>
            <li><a href="/drift">/drift</a> - live payload drift summary</li>
          </ul>
        </div>
        <div class="card">
          <h2>Sample JSON for /predict</h2>
          <pre>{{
  "monthly_spend": 149.0,
  "tenure_months": 8,
  "support_tickets": 3,
  "logins_last_30d": 11,
  "discount_ratio": 0.15,
  "payment_failures": 1
}}</pre>
        </div>
      </body>
    </html>
    """
