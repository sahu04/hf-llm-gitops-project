"""
Hugging Face LLM Inference API
- /predict   : run text generation on a small, CPU-friendly Hugging Face model
- /health    : liveness/readiness probe target for Kubernetes
- /metrics   : Prometheus scrape target (auto-added by the instrumentator)

Security:
- Requires an API key sent in the 'X-API-Key' header, read from the API_KEY env var
  (in Kubernetes this env var is populated from a Secret — see k8s/secret-example.yaml)
"""

import os
import logging
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from transformers import pipeline
from prometheus_fastapi_instrumentator import Instrumentator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hf-llm-api")

# ---- Config ----
MODEL_NAME = os.getenv("MODEL_NAME", "distilgpt2")  # small, CPU-friendly model
API_KEY = os.getenv("API_KEY", "")  # set via Kubernetes Secret in production
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "50"))

app = FastAPI(title="HF LLM Inference API", version="1.0.0")

# Prometheus metrics exposed automatically at GET /metrics
Instrumentator().instrument(app).expose(app)

# ---- Load model once at startup (not per-request) ----
logger.info(f"Loading model: {MODEL_NAME}")
generator = pipeline("text-generation", model=MODEL_NAME)
logger.info("Model loaded successfully")


class PredictRequest(BaseModel):
    prompt: str


class PredictResponse(BaseModel):
    prompt: str
    generated_text: str


def verify_api_key(x_api_key: str = Header(default="")):
    if not API_KEY:
        # If no API_KEY is configured, fail closed rather than open.
        raise HTTPException(status_code=500, detail="Server misconfigured: API_KEY not set")
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@app.get("/health")
def health():
    """Used by Kubernetes readiness/liveness probes."""
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest, x_api_key: str = Header(default="")):
    verify_api_key(x_api_key)

    if not request.prompt or len(request.prompt.strip()) == 0:
        raise HTTPException(status_code=400, detail="prompt must not be empty")

    if len(request.prompt) > 1000:
        raise HTTPException(status_code=400, detail="prompt too long (max 1000 chars)")

    result = generator(
        request.prompt,
        max_new_tokens=MAX_NEW_TOKENS,
        num_return_sequences=1,
        truncation=True,
    )
    generated = result[0]["generated_text"]

    return PredictResponse(prompt=request.prompt, generated_text=generated)
