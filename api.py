"""
api.py — Wrapper Mínimo de Inferencia con FastAPI
==================================================
Este wrapper permite exponer el pipeline para desarrollo y pruebas de integración.

En producción en un robot, el nodo de ROS 2 llama a InferenceEngine
directamente (sin overhead de HTTP). Este wrapper existe para desarrollo,
testing y para demostrar el pipeline a stakeholders.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from functools import partial
import asyncio
import threading
import time
import logging
from pathlib import Path
from inference_engine import InferenceEngine, BenchmarkResult

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

MODEL_DIR = Path("models")
MODELS = {
    "fp32": MODEL_DIR / "mobilenet_v2_fp32.onnx",
    "int8": MODEL_DIR / "mobilenet_v2_int8.onnx",
}

_engines: dict[str, InferenceEngine] = {}
_engines_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-calienta el motor FP32 al arrancar si está disponible."""
    path = MODELS["fp32"]
    if path.exists():
        log.info("Pre-calentando motor fp32 …")
        _engines["fp32"] = InferenceEngine(str(path), provider="cpu")
        _engines["fp32"].predict()
        log.info("Motor listo.")
    else:
        log.warning("Modelo FP32 no encontrado. Ejecutar export_model.py primero.")
    yield
    _engines.clear()


app = FastAPI(
    title="API de Inferencia para Robótica Edge",
    description="Wrapper mínimo de inferencia para el pipeline ONNX. La lógica principal está en inference_engine.py.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(RuntimeError)
async def runtime_error_handler(request, exc):
    log.error(f"RuntimeError: {exc}")
    return JSONResponse(status_code=500, content={"detail": str(exc)})


# ── Esquemas ───────────────────────────────────────────────────────────────────

class PredictionRequest(BaseModel):
    model: str = Field("fp32", pattern="^(fp32|int8)$",
                       description="Variante de precisión del modelo")
    top_k: int = Field(5, ge=1, le=1000,
                       description="Clases top-k a devolver (máx. 1000 para ImageNet)")


class ClassPrediction(BaseModel):
    class_id:   int
    confidence: float


class PredictionResponse(BaseModel):
    model:       str
    precision:   str
    provider:    str
    latency_ms:  float
    predictions: list[ClassPrediction]


class BenchmarkRequest(BaseModel):
    model:    str = Field("fp32", pattern="^(fp32|int8)$")
    n_warmup: int = Field(10, ge=1, le=100,
                          description="Ejecuciones de warmup")
    n_runs:   int = Field(100, ge=10, le=2000,
                          description="Ejecuciones medidas")


class BenchmarkResponse(BaseModel):
    model_path:      str
    precision:       str
    provider:        str
    n_runs:          int
    mean_latency_ms: float
    p50_latency_ms:  float
    p95_latency_ms:  float
    p99_latency_ms:  float
    min_latency_ms:  float
    max_latency_ms:  float
    throughput_fps:  float


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_engine(model_key: str) -> InferenceEngine:
    """
    Carga el motor de forma lazy en la primera petición; lo cachea para las siguientes.
    Thread-safe mediante double-checked locking para evitar creación duplicada cuando
    llegan peticiones concurrentes antes de que el motor esté cacheado.
    """
    if model_key in _engines:
        return _engines[model_key]
    with _engines_lock:
        if model_key in _engines:
            return _engines[model_key]
        path = MODELS.get(model_key)
        if path is None or not path.exists():
            raise HTTPException(
                status_code=503,
                detail=f"Modelo '{model_key}' no encontrado en {path}. Ejecutar export_model.py primero."
            )
        _engines[model_key] = InferenceEngine(str(path), provider="cpu")
    return _engines[model_key]


# ── Rutas ──────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Sistema"])
async def health():
    """Sonda de disponibilidad compatible con Kubernetes y watchdogs de ROS 2."""
    return {
        "estado":            "ok",
        "motores_cargados":  list(_engines.keys()),
        "modelos_disponibles": {k: {"ruta": str(v), "existe": v.exists()} for k, v in MODELS.items()},
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Inferencia"])
async def predict(req: PredictionRequest):
    """Ejecuta un forward pass sobre un frame simulado."""
    engine = _get_engine(req.model)
    t0 = time.perf_counter()
    top_k = engine.predict_top_k(frame=None, k=req.top_k)
    latency_ms = (time.perf_counter() - t0) * 1_000
    return PredictionResponse(
        model       = req.model,
        precision   = "INT8" if req.model == "int8" else "FP32",
        provider    = engine.session.get_providers()[0],
        latency_ms  = round(latency_ms, 3),
        predictions = [ClassPrediction(**p) for p in top_k],
    )


@app.post("/benchmark", response_model=BenchmarkResponse, tags=["Profiling"])
async def benchmark(req: BenchmarkRequest):
    """
    Ejecuta el benchmark completo delegado al thread pool — evita bloquear
    el event loop asíncrono durante las ejecuciones CPU-intensivas de inferencia.
    """
    engine    = _get_engine(req.model)
    precision = "INT8" if req.model == "int8" else "FP32"
    loop      = asyncio.get_running_loop()
    result: BenchmarkResult = await loop.run_in_executor(
        None,
        partial(engine.benchmark, n_warmup=req.n_warmup, n_runs=req.n_runs, precision=precision),
    )
    return BenchmarkResponse(
        model_path      = result.model_path,
        precision       = result.precision,
        provider        = result.provider,
        n_runs          = result.n_runs,
        mean_latency_ms = result.mean_latency_ms,
        p50_latency_ms  = result.p50_latency_ms,
        p95_latency_ms  = result.p95_latency_ms,
        p99_latency_ms  = result.p99_latency_ms,
        min_latency_ms  = result.min_latency_ms,
        max_latency_ms  = result.max_latency_ms,
        throughput_fps  = result.throughput_fps,
    )
