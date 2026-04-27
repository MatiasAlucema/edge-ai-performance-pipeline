"""
run_pipeline.py — Orquestador del Pipeline Completo
====================================================
Ejecuta todos los pasos en el orden correcto con comprobación de dependencias.
Seguro para ejecutar varias veces — omite los pasos cuya salida ya existe.

Uso:
    python run_pipeline.py              # ejecutar todos los pasos
    python run_pipeline.py --force      # re-ejecutar aunque las salidas ya existan
    python run_pipeline.py --skip-trt   # omitir TensorRT (por defecto en PC sin GPU NVIDIA)

Pasos:
    1. Export    : PyTorch MobileNetV2 → ONNX FP32
    2. Pruning   : Mostrar estadísticas de pruning (estructurado vs no estructurado)
    3. Cuantizar : ONNX FP32 → INT8 con formato QDQ
    4. Validar   : Tabla comparativa precisión vs eficiencia
    5. Benchmark : Comparativa de latencia / FPS entre FP32 e INT8
    6. TRT       : Compilar engine TensorRT (solo Jetson, omitido si no disponible)
    7. API       : Instrucciones para arrancar el servidor FastAPI
"""

import sys
import time
import logging
import argparse
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

MODEL_DIR = Path("models")
FP32_PATH = MODEL_DIR / "mobilenet_v2_fp32.onnx"
INT8_PATH = MODEL_DIR / "mobilenet_v2_int8.onnx"


def header(title: str) -> None:
    log.info("")
    log.info("═" * 56)
    log.info(f"  {title}")
    log.info("═" * 56)


def step_export(force: bool) -> bool:
    header("Paso 1 — Export PyTorch → ONNX (FP32)")
    if FP32_PATH.exists() and not force:
        log.info(f"  Omitiendo — {FP32_PATH} ya existe. Usar --force para re-ejecutar.")
        return True
    try:
        from export_model import load_pytorch_model, export_to_onnx
        MODEL_DIR.mkdir(exist_ok=True)
        model = load_pytorch_model("mobilenet_v2")
        export_to_onnx(model, FP32_PATH)
        return True
    except Exception as e:
        log.error(f"Export fallido: {e}")
        return False


def step_prune() -> bool:
    header("Paso 2 — Pruning (estadísticas)")
    try:
        from pruning import load_model, apply_unstructured_pruning, apply_structured_pruning
        from pruning import make_pruning_permanent, compare_model_sizes
        original = load_model()
        unstructured = apply_unstructured_pruning(original, amount=0.3)
        unstructured = make_pruning_permanent(unstructured)
        compare_model_sizes(original, unstructured, technique="No estructurado — 30%")
        structured = apply_structured_pruning(original, amount=0.25)
        structured = make_pruning_permanent(structured)
        compare_model_sizes(original, structured, technique="Estructurado — 25% canales")
        log.info("  Para exportar un modelo podado: ver docstring de pruning.py.")
        return True
    except Exception as e:
        log.error(f"Paso de pruning fallido: {e}")
        return False


def step_quantize(force: bool) -> bool:
    header("Paso 3 — Cuantización FP32 → INT8 (formato QDQ)")
    if INT8_PATH.exists() and not force:
        log.info(f"  Omitiendo — {INT8_PATH} ya existe. Usar --force para re-ejecutar.")
        return True
    if not FP32_PATH.exists():
        log.error(f"  Modelo FP32 no encontrado: {FP32_PATH}. Ejecutar el paso 1 primero.")
        return False
    try:
        from quantize_model import quantize_model
        quantize_model(FP32_PATH, INT8_PATH)
        return True
    except Exception as e:
        log.error(f"Cuantización fallida: {e}")
        return False


def step_validate() -> bool:
    header("Paso 4 — Validación Precisión vs Eficiencia")
    if not FP32_PATH.exists():
        log.warning("  Modelo FP32 no encontrado — omitiendo validación.")
        return False
    try:
        from validate_accuracy import validate_model, print_comparison_table
        results = []
        results.append(validate_model(FP32_PATH, "MobileNetV2", "FP32", n_samples=100))
        if INT8_PATH.exists():
            results.append(validate_model(INT8_PATH, "MobileNetV2", "INT8", n_samples=100))
        else:
            log.warning("  Modelo INT8 no encontrado — comparando solo FP32.")
        print_comparison_table(results)
        return True
    except Exception as e:
        log.error(f"Validación fallida: {e}")
        return False


def step_benchmark() -> bool:
    header("Paso 5 — Benchmark CLI (latencia / FPS)")
    if not FP32_PATH.exists():
        log.warning("  Modelo FP32 no encontrado — omitiendo benchmark.")
        return False
    try:
        from inference_engine import InferenceEngine
        log.info("  Benchmarking FP32 …")
        engine_fp32 = InferenceEngine(str(FP32_PATH), provider="cpu")
        result_fp32 = engine_fp32.benchmark(n_warmup=10, n_runs=100, precision="FP32")

        if INT8_PATH.exists():
            log.info("  Benchmarking INT8 …")
            engine_int8 = InferenceEngine(str(INT8_PATH), provider="cpu")
            result_int8 = engine_int8.benchmark(n_warmup=10, n_runs=100, precision="INT8")
            speedup = result_fp32.mean_latency_ms / result_int8.mean_latency_ms
            log.info(f"  Speedup INT8 vs FP32: {speedup:.2f}×")
        return True
    except Exception as e:
        log.error(f"Benchmark fallido: {e}")
        return False


def step_trt() -> bool:
    header("Paso 6 — Engine TensorRT (solo Jetson)")
    try:
        import tensorrt  # noqa: F401
    except ImportError:
        log.info("  TensorRT no disponible en este entorno — omitiendo.")
        log.info("  Ejecutar trt_engine.py directamente en el Jetson destino con JetPack instalado.")
        return True  # No es un fallo — esperado en máquinas de desarrollo

    try:
        from trt_engine import build_trt_engine
        build_trt_engine(FP32_PATH, MODEL_DIR / "mobilenet_v2_fp16.engine", precision="fp16")
        if INT8_PATH.exists():
            build_trt_engine(INT8_PATH, MODEL_DIR / "mobilenet_v2_int8.engine", precision="int8")
        return True
    except Exception as e:
        log.error(f"Paso TensorRT fallido: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Orquestador del Pipeline de Robótica Edge")
    parser.add_argument("--force",    action="store_true",
                        help="Re-ejecutar todos los pasos aunque las salidas ya existan")
    parser.add_argument("--skip-trt", action="store_true",
                        help="Omitir la compilación del engine TensorRT")
    args = parser.parse_args()

    log.info("Pipeline de AI Performance Engineering para Robótica Edge")
    log.info(f"Directorio de modelos: {MODEL_DIR.resolve()}")

    t_start = time.perf_counter()
    results = {}

    results["export"]    = step_export(args.force)
    results["pruning"]   = step_prune()
    results["cuantizar"] = step_quantize(args.force)
    results["validar"]   = step_validate()
    results["benchmark"] = step_benchmark()

    if not args.skip_trt:
        results["trt"] = step_trt()

    # ── Resumen ────────────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t_start
    header("Resumen del Pipeline")
    all_ok = True
    for step, ok in results.items():
        status = "✓" if ok else "✗"
        log.info(f"  {status}  {step}")
        if not ok:
            all_ok = False

    log.info("")
    log.info(f"  Tiempo total: {elapsed:.1f} s")
    log.info("")

    if all_ok:
        log.info("  Todos los pasos completados. Para arrancar el servidor FastAPI:")
        log.info("    uvicorn api:app --host 0.0.0.0 --port 8000")
        log.info("  Abrir en el navegador: http://localhost:8000/docs")
    else:
        log.error("  Uno o más pasos han fallado. Ver salida anterior para detalles.")
        sys.exit(1)


if __name__ == "__main__":
    main()