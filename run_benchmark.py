"""
run_benchmark.py — Herramienta CLI de Profiling
================================================
Ejecuta el benchmark de latencia y throughput desde la línea de comandos,
sin necesitar el servidor FastAPI. Útil para comparar FP32 vs INT8 antes
de desplegar en el robot.

Uso:
    python run_benchmark.py                        # benchmark FP32 por defecto
    python run_benchmark.py --model int8           # benchmark INT8
    python run_benchmark.py --compare              # comparativa FP32 vs INT8
    python run_benchmark.py --model fp32 --runs 500
"""

import argparse
import sys
from pathlib import Path
from inference_engine import InferenceEngine

MODEL_DIR = Path("models")
PRESETS = {
    "fp32": MODEL_DIR / "mobilenet_v2_fp32.onnx",
    "int8": MODEL_DIR / "mobilenet_v2_int8.onnx",
}


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark de inferencia ONNX para robótica edge")
    p.add_argument("--model",    choices=["fp32", "int8"], default="fp32",
                   help="Variante de precisión del modelo")
    p.add_argument("--provider", choices=["cpu", "cuda", "tensorrt"], default="cpu",
                   help="Proveedor de ejecución de ONNX Runtime")
    p.add_argument("--warmup",   type=int, default=20,
                   help="Ejecuciones de warmup antes de medir (mín. 1)")
    p.add_argument("--runs",     type=int, default=200,
                   help="Ejecuciones medidas (mín. 10 para percentiles estadísticamente válidos)")
    p.add_argument("--compare",  action="store_true",
                   help="Ejecutar FP32 e INT8 e imprimir tabla comparativa")
    args = p.parse_args()

    if args.warmup < 1:
        p.error("--warmup debe ser >= 1")
    if args.runs < 10:
        p.error("--runs debe ser >= 10 para percentiles estadísticamente significativos")

    return args


def run_single(model_key: str, provider: str, warmup: int, runs: int):
    path = PRESETS[model_key]
    if not path.exists():
        print(f"[ERROR] Modelo no encontrado: {path}")
        print("  → Ejecutar: python export_model.py")
        if model_key == "int8":
            print("  → Luego: python quantize_model.py")
        sys.exit(1)

    engine = InferenceEngine(str(path), provider=provider)
    precision = "INT8" if model_key == "int8" else "FP32"
    return engine.benchmark(n_warmup=warmup, n_runs=runs, precision=precision)


def compare(provider: str, warmup: int, runs: int):
    results = {}
    for key in ["fp32", "int8"]:
        path = PRESETS[key]
        if not path.exists():
            print(f"[OMITIDO] Modelo {key} no encontrado en {path}.")
            continue
        results[key] = run_single(key, provider, warmup, runs)

    if len(results) < 2:
        print("\n[ERROR] Se necesitan ambos modelos (fp32 e int8) para la comparativa.")
        print("  → Ejecutar: python export_model.py && python quantize_model.py")
        sys.exit(1)

    fp32 = results["fp32"]
    int8 = results["int8"]
    speedup = fp32.mean_latency_ms / int8.mean_latency_ms

    print("\n" + "═" * 56)
    print("  Comparativa FP32 vs INT8")
    print("═" * 56)
    print(f"  {'Métrica':<24} {'FP32':>10} {'INT8':>10} {'Δ':>8}")
    print("─" * 56)
    rows = [
        ("Latencia media (ms)",  fp32.mean_latency_ms,  int8.mean_latency_ms),
        ("Latencia p95 (ms)",    fp32.p95_latency_ms,   int8.p95_latency_ms),
        ("Latencia p99 (ms)",    fp32.p99_latency_ms,   int8.p99_latency_ms),
        ("Throughput (FPS)",     fp32.throughput_fps,    int8.throughput_fps),
    ]
    for label, v_fp32, v_int8 in rows:
        delta = v_int8 / v_fp32
        symbol = "↓" if "Latencia" in label else "↑"
        print(f"  {label:<24} {v_fp32:>10.2f} {v_int8:>10.2f} {delta:>6.2f}× {symbol}")
    print("─" * 56)
    print(f"  Speedup total: {speedup:.2f}× más rápido con INT8")
    print(f"  (Objetivo: >= 30 FPS para control de robot en tiempo real)")
    print("═" * 56 + "\n")


if __name__ == "__main__":
    args = parse_args()
    if args.compare:
        compare(args.provider, args.warmup, args.runs)
    else:
        run_single(args.model, args.provider, args.warmup, args.runs)
