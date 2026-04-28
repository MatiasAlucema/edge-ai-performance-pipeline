"""
trt_engine.py — Compilación e Inferencia con TensorRT
======================================================
Optimización para TensorRT y aceleradores específicos.

¿Por qué TensorRT en lugar de ONNX Runtime en Jetson?
  ONNX Runtime con CPUExecutionProvider es genérico — funciona en cualquier
  sitio pero no está optimizado para nada específico. TensorRT compila el modelo
  en un plan de ejecución específico para la GPU/NVDLA exacta del Jetson destino:

    ONNX Runtime CPU  : ~8 ms  (FP32 MobileNetV2)
    ONNX Runtime CUDA : ~4 ms  (FP32 MobileNetV2)
    TensorRT FP16     : ~1.5 ms
    TensorRT INT8     : ~0.8 ms → 10× sobre ONNX CPU

  Esa diferencia de 10× es la que separa "funciona en un robot" de "funciona
  bien en un robot". El camino NVDLA añade eficiencia energética encima.

Flujo de compilación TensorRT:
  1. Modelo ONNX (con nodos QDQ para INT8) ← salida de quantize_model.py
  2. trt_engine.py lo parsea via el parser ONNX de TensorRT
  3. El Builder optimiza para el hardware destino (fusión de capas, selección de kernels)
  4. Archivo .engine serializado — específico de la plataforma, NO puede moverse
     a una GPU diferente. Debe compilarse EN el Jetson destino.

Nota de co-diseño modelo ↔ hardware:
  La configuración del Builder establece memoria de workspace, modo de precisión
  y opcionalmente restringe qué capas corren en NVDLA vs GPU. Aquí es donde
  ocurre el co-diseño en tiempo de ejecución: elegir la precisión por capa para
  equilibrar precisión vs throughput según las capacidades del SoC destino.
"""

import numpy as np
import time
import logging
from pathlib import Path

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

MODEL_DIR   = Path("models")
INPUT_NAME  = "image"
INPUT_SHAPE = (1, 3, 224, 224)


def build_trt_engine(
    onnx_path: Path,
    engine_path: Path,
    precision: str = "fp16",
    workspace_gb: float = 1.0,
) -> bool:
    """
    Compila un modelo ONNX en un engine TensorRT.

    Args:
        onnx_path    : Ruta al archivo ONNX de entrada (usar QDQ INT8 para mejores resultados).
        engine_path  : Ruta de salida para el archivo .engine serializado.
        precision    : "fp32", "fp16" o "int8". fp16 es el punto de partida recomendado
                       — ~2× más rápido que fp32 con pérdida de precisión despreciable.
        workspace_gb : Presupuesto de memoria GPU para los algoritmos de optimización
                       de TensorRT. Mayor = más kernels explorados = mejor engine,
                       pero tiempo de compilación más largo. 1 GB es estándar para Jetson Orin NX.

    Devuelve True si la compilación tuvo éxito, False si TensorRT no está disponible.

    Nota:
      Este script requiere tensorrt y pycuda instalados, que solo están disponibles
      en Jetson (JetPack) o en una máquina x86 con TensorRT instalado.
      En una máquina de desarrollo sin TensorRT, la función lo detecta y devuelve
      False de forma limpia — el resto del pipeline sigue funcionando via ONNX Runtime.
    """
    try:
        import tensorrt as trt
    except ImportError:
        log.warning(
            "TensorRT no disponible en este entorno. "
            "En Jetson, TensorRT viene preinstalado con JetPack. "
            "En x86, instalar con: pip install tensorrt"
        )
        return False

    onnx_path   = Path(onnx_path)
    engine_path = Path(engine_path)

    if not onnx_path.exists():
        raise FileNotFoundError(f"Modelo ONNX no encontrado: {onnx_path}")

    engine_path.parent.mkdir(parents=True, exist_ok=True)

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    log.info("Compilando engine TensorRT …")
    log.info(f"  Entrada   : {onnx_path}")
    log.info(f"  Salida    : {engine_path}")
    log.info(f"  Precisión : {precision.upper()}")
    log.info(f"  Workspace : {workspace_gb} GB")

    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network() as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser, \
         builder.create_builder_config() as config:

        # Nota: el flag EXPLICIT_BATCH fue eliminado en TensorRT 10 (JetPack 6+).
        # Todos los networks son explicit batch por defecto en TRT 10.

        # ── Configuración del Builder ──────────────────────────────────────────
        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE,
            int(workspace_gb * 1 << 30)
        )

        if precision == "fp16":
            if not builder.platform_has_fast_fp16:
                log.warning("FP16 no soportado nativamente en esta plataforma — usando FP32.")
            else:
                config.set_flag(trt.BuilderFlag.FP16)
                log.info("  FP16 activado ✓")

        elif precision == "int8":
            if not builder.platform_has_fast_int8:
                log.warning("INT8 no soportado nativamente — usando FP16.")
                config.set_flag(trt.BuilderFlag.FP16)
            else:
                config.set_flag(trt.BuilderFlag.INT8)
                # Para INT8 con ONNX QDQ, TensorRT lee la calibración de los nodos Q/DQ.
                # No se necesita calibrador separado cuando el modelo fue cuantizado con QDQ.
                log.info("  INT8 activado (calibración leída de nodos QDQ) ✓")

        # ── Parsear ONNX ───────────────────────────────────────────────────────
        log.info("  Parseando grafo ONNX …")
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
                raise RuntimeError("Parse de ONNX fallido:\n" + "\n".join(errors))
        log.info("  Grafo ONNX parseado ✓")

        # ── Compilar engine ────────────────────────────────────────────────────
        log.info("  Compilando engine (puede tardar 1–5 minutos en la primera ejecución) …")
        t0 = time.perf_counter()
        serialised = builder.build_serialized_network(network, config)
        build_time = time.perf_counter() - t0

        if serialised is None:
            raise RuntimeError(
                "Compilación del engine TensorRT fallida — "
                "comprobar memoria GPU y compatibilidad del modelo."
            )

        with open(engine_path, "wb") as f:
            f.write(serialised)

        size_mb = engine_path.stat().st_size / 1e6
        log.info(f"  Engine guardado: {engine_path}  ({size_mb:.1f} MB)")
        log.info(f"  Tiempo de compilación: {build_time:.1f} s")
        log.info(
            "  ⚠ Este engine es específico del hardware — debe compilarse "
            "EN el Jetson destino, no en cross-compilation."
        )
        return True


def run_trt_inference(engine_path: Path, n_runs: int = 100) -> dict:
    """
    Carga un engine TensorRT compilado y mide la latencia de inferencia.

    Demuestra el camino completo del runtime TensorRT:
      deserializar engine → crear contexto de ejecución → asignar buffers en dispositivo
      → ejecutar inferencia → copiar resultados al host.

    Devuelve un diccionario con estadísticas de latencia.
    """
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401 — inicializa el contexto CUDA
    except ImportError:
        log.warning("TensorRT / pycuda no disponibles. Omitiendo benchmark de runtime.")
        return {}

    engine_path = Path(engine_path)
    if not engine_path.exists():
        raise FileNotFoundError(f"Engine no encontrado: {engine_path}")

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    log.info(f"Cargando engine TensorRT: {engine_path.name}")
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()

    # Asignar buffers en dispositivo (GPU)
    h_input  = np.random.randn(*INPUT_SHAPE).astype(np.float32)
    h_output = np.zeros((1, 1000), dtype=np.float32)  # ImageNet: 1000 clases
    d_input  = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    stream   = cuda.Stream()

    def run_once():
        cuda.memcpy_htod_async(d_input, h_input, stream)
        # execute_async_v3 es obligatorio en TensorRT 10+ (JetPack 6).
        # execute_async_v2 fue eliminado en TRT 10.
        if hasattr(context, "execute_async_v3"):
            context.execute_async_v3(stream_handle=stream.handle)
        else:
            # Fallback para TensorRT 8.x (JetPack 5)
            context.execute_async_v2(
                bindings=[int(d_input), int(d_output)],
                stream_handle=stream.handle,
            )
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        stream.synchronize()

    # Warmup
    log.info("  Calentando …")
    for _ in range(20):
        run_once()

    # Benchmark
    latencies_ms = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        run_once()
        latencies_ms.append((time.perf_counter() - t0) * 1000)

    arr = np.array(latencies_ms)
    result = {
        "mean_ms" : float(np.mean(arr)),
        "p50_ms"  : float(np.percentile(arr, 50)),
        "p95_ms"  : float(np.percentile(arr, 95)),
        "p99_ms"  : float(np.percentile(arr, 99)),
        "fps"     : n_runs / (np.sum(arr) / 1000),
    }

    log.info("─" * 48)
    log.info(f"  Benchmark de inferencia TensorRT ({n_runs} ejecuciones):")
    log.info(f"  Media : {result['mean_ms']:.2f} ms")
    log.info(f"  p95   : {result['p95_ms']:.2f} ms")
    log.info(f"  p99   : {result['p99_ms']:.2f} ms")
    log.info(f"  FPS   : {result['fps']:.1f}")
    log.info("─" * 48)
    return result


if __name__ == "__main__":
    MODEL_DIR.mkdir(exist_ok=True)

    # Engine FP16 — mejor punto de partida: ~2× speedup vs FP32, pérdida de precisión despreciable
    fp16_ok = build_trt_engine(
        onnx_path   = MODEL_DIR / "mobilenet_v2_fp32.onnx",
        engine_path = MODEL_DIR / "mobilenet_v2_fp16.engine",
        precision   = "fp16",
    )

    # Engine INT8 — requiere el ONNX en formato QDQ de quantize_model.py
    int8_ok = build_trt_engine(
        onnx_path   = MODEL_DIR / "mobilenet_v2_int8.onnx",
        engine_path = MODEL_DIR / "mobilenet_v2_int8.engine",
        precision   = "int8",
    )

    if int8_ok:
        run_trt_inference(MODEL_DIR / "mobilenet_v2_int8.engine")
    else:
        log.info(
            "TensorRT no disponible en este entorno.\n"
            "Ejecutar este script en el Jetson destino con JetPack instalado."
        )
