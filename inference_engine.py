"""
inference_engine.py — Motor de Inferencia con ONNX Runtime
===========================================================
Nota de arquitectura para robótica edge:
  Esta clase está diseñada para instanciarse UNA SOLA VEZ al arrancar el nodo
  de ROS 2 y reutilizarse para cada frame de cámara. Las operaciones costosas
  (creación de sesión, pre-asignación de memoria) ocurren en __init__; el
  camino crítico (predict / benchmark) se mantiene lo más ligero posible.

  Alternativa cloud (por qué NO la usamos en un robot):
    ✗ Round-trip de red: 50–200 ms de latencia — imposible para control a 30 Hz
    ✗ Dependencia de conectividad: los robots operan en zonas sin GPS ni RF
    ✗ Privacidad de datos: los frames de cámara salen del robot (riesgo legal e IP)
    ✗ Ancho de banda: stream 1080p a 30 FPS ≈ 500 Mbps — inviable por WiFi
    ✓ ONNX Runtime local: 3–8 ms de latencia, sin dependencia de red, air-gapped
"""

import time
import numpy as np
import onnxruntime as ort
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

INPUT_SHAPE = (1, 3, 224, 224)  # NCHW


@dataclass
class BenchmarkResult:
    """Informe de profiling estructurado — listo para volcar en un dashboard."""
    model_path:      str
    n_runs:          int
    mean_latency_ms: float
    p50_latency_ms:  float
    p95_latency_ms:  float
    p99_latency_ms:  float
    min_latency_ms:  float
    max_latency_ms:  float
    throughput_fps:  float
    provider:        str
    precision:       str = "FP32"
    extra_notes:     list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "─" * 52,
            f"  Modelo    : {Path(self.model_path).name}",
            f"  Proveedor : {self.provider}  ({self.precision})",
            f"  Ejecuciones: {self.n_runs}",
            "  Latencia (ms):",
            f"    media={self.mean_latency_ms:.2f}  p50={self.p50_latency_ms:.2f}"
            f"  p95={self.p95_latency_ms:.2f}  p99={self.p99_latency_ms:.2f}",
            f"    min={self.min_latency_ms:.2f}   max={self.max_latency_ms:.2f}",
            f"  Throughput: {self.throughput_fps:.1f} FPS",
        ]
        for note in self.extra_notes:
            lines.append(f"  ℹ  {note}")
        lines.append("─" * 52)
        return "\n".join(lines)


class InferenceEngine:
    """
    Wrapper ligero de inferencia con ONNX Runtime, optimizado para hardware
    edge tipo Jetson. Soporta proveedores CPU, CUDA y TensorRT.

    Estrategia de huella de memoria:
      - El tensor de entrada se pre-asigna una vez en __init__ y se reutiliza —
        evita el malloc de numpy por frame (sorprendentemente costoso a 30 Hz).
      - El tensor de salida lo asigna session.run() por llamada. Para despliegues
        en GPU, el siguiente paso sería IO binding de OrtValue para mantener los
        tensores en memoria de dispositivo y eliminar las copias CPU↔GPU.
    """

    PROVIDER_MAP = {
        "tensorrt" : ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
        "cuda"     : ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "cpu"      : ["CPUExecutionProvider"],
    }

    def __init__(
        self,
        model_path: str | Path,
        provider: str = "cpu",
        intra_op_threads: int = 4,
        inter_op_threads: int = 1,
    ):
        self.model_path = Path(model_path)
        self.provider   = provider.lower()

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Modelo ONNX no encontrado: {self.model_path}\n"
                "Ejecutar export_model.py (y quantize_model.py para INT8) primero."
            )

        if self.provider not in self.PROVIDER_MAP:
            raise ValueError(
                f"Proveedor desconocido '{provider}'. "
                f"Opciones: {list(self.PROVIDER_MAP.keys())}"
            )

        # ── Opciones de sesión ─────────────────────────────────────────────────
        # En Jetson Orin con 12 cores ARM Cortex-A78AE, 4 hilos intra-op es el
        # punto óptimo para cargas de trabajo de un solo modelo. Más hilos →
        # contención de caché. inter_op=1 porque ejecutamos un modelo, no un grafo
        # de pipeline.
        opts = ort.SessionOptions()
        opts.intra_op_num_threads        = intra_op_threads
        opts.inter_op_num_threads        = inter_op_threads
        opts.execution_mode              = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level    = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.enable_mem_pattern          = True   # Cachea patrones de asignación de memoria
        opts.enable_cpu_mem_arena        = True

        providers = self.PROVIDER_MAP.get(self.provider, ["CPUExecutionProvider"])

        log.info(f"Cargando modelo: {self.model_path.name}")
        log.info(f"  Proveedor solicitado  : {provider}")
        log.info(f"  Proveedores disponibles: {ort.get_available_providers()}")

        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=opts,
            providers=providers,
        )

        active = self.session.get_providers()[0]
        log.info(f"  Proveedor activo      : {active}")

        self.input_name  = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # Buffer de entrada pre-asignado — evita malloc por frame
        self._input_buffer = np.zeros(INPUT_SHAPE, dtype=np.float32)
        log.info("  Motor listo. Buffer de entrada pre-asignado.")

    # ── API pública ────────────────────────────────────────────────────────────

    def preprocess(self, raw_frame: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Normaliza un frame uint8 HWC a float32 NCHW.

        La normalización ImageNet está integrada aquí para que el grafo ONNX
        permanezca limpio. Si exportas un modelo personalizado, puede ser
        conveniente incluir la normalización como op Normalize en el grafo ONNX
        para ahorrar una llamada Python en el camino crítico.
        """
        if raw_frame is None:
            # Simulamos un frame RGB 224×224 como si viniera de un mensaje ROS Image.
            # El límite superior de randint es exclusivo, así que 256 da el rango
            # completo [0, 255] de uint8.
            raw_frame = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)

        # ── Validación de entrada ──────────────────────────────────────────────
        if not isinstance(raw_frame, np.ndarray):
            raise TypeError(f"Se esperaba np.ndarray, se recibió {type(raw_frame)}")
        if raw_frame.ndim != 3 or raw_frame.shape[2] != 3:
            raise ValueError(
                f"Se esperaba frame HWC con 3 canales, se recibió forma {raw_frame.shape}"
            )
        if raw_frame.shape[:2] != (224, 224):
            raise ValueError(
                f"Se esperaba frame 224×224, se recibió {raw_frame.shape[:2]}. "
                "Redimensionar antes de llamar a preprocess()."
            )

        img = raw_frame.astype(np.float32) / 255.0

        # Media y std por canal de ImageNet (RGB)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img  = (img - mean) / std

        # HWC → NCHW
        img = img.transpose(2, 0, 1)[np.newaxis, ...]  # (1, 3, 224, 224)

        # Escribir en buffer pre-asignado (zero-copy para el downstream)
        np.copyto(self._input_buffer, img)
        return self._input_buffer

    def predict(self, frame: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Ejecuta un forward pass. Devuelve logits brutos (1, 1000) para ImageNet.

        Desglose de costes del camino crítico (CPU, MobileNetV2 INT8 en Jetson Orin):
          preprocess  : ~0.3 ms
          session.run : ~3.5 ms
          Total       : ~3.8 ms → 263 FPS máximo teórico (single-threaded)
        """
        tensor = self.preprocess(frame)
        try:
            outputs = self.session.run(
                [self.output_name],
                {self.input_name: tensor},
            )
        except Exception as e:
            raise RuntimeError(f"Inferencia ONNX Runtime fallida: {e}") from e
        return outputs[0]

    def predict_top_k(self, frame: Optional[np.ndarray] = None, k: int = 5) -> list[dict]:
        """Devuelve los k índices de clase con mayor confianza (softmax)."""
        logits = self.predict(frame).squeeze()   # (n_clases,)
        n_classes = logits.shape[0]
        if k > n_classes:
            raise ValueError(
                f"k={k} supera el número de clases de salida ({n_classes}). "
                f"Usar k <= {n_classes}."
            )
        exp    = np.exp(logits - logits.max())   # Softmax numéricamente estable
        probs  = exp / exp.sum()
        top_k  = np.argsort(probs)[::-1][:k]
        return [{"class_id": int(i), "confidence": float(probs[i])} for i in top_k]

    # ── Benchmarking ───────────────────────────────────────────────────────────

    def benchmark(
        self,
        n_warmup: int = 20,
        n_runs:   int = 200,
        precision: str = "FP32",
    ) -> BenchmarkResult:
        """
        Mide latencia y throughput de inferencia con rigor estadístico.

        Metodología de profiling:
          1. Runs de warmup: deja que los cachés de JIT/kernels alcancen estado
             estable. Sin warmup, la compilación del primer run infla el p50.
          2. Reloj monotónico (time.perf_counter): resolución ~1 ns, no afectado
             por ajustes NTP — seguro para mediciones sub-ms.
          3. p95/p99: más relevantes que media/máximo en sistemas de tiempo real.
             El loop de control de un robot necesita saber si el 99% de los frames
             termina a tiempo, no el promedio — la latencia de cola mata el
             determinismo.

        Punto para la entrevista:
          "En Jetson también usamos CUDA events (cudaEventRecord) para timing en GPU,
           ya que perf_counter incluye el overhead de CPU. Esto nos da el tiempo real
           de ejecución del kernel separado del coste de transferencia H2D/D2H."
        """
        log.info(f"Benchmarking: {n_warmup} warmup + {n_runs} ejecuciones medidas …")

        # Warmup — rellena el predictor de ramas de CPU, calienta cachés de ONNX Runtime
        for _ in range(n_warmup):
            self.predict()

        latencies_ms: list[float] = []
        t_wall_start = time.perf_counter()

        for _ in range(n_runs):
            t0 = time.perf_counter()
            self.predict()
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1_000)

        t_wall_end = time.perf_counter()
        wall_seconds = t_wall_end - t_wall_start
        # Nota: wall_seconds incluye el overhead de perf_counter() por iteración (~50 ns).
        # Para n_runs=200 esto añade ~20 µs en total — despreciable frente a latencias
        # de inferencia en ms. El throughput real (sin instrumentación) sería algo mayor.

        arr = np.array(latencies_ms)
        result = BenchmarkResult(
            model_path      = str(self.model_path),
            n_runs          = n_runs,
            mean_latency_ms = float(np.mean(arr)),
            p50_latency_ms  = float(np.percentile(arr, 50)),
            p95_latency_ms  = float(np.percentile(arr, 95)),
            p99_latency_ms  = float(np.percentile(arr, 99)),
            min_latency_ms  = float(np.min(arr)),
            max_latency_ms  = float(np.max(arr)),
            throughput_fps  = n_runs / wall_seconds,
            provider        = self.session.get_providers()[0],
            precision       = precision,
            extra_notes     = [
                "Jitter (p99-p50): {:.2f} ms".format(
                    float(np.percentile(arr, 99) - np.percentile(arr, 50))
                ),
                "Desviación típica: {:.2f} ms".format(float(np.std(arr))),
            ],
        )

        log.info("\n" + result.summary())
        return result
