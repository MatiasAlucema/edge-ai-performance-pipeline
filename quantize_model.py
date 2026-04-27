"""
quantize_model.py — Cuantización Estática ONNX FP32 → INT8
============================================================
Contexto de ingeniería de rendimiento:
  La cuantización INT8 es la optimización de mayor impacto en robótica edge
  porque ataca tres cuellos de botella simultáneamente:

    1. Throughput de cómputo: INT8 GEMM en el acelerador de deep learning de Jetson
                              (NVDLA o TensorRT INT8) es 4–8× más rápido que
                              las CUDA cores en FP32.

    2. Ancho de banda de memoria: Las activaciones y pesos que se mueven por el chip
                                  son 4× más pequeños, reduciendo la presión sobre
                                  la DRAM — el cuello de botella principal en SoCs
                                  embebidos.

    3. Consumo energético: Menos movimiento de datos = menos transacciones de memoria
                           = menos energía. Crítico en robots con batería.

  Nota sobre el tamaño del archivo con formato QDQ:
    El archivo .onnx producido por cuantización estática con formato QDQ NO es
    más pequeño que el FP32 en disco — suele tener el mismo tamaño o algo mayor
    (~14–17 MB), porque ONNX sigue almacenando los pesos como FP32 más los nodos
    Q/DQ añadidos al grafo. El ahorro ocurre en tiempo de ejecución: TensorRT lee
    el grafo QDQ y compila un engine INT8 que carga ~3.5 MB de pesos enteros en
    memoria de dispositivo. Ahí es donde importa la reducción — en RAM, no en disco.

  Trade-off: ~0.5–1.5% de caída de precisión en ImageNet — aceptable para la mayoría
  de tareas de detección/clasificación en entornos de robot controlados.
"""

import numpy as np
from pathlib import Path
from onnxruntime.quantization import (
    quantize_static,
    CalibrationDataReader,
    QuantType,
    QuantFormat,
)
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

MODEL_DIR   = Path("models")
INPUT_NAME  = "image"
INPUT_SHAPE = (1, 3, 224, 224)


class DummyCalibrationReader(CalibrationDataReader):
    """
    Proporciona datos de calibración para la cuantización estática.

    Nota para producción:
      Reemplazar los tensores dummy con 100–500 frames representativos del
      stream real de cámara del robot. Los datos de calibración deben cubrir
      el rango completo de iluminación, distancias y tipos de objeto que el
      robot encontrará. Esto es lo que diferencia un modelo INT8 de producción
      de una demo de prueba.

    ¿Por qué cuantización estática y no dinámica?
      La cuantización estática pre-calcula los rangos de activación offline, así
      que en inferencia ese cálculo ya está hecho. La cuantización dinámica los
      recalcula en cada forward pass — añadiendo ~5–10% de latencia extra, lo que
      anula su propósito en un loop de robot a 30 Hz.
    """

    def __init__(self, n_samples: int = 50):
        self.n_samples = n_samples
        self._idx = 0
        log.info(f"Lector de calibración inicializado con {n_samples} frames sintéticos.")

    def get_next(self):
        if self._idx >= self.n_samples:
            return None  # Señal de fin de datos de calibración

        # Generamos un tensor aleatorio en el rango esperado por el modelo tras
        # la normalización ImageNet (media≈0, std≈1 por canal). Es ruido gaussiano —
        # no frames reales — y produce una calibración estadísticamente válida pero
        # subóptima. En producción, reemplazar con frames reales de la cámara del robot.
        dummy = np.random.randn(*INPUT_SHAPE).astype(np.float32)
        self._idx += 1
        return {INPUT_NAME: dummy}

    def rewind(self):
        self._idx = 0


def quantize_model(
    fp32_path: Path,
    int8_path: Path,
    n_calibration_samples: int = 50,
) -> Path:
    """
    Ejecuta cuantización estática INT8 por canal con formato QDQ.

    Formato QDQ (Quantize-DeQuantize):
      Inserta nodos Q/DQ explícitos alrededor de cada operación cuantizable.
      Esto es obligatorio para TensorRT, ya que necesita ver los límites de
      cuantización en el grafo para realizar fusión de capas a través de ellos.
      Sin QDQ, TensorRT no puede generar un engine INT8 desde el archivo ONNX.

    Nota sobre el tamaño del archivo de salida:
      El archivo .onnx resultante tiene un tamaño similar al FP32 original
      porque ONNX almacena los pesos como FP32 más los nodos Q/DQ extra.
      El ahorro real de memoria INT8 (~4×) se realiza cuando TensorRT
      compila el engine, no en el archivo .onnx.
    """
    fp32_path = Path(fp32_path)
    int8_path = Path(int8_path)

    if not fp32_path.exists():
        raise FileNotFoundError(f"Modelo FP32 no encontrado: {fp32_path}")

    int8_path.parent.mkdir(parents=True, exist_ok=True)
    reader = DummyCalibrationReader(n_calibration_samples)

    log.info("Ejecutando cuantización estática INT8 …")
    log.info(f"  Entrada : {fp32_path}")
    log.info(f"  Salida  : {int8_path}")

    try:
        quantize_static(
            model_input=str(fp32_path),
            model_output=str(int8_path),
            calibration_data_reader=reader,
            quant_format=QuantFormat.QDQ,      # Formato compatible con TensorRT
            per_channel=True,                   # Por canal > por tensor en precisión
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            # optimize_model fue eliminado en onnxruntime >= 1.16 — las optimizaciones
            # del grafo ahora se aplican automáticamente en el pipeline de cuantización.
        )
    except Exception as e:
        raise RuntimeError(f"Cuantización fallida: {e}") from e

    if not int8_path.exists():
        raise RuntimeError(f"Cuantización completada pero archivo de salida no encontrado: {int8_path}")

    fp32_mb = fp32_path.stat().st_size / 1e6
    int8_mb = int8_path.stat().st_size / 1e6

    log.info("  Cuantización completada ✓")
    log.info(f"  Tamaño FP32 (disco) : {fp32_mb:.1f} MB")
    log.info(f"  Tamaño INT8 (disco) : {int8_mb:.1f} MB  "
             f"(overhead del grafo QDQ — el ahorro real lo realiza TensorRT en RAM)")
    log.info("  Memoria de pesos INT8 en runtime: ~{:.1f} MB (4× menos que FP32)".format(fp32_mb / 4))
    return int8_path


if __name__ == "__main__":
    fp32 = MODEL_DIR / "mobilenet_v2_fp32.onnx"
    int8 = MODEL_DIR / "mobilenet_v2_int8.onnx"

    if not fp32.exists():
        raise FileNotFoundError(
            f"{fp32} no encontrado. Ejecutar export_model.py primero."
        )

    quantize_model(fp32, int8)
    log.info("Listo. Usar la ruta del modelo INT8 en InferenceEngine para despliegue edge.")
