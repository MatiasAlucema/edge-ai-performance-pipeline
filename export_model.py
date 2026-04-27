"""
export_model.py — Conversión de PyTorch a ONNX
===============================================
Nota de robótica edge:
  Exportar a ONNX desacopla el runtime de inferencia del framework de
  entrenamiento. En un Jetson Orin ejecutando ROS 2, no queremos el runtime
  completo de PyTorch (~800 MB) cargado dentro de un nodo ROS. ONNX Runtime
  con el proveedor TensorRT reduce eso a ~80 MB y elimina el overhead del GIL
  de Python en el camino crítico de inferencia.
"""

import torch
import torchvision.models as models
import onnx
import logging
import warnings
from pathlib import Path

try:
    import onnxsim
    _ONNXSIM_AVAILABLE = True
except ImportError:
    _ONNXSIM_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── Constantes ─────────────────────────────────────────────────────────────────
INPUT_SHAPE   = (1, 3, 224, 224)   # NCHW — estándar para MobileNet/YOLO
OPSET_VERSION = 17                  # Opset estable más reciente para ONNX Runtime 1.17+
MODEL_DIR     = Path("models")

_SUPPORTED_MODELS = {
    "mobilenet_v2": (models.mobilenet_v2, models.MobileNet_V2_Weights.DEFAULT),
}


def load_pytorch_model(model_name: str = "mobilenet_v2") -> torch.nn.Module:
    """
    Carga un backbone preentrenado en ImageNet de bajo coste computacional.

    MobileNetV2 se elige deliberadamente:
      - Convoluciones depthwise-separables → ~3.4M parámetros (vs 25M de ResNet-50)
      - Diseñado para inferencia en dispositivos móviles/embebidos — ideal para Jetson
      - Los residuales invertidos mantienen las activaciones pequeñas → menor ancho
        de banda de DRAM por frame
    """
    if model_name not in _SUPPORTED_MODELS:
        raise ValueError(
            f"Modelo '{model_name}' no soportado. "
            f"Opciones disponibles: {list(_SUPPORTED_MODELS.keys())}"
        )

    model_fn, weights = _SUPPORTED_MODELS[model_name]
    log.info(f"Cargando {model_name} preentrenado desde torchvision …")
    # .eval() pone las capas BatchNorm en modo inferencia (usa estadísticas acumuladas)
    # y desactiva dropout. Modifica in-place y devuelve self.
    model = model_fn(weights=weights).eval()
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"  Parámetros : {n_params:,}")
    log.info(f"  Tamaño     : ~{n_params * 4 / 1e6:.1f} MB (estimado FP32)")
    return model


def export_to_onnx(
    model: torch.nn.Module,
    output_path: Path,
    simplify: bool = True,
) -> Path:
    """
    Traza el modelo y escribe un archivo .onnx autocontenido.

    ¿Por qué formas estáticas?
      Los nodos de ROS 2 procesan una resolución de cámara fija (p. ej. 640×480
      redimensionada a 224×224). Con formas estáticas, TensorRT puede pre-compilar
      los kernels CUDA en el arranque en lugar de reconstruirlos cada frame —
      crítico para cumplir el deadline de un loop de control a 30 Hz.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        log.warning(f"  Sobreescribiendo archivo existente: {output_path}")

    dummy_input = torch.randn(*INPUT_SHAPE)

    log.info(f"Trazando modelo → {output_path} …")
    # Suprimimos el warning de deprecación de torch.onnx introducido en PyTorch 2.1.
    # La API clásica sigue funcionando correctamente — el warning avisa de un cambio
    # futuro, no de un problema actual.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="torch.onnx")
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            opset_version=OPSET_VERSION,
            input_names=["image"],          # Nombres explícitos para mapeo con mensajes ROS 2
            output_names=["class_logits"],
            dynamic_axes=None,              # Formas estáticas — ver docstring
            export_params=True,
            do_constant_folding=True,       # Fusiona sub-grafos estáticos en constantes,
                                            # reduciendo el tamaño del grafo exportado.
        )
    log.info("  Export completado.")

    # Validar integridad del grafo
    onnx_model = onnx.load(str(output_path))
    try:
        onnx.checker.check_model(onnx_model)
        log.info("  Grafo ONNX validado ✓")
    except onnx.checker.ValidationError as e:
        raise RuntimeError(f"Validación del grafo ONNX fallida: {e}") from e

    if simplify:
        if not _ONNXSIM_AVAILABLE:
            log.warning("  onnx-simplifier no instalado (pip install onnxsim); omitiendo.")
        else:
            log.info("  Ejecutando onnx-simplifier (fusiona ops redundantes) …")
            try:
                simplified, ok = onnxsim.simplify(onnx_model)
                if ok:
                    onnx.save(simplified, str(output_path))
                    log.info("  Grafo simplificado ✓")
                else:
                    log.warning("  Simplificación omitida — el modelo no puede simplificarse más.")
            except Exception as e:
                log.warning(f"  onnx-simplifier falló ({e}); conservando grafo original.")

    size_mb = output_path.stat().st_size / 1e6
    log.info(f"  Guardado: {output_path}  ({size_mb:.1f} MB)")
    return output_path


if __name__ == "__main__":
    MODEL_NAME = "mobilenet_v2"
    MODEL_DIR.mkdir(exist_ok=True)   # Crear directorio de salida solo al ejecutar como script
    model = load_pytorch_model(MODEL_NAME)
    export_to_onnx(model, MODEL_DIR / f"{MODEL_NAME}_fp32.onnx")
    log.info("Listo. Siguiente paso: ejecutar quantize_model.py para generar la variante INT8.")
