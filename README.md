# Pipeline de AI Performance Engineering para Robótica Edge

Prueba de concepto que cubre las responsabilidades principales de un
AI Performance Engineer para robótica edge: compresión de modelos,
aceleración con TensorRT, benchmarking y validación de precisión vs
eficiencia — orientado a despliegue en hardware NVIDIA Jetson.

<img width="1851" height="903" alt="Captura de pantalla 2026-04-27 043642" src="https://github.com/user-attachments/assets/56f1f03a-bf99-44c6-81da-d28c234a2096" />

```
edge_robot_pipeline/
├── run_pipeline.py       # ← EMPEZAR AQUÍ: orquesta todos los pasos en orden
├── export_model.py       # PyTorch → ONNX (FP32)
├── pruning.py            # Pruning estructurado y no estructurado
├── quantize_model.py     # Cuantización estática FP32 → INT8 (formato QDQ)
├── trt_engine.py         # Compilación e inferencia con engine TensorRT
├── inference_engine.py   # Sesión ONNX Runtime, preprocesado, benchmarking
├── validate_accuracy.py  # Tabla comparativa precisión vs eficiencia
├── run_benchmark.py      # Herramienta CLI de profiling (latencia, FPS, percentiles)
├── api.py                # Wrapper mínimo FastAPI para pruebas de integración
├── models/               # Archivos .onnx y .engine generados (se crea en el primer run)
└── requirements.txt
```

---

## Conceptos cubiertos (alineados con las responsabilidades de la vacante)

| Responsabilidad (vacante)                           | Archivo                |
|-----------------------------------------------------|------------------------|
| Cuantización                                        | `quantize_model.py`    |
| Pruning                                             | `pruning.py`           |
| Optimización para TensorRT y aceleradores           | `trt_engine.py`        |
| Benchmarking comparativo en plataformas hardware    | `run_benchmark.py`     |
| Profiling en placa (latencia, memoria)              | `inference_engine.py`  |
| Optimización de pipelines completos                 | `inference_engine.py`  |
| Validación precisión vs eficiencia                  | `validate_accuracy.py` |
| Co-diseño modelo ↔ hardware                         | `pruning.py`, `trt_engine.py` |

---

## Inicio rápido

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Ejecutar el pipeline completo con un solo comando
python run_pipeline.py
#  Ejecuta todos los pasos en orden:
#  export → pruning → cuantizar → validar → benchmark → trt (si hay Jetson)
#  Seguro para ejecutar varias veces — omite pasos cuya salida ya existe.

# Opciones adicionales:
python run_pipeline.py --force      # re-ejecutar aunque las salidas ya existan
python run_pipeline.py --skip-trt   # omitir TensorRT (recomendado en PC sin GPU NVIDIA)
```

O ejecutar los pasos individualmente:

```bash
python export_model.py              # Paso 1: PyTorch → ONNX FP32 (~14 MB)
python pruning.py                   # Paso 2: estadísticas de pruning estructurado vs no estructurado
python quantize_model.py            # Paso 3: ONNX FP32 → INT8 QDQ (~14-17 MB en disco)
python validate_accuracy.py         # Paso 4: tabla comparativa precisión vs eficiencia
python run_benchmark.py --compare   # Paso 5: comparativa FP32 vs INT8 en latencia / FPS
python trt_engine.py                # Paso 6: engine TensorRT — solo en Jetson con JetPack

# Opcional: wrapper FastAPI para pruebas de integración
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 1
# Abrir en el navegador: http://localhost:8000/docs
```

---

## Decisiones de arquitectura

### Pipeline de compresión — ¿por qué tres técnicas?

Cada técnica ataca un cuello de botella diferente:

| Técnica        | Qué reduce               | Cuándo usarla                           |
|----------------|--------------------------|----------------------------------------|
| Pruning        | FLOPs, parámetros        | Cuando la inferencia está limitada por cómputo |
| Cuantización   | Ancho de banda de memoria| Siempre — coste mínimo en precisión    |
| Distillation   | Profundidad/anchura      | Cuando hay presupuesto de entrenamiento |

El orden recomendado para compresión máxima es:
**Pruning → fine-tune → cuantización**. Cada paso multiplica los ahorros del anterior.

### Pruning estructurado vs no estructurado

El pruning no estructurado (poner a cero pesos individuales) crea dispersión en
papel pero no da speedup real en la GPU o el NVDLA de Jetson — operan con tensores
densos. El pruning estructurado elimina canales de salida enteros, produciendo un
modelo más pequeño y denso que es más rápido en cualquier hardware destino.

### ¿Por qué formato QDQ para la cuantización INT8?

QDQ (Quantize-DeQuantize) inserta nodos de cuantización explícitos en el grafo ONNX.
TensorRT necesita esto para ver los límites de cuantización y realizar fusión de capas
a través de ellos. Sin QDQ, TensorRT no puede generar un engine INT8 desde el archivo ONNX.

### Tamaño en disco INT8 vs memoria en runtime

El archivo `.onnx` tras la cuantización QDQ tiene un tamaño similar al FP32 original
(~14–17 MB) porque ONNX almacena los pesos como FP32 más los nodos Q/DQ extra. El
ahorro real de 4× ocurre en runtime, cuando TensorRT compila el engine INT8 y carga
~3.5 MB de pesos enteros en la SRAM del dispositivo en lugar de ~14 MB en coma flotante.
Eso reduce el ancho de banda de DRAM por frame, que es el cuello de botella real en
SoCs embebidos.

### TensorRT — por qué marca la diferencia

```
ONNX Runtime CPU  : ~8 ms  (FP32 MobileNetV2)
ONNX Runtime CUDA : ~4 ms
TensorRT FP16     : ~1.5 ms
TensorRT INT8     : ~0.8 ms  → 10× sobre ONNX CPU
```

TensorRT compila el modelo en un plan de ejecución específico del hardware —
fusionando capas, seleccionando kernels CUDA óptimos, y enrutando operaciones
INT8 al NVDLA. El archivo `.engine` es específico de la plataforma y debe compilarse
EN el Jetson destino.

### Co-diseño modelo ↔ hardware

Los ratios de pruning se eligen para alinearse con el tile size del NVDLA del
Jetson Orin NX (16 canales de salida). Podar a múltiplos de 16 canales por capa
maximiza la utilización del NVDLA. Esto es co-diseño: la arquitectura del modelo
está condicionada por las restricciones del hardware destino, no de forma independiente.

### ¿Por qué no una API cloud?

```
Cámara del robot (30 FPS)
       |
  codificar JPEG --> HTTP POST --> [WiFi] --> Endpoint Cloud
                                                | ~150 ms RTT
  recibir JSON <---------------------------------/

Tiempo total: ~200 ms = 5 FPS máximo.
Loop de control del robot necesita: >= 30 FPS (33 ms de presupuesto).
La cloud falla por un factor de 6×.

ONNX Runtime local:
  preprocesado --> session.run --> postprocesado
     0.3 ms          3.5 ms          0.1 ms = 3.9 ms total
Supera el requisito por 8×. Sin WiFi. Sin salida de datos. Air-gapped.
```

### Patrón de integración con ROS 2

```python
import cv2
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Image
from inference_engine import InferenceEngine

class VisionNode(Node):
    def __init__(self):
        super().__init__('vision_node')
        # Cargar UNA SOLA VEZ al arrancar — crear la sesión cuesta ~40-200 ms
        self.engine = InferenceEngine('models/mobilenet_v2_int8.onnx')
        self.sub = self.create_subscription(Image, '/camera/image_raw', self.cb, 10)

    def cb(self, msg: Image):
        frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        # Redimensionar al tamaño de entrada del modelo — el engine valida (224, 224)
        frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)
        predictions = self.engine.predict_top_k(frame, k=3)
        # publicar en /vision/detections ...
```

---

## Objetivos de rendimiento (Jetson Orin NX 8GB)

Estimaciones basadas en benchmarks publicados de Jetson. Los resultados reales
dependen de la versión de JetPack, el estado térmico y las cargas de trabajo concurrentes.

| Modelo   | Técnica               | Media (ms) | p99 (ms) | FPS   | RSS del proceso |
|----------|-----------------------|------------|----------|-------|-----------------|
| FP32     | Línea base            | ~8 ms      | ~12 ms   | ~90   | ~120 MB         |
| INT8     | Cuantización          | ~2.5 ms    | ~4 ms    | ~280  | ~40 MB          |
| INT8     | Pruning + Cuantización| ~1.5 ms    | ~2.5 ms  | ~450  | ~25 MB          |
| INT8     | Engine TensorRT       | ~0.8 ms    | ~1.2 ms  | ~800  | ~120 MB RSS + ~60 MB dispositivo |
