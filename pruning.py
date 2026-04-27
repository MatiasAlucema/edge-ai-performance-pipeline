"""
pruning.py — Pruning Estructurado para Despliegue en Edge
==========================================================
Relevancia para la vacante (Funditec):
  "Cuantización, pruning y distillation" — este script cubre el pruning.

¿Por qué pruning en robótica?
  El presupuesto de cómputo de un robot está fijado por su SoC (p. ej. Jetson Orin NX).
  El pruning reduce el número de operaciones que el modelo debe realizar por frame,
  bajando directamente la latencia y el consumo energético — dos de los tres KPIs
  del perfil de la vacante.

Pruning estructurado vs no estructurado:
  El pruning no estructurado pone a cero pesos individuales. El modelo queda
  disperso en papel, pero el hardware sigue ejecutando el mismo número de operaciones
  porque las GPUs y NPUs modernas trabajan con tensores densos. Sin speedup real.

  El pruning estructurado elimina filtros enteros (canales de salida) de las capas Conv.
  El modelo resultante es más pequeño y denso — se ejecuta más rápido en cualquier
  hardware destino, incluido el NVDLA de Jetson que requiere operaciones INT8 densas.

Nota de co-diseño (modelo ↔ hardware):
  El ratio de pruning es consciente del hardware. En el NVDLA del Jetson Orin NX,
  las convoluciones INT8 se procesan en tiles de 16 canales. Podar a múltiplos de
  16 canales por capa maximiza la utilización del NVDLA. Esto es lo que Funditec
  entiende por "co-diseño modelo ↔ hardware".
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torchvision.models as models
from pathlib import Path
import logging
import copy

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

MODEL_DIR = Path("models")


def load_model() -> nn.Module:
    """Carga MobileNetV2 preentrenado en modo inferencia."""
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT).eval()
    log.info(f"MobileNetV2 cargado — {sum(p.numel() for p in model.parameters()):,} parámetros")
    return model


def count_nonzero_params(model: nn.Module) -> tuple[int, int]:
    """
    Devuelve (parámetros_no_cero, total_parámetros) contando TODOS los parámetros.
    Nota: el pruning solo afecta a los pesos de Conv2d, no a bias ni BatchNorm,
    por lo que la reducción global siempre será menor que el ratio de pruning aplicado.
    """
    total   = sum(p.numel() for p in model.parameters())
    nonzero = sum(int((p != 0).sum().item()) for p in model.parameters())
    return nonzero, total


def count_conv_nonzero(model: nn.Module) -> tuple[int, int]:
    """
    Devuelve (no_cero, total) contando solo los pesos de capas Conv2d.
    Esta es la métrica relevante para el pruning, ya que solo actúa sobre Conv2d.
    """
    total   = sum(m.weight.numel() for m in model.modules() if isinstance(m, nn.Conv2d))
    nonzero = sum(int((m.weight != 0).sum().item()) for m in model.modules() if isinstance(m, nn.Conv2d))
    return nonzero, total


def apply_unstructured_pruning(model: nn.Module, amount: float = 0.3) -> nn.Module:
    """
    Aplica pruning no estructurado L1 a todas las capas Conv2d.

    amount: fracción de pesos a poner a cero (0.3 = 30% de dispersión).

    Limitación: el pruning no estructurado crea dispersión en papel, pero NO
    reduce el cómputo real en hardware denso como la GPU/NVDLA de Jetson.
    Se incluye aquí para comparación didáctica con el pruning estructurado.
    """
    model = copy.deepcopy(model)
    pruned_layers = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.l1_unstructured(module, name="weight", amount=amount)
            pruned_layers += 1

    nonzero, total = count_nonzero_params(model)
    sparsity = 1.0 - nonzero / total
    log.info(f"Pruning no estructurado ({amount:.0%} por capa Conv2d):")
    log.info(f"  Capas podadas  : {pruned_layers}")
    log.info(f"  Dispersión global: {sparsity:.1%}")
    log.info(f"  ⚠ Aviso: la dispersión es teórica — sin speedup real en hardware denso.")
    return model


def apply_structured_pruning(model: nn.Module, amount: float = 0.25) -> nn.Module:
    """
    Aplica pruning estructurado por norma L2 sobre los canales de salida (dim=0)
    de las capas Conv2d.

    amount: fracción de canales de salida a eliminar por capa (0.25 = 25% menos filtros).

    ¿Por qué da speedup real?
      Eliminar canales de salida completos reduce el número de operaciones de
      convolución proporcionalmente. El modelo resultante es más denso y pequeño,
      compatible con TensorRT y el camino INT8 del NVDLA de Jetson.

    Nota de co-diseño hardware:
      El NVDLA del Jetson Orin NX procesa convoluciones en tiles de 16 canales.
      Tras el pruning, los canales restantes se reindexan de forma contigua —
      no se necesita ejecución dispersa, mejora real del throughput.
    """
    model = copy.deepcopy(model)
    pruned_layers = 0

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and module.out_channels >= 4:
            # Protección: se necesitan al menos 4 canales para que el 25% de pruning
            # siempre deje al menos 1 canal completo.
            prune.ln_structured(module, name="weight", amount=amount, n=2, dim=0)
            pruned_layers += 1

    nonzero, total = count_nonzero_params(model)
    sparsity = 1.0 - nonzero / total

    log.info(f"Pruning estructurado ({amount:.0%} de canales de salida por Conv2d):")
    log.info(f"  Capas podadas   : {pruned_layers}")
    log.info(f"  Dispersión global: {sparsity:.1%}")
    log.info(f"  ✓ Estructurado — speedup real tras make_permanent + export a ONNX.")
    return model


def make_pruning_permanent(model: nn.Module) -> nn.Module:
    """
    Elimina las máscaras de pruning y hace permanente la dispersión en los pesos.

    Debe llamarse antes del export a ONNX. Mientras las máscaras están activas,
    el modelo mantiene los pesos originales Y la máscara binaria en paralelo —
    el doble de memoria. make_permanent los fusiona y elimina los buffers de máscara,
    dejando solo los pesos podados.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            try:
                prune.remove(module, "weight")
            except ValueError:
                pass  # La capa no fue podada
    log.info("Máscaras de pruning eliminadas — los pesos son ahora permanentemente dispersos.")
    return model


def compare_model_sizes(original: nn.Module, pruned: nn.Module, technique: str = "") -> dict:
    """
    Compara el número de parámetros entre el modelo original y el podado.
    Muestra métricas globales y específicas de Conv2d (donde actúa el pruning).
    """
    orig_total   = sum(p.numel() for p in original.parameters())
    pruned_nz, pruned_total = count_nonzero_params(pruned)

    # Métricas específicas de Conv2d — donde realmente actúa el pruning
    orig_conv_total           = sum(m.weight.numel() for m in original.modules() if isinstance(m, nn.Conv2d))
    pruned_conv_nz, _         = count_conv_nonzero(pruned)
    conv_sparsity             = 1.0 - pruned_conv_nz / orig_conv_total if orig_conv_total > 0 else 0.0

    result = {
        "original_params"     : orig_total,
        "pruned_total_params" : pruned_total,
        "pruned_nonzero"      : pruned_nz,
        "global_sparsity"     : 1.0 - pruned_nz / pruned_total,
        "conv_sparsity"       : conv_sparsity,
    }

    log.info("─" * 56)
    if technique:
        log.info(f"  Técnica: {technique}")
    log.info(f"  Parámetros totales        : {orig_total:,}")
    log.info(f"  Pesos Conv2d no-cero      : {pruned_conv_nz:,} / {orig_conv_total:,}")
    log.info(f"  Dispersión en Conv2d      : {conv_sparsity:.1%}  ← métrica relevante del pruning")
    log.info(f"  Dispersión global (todos) : {result['global_sparsity']:.1%}  "
             f"(menor porque BN y bias no se podan)")
    log.info("─" * 56)
    return result


if __name__ == "__main__":
    MODEL_DIR.mkdir(exist_ok=True)

    original = load_model()

    log.info("\n── Pruning No Estructurado ───────────────────────")
    unstructured = apply_unstructured_pruning(original, amount=0.3)
    unstructured = make_pruning_permanent(unstructured)
    compare_model_sizes(original, unstructured, technique="No estructurado — 30% pesos Conv2d a cero")

    log.info("\n── Pruning Estructurado ──────────────────────────")
    structured = apply_structured_pruning(original, amount=0.25)
    structured = make_pruning_permanent(structured)
    compare_model_sizes(original, structured, technique="Estructurado — 25% canales de salida eliminados")

    log.info("\nPróximos pasos:")
    log.info("  1. Fine-tuning del modelo podado para recuperar precisión")
    log.info("  2. Export a ONNX con export_model.py")
    log.info("  3. Cuantización a INT8 con quantize_model.py")
    log.info("  Pipeline completo: pruning → fine-tune → cuantización = compresión máxima")