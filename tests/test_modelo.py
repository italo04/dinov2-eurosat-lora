import pytest
import torch
import torch.nn as nn
from eurosat_lora.modelo import (
    LoRALinear,
    aplicar_lora,
    adelante_lora,
    fusionar_lora,
    representacion,
    salida_bloque,
    contar_entrenables,
)


def test_lora_b_cero_salida_identica(modelo_chico):
    x = torch.randn(2, 3, 28, 28)
    with torch.no_grad():
        out_base = representacion(modelo_chico(x).last_hidden_state)
    aplicar_lora(modelo_chico, bloques=(2, 3), proyecciones=("q", "v"), r=4, alpha=4.0)
    with torch.no_grad():
        out_lora = adelante_lora(modelo_chico, x, primer_bloque=2)
    diff = (out_lora - out_base).abs().max().item()
    assert diff == pytest.approx(0.0, abs=1e-5)


def test_parametros_lora_qv_dos_bloques():
    base_lin = nn.Linear(384, 384)
    lora = LoRALinear(base_lin, r=8, alpha=8.0)
    params_por_capa = sum(p.numel() for p in lora.parameters() if p.requires_grad)
    assert params_por_capa == 8 * (384 + 384)
    assert 2 * 2 * params_por_capa == 24_576


def test_aplicar_lora_solo_q(modelo_chico):
    reemplazados = aplicar_lora(modelo_chico, bloques=(2, 3), proyecciones=("q",), r=4, alpha=4.0)
    assert len(reemplazados) == 2
    assert all("value" not in name for name in reemplazados)


def test_aplicar_lora_cuatro_bloques(modelo_chico):
    reemplazados = aplicar_lora(modelo_chico, bloques=(0, 1, 2, 3), proyecciones=("q", "v"), r=4, alpha=4.0)
    assert len(reemplazados) == 8


def test_adelante_lora_igual_forward_completo(modelo_chico):
    x = torch.randn(2, 3, 28, 28)
    aplicar_lora(modelo_chico, bloques=(2, 3), proyecciones=("q", "v"), r=4, alpha=4.0)
    out_adelante = adelante_lora(modelo_chico, x, primer_bloque=2)
    out_forward = representacion(modelo_chico(x).last_hidden_state)
    assert torch.allclose(out_adelante, out_forward, atol=1e-5)


def test_fusion_equivalente(modelo_chico):
    x = torch.randn(2, 3, 28, 28)
    aplicar_lora(modelo_chico, bloques=(2, 3), proyecciones=("q", "v"), r=4, alpha=4.0)
    for m in modelo_chico.modules():
        if isinstance(m, LoRALinear):
            nn.init.normal_(m.B, std=0.02)
    with torch.no_grad():
        out_antes = adelante_lora(modelo_chico, x, primer_bloque=2)
    n_fusionadas = fusionar_lora(modelo_chico)
    assert n_fusionadas == 4
    with torch.no_grad():
        out_despues = representacion(modelo_chico(x).last_hidden_state)
    assert torch.allclose(out_antes, out_despues, atol=1e-5)
    assert not any(isinstance(m, LoRALinear) for m in modelo_chico.modules())


def test_salida_bloque_tupla_o_tensor():
    t = torch.randn(2, 4)
    assert salida_bloque((t,)) is t
    assert salida_bloque(t) is t
