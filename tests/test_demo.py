"""tests/test_demo.py — Tests de la demo Streamlit y su núcleo."""

import copy
import os
import pytest
from PIL import Image
import torch
import torch.nn as nn

from app.demo_nucleo import (
    checkpoints_disponibles,
    cargar_modelos,
    predecir_imagen,
    CLASES_EUROSAT,
)
from eurosat_lora.modelo import aplicar_lora


def test_sin_checkpoints(tmp_path):
    """Comprueba que una ruta vacía no tiene checkpoints disponibles."""
    assert checkpoints_disponibles(str(tmp_path)) is False


@pytest.fixture
def modelos_chicos(modelo_chico):
    """Crea un diccionario de modelos sintéticos chicos para testear inferencia."""
    m_base = copy.deepcopy(modelo_chico)
    m_lora = copy.deepcopy(modelo_chico)
    aplicar_lora(m_lora, bloques=(2, 3), proyecciones=("q", "v"), r=4, alpha=4.0)

    in_dim = modelo_chico.config.hidden_size * 2
    cabeza_base = nn.Linear(in_dim, len(CLASES_EUROSAT))
    cabeza_lora = nn.Linear(in_dim, len(CLASES_EUROSAT))

    return {
        "base": m_base,
        "cabeza_base": cabeza_base,
        "lora": m_lora,
        "cabeza_lora": cabeza_lora,
        "primer_bloque": 2,
        "clases": CLASES_EUROSAT,
    }


def test_predecir_imagen_top3(modelos_chicos):
    """Comprueba que predecir_imagen devuelve 3 pares con probabilidades decrecientes y suma <= 1."""
    im = Image.new("RGB", (28, 28), color=(100, 150, 200))
    res = predecir_imagen(modelos_chicos, im, device="cpu", img_size=28)

    assert "base" in res and "lora" in res
    for cond in ["base", "lora"]:
        top3 = res[cond]
        assert len(top3) == 3, f"Esperados 3 pares en {cond}, obtenidos {len(top3)}"

        probs = [p for _, p in top3]
        # Probabilidades descendentes
        assert probs[0] >= probs[1] >= probs[2]
        # Suma de top-3 <= 1.0 (con pequeña tolerancia float)
        assert sum(probs) <= 1.0001
        for p in probs:
            assert 0.0 <= p <= 1.0


def test_cargar_modelos_con_checkpoints_sinteticos(tmp_path, modelo_chico, monkeypatch):
    """Comprueba que cargar_modelos funciona cuando los checkpoints están presentes."""
    in_dim = modelo_chico.config.hidden_size * 2
    cabeza = nn.Linear(in_dim, len(CLASES_EUROSAT))

    # Guardar cabeza base y lora
    torch.save(cabeza.state_dict(), tmp_path / "cabeza.pt")
    lora_dict = {
        "cabeza": cabeza.state_dict(),
        "lora": {},
    }
    torch.save(lora_dict, tmp_path / "lora_y_cabeza.pt")

    assert checkpoints_disponibles(str(tmp_path)) is True

    # Monkeypatch cargar_dinov2 para devolver modelo chico
    import app.demo_nucleo as dn
    monkeypatch.setattr(dn, "cargar_dinov2", lambda *a, **kw: copy.deepcopy(modelo_chico))

    modelos = cargar_modelos(str(tmp_path), device="cpu")
    assert "base" in modelos
    assert "lora" in modelos
    assert len(modelos["clases"]) == 10
