import copy
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from eurosat_lora.config import Config
from eurosat_lora.entrenamiento import (
    entrenar_base,
    predecir_base,
    entrenar_lora,
    predecir_lora,
)
from eurosat_lora.modelo import aplicar_lora


@pytest.fixture
def datos_sinteticos():
    torch.manual_seed(42)
    x_c0 = torch.randn(20, 3, 28, 28) - 1.5
    y_c0 = torch.zeros(20, dtype=torch.long)
    x_c1 = torch.randn(20, 3, 28, 28) + 1.5
    y_c1 = torch.ones(20, dtype=torch.long)
    x = torch.cat([x_c0, x_c1], dim=0)
    y = torch.cat([y_c0, y_c1], dim=0)

    idx_tr = list(range(15)) + list(range(20, 35))
    idx_va = list(range(15, 20)) + list(range(35, 40))

    dl_tr = DataLoader(TensorDataset(x[idx_tr], y[idx_tr]), batch_size=8, shuffle=True)
    dl_va = DataLoader(TensorDataset(x[idx_va], y[idx_va]), batch_size=8, shuffle=False)
    return dl_tr, dl_va, x[idx_va], y[idx_va]


def test_entrenar_base_historial_y_seleccion():
    torch.manual_seed(42)
    X_tr = torch.randn(30, 64)
    y_tr = torch.tensor([0] * 15 + [1] * 15, dtype=torch.long)
    X_va = torch.randn(10, 64)
    y_va = torch.tensor([0] * 5 + [1] * 5, dtype=torch.long)
    cache = {"train": (X_tr, y_tr), "val": (X_va, y_va)}

    cfg = Config(epocas=2, batch=8)
    cabeza, hist = entrenar_base(cache, n_clases=2, cfg=cfg, seed=42)

    assert len(hist) == 2
    assert set(hist.columns) == {"epoca", "loss_train", "val_accuracy", "val_macro_f1"}
    preds_va, _ = predecir_base(cabeza, X_va)
    from eurosat_lora.metricas import metricas
    f1_actual = metricas(y_va.numpy(), preds_va)["macro_f1"]
    assert f1_actual == pytest.approx(hist["val_macro_f1"].max(), abs=1e-5)


def test_entrenar_lora_restaura_mejor(modelo_chico, datos_sinteticos):
    dl_tr, dl_va, x_va, y_va = datos_sinteticos
    attn_layer = modelo_chico.encoder.layer[2].attention
    attn_sub = attn_layer.attention if hasattr(attn_layer, "attention") else attn_layer
    q_name = "query" if hasattr(attn_sub, "query") else "q_proj"
    w_query_orig = getattr(attn_sub, q_name).weight.clone()

    aplicar_lora(modelo_chico, bloques=(2, 3), proyecciones=("q", "v"), r=4, alpha=4.0)
    cfg = Config(epocas=2, batch=8, lora_bloques=(2, 3))
    cabeza, hist, pesos = entrenar_lora(modelo_chico, dl_tr, dl_va, n_clases=2, cfg=cfg, seed=42)

    preds_va, _ = predecir_lora(modelo_chico, cabeza, dl_va, primer_bloque=2)
    from eurosat_lora.metricas import metricas
    f1_final = metricas(y_va.numpy(), preds_va)["macro_f1"]
    assert f1_final == pytest.approx(hist["val_macro_f1"].max(), abs=1e-5)
    lora_layer = getattr(attn_sub, q_name)
    assert torch.equal(lora_layer.base.weight, w_query_orig)


def test_entrenar_lora_determinista(modelo_chico, datos_sinteticos):
    _, dl_va, _, _ = datos_sinteticos
    torch.manual_seed(42)
    x_c0 = torch.randn(15, 3, 28, 28) - 1.5
    x_c1 = torch.randn(15, 3, 28, 28) + 1.5
    x = torch.cat([x_c0, x_c1])
    y = torch.tensor([0] * 15 + [1] * 15, dtype=torch.long)

    m1 = copy.deepcopy(modelo_chico)
    aplicar_lora(m1, bloques=(2, 3), proyecciones=("q", "v"), r=4, alpha=4.0)
    m2 = copy.deepcopy(m1)

    cfg = Config(epocas=1, batch=8, lora_bloques=(2, 3))

    g1 = torch.Generator(); g1.manual_seed(0)
    dl_tr_seed1 = DataLoader(TensorDataset(x, y), batch_size=8, shuffle=True, generator=g1)
    _, _, p1 = entrenar_lora(m1, dl_tr_seed1, dl_va, n_clases=2, cfg=cfg, seed=0)

    g2 = torch.Generator(); g2.manual_seed(0)
    dl_tr_seed2 = DataLoader(TensorDataset(x, y), batch_size=8, shuffle=True, generator=g2)
    _, _, p2 = entrenar_lora(m2, dl_tr_seed2, dl_va, n_clases=2, cfg=cfg, seed=0)

    for k in p1["lora"]:
        assert torch.allclose(p1["lora"][k], p2["lora"][k]), f"Pesos divergen en {k}"


def test_predecir_devuelve_prob_en_0_1(modelo_chico, datos_sinteticos):
    _, dl_va, _, y_va = datos_sinteticos
    aplicar_lora(modelo_chico, bloques=(2, 3), proyecciones=("q", "v"), r=4, alpha=4.0)
    cabeza = torch.nn.Linear(modelo_chico.config.hidden_size * 2, 2)
    preds, probs = predecir_lora(modelo_chico, cabeza, dl_va, primer_bloque=2)

    assert len(preds) == len(y_va)
    assert len(probs) == len(y_va)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
