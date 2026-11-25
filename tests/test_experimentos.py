"""tests/test_experimentos.py — Tests de la matriz de experimentos."""

import json
import os
import copy
import pytest
import torch
from torch.utils.data import TensorDataset, DataLoader
from PIL import Image

from eurosat_lora.config import Config
from eurosat_lora.experimentos import (
    MATRIZ_E3,
    Corrida,
    ejecutar_corrida,
    verificar_predicciones,
    Contexto,
)
from eurosat_lora.modelo import aplicar_lora


def test_matriz_e3():
    """20 corridas, ids únicos, 14 LoRA + 6 base, 12 con semillas {42,0,1}."""
    assert len(MATRIZ_E3) == 20
    ids = [c.id for c in MATRIZ_E3]
    assert len(ids) == len(set(ids)), "IDs duplicados en MATRIZ_E3"

    n_lora = sum(1 for c in MATRIZ_E3 if c.condicion == "lora")
    n_base = sum(1 for c in MATRIZ_E3 if c.condicion == "base")
    assert n_lora == 14, f"Esperadas 14 corridas LoRA, hay {n_lora}"
    assert n_base == 6, f"Esperadas 6 corridas base, hay {n_base}"

    n_con_tres_semillas = sum(
        1 for c in MATRIZ_E3 if c.semilla in {42, 0, 1}
        and c.rango == 8 and set(c.bloques) == {10, 11} and set(c.proyecciones) == {"q", "v"}
    )
    assert n_con_tres_semillas == 12, f"Esperadas 12 con semillas {{42,0,1}}, hay {n_con_tres_semillas}"


@pytest.fixture
def ctx_chico(tmp_path, modelo_chico):
    """Contexto sintético con dataset ImageFolder y modelo chico."""
    # Crear dataset ImageFolder mínimo
    from torchvision.datasets import ImageFolder
    for cls in ["ClaseA", "ClaseB"]:
        (tmp_path / cls).mkdir()
        for i in range(20):
            img = Image.new("RGB", (28, 28), color=(i * 5, i * 10, i * 3))
            img.save(tmp_path / cls / f"{cls}_{i}.jpg")

    from torchvision import transforms
    tf = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])
    dataset = ImageFolder(str(tmp_path), transform=tf)
    clases = dataset.classes
    n = len(dataset)
    # Partición mínima
    idx_all = list(range(n))
    n_tr = int(n * 0.6)
    n_va = int(n * 0.2)
    idx_tr = idx_all[:n_tr]
    idx_va = idx_all[n_tr:n_tr + n_va]
    idx_te = idx_all[n_tr + n_va:]

    from eurosat_lora.datos import archivos_de
    particion_interna = {
        "train": archivos_de(dataset, idx_tr),
        "val": archivos_de(dataset, idx_va),
        "test": archivos_de(dataset, idx_te),
    }
    cfg = Config(epocas=1, batch=8, max_por_clase=10)
    ctx = Contexto(
        dataset=dataset,
        clases=clases,
        particiones={"interna": particion_interna},
        cfg=cfg,
        device="cpu",
        amp=False,
    )
    return ctx


def test_corrida_idempotente(tmp_path, ctx_chico, modelo_chico, monkeypatch):
    """1ª llamada crea metricas.json; 2ª no reentrena."""
    corrida = Corrida.crear("A2", "interna", "base", 42)
    salida = str(tmp_path / "salida")

    # Monkeypatch cargar_dinov2 para que devuelva el modelo chico
    import eurosat_lora.experimentos as exp_mod
    monkeypatch.setattr(exp_mod, "cargar_dinov2", lambda *a, **kw: copy.deepcopy(modelo_chico))

    res1 = ejecutar_corrida(corrida, ctx_chico, salida)
    assert os.path.exists(os.path.join(salida, corrida.id, "metricas.json"))

    # Monkeypatch entrenar_base para que falle en 2ª llamada
    monkeypatch.setattr(exp_mod, "entrenar_base", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("no debería llamarse")))

    res2 = ejecutar_corrida(corrida, ctx_chico, salida)
    assert res1["id"] == res2["id"]


def test_corrida_parcial_se_rehace(tmp_path, ctx_chico, modelo_chico, monkeypatch):
    """Carpeta con historial.csv pero sin metricas.json → se rehace."""
    corrida = Corrida.crear("A2", "interna", "base", 0)
    salida = str(tmp_path / "salida")
    carpeta = os.path.join(salida, corrida.id)
    os.makedirs(carpeta, exist_ok=True)
    # Crear historial sin metricas.json (simulando carpeta parcial)
    with open(os.path.join(carpeta, "historial.csv"), "w") as f:
        f.write("epoca,loss_train\n1,0.5\n")

    import eurosat_lora.experimentos as exp_mod
    monkeypatch.setattr(exp_mod, "cargar_dinov2", lambda *a, **kw: copy.deepcopy(modelo_chico))

    res = ejecutar_corrida(corrida, ctx_chico, salida)
    assert "id" in res
    assert os.path.exists(os.path.join(carpeta, "metricas.json"))


def test_verificar_predicciones_falta(tmp_path):
    """Log sin '| A6 |' → RuntimeError con 'A6'."""
    log = tmp_path / "DESIGN_LOG.md"
    log.write_text("| A5 |\n| alguna fila |\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="A6"):
        verificar_predicciones({"A5", "A6"}, ruta_log=str(log))
