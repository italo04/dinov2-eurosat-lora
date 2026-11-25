import numpy as np
import pytest
from PIL import Image
from torchvision.datasets import ImageFolder
from eurosat_lora.datos import (
    particion_estratificada,
    archivos_de,
    indices_de,
)

CONTEOS = dict(
    AnnualCrop=3000,
    Forest=3000,
    HerbaceousVegetation=3000,
    Highway=2500,
    Industrial=2500,
    Pasture=2000,
    PermanentCrop=2500,
    Residential=3000,
    River=2500,
    SeaLake=3000,
)


def test_particion_tamanos_e2():
    y = np.concatenate([[i] * n for i, n in enumerate(CONTEOS.values())])
    tr, va, te = particion_estratificada(y, frac_val=0.15, frac_test=0.15, seed=42)
    assert len(tr) == 18_900
    assert len(va) == 4_050
    assert len(te) == 4_050
    assert set(tr).isdisjoint(set(va))
    assert set(tr).isdisjoint(set(te))
    assert set(va).isdisjoint(set(te))
    assert len(set(tr) | set(va) | set(te)) == 27_000


def test_particion_determinista():
    y = np.concatenate([[i] * n for i, n in enumerate(CONTEOS.values())])
    tr1, va1, te1 = particion_estratificada(y, seed=42)
    tr2, va2, te2 = particion_estratificada(y, seed=42)
    assert tr1 == tr2
    assert va1 == va2
    assert te1 == te2


def test_archivos_indices_ida_vuelta(tmp_path):
    for c in ["ClaseA", "ClaseB", "ClaseC"]:
        (tmp_path / c).mkdir()
        for i in range(4):
            img = Image.new("RGB", (64, 64), color=(i * 10, i * 20, i * 30))
            img.save(tmp_path / c / f"{c}_{i}.jpg")

    dataset = ImageFolder(str(tmp_path))
    indices = [0, 2, 5, 8, 11]
    archivos = archivos_de(dataset, indices)
    assert len(archivos) == 5
    assert all("/" in a for a in archivos)
    recuperados = indices_de(dataset, archivos)
    assert recuperados == indices


def test_indices_de_archivo_faltante(tmp_path):
    (tmp_path / "ClaseA").mkdir()
    img = Image.new("RGB", (64, 64))
    img.save(tmp_path / "ClaseA" / "ClaseA_0.jpg")
    dataset = ImageFolder(str(tmp_path))

    with pytest.raises(KeyError):
        indices_de(dataset, ["ClaseA/ClaseA_inexistente.jpg"])
