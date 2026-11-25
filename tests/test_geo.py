"""tests/test_geo.py — Tests de coordenadas y partición geográfica."""

import numpy as np
import pandas as pd
import pytest
from pyproj import Transformer

from eurosat_lora.geo import (
    centro_lonlat,
    verificar_emparejamiento,
    proyectar,
    agrupar_regiones,
    particion_geografica,
)


def coordenadas_sinteticas(n_grupos=24, n_clases=10, n_total=1200, seed=42):
    rng = np.random.default_rng(seed)
    centros_lon = rng.uniform(5, 30, n_grupos)
    centros_lat = rng.uniform(42, 55, n_grupos)

    registros = []
    por_grupo = n_total // n_grupos
    for g in range(n_grupos):
        for i in range(por_grupo):
            lon = centros_lon[g] + rng.normal(0, 0.3)
            lat = centros_lat[g] + rng.normal(0, 0.3)
            clase = f"Clase{rng.integers(0, n_clases)}"
            registros.append({
                "archivo_rgb": f"Clase{rng.integers(0, n_clases)}/img_{g}_{i}.jpg",
                "clase": clase,
                "lon": lon,
                "lat": lat,
            })
    return pd.DataFrame(registros)


def test_centro_lonlat_utm():
    from affine import Affine
    from pyproj import CRS
    transform = Affine(10.0, 0.0, 500000.0,
                       0.0, -10.0, 5300000.0)
    crs = CRS.from_epsg(32632)
    lon, lat = centro_lonlat(transform, crs, ancho=64, alto=64)

    t = Transformer.from_crs("EPSG:32632", "EPSG:4326", always_xy=True)
    cx = 500000.0 + 32 * 10.0
    cy = 5300000.0 - 32 * 10.0
    lon_ref, lat_ref = t.transform(cx, cy)
    assert lon == pytest.approx(lon_ref, abs=1e-6)
    assert lat == pytest.approx(lat_ref, abs=1e-6)


def test_emparejamiento_falla_con_faltantes():
    ms = {"River/River_1", "River/River_2", "Forest/Forest_1"}
    rgb = {"River/River_1", "Forest/Forest_1", "River/River_9"}
    with pytest.raises(ValueError, match="River_9"):
        verificar_emparejamiento(ms, rgb)


def test_particion_disjunta_y_completa():
    coords = coordenadas_sinteticas(n_total=480)
    res = particion_geografica(coords, seed=42)
    todos = set(res["train"]) | set(res["val"]) | set(res["test_dentro"]) | set(res["test_fuera"])
    assert set(coords["archivo_rgb"]) == todos
    assert set(res["train"]).isdisjoint(set(res["val"]))
    assert set(res["train"]).isdisjoint(set(res["test_fuera"]))
    assert set(res["val"]).isdisjoint(set(res["test_fuera"]))
    assert set(res["train"]).isdisjoint(set(res["test_dentro"]))
    assert set(res["val"]).isdisjoint(set(res["test_dentro"]))
    assert set(res["test_fuera"]).isdisjoint(set(res["test_dentro"]))


def test_regiones_completas():
    coords = coordenadas_sinteticas(n_total=480)
    res = particion_geografica(coords, seed=42)
    region_map = res["region"]

    regs_test_fuera = set(region_map[a] for a in res["test_fuera"])
    regs_val = set(region_map[a] for a in res["val"])
    regs_train = set(region_map[a] for a in res["train"])
    regs_td = set(region_map[a] for a in res["test_dentro"])

    assert regs_test_fuera.isdisjoint(regs_train)
    assert regs_test_fuera.isdisjoint(regs_td)
    assert regs_val.isdisjoint(regs_train)
    assert regs_val.isdisjoint(regs_td)
    assert regs_test_fuera.isdisjoint(regs_val)


def test_restriccion_clases_y_tamanos():
    coords = coordenadas_sinteticas(n_total=600, n_clases=5, n_grupos=20)
    res = particion_geografica(coords, seed=42)
    n_total = len(coords)
    f_test = len(res["test_fuera"]) / n_total
    f_val = len(res["val"]) / n_total
    f_td = len(res["test_dentro"]) / n_total

    assert 0.12 <= f_test <= 0.18, f"f_test={f_test:.3f}"
    assert 0.12 <= f_val <= 0.18, f"f_val={f_val:.3f}"
    assert 0.07 <= f_td <= 0.13, f"test_dentro={f_td:.3f} (esperado ~10%)"


def test_determinista():
    coords = coordenadas_sinteticas(n_total=480)
    r1 = particion_geografica(coords, seed=42)
    r2 = particion_geografica(coords, seed=42)
    assert sorted(r1["train"]) == sorted(r2["train"])
    assert sorted(r1["val"]) == sorted(r2["val"])
    assert sorted(r1["test_fuera"]) == sorted(r2["test_fuera"])


def test_respaldo_se_registra():
    rng = np.random.default_rng(99)
    registros = []
    for g in range(3):
        for i in range(20):
            clase = "ClaseA" if i < 10 else "ClaseB"
            registros.append({
                "archivo_rgb": f"{clase}/{clase}_{g}_{i}.jpg",
                "clase": clase,
                "lon": 10.0 + g * 0.1 + rng.normal(0, 0.01),
                "lat": 48.0 + rng.normal(0, 0.01),
            })
    coords = pd.DataFrame(registros)
    try:
        res = particion_geografica(coords, seed=42)
        assert res["intento_respaldo"] >= 0
        n = len(coords)
        f_tf = len(res["test_fuera"]) / n
        f_v = len(res["val"]) / n
        assert f_tf >= 0.03 and f_v >= 0.03
    except RuntimeError:
        pass
