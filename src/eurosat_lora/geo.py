"""geo.py — Coordenadas, proyección, regiones y partición geográfica."""

import os
from typing import Optional

import numpy as np
import pandas as pd
from pyproj import Transformer
from sklearn.cluster import KMeans
from scipy.spatial import cKDTree

from eurosat_lora.datos import particion_estratificada

EUROSAT_MS_URL = "https://zenodo.org/records/7711810/files/EuroSAT_MS.zip"
RESPALDO = [(12, 0.05), (15, 0.05), (20, 0.05), (20, 0.03)]


def centro_lonlat(transform, crs, ancho: int, alto: int) -> tuple[float, float]:
    """Calcula el centro del tile (lon, lat) a partir del affine transform y CRS."""
    # El centro del tile en coordenadas del raster
    cx = transform.c + (ancho / 2) * transform.a + (alto / 2) * transform.b
    cy = transform.f + (ancho / 2) * transform.d + (alto / 2) * transform.e

    # Proyectar desde el CRS del raster a WGS84
    if hasattr(crs, "to_epsg"):
        epsg = crs.to_epsg()
    else:
        epsg = int(str(crs).split("EPSG:")[-1].strip().rstrip("}").rstrip('"'))

    transformer = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(cx, cy)
    return float(lon), float(lat)


def verificar_emparejamiento(nombres_ms: set, nombres_rgb: set) -> None:
    """Compara conjuntos de nombres (Clase/Clase_N) sin extensión.
    Levanta ValueError con hasta 20 nombres faltantes de cada lado.
    """
    solo_ms = nombres_ms - nombres_rgb
    solo_rgb = nombres_rgb - nombres_ms

    if solo_ms or solo_rgb:
        msg_parts = []
        if solo_ms:
            muestra = sorted(solo_ms)[:20]
            msg_parts.append(f"Solo en MS ({len(solo_ms)}): {muestra}")
        if solo_rgb:
            muestra = sorted(solo_rgb)[:20]
            msg_parts.append(f"Solo en RGB ({len(solo_rgb)}): {muestra}")
        raise ValueError("Emparejamiento MS↔RGB fallido. " + "; ".join(msg_parts))


def proyectar(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Proyecta lon/lat WGS84 a EPSG:3035 (ETRS89-LAEA). Devuelve array (N, 2)."""
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    x, y = transformer.transform(lon, lat)
    return np.column_stack([x, y])


def agrupar_regiones(xy: np.ndarray, k: int, seed: int = 42) -> np.ndarray:
    """Aplica KMeans sobre coordenadas proyectadas. Devuelve etiquetas (N,)."""
    km = KMeans(n_clusters=k, n_init=10, random_state=seed)
    return km.fit_predict(xy)


def asignar_regiones(
    region: np.ndarray,
    y: np.ndarray,
    k: int,
    min_frac_clase: float,
    rango_frac: tuple[float, float] = (0.12, 0.18),
    n_intentos: int = 20_000,
    seed: int = 42,
) -> Optional[dict]:
    """Busca asignación de regiones completas a test_fuera y val.

    Devuelve {'test_fuera': [regiones], 'val': [regiones]} o None si no encuentra.
    Minimiza |f_test-0.15| + |f_val-0.15| entre las válidas.
    """
    rng = np.random.default_rng(seed)
    n_total = len(y)
    regiones_todas = list(range(k))
    clases = np.unique(y)
    lo, hi = rango_frac

    mejor = None
    mejor_coste = float("inf")

    for _ in range(n_intentos):
        # Seleccionar subconjuntos aleatorios de regiones
        perm = rng.permutation(regiones_todas)
        # Partir en test_fuera y val de forma aleatoria
        n_test = rng.integers(1, k - 1)
        n_val = rng.integers(1, k - n_test)
        regs_test = set(perm[:n_test].tolist())
        regs_val = set(perm[n_test:n_test + n_val].tolist())

        if not regs_test or not regs_val or regs_test & regs_val:
            continue

        mask_test = np.isin(region, list(regs_test))
        mask_val = np.isin(region, list(regs_val))
        f_test = mask_test.sum() / n_total
        f_val = mask_val.sum() / n_total

        if not (lo <= f_test <= hi and lo <= f_val <= hi):
            continue

        # Verificar restricción de clases
        ok = True
        for c in clases:
            mask_c = y == c
            n_c = mask_c.sum()
            if n_c == 0:
                continue
            if (mask_test & mask_c).sum() / n_c < min_frac_clase:
                ok = False
                break
            if (mask_val & mask_c).sum() / n_c < min_frac_clase:
                ok = False
                break
        if not ok:
            continue

        coste = abs(f_test - 0.15) + abs(f_val - 0.15)
        if coste < mejor_coste:
            mejor_coste = coste
            mejor = {
                "test_fuera": sorted(regs_test),
                "val": sorted(regs_val),
            }

    return mejor


def particion_geografica(coords: pd.DataFrame, seed: int = 42) -> dict:
    """Genera la partición geográfica completa con respaldo.

    coords: DataFrame con columnas [archivo_rgb, clase, lon, lat]
    Devuelve dict con claves: train, val, test_dentro, test_fuera,
    region (por archivo), k, min_frac_clase, intento_respaldo.
    """
    lon = coords["lon"].values
    lat = coords["lat"].values
    y_str = coords["clase"].values
    # Mapear clases a enteros
    clases_unicas = sorted(set(y_str))
    clase_a_int = {c: i for i, c in enumerate(clases_unicas)}
    y = np.array([clase_a_int[c] for c in y_str])
    xy = proyectar(lon, lat)
    archivos = coords["archivo_rgb"].values

    # Escalar RESPALDO: el primero es k=12 que ya está en RESPALDO
    niveles = [(12, 0.05, 20_000)] + [
        (kr, mf, 20_000) for kr, mf in RESPALDO[1:]
    ]

    intento_respaldo = -1
    for idx_nivel, (k, min_frac, n_intentos) in enumerate(niveles):
        region = agrupar_regiones(xy, k=k, seed=seed)
        asig = asignar_regiones(
            region, y, k=k, min_frac_clase=min_frac,
            n_intentos=n_intentos, seed=seed
        )
        if asig is not None:
            intento_respaldo = idx_nivel
            break
    else:
        raise RuntimeError(
            "No se encontró una partición geográfica válida en ningún nivel de respaldo."
        )

    regs_test_fuera = asig["test_fuera"]
    regs_val = asig["val"]
    regs_restantes = [r for r in range(k) if r not in regs_test_fuera and r not in regs_val]

    mask_test_fuera = np.isin(region, regs_test_fuera)
    mask_val = np.isin(region, regs_val)
    mask_restantes = np.isin(region, regs_restantes)

    # test_dentro: 10% del total, estratificado de regiones de train
    idx_restantes = np.where(mask_restantes)[0]
    y_rest = y[idx_restantes]
    idx_tr_rel, _, idx_td_rel = particion_estratificada(
        y_rest, frac_val=0.0, frac_test=0.1 / mask_restantes.mean(), seed=seed
    )
    idx_train_global = idx_restantes[idx_tr_rel]
    idx_td_global = idx_restantes[idx_td_rel]

    return {
        "train": archivos[idx_train_global].tolist(),
        "val": archivos[mask_val].tolist(),
        "test_dentro": archivos[idx_td_global].tolist(),
        "test_fuera": archivos[mask_test_fuera].tolist(),
        "region": {archivos[i]: int(region[i]) for i in range(len(archivos))},
        "k": k,
        "min_frac_clase": min_frac,
        "intento_respaldo": intento_respaldo,
    }


def distancias_test_train(xy_test: np.ndarray, xy_train: np.ndarray) -> dict:
    """Calcula distancia mínima y mediana en km de test a train más cercano."""
    tree = cKDTree(xy_train)
    dist_m, _ = tree.query(xy_test, k=1)
    dist_km = dist_m / 1000.0
    return {
        "min_km": float(np.min(dist_km)),
        "mediana_km": float(np.median(dist_km)),
    }
