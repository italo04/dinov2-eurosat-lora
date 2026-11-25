"""analisis.py — Tablas de resultados, veredicto A4 y diagnósticos."""

import os
import json
import numpy as np
import pandas as pd

from eurosat_lora.metricas import metricas, mcnemar_exacto


def cargar_corridas(salida: str) -> pd.DataFrame:
    """Carga todas las carpetas de corridas en un DataFrame."""
    filas = []
    if not os.path.exists(salida):
        return pd.DataFrame()
    for nombre in os.listdir(salida):
        ruta_m = os.path.join(salida, nombre, "metricas.json")
        if not os.path.exists(ruta_m):
            continue
        with open(ruta_m, encoding="utf-8") as f:
            d = json.load(f)
        for conjunto, m_dict in d.get("metricas", {}).items():
            fila = {
                "id": d["id"],
                "experimento": d["experimento"],
                "particion": d["particion"],
                "condicion": d["condicion"],
                "semilla": d["semilla"],
                "rango": d["rango"],
                "conjunto": conjunto,
                "accuracy": m_dict.get("accuracy"),
                "macro_f1": m_dict.get("macro_f1"),
            }
            filas.append(fila)
    return pd.DataFrame(filas)


def tabla_semillas(df: pd.DataFrame) -> pd.DataFrame:
    """C2: por partición, semilla y condición."""
    return (
        df[df["conjunto"] != "val"]
        .groupby(["particion", "condicion", "semilla", "conjunto"])["macro_f1"]
        .mean()
        .reset_index()
    )


def tabla_mcnemar(salida: str, df: pd.DataFrame | None = None) -> pd.DataFrame:
    """C3: McNemar exacto base vs LoRA r8 por semilla y conjunto."""
    filas = []
    for semilla in [42, 0, 1]:
        for particion in ["interna", "geo"]:
            conjunto_test = "test" if particion == "interna" else "test_fuera"
            id_base = f"{particion}_base_r8_b10-11_q+v_s{semilla}"
            id_lora = f"{particion}_lora_r8_b10-11_q+v_s{semilla}"
            ruta_base = os.path.join(salida, id_base, f"predicciones_{conjunto_test}.csv")
            ruta_lora = os.path.join(salida, id_lora, f"predicciones_{conjunto_test}.csv")
            if not (os.path.exists(ruta_base) and os.path.exists(ruta_lora)):
                continue
            df_b = pd.read_csv(ruta_base)
            df_l = pd.read_csv(ruta_lora)
            if len(df_b) != len(df_l):
                continue
            res = mcnemar_exacto(df_b["real"].values, df_b["predicho"].values, df_l["predicho"].values)
            filas.append({
                "particion": particion, "semilla": semilla, "conjunto": conjunto_test,
                **res
            })
    return pd.DataFrame(filas)


def tabla_fase_c(df: pd.DataFrame) -> pd.DataFrame:
    """C4: Por semilla y condición, macro-F1 en test_dentro y test_fuera, caída y ventaja."""
    geo = df[df["particion"] == "geo"]
    filas = []
    for semilla in geo["semilla"].unique():
        for condicion in ["base", "lora"]:
            sub = geo[(geo["semilla"] == semilla) & (geo["condicion"] == condicion)]
            dentro = sub[sub["conjunto"] == "test_dentro"]["macro_f1"].values
            fuera = sub[sub["conjunto"] == "test_fuera"]["macro_f1"].values
            if len(dentro) == 0 or len(fuera) == 0:
                continue
            f1_dentro = dentro[0]
            f1_fuera = fuera[0]
            filas.append({
                "semilla": semilla,
                "condicion": condicion,
                "f1_dentro": f1_dentro,
                "f1_fuera": f1_fuera,
                "caida": f1_dentro - f1_fuera,
            })
    return pd.DataFrame(filas)


def veredicto_a4(tabla_fc: pd.DataFrame) -> dict:
    """Veredicto según la regla de decisión 5.10 del spec."""
    base = tabla_fc[tabla_fc["condicion"] == "base"]
    lora = tabla_fc[tabla_fc["condicion"] == "lora"]

    if base.empty or lora.empty:
        return {"veredicto": "intermedio", "error": "datos insuficientes"}

    caida_media_base = base["caida"].mean()
    caida_media_lora = lora["caida"].mean()

    # Ventaja LoRA fuera = F1_lora_fuera - F1_base_fuera (por semilla) → media
    semillas = sorted(set(base["semilla"]) & set(lora["semilla"]))
    ventajas = []
    signos = []
    for s in semillas:
        f1b = base[base["semilla"] == s]["f1_fuera"].values
        f1l = lora[lora["semilla"] == s]["f1_fuera"].values
        if len(f1b) > 0 and len(f1l) > 0:
            v = f1l[0] - f1b[0]
            ventajas.append(v)
            signos.append(np.sign(v))
    ventaja_media_fuera = float(np.mean(ventajas)) if ventajas else 0.0
    signo_consistente = len(set(signos)) == 1 if signos else False

    # Regla de 5.10
    if (3.0 <= caida_media_base <= 10.0 and 3.0 <= caida_media_lora <= 10.0
            and ventaja_media_fuera <= 0.5):
        veredicto = "se_cumple"
    elif caida_media_base < 1.0 or caida_media_lora < 1.0 or ventaja_media_fuera >= 1.3:
        veredicto = "se_rechaza"
    else:
        veredicto = "intermedio"

    return {
        "veredicto": veredicto,
        "caida_media_base": float(caida_media_base),
        "caida_media_lora": float(caida_media_lora),
        "ventaja_media_fuera": ventaja_media_fuera,
        "caida_base_media": float(caida_media_base),
        "caida_lora_media": float(caida_media_lora),
        "ventaja_fuera_media": ventaja_media_fuera,
        "signo_consistente": signo_consistente,
    }


def f1_por_clase_geo(salida: str) -> pd.DataFrame:
    """C5: F1 por clase en test_fuera."""
    return pd.DataFrame()  # placeholder; requiere predicciones completas


def f1_por_region(salida: str) -> pd.DataFrame:
    """C6: F1 por región."""
    return pd.DataFrame()  # placeholder


def tabla_ablaciones(df: pd.DataFrame) -> pd.DataFrame:
    """C7: Ablaciones A3/A5/A6 vs r8 en la partición interna, y A3g/A5g/A6g en geográfica."""
    return df[df["experimento"].isin(["A3", "A5", "A6", "A3g", "A5g", "A6g"])]
