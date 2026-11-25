import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score


def metricas(y: np.ndarray, p: np.ndarray) -> dict:
    y_arr = np.asarray(y)
    p_arr = np.asarray(p)
    return {
        "accuracy": float(accuracy_score(y_arr, p_arr)),
        "macro_f1": float(f1_score(y_arr, p_arr, average="macro", zero_division=0)),
    }


def f1_por_clase(y: np.ndarray, p: np.ndarray, clases: list[str]) -> pd.Series:
    y_arr = np.asarray(y)
    p_arr = np.asarray(p)
    scores = f1_score(y_arr, p_arr, average=None, zero_division=0)
    return pd.Series(scores, index=clases)


def mcnemar_exacto(y: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> dict:
    y_arr = np.asarray(y)
    p1_arr = np.asarray(p1)
    p2_arr = np.asarray(p2)

    b = int(np.sum((p1_arr == y_arr) & (p2_arr != y_arr)))
    c = int(np.sum((p1_arr != y_arr) & (p2_arr == y_arr)))
    n_discordantes = b + c

    if n_discordantes == 0:
        p_valor = 1.0
    else:
        p_valor = float(binomtest(b, n_discordantes, 0.5).pvalue)

    return {
        "b": b,
        "c": c,
        "n_discordantes": n_discordantes,
        "p_valor": p_valor,
    }
