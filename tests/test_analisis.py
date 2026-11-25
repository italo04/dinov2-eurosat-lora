"""tests/test_analisis.py — Tests de veredicto A4 y norma de tokens."""

import numpy as np
import pandas as pd
import pytest

from eurosat_lora.analisis import veredicto_a4
from eurosat_lora.diagnosticos import norma_tokens


def _tabla(semillas, f1_base_dentro, f1_base_fuera, f1_lora_dentro, f1_lora_fuera):
    filas = []
    for s, bd, bf, ld, lf in zip(semillas, f1_base_dentro, f1_base_fuera, f1_lora_dentro, f1_lora_fuera):
        filas.append({"semilla": s, "condicion": "base", "f1_dentro": bd, "f1_fuera": bf, "caida": bd - bf})
        filas.append({"semilla": s, "condicion": "lora", "f1_dentro": ld, "f1_fuera": lf, "caida": ld - lf})
    return pd.DataFrame(filas)


def test_a4_se_cumple():
    t = _tabla([42, 0, 1], [80, 81, 79], [75, 75, 76], [81, 82, 80], [75.5, 75.3, 76.0])
    res = veredicto_a4(t)
    assert res["veredicto"] == "se_cumple"


def test_a4_rechazo_sin_caida():
    t = _tabla([42, 0, 1], [80, 80, 80], [79.5, 79.7, 79.8], [81, 81, 81], [80, 80, 80])
    res = veredicto_a4(t)
    assert res["veredicto"] == "se_rechaza"


def test_a4_rechazo_ventaja():
    t = _tabla([42, 0, 1], [80, 80, 80], [70, 70, 70], [82, 82, 82], [71.5, 71.4, 71.3])
    res = veredicto_a4(t)
    assert res["veredicto"] == "se_rechaza"


def test_a4_intermedio():
    t = _tabla([42, 0, 1], [80, 80, 80], [78, 78, 78], [82, 82, 82], [78.3, 78.3, 78.3])
    res = veredicto_a4(t)
    assert res["veredicto"] == "intermedio"


def test_a4_signo_inconsistente():
    t = _tabla([42, 0, 1], [82, 82, 82], [76, 76, 76], [83, 81, 82.5], [76.4, 75.7, 76.2])
    res = veredicto_a4(t)
    assert res["signo_consistente"] is False


def test_norma_tokens_forma(modelo_chico):
    x = torch.randn(2, 3, 28, 28)
    normas = norma_tokens(modelo_chico, x)
    assert normas.shape == (2, 5)
