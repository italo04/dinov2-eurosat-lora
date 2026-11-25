import numpy as np
import pytest
from scipy.stats import binomtest
from eurosat_lora.metricas import metricas, mcnemar_exacto


def test_mcnemar_conteos():
    y = np.array([0, 0, 0, 0, 1, 1])
    p1 = np.array([0, 0, 1, 1, 1, 0])
    p2 = np.array([0, 1, 0, 0, 1, 1])
    res = mcnemar_exacto(y, p1, p2)
    assert res["b"] == 1
    assert res["c"] == 3
    assert res["n_discordantes"] == 4


def test_mcnemar_valor_conocido():
    # Creamos un caso con b=10 y c=30
    # y=0 para todos, p1 acierta 10 donde p2 falla, p2 acierta 30 donde p1 falla
    y = np.zeros(40, dtype=int)
    p1 = np.array([0] * 10 + [1] * 30)
    p2 = np.array([1] * 10 + [0] * 30)
    res = mcnemar_exacto(y, p1, p2)
    assert res["b"] == 10
    assert res["c"] == 30
    esperado = binomtest(10, 40, 0.5).pvalue
    assert res["p_valor"] == pytest.approx(esperado)


def test_mcnemar_sin_discordantes():
    y = np.array([0, 1, 2, 3])
    p1 = np.array([0, 1, 2, 3])
    p2 = np.array([0, 1, 2, 3])
    res = mcnemar_exacto(y, p1, p2)
    assert res["n_discordantes"] == 0
    assert res["p_valor"] == 1.0


def test_metricas_perfectas():
    y = np.array([0, 1, 2, 3, 4])
    p = np.array([0, 1, 2, 3, 4])
    res = metricas(y, p)
    assert res["accuracy"] == 1.0
    assert res["macro_f1"] == 1.0
