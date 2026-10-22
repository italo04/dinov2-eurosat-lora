# DINOv2 + LoRA para clasificación de cobertura del suelo (EuroSAT RGB)

Proyecto de la tarea académica final de **1INF52 Deep Learning (PUCP, 2026-2)**.

**Pregunta de investigación:** ¿añadir adaptadores LoRA mejora la clasificación en regiones geográficas nuevas frente a usar DINOv2 congelado?

| Fase | Contenido | Estado |
|---|---|---|
| A · Teardown | Parámetros y FLOPs por bloque, latencia y memoria, PCA de tokens, atención CLS, campo receptivo efectivo | E2 |
| B · Intervención | LoRA (r = 8) en Q y V de los bloques 11–12 vs linear probing, mismo presupuesto | E2 (preliminar, partición interna) |
| C · Estrés | Evaluación en regiones geográficas reservadas y análisis de fallos | E3 |

## Resultados del E2 (Colab, Tesla T4, semilla 42)

| Condición | Parámetros entrenables | Accuracy test | Macro-F1 test |
|---|---|---|---|
| Linear probing (DINOv2 congelado) | 7,690 | 0.9600 | 0.9580 |
| LoRA r=8 en Q, V (bloques 11–12) | 32,266 | **0.9728** | **0.9714** |

- LoRA mejora +1.33 pts de macro-F1 en la **partición interna** (aleatoria). Las mayores mejoras están en Highway y River; aparecen nuevas confusiones entre clases de vegetación.
- Teardown: 22.06 M parámetros, 12.25 GFLOPs por imagen a 224 px, 8.9 ms (batch 1, fp32) y 854 img/s (batch 128, fp16) en una T4.
- Detalle, interpretación y limitaciones en `DESIGN_LOG.md`; tablas en `results/` y figuras en `figures/`.

## Estructura

```
├── E2_teardown_lora.ipynb   # notebook principal del E2
├── DESIGN_LOG.md            # cuaderno de decisiones (decisiones, hipótesis, resultados, uso de IA)
├── requirements.txt
├── splits/                  # particiones con semilla fija (se generan al ejecutar)
├── results/                 # tablas CSV/JSON (se generan al ejecutar)
├── figures/                 # figuras PNG (se generan al ejecutar)
└── checkpoints/             # cabeza lineal y adaptadores LoRA (se generan al ejecutar; no incluidos en E2)
```

## Cómo ejecutar

### Opción recomendada: Google Colab (GPU gratuita)
1. Abrir el notebook en Colab: `https://colab.research.google.com/github/jflxx/deep-learning/E2_teardown_lora.ipynb`
2. *Entorno de ejecución → Ejecutar todo*. El notebook descarga EuroSAT RGB (~90 MB, Zenodo) y los pesos de DINOv2 (Hugging Face). Tarda ~30–60 min.
3. Descargar las carpetas `results/`, `figures/`, `splits/` y `checkpoints/` y subirlas al repositorio.

### Opción local
```bash
python -m venv .venv && source .venv/bin/activate   # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook E2_teardown_lora.ipynb
```
Sin GPU NVIDIA, poner `MAX_POR_CLASE = 200` en la celda de configuración (subconjunto de 2,000 imágenes). Las GPU AMD de la serie RX 500 no son compatibles con PyTorch.

## Reproducibilidad
- Semilla fija (`SEMILLA = 42`) en Python, NumPy y PyTorch; cuDNN determinista.
- La partición se guarda en `splits/split_seed42.json` y se reutiliza en todas las ejecuciones.
- Versiones y hardware de cada ejecución en `results/entorno.json`.
- Todos los hiperparámetros están en la celda de configuración del notebook y en `DESIGN_LOG.md`.

## Datos y modelos (licencias)
- **EuroSAT** — Helber, P., Bischke, B., Dengel, A., Borth, D. (2019). *EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification.* IEEE JSTARS 12(7), 2217–2226. [GitHub](https://github.com/phelber/EuroSAT) · [Zenodo](https://zenodo.org/records/7711810). Licencia MIT; imágenes Copernicus Sentinel-2 sujetas a sus términos de uso.
- **DINOv2** — Oquab, M. et al. (2023). *DINOv2: Learning Robust Visual Features without Supervision.* [arXiv:2304.07193](https://arxiv.org/abs/2304.07193). Código y pesos: [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2), licencia Apache-2.0. Se usa la versión de Hugging Face `facebook/dinov2-small`.
- **LoRA** — Hu, E. J. et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* [arXiv:2106.09685](https://arxiv.org/abs/2106.09685). Implementación propia en el notebook.
- **Campo receptivo efectivo** — Luo, W. et al. (2016). *Understanding the Effective Receptive Field in Deep Convolutional Neural Networks.* NeurIPS.

## Uso de asistencia de IA
Se usaron asistentes de IA como herramienta de trabajo. El detalle (qué, dónde y qué revisó el equipo) está en la sección 4 de `DESIGN_LOG.md`.

## Equipo
- Alfredo Jimenez
- Jhonatan Davila
- Italo Mijail
