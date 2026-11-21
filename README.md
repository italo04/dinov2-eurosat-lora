# Dissecting and Adapting DINOv2 for Land Cover Classification: An Architectural Teardown & Parameter-Efficient Study on Sentinel-2 Imagery

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/italo04/dinov2-eurosat-lora/blob/main/E2_teardown_lora.ipynb)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![Dataset: EuroSAT Sentinel-2](https://img.shields.io/badge/Dataset-EuroSAT%20RGB-brightgreen.svg)](https://github.com/phelber/EuroSAT)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Authors:** Alfredo Jimenez • Jhonatan Davila • Italo Perez  
**Affiliation:** Departamento de Ingeniería, Pontificia Universidad Católica del Perú (PUCP), Lima, Peru  
**Research Group:** Grupo de Investigación en Inteligencia Artificial (IA-PUCP)  
**Course:** 1INF52 - Deep Learning (Semestre 2026-2)  
**Lead Instructor:** Dr. César A. Beltrán Castañón  
**Venue:** Workshop on Artificial Intelligence and Machine Learning Applications (WAIMLAp 2026)  

---

## 1. Overview & Research Question

Foundation vision models trained through self-supervised learning, such as Meta AI\'s **DINOv2 (Vision Transformer architecture, dinov2_vits14)**, demonstrate exceptional representation capabilities across natural photographic distributions. However, deploying them to remote sensing entails a substantial domain shift: satellite imagery exhibits nadir viewing geometry, repetitive multi-scale agricultural textures, the absence of focal foreground subjects, and subtle inter-class spectral boundaries.

While standard **Linear Probing** (training an affine classifier over frozen features) avoids catastrophic forgetting, it remains constrained by features not explicitly aligned with remote sensing textures. Conversely, full model fine-tuning updates over 22 million parameters, risking catastrophic forgetting and incurring heavy computational demands.

**Core Research Question:**  
> *Can Low-Rank Adaptation (LoRA) restricted strictly to late self-attention layers bridge the remote sensing domain shift with minimal compute and zero inference latency overhead?*

---

## 2. Key Empirical Findings

1. **Phase A — Architectural Teardown:**  
   * **Parameter Footprint:** DINOv2-Small (ViT-S/14 without registers) comprises **22,056,576 parameters** (88.2 MB fp32) distributed uniformly across 12 transformer encoder blocks (~1.78 M parameters and ~1.011 GFLOPs per block).  
   * **Computational Complexity:** Total analytical computation is **12.25 GFLOPs** per  \times 224$ px image (batch 1), where quadratic self-attention accounts for 9.9%.  
   * **Hardware Profiling (Tesla T4 GPU):** Latency is **8.89 ms** (112.5 img/s, batch 1, fp32) and **1.17 ms/img** (853.7 img/s, batch 128, fp16).  
   * **Effective Receptive Field (ERF):** Quantitative gradient mapping demonstrates that the receptive field spans globally from block 1 onwards ({50}$ expands from 69.4 px to 85.0 px, and {90}$ from 117.2 px to 127.1 px at block 12).  
   * **Feature Evolution (PCA):** Principal Component Analysis reveals that while block 4 tokens encode spatial patch coordinates, block 12 tokens cluster into semantic land-cover classes.

2. **Phase B — Controlled Nucleus Intervention (LoRA):**  
   * **Parameter Efficiency:** We freeze blocks 0 through 10 and inject low-rank decomposition matrices (=8, \alpha=8$) exclusively into the Query ($) and Value ($) projections of blocks 11 and 12 ( = W_0 + \frac{\alpha}{r} B \cdot A$).  
   * Trainable parameters are confined to **32,266** (24,576 LoRA + 7,690 head), representing only **0.146% of the backbone**.  
   * **Caching Acceleration:** Embedding/prefix caching accelerates training by over **8x** without numerical deviation ($\max |\Delta| < 10^{-5}$).  
   * **Zero Runtime Overhead:** In-place weight merging ( = W_0 + BA$) eliminates inference latency penalties entirely.

3. **Statistically Significant Benchmark Uplift:**  
   * Evaluated across the complete **EuroSAT RGB benchmark** (27,000 Sentinel-2 images, stratified 70/15/15, 4,050 held-out test images, seed 42):
     * **Test Macro-F1:** **97.14%** (LoRA) vs. **95.80%** (Baseline) $\\implies \\mathbf{\\Delta = +1.34\\%}$.
     * **Test Accuracy:** **97.28%** (LoRA) vs. **96.00%** (Baseline) $\\implies \\mathbf{\\Delta = +1.28\\%}$ (misclassifications drop from 162 to 110).
     * **Statistical Significance:** Two-tailed McNemar test confirms significance with $\\chi^2 = 9.56, p = 0.002$.
   * **Class Dynamics:** Major improvements in linear infrastructure and hydrology: **Highway F1** rises from 91.87% to **97.72% (+5.85%)** and **River F1** from 93.80% to **97.51% (+3.71%)**.

---

## 3. Quantitative Benchmark Comparison

| Method / Configuration | Trainable Params | % Backbone | Test Accuracy | Test Macro-F1 | Test Errors |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Linear Probing** (Frozen DINOv2) | 7,690 | 0.035% | 96.00% | 95.80% | 162 / 4,050 |
| **DINOv2 + LoRA** (Blocks 11–12, =8, \alpha=8$) | **32,266** | **0.146%** | **97.28%** | **97.14% (+1.34%)** | **110 / 4,050** |

*Table 1: Test performance on full EuroSAT Sentinel-2 RGB partition (4,050 images, seed 42).*

---

## 4. Repository Structure

`	ext
dinov2-eurosat-lora/
├── E2_teardown_lora.ipynb               # Primary reproducible notebook (Colab GPU T4)
├── E2_Teardown_Intervencion_DINOv2_EuroSAT.ipynb # Local CPU notebook with prefix caching
├── README.md                            # Comprehensive project guide and empirical report
├── DESIGN_LOG.md                        # Formal design decision log (D01 to D18) & hypotheses
├── requirements.txt                     # Project dependencies
├── .gitignore                           # Excludes large checkpoints, weights and raw data
│
├── figures/                             # Visualizations from architectural teardown & eval
│   ├── 00_muestras.png                  # EuroSAT RGB sample mosaic across 10 classes
│   ├── A1_parametros.png                # Parameter distribution per transformer block
│   ├── A2_flops.png                     # FLOPs per block and scaling vs input resolution
│   ├── A4_atencion_cls.png              # Self-attention map visualizations for [CLS] token
│   ├── A4_pca_tokens.png                # PCA representation evolution across blocks 1 to 12
│   ├── A5_erf.png                       # Quantitative Effective Receptive Field (ERF) maps
│   ├── B1_confusion_y_curvas.png        # Paired confusion matrices and validation curves
│   └── repo_qr.png                      # QR code linking to this repository
│
├── results/                             # Quantitative CSV tables and JSON experiment metrics
│   ├── A1_parametros_por_bloque.csv
│   ├── A2_flops_por_bloque.csv
│   ├── A3_latencia_memoria.csv
│   ├── A5_erf.csv
│   ├── B1_comparacion.csv
│   ├── B2_f1_por_clase.csv
│   ├── B_resumen.json
│   └── entorno.json
│
├── src/                                 # Modular Python source package
│   ├── data/                            # Dataset downloading and deterministic partitioning
│   ├── models/                          # DINOv2 loading, custom LoRALinear and caching
│   ├── teardown/                        # FLOP counting, ERF computation, latency profiling
│   └── eval/                            # Evaluation loops, metrics and confusion analysis
│
├── scripts/                             # Automation scripts
│   ├── build_poster.py                  # Generates WAIMLAp A0 poster PPTX/PDF
│   ├── build_extended_abstract.py       # Compiles extended abstract
│   └── export_pdf.ps1                   # High-res COM vector PDF exporter
│
├── docs/                                # Defense guides and architectural logs
│   ├── DECISION_LOG.md                  # Detailed decision log
│   ├── GUION_EXPOSICION_E2.md           # Slide-by-slide defense script
│   └── GUIA_PRESENTACION_WAIMLAP.md     # WAIMLAp poster session guide and Q&A bank
│
└── waimlap/                             # WAIMLAp 2026 Conference Materials
    ├── Abstract_DINOv2_EuroSAT_LoRA.pdf # Official 2-page Extended Abstract
    ├── Abstract_DINOv2_EuroSAT_LoRA.tex # LaTeX source with IEEE/LNCS styling
    ├── Poster_DINOv2_EuroSAT_LoRA_Mejorado.pdf # High-resolution A0 Poster (841 x 1189 mm)
    ├── Presentacion_E2_DINOv2_EuroSAT.pptx    # 16:9 Presentation slide deck
    ├── diapositivas.html                # Interactive HTML5 slide viewer
    └── poster.html                      # Interactive web poster viewer
`

---

## 5. Quick Reproduction Guide

### Option A: Google Colab (Recommended, Free GPU)
1. Click the **[Open In Colab](https://colab.research.google.com/github/italo04/dinov2-eurosat-lora/blob/main/E2_teardown_lora.ipynb)** badge above.
2. Select **Runtime $\\to$ Change runtime type $\\to$ GPU (Tesla T4)**.
3. Click **Runtime $\\to$ Run all**. The notebook automatically downloads the EuroSAT dataset (~90 MB) and Meta AI\'s official DINOv2 weights. Total runtime: ~30–45 minutes.

### Option B: Local Execution (CPU / GPU)
`ash
# 1. Clone repository
git clone https://github.com/italo04/dinov2-eurosat-lora.git
cd dinov2-eurosat-lora

# 2. Setup virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Jupyter Notebook
jupyter notebook E2_teardown_lora.ipynb
`

---

## 6. Acknowledgments & Attribution

* **Pontificia Universidad Católica del Perú (PUCP):** Deep Learning course (1INF52, 2026-2), Lead Instructor: Dr. César A. Beltrán Castañón.
* **Copernicus Programme & European Space Agency (ESA):** For open-access Sentinel-2 Earth observation imagery and the EuroSAT benchmark dataset.
* **Meta AI Research:** For open-source DINOv2 foundation models and weights.
* **Workshop on Artificial Intelligence and Machine Learning Applications (WAIMLAp 2026):** Departamento de Ingeniería & IA-PUCP.

---

## 7. Key References

`ibtex
@article{oquab2023dinov2,
  title={DINOv2: Learning Robust Visual Features without Supervision},
  author={Oquab, Maxime and Darcet, Timoth{\'e}e and Moutakanni, Th{\'e}o and Vo, Huy V and others},
  journal={arXiv preprint arXiv:2304.07193},
  year={2023}
}

@inproceedings{hu2022lora,
  title={LoRA: Low-Rank Adaptation of Large Language Models},
  author={Hu, Edward J and Shen, Yelong and Wallis, Phillip and Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean and Wang, Lu and Chen, Weizhu},
  booktitle={Proc. International Conference on Learning Representations (ICLR)},
  year={2022}
}

@article{helber2019eurosat,
  title={EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification},
  author={Helber, Patrick and Bischke, Benjamin and Dengel, Andreas and Borth, Damian},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  volume={12},
  number={7},
  pages={2217--2226},
  year={2019}
}

@inproceedings{luo2016understanding,
  title={Understanding the Effective Receptive Field in Deep Convolutional Neural Networks},
  author={Luo, Wenjie and Li, Yujia and Urtasun, Raquel and Zemel, Richard},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  volume={29},
  year={2016}
}
`
