import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import pathlib
from PIL import Image
import json

ROOT = pathlib.Path(r"c:\Users\italo\Documents\Ciclo 2026-2\Deep Learning\TA\Entregable_2")
FIG_OUT = ROOT / "presentacion_waimlap" / "figures"
FIG_OUT.mkdir(parents=True, exist_ok=True)
SCRATCH_RESULTS = pathlib.Path(r"C:\Users\italo\.gemini\antigravity\brain\33baf598-a1d5-4ede-8b5f-7db59ef93aff\scratch\partner_repo\results")
SCRATCH_FIGS = pathlib.Path(r"C:\Users\italo\.gemini\antigravity\brain\33baf598-a1d5-4ede-8b5f-7db59ef93aff\scratch\partner_repo\figures")

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.edgecolor'] = '#cbd5e1'
plt.rcParams['axes.linewidth'] = 1.2

# ==============================================================================
# FIG 1: DATASET SAMPLES 10-CLASS STRIP (Aesthetic Banner for Card 1)
# ==============================================================================
def make_fig1():
    print("Generating Fig 1: Dataset Samples Strip...")
    img_src = ROOT / "outputs" / "figures" / "00_muestras.png"
    if not img_src.exists():
        img_src = SCRATCH_FIGS / "00_muestras.png"
    
    # We load 00_muestras.png and enhance it with a clean border and title
    img = Image.open(img_src)
    fig, ax = plt.subplots(figsize=(14, 4.5), dpi=300)
    ax.imshow(img)
    ax.axis('off')
    plt.tight_layout(pad=0.2)
    out_p = FIG_OUT / "poster_art_dataset.png"
    plt.savefig(out_p, dpi=300, bbox_inches='tight', facecolor='#ffffff')
    plt.close()
    print("Saved:", out_p)

# ==============================================================================
# FIG 2: METHODOLOGY & TEARDOWN (For Card 3)
# ==============================================================================
def make_fig2():
    print("Generating Fig 2: Architecture & Teardown Breakdown...")
    # Load parameters & flops data
    p_df = pd.read_csv(SCRATCH_RESULTS / "A1_parametros_por_bloque.csv")
    f_df = pd.read_csv(SCRATCH_RESULTS / "A2_flops_por_bloque.csv")

    fig = plt.figure(figsize=(14, 5.2), dpi=300)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.0], wspace=0.25)

    # Panel A: Visual Schematic of ViT-S/14 and LoRA injection
    ax1 = fig.add_subplot(gs[0])
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 8.5)
    ax1.axis('off')

    # Draw Architecture flow
    # Input
    ax1.add_patch(patches.FancyBboxPatch((0.5, 6.8), 9.0, 1.2, boxstyle="round,pad=0.2", fc="#f1f5f9", ec="#94a3b8", lw=1.5))
    ax1.text(5.0, 7.4, "Input: Sentinel-2 RGB (224 x 224 x 3) → Patch Embed: 256 tokens (14x14) + [CLS] token (d=384)", 
             ha="center", va="center", fontsize=10.5, fontweight="bold", color="#0f172a")

    # Frozen Blocks 0-10
    ax1.add_patch(patches.FancyBboxPatch((0.5, 3.8), 9.0, 2.4, boxstyle="round,pad=0.2", fc="#e2e8f0", ec="#64748b", lw=1.5))
    ax1.text(5.0, 5.4, "Blocks 0 to 10 (Frozen DINOv2 Backbone — 19.53M Params)", 
             ha="center", va="center", fontsize=11, fontweight="bold", color="#1e293b")
    ax1.text(5.0, 4.4, "• MSA (6 heads, d_h=64) + MLP (dim 1536)  [Precomputed Embeddings Cache]\n• Global Receptive Field established by Block 4 (r50 = 57.4 px → 67.4 px)", 
             ha="center", va="center", fontsize=9.5, color="#334155")

    # Arrow
    ax1.annotate("", xy=(5.0, 3.2), xytext=(5.0, 3.7), arrowprops=dict(arrowstyle="->", lw=2, color="#059669"))

    # Adapted Blocks 11-12
    ax1.add_patch(patches.FancyBboxPatch((0.5, 0.4), 9.0, 2.6, boxstyle="round,pad=0.2", fc="#ecfdf5", ec="#059669", lw=2.0))
    ax1.text(5.0, 2.3, "Blocks 11 & 12: Nucleus Intervention with LoRA (r=8, α=8)", 
             ha="center", va="center", fontsize=11, fontweight="bold", color="#065f46")
    ax1.text(5.0, 1.4, "• W_adapted = W_0 + (α/r)(B · A) on Query & Value projections\n• 4 layers × 6,144 = 24,576 adapter parameters\n• Trainable Budget: 32,266 params (0.146% of model) | Zero latency upon merge", 
             ha="center", va="center", fontsize=9.5, color="#047857")

    ax1.set_title("(A) DINOv2 (ViT-S/14) Backbone & LoRA Intervention", fontsize=12, fontweight="bold", color="#0f172a", pad=10)

    # Panel B: Parameters & FLOPs by Block
    ax2 = fig.add_subplot(gs[1])
    blocks = [f"B{i}" for i in range(1, 13)]
    mlp_params = [1.182] * 12
    attn_params = [0.591] * 12
    
    # Stacked bar
    b1 = ax2.bar(blocks, attn_params, label="Attention (33%)", color="#3b82f6", width=0.6)
    b2 = ax2.bar(blocks, mlp_params, bottom=attn_params, label="MLP (67%)", color="#0f172a", width=0.6)

    # Highlight blocks 11 and 12 with LoRA badge
    ax2.annotate("LoRA (Q,V)\nr=8, α=8", xy=(10.5, 1.8), xytext=(10.5, 2.15),
                 ha="center", fontsize=9, fontweight="bold", color="#059669",
                 arrowprops=dict(arrowstyle="->", color="#059669", lw=1.5))

    ax2.set_ylabel("Parameters per Block (Millions)", fontsize=10, fontweight="bold", color="#0f172a")
    ax2.set_ylim(0, 2.5)
    ax2.set_title("(B) Parameter Distribution & Target Blocks", fontsize=12, fontweight="bold", color="#0f172a", pad=10)
    ax2.legend(loc="upper left", frameon=True, fontsize=9.5)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    out_p = FIG_OUT / "poster_art_methodology.png"
    plt.savefig(out_p, dpi=300, bbox_inches='tight', facecolor='#ffffff')
    plt.close()
    print("Saved:", out_p)

# ==============================================================================
# FIG 3: COMPLETE RESULTS COMPOSITE (For Card 4)
# ==============================================================================
def make_fig3():
    print("Generating Fig 3: Results Composite (Curves + Per-Class Delta)...")
    # Load F1 per class
    df_f1 = pd.read_csv(SCRATCH_RESULTS / "B2_f1_por_clase.csv")
    df_f1 = df_f1.sort_values(by="delta", ascending=True)

    # Load history
    h_base = pd.read_csv(SCRATCH_RESULTS / "B0_historial_base.csv")
    h_lora = pd.read_csv(SCRATCH_RESULTS / "B0_historial_lora.csv")

    fig = plt.figure(figsize=(14, 5.2), dpi=300)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.2], wspace=0.28)

    # Panel A: Learning Curves (Validation Macro-F1)
    ax1 = fig.add_subplot(gs[0])
    epochs = h_base["epoca"]
    ax1.plot(epochs, h_base["val_macro_f1"] * 100, marker='o', lw=2.2, color="#64748b", label="Linear Probe (Baseline)")
    ax1.plot(epochs, h_lora["val_macro_f1"] * 100, marker='s', lw=2.5, color="#059669", label="DINOv2 + LoRA (Blocks 11-12)")

    ax1.axhline(y=95.80, color="#64748b", linestyle=":", alpha=0.8, label="Test Base: 95.80%")
    ax1.axhline(y=97.14, color="#059669", linestyle="--", alpha=0.8, label="Test LoRA: 97.14% (+1.34%)")

    ax1.set_xlabel("Epoch", fontsize=10.5, fontweight="bold", color="#0f172a")
    ax1.set_ylabel("Validation Macro-F1 (%)", fontsize=10.5, fontweight="bold", color="#0f172a")
    ax1.set_title("(A) Validation Dynamics (8 Epochs, AdamW)", fontsize=12, fontweight="bold", color="#0f172a", pad=10)
    ax1.set_ylim(94.0, 98.2)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right", fontsize=9, frameon=True)

    # Panel B: Per-Class F1 Improvement (Delta Bar Chart)
    ax2 = fig.add_subplot(gs[1])
    classes = df_f1["clase"]
    deltas = df_f1["delta"] * 100

    colors = ["#059669" if d >= 1.0 else ("#10b981" if d > 0 else "#94a3b8") for d in deltas]
    bars = ax2.barh(classes, deltas, color=colors, height=0.65, edgecolor="#0f172a", lw=0.8)

    # Add data labels
    for bar in bars:
        w = bar.get_width()
        x_pos = w + 0.12 if w >= 0 else w - 0.4
        ax2.text(x_pos, bar.get_y() + bar.get_height()/2, f"{w:+.2f}%", 
                 va="center", ha="left" if w >= 0 else "right", 
                 fontsize=9, fontweight="bold", color="#0f172a")

    ax2.set_xlabel("Macro-F1 Gain over Linear Probing (%)", fontsize=10.5, fontweight="bold", color="#0f172a")
    ax2.set_xlim(-0.2, 7.0)
    ax2.set_title("(B) Per-Class Uplift: Highway (+5.85%) & River (+3.71%)", fontsize=12, fontweight="bold", color="#0f172a", pad=10)
    ax2.grid(axis='x', linestyle="--", alpha=0.5)

    plt.tight_layout()
    out_p = FIG_OUT / "poster_art_results.png"
    plt.savefig(out_p, dpi=300, bbox_inches='tight', facecolor='#ffffff')
    plt.close()
    print("Saved:", out_p)

# ==============================================================================
# FIG 4: ERF & TOKEN PROGRESSION (For Card 5)
# ==============================================================================
def make_fig4():
    print("Generating Fig 4: ERF & Features Progression...")
    erf_img_path = SCRATCH_FIGS / "A5_erf.png"
    if not erf_img_path.exists():
        erf_img_path = ROOT / "outputs" / "figures" / "A5_erf.png"
    
    img = Image.open(erf_img_path)
    fig, ax = plt.subplots(figsize=(14, 3.8), dpi=300)
    ax.imshow(img)
    ax.axis('off')
    ax.set_title("Effective Receptive Field (ERF) Expansion across Blocks: r50 = 69.4 px (B1) → 85.0 px (B12)", 
                 fontsize=11.5, fontweight="bold", color="#0f172a", pad=8)
    plt.tight_layout(pad=0.2)
    out_p = FIG_OUT / "poster_art_erf.png"
    plt.savefig(out_p, dpi=300, bbox_inches='tight', facecolor='#ffffff')
    plt.close()
    print("Saved:", out_p)

if __name__ == "__main__":
    make_fig1()
    make_fig2()
    make_fig3()
    make_fig4()
