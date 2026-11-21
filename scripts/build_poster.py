import pptx
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from PIL import Image
import pathlib
import os

ROOT = pathlib.Path(r"c:\Users\italo\Documents\Ciclo 2026-2\Deep Learning\TA\Entregable_2")
TEMPLATE_PATH = ROOT / "templates" / "template_poster.pptx"
OUTPUT_PPTX_WAIMLAP = ROOT / "presentacion_waimlap" / "Poster_DINOv2_EuroSAT_LoRA_Mejorado.pptx"
OUTPUT_PPTX_ROOT = ROOT / "Poster_DINOv2_EuroSAT_LoRA_Mejorado.pptx"
FIGS_DIR = ROOT / "outputs" / "figures"

prs = pptx.Presentation(str(TEMPLATE_PATH))
slide = prs.slides[0]

def set_tf_text(tf, paragraphs_data):
    tf.clear()
    for i, pdata in enumerate(paragraphs_data):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(pdata.get("space_after", 4))
        p.space_before = Pt(pdata.get("space_before", 0))
        p.alignment = pdata.get("align", PP_ALIGN.LEFT)
        r = p.add_run()
        r.text = pdata["text"]
        r.font.name = "Times New Roman"
        r.font.size = Pt(pdata.get("size", 15))
        r.font.bold = pdata.get("bold", False)
        c = pdata.get("color", (30, 41, 59))
        r.font.color.rgb = RGBColor(*c)

def add_fitted_picture(slide, img_path, left_in, top_in, max_w_in, max_h_in):
    """
    Guarantees PROPORTIONAL scaling with zero distortion,
    centering the image within the bounding box.
    """
    img = Image.open(str(img_path))
    img_w, img_h = img.size
    aspect = img_w / img_h
    box_aspect = max_w_in / max_h_in

    if aspect > box_aspect:
        # Constrained by width
        w = max_w_in
        h = max_w_in / aspect
        x = left_in
        y = top_in + (max_h_in - h) / 2
    else:
        # Constrained by height
        h = max_h_in
        w = max_h_in * aspect
        x = left_in + (max_w_in - w) / 2
        y = top_in

    pic = slide.shapes.add_picture(str(img_path), Inches(x), Inches(y), Inches(w), Inches(h))
    return pic

# ==============================================================================
# HEADER: TITLE, AUTHORS, AFFILIATION
# ==============================================================================
shape_title = slide.shapes[2]
set_tf_text(shape_title.text_frame, [
    {
        "text": "Dissecting and Adapting DINOv2 for Land Cover Classification:\nAn Architectural Teardown and Parameter-Efficient Study on Sentinel-2 Imagery",
        "size": 38,
        "bold": True,
        "color": (255, 255, 255),
        "align": PP_ALIGN.CENTER,
        "space_after": 0
    }
])

shape_authors = slide.shapes[3]
shape_authors.left = Inches(1.5)
shape_authors.width = Inches(30.1)
set_tf_text(shape_authors.text_frame, [
    {
        "text": "Alfredo Jimenez  •  Jhonatan Davila  •  Italo Perez",
        "size": 22,
        "bold": True,
        "color": (224, 231, 255),
        "align": PP_ALIGN.CENTER,
        "space_after": 4
    },
    {
        "text": "Grupo de Investigación en Inteligencia Artificial (IA-PUCP)  |  Departamento de Ingeniería, PUCP  |  Contact: italo.perez@pucp.edu.pe",
        "size": 16,
        "bold": False,
        "color": (203, 213, 225),
        "align": PP_ALIGN.CENTER,
        "space_after": 0
    }
])

# Replace template placeholder banner (Rounded Rectangle 7) with executive summary highlights
shape_logos = slide.shapes[4]
shape_logos.fill.solid()
shape_logos.fill.fore_color.rgb = RGBColor(241, 245, 249)
shape_logos.line.color.rgb = RGBColor(203, 213, 225)
shape_logos.line.width = Pt(1.5)
shape_logos.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
set_tf_text(shape_logos.text_frame, [
    {
        "text": "DATASET: Sentinel-2 EuroSAT (27,000 images, 10 LULC classes)   •   MODEL: Meta AI DINOv2 ViT-S/14 (22.06M params, 12.25 GFLOPs)",
        "size": 16,
        "bold": True,
        "color": (30, 58, 138),
        "align": PP_ALIGN.CENTER,
        "space_after": 4
    },
    {
        "text": "INTERVENTION: Nucleus LoRA (Q & V in blocks 11–12, r=8, α=8, 0.146% params)   •   IMPACT: +1.34% Test Macro-F1 (p = 0.002, Tesla T4)",
        "size": 16,
        "bold": True,
        "color": (5, 150, 105),
        "align": PP_ALIGN.CENTER,
        "space_after": 0
    }
])

# Remove template instruction banner (TextBox 41)
for s in slide.shapes:
    if s.has_text_frame and "A0 PORTRAIT" in s.text_frame.text:
        s.text_frame.clear()

# ==============================================================================
# CARD 1: MOTIVATION & RESEARCH PROBLEM (Left, Top)
# Card top = 7.18", height = 6.00", left = 1.20", width = 15.03"
# ==============================================================================
shape_c1 = slide.shapes[8]
shape_c1.top = Inches(8.0)
shape_c1.height = Inches(2.1)
set_tf_text(shape_c1.text_frame, [
    {"text": "• Foundation Model Paradigm: DINOv2 acquires versatile visual features from web-scale photos, but satellite imagery (Sentinel-2 EuroSAT) exhibits nadir geometry, repetitive spatial textures, and absence of central foreground objects.", "size": 13.5, "bold": False, "space_after": 3},
    {"text": "• Linear Probing vs Fine-Tuning: Static linear probes cannot adapt representations to remote sensing spatial distributions, while full fine-tuning modifies 22M+ parameters, risking catastrophic forgetting.", "size": 13.5, "bold": False, "space_after": 3},
    {"text": "• Core Question: Can low-rank adaptation (LoRA) restricted strictly to late self-attention layers bridge the domain shift with minimal compute and zero inference latency overhead?", "size": 13.5, "bold": True, "color": (15, 23, 42), "space_after": 0}
])
# Embed 00_muestras.png directly from notebook
add_fitted_picture(slide, FIGS_DIR / "00_muestras.png", 1.5, 10.25, 14.4, 2.75)

# ==============================================================================
# CARD 2: OBJECTIVES & KEY CONTRIBUTIONS (Right, Top)
# Card top = 7.18", height = 6.00", left = 16.88", width = 15.03"
# ==============================================================================
shape_c2 = slide.shapes[12]
shape_c2.top = Inches(8.0)
shape_c2.height = Inches(4.9)
set_tf_text(shape_c2.text_frame, [
    {"text": "1. Deep Architectural Teardown (Phase A):", "size": 15.5, "bold": True, "color": (30, 58, 138), "space_after": 2},
    {"text": "   Exhaustive parameter profiling (22.06M), FLOP analysis (12.25 GFLOPs), and hardware benchmarking on Tesla T4 (853.7 img/s in fp16).", "size": 13.5, "bold": False, "space_after": 5},
    {"text": "2. Controlled Nucleus Intervention (Phase B):", "size": 15.5, "bold": True, "color": (30, 58, 138), "space_after": 2},
    {"text": "   Handcrafted LoRA (r=8, α=8) injected into Query and Value projections of blocks 11 and 12, modifying only 32,266 parameters (0.146% of backbone).", "size": 13.5, "bold": False, "space_after": 5},
    {"text": "3. Representation Caching & Zero Overhead:", "size": 15.5, "bold": True, "color": (30, 58, 138), "space_after": 2},
    {"text": "   Precomputing frozen representations preserves strict equivalence and slashes training time from ~8 minutes to 61 seconds.", "size": 13.5, "bold": False, "space_after": 5},
    {"text": "4. Statistically Validated Benchmark Uplift:", "size": 15.5, "bold": True, "color": (5, 150, 105), "space_after": 2},
    {"text": "   Macro-F1 improves from 95.80% to 97.14% (+1.34%, p = 0.002) on 4,050 test images across all 27,000 EuroSAT samples, led by Highway (+5.85%) and River (+3.71%).", "size": 13.5, "bold": True, "color": (5, 150, 105), "space_after": 0}
])

# ==============================================================================
# CARD 3: METHODOLOGY & TEARDOWN (Left, Middle)
# Card top = 13.83", height = 10.20", left = 1.20", width = 15.03"
# ==============================================================================
shape_c3 = slide.shapes[16]
shape_c3.top = Inches(14.6)
shape_c3.height = Inches(2.2)
set_tf_text(shape_c3.text_frame, [
    {"text": "• Backbone Teardown: DINOv2 ViT-S/14 (12 blocks, d=384, 6 heads, MLP=1536). Total: 22,056,576 parameters, 12.25 GFLOPs at 224 px.", "size": 13.5, "bold": False, "space_after": 3},
    {"text": "• LoRA Formulation: W = W₀ + (α/r)(B · A), with B ∈ ℝ^(384×8) initialized to 0, A ∈ ℝ^(8×384) Kaiming uniform. Applied to Q and V of blocks 11 & 12.", "size": 13.5, "bold": True, "color": (15, 23, 42), "space_after": 3},
    {"text": "• Experimental Control: 27,000 EuroSAT images, stratified 70/15/15 (seed 42). AdamW (lr=1e-3, weight decay=1e-4, batch 64, 8 epochs).", "size": 13.5, "bold": False, "space_after": 0}
])
# Embed A1_parametros.png and A2_flops.png directly from notebook
# Stack them with clean proportions
add_fitted_picture(slide, FIGS_DIR / "A1_parametros.png", 1.5, 17.0, 14.4, 3.3)
add_fitted_picture(slide, FIGS_DIR / "A2_flops.png", 1.5, 20.4, 14.4, 3.3)

# ==============================================================================
# CARD 4: RESULTS & DISCUSSION (Right, Middle)
# Card top = 13.83", height = 10.20", left = 16.88", width = 15.03"
# ==============================================================================
shape_c4 = slide.shapes[20]
shape_c4.top = Inches(14.6)
shape_c4.height = Inches(2.2)
set_tf_text(shape_c4.text_frame, [
    {"text": "• Benchmark Uplift: Test Macro-F1: 97.14% (LoRA) vs 95.80% (Baseline)  ⟹  Δ = +1.34%. Accuracy: 97.28% vs 96.00% (McNemar p = 0.002).", "size": 13.5, "bold": True, "color": (5, 150, 105), "space_after": 3},
    {"text": "• Infrastructure & Hydrology Gains: Highway F1 leaps from 91.87% to 97.72% (+5.85%); River F1 rises from 93.80% to 97.51% (+3.71%).", "size": 13.5, "bold": False, "space_after": 3},
    {"text": "• Inference Throughput: Merged weights yield zero latency penalty: 1.17 ms/img (853.7 img/s in fp16 batch 128 on Tesla T4).", "size": 13.5, "bold": False, "space_after": 0}
])
# Embed B1_confusion_y_curvas.png directly from notebook
add_fitted_picture(slide, FIGS_DIR / "B1_confusion_y_curvas.png", 17.0, 17.0, 14.6, 4.4)

# Add a clean, native PowerPoint comparison table below the curves
table_shape = slide.shapes.add_table(3, 5, Inches(17.2), Inches(21.6), Inches(14.2), Inches(1.8))
table = table_shape.table
table.columns[0].width = Inches(3.6)
table.columns[1].width = Inches(2.6)
table.columns[2].width = Inches(2.6)
table.columns[3].width = Inches(2.8)
table.columns[4].width = Inches(2.6)

headers = ["Condition", "Trainable Params", "Test Accuracy", "Test Macro-F1", "Test Errors"]
for col_idx, h in enumerate(headers):
    cell = table.cell(0, col_idx)
    cell.text = h
    cell.fill.solid()
    cell.fill.fore_color.rgb = RGBColor(15, 23, 42)
    p = cell.text_frame.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    p.runs[0].font.name = "Times New Roman"
    p.runs[0].font.size = Pt(12)
    p.runs[0].font.bold = True
    p.runs[0].font.color.rgb = RGBColor(255, 255, 255)

row_data = [
    ["Linear Probing (Frozen)", "7,690 (0.035%)", "96.00%", "95.80%", "162 / 4,050"],
    ["DINOv2 + LoRA (Bl. 11-12)", "32,266 (0.146%)", "97.28%", "97.14% (+1.34%)", "110 / 4,050"]
]
for r_idx, rvals in enumerate(row_data):
    for c_idx, val in enumerate(rvals):
        cell = table.cell(r_idx + 1, c_idx)
        cell.text = val
        cell.fill.solid()
        bg_col = RGBColor(248, 250, 252) if r_idx == 0 else RGBColor(240, 253, 244)
        cell.fill.fore_color.rgb = bg_col
        p = cell.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.LEFT if c_idx == 0 else PP_ALIGN.CENTER
        p.runs[0].font.name = "Times New Roman"
        p.runs[0].font.size = Pt(12)
        if r_idx == 1 and c_idx in [2, 3]:
            p.runs[0].font.bold = True
            p.runs[0].font.color.rgb = RGBColor(5, 150, 105)

# ==============================================================================
# CARD 5: CONCLUSIONS & ANALYSIS (Left, Row 3)
# Card top = 24.68", height = 6.30", left = 1.20", width = 15.03"
# ==============================================================================
shape_c5 = slide.shapes[24]
shape_c5.top = Inches(25.5)
shape_c5.height = Inches(2.2)
set_tf_text(shape_c5.text_frame, [
    {"text": "• Core Finding: Adapting 0.146% of parameters in the last 2 self-attention blocks yields statistically significant domain adaptation (+1.34% F1, p = 0.002).", "size": 13.5, "bold": True, "color": (15, 23, 42), "space_after": 3},
    {"text": "• Epistemological Honesty: Evaluated on stratified random partition (seed 42). Random splits do not measure spatial autocorrelation across geographic regions.", "size": 13.5, "bold": False, "space_after": 3},
    {"text": "• Phase C Roadmap: Evaluate out-of-distribution shifts using geographically disjoint European regional tiles and multi-modal Sentinel-1 SAR imagery.", "size": 13.5, "bold": False, "space_after": 0}
])
# Embed A5_erf.png directly from notebook
add_fitted_picture(slide, FIGS_DIR / "A5_erf.png", 1.5, 27.9, 14.4, 2.9)

# ==============================================================================
# CARD 6: KEY REFERENCES (Right, Row 3)
# Card top = 24.68", height = 6.30", left = 16.88", width = 15.03"
# ==============================================================================
shape_c6 = slide.shapes[28]
shape_c6.top = Inches(25.5)
shape_c6.height = Inches(5.0)
set_tf_text(shape_c6.text_frame, [
    {"text": "[1] M. Oquab et al., 'DINOv2: Learning Robust Visual Features without Supervision,' arXiv preprint arXiv:2304.07193, 2023.", "size": 13.5, "bold": False, "space_after": 6},
    {"text": "[2] E. J. Hu et al., 'LoRA: Low-Rank Adaptation of Large Language Models,' in Proc. Int. Conf. Learn. Represent. (ICLR), 2022.", "size": 13.5, "bold": False, "space_after": 6},
    {"text": "[3] P. Helber et al., 'EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification,' IEEE JSTARS, 12(7):2217-2226, 2019.", "size": 13.5, "bold": False, "space_after": 6},
    {"text": "[4] W. Luo et al., 'Understanding the Effective Receptive Field in Deep Convolutional Neural Networks,' in Proc. NeurIPS, 2016.", "size": 13.5, "bold": False, "space_after": 0}
])

# ==============================================================================
# CARD 7: ACKNOWLEDGMENTS (Left, Bottom)
# Card top = 31.63", height = 6.00", left = 1.20", width = 15.03"
# ==============================================================================
shape_c7 = slide.shapes[32]
shape_c7.top = Inches(32.4)
shape_c7.height = Inches(4.8)
set_tf_text(shape_c7.text_frame, [
    {"text": "• Pontificia Universidad Católica del Perú (PUCP):", "size": 14.5, "bold": True, "color": (30, 58, 138), "space_after": 2},
    {"text": "   Course: Deep Learning (1INF52) - 2026-2. Lead Instructor: Dr. César A. Beltrán Castañón.", "size": 13, "bold": False, "space_after": 6},
    {"text": "• Copernicus Programme & European Space Agency (ESA):", "size": 14.5, "bold": True, "color": (30, 58, 138), "space_after": 2},
    {"text": "   For open-access Sentinel-2 Earth observation data and the EuroSAT benchmark dataset.", "size": 13, "bold": False, "space_after": 6},
    {"text": "• Meta AI Research & Open-Source Community:", "size": 14.5, "bold": True, "color": (30, 58, 138), "space_after": 2},
    {"text": "   For self-supervised DINOv2 vision foundation models and the PyTorch ecosystem.", "size": 13, "bold": False, "space_after": 6},
    {"text": "• Research Infrastructure & Compute:", "size": 14.5, "bold": True, "color": (30, 58, 138), "space_after": 2},
    {"text": "   Google Colab Research (Tesla T4 GPU) and the WAIMLAp 2026 academic symposium (IA-PUCP).", "size": 13, "bold": False, "space_after": 0}
])

# ==============================================================================
# CARD 8: REPRODUCIBILITY & CONTACT (Right, Bottom)
# Card top = 31.63", height = 6.00", left = 16.88", width = 15.03"
# ==============================================================================
shape_c8 = slide.shapes[36]
shape_c8.top = Inches(32.4)
shape_c8.height = Inches(2.2)
set_tf_text(shape_c8.text_frame, [
    {"text": "• Reproducible Open-Source Repository:", "size": 14.5, "bold": True, "color": (5, 150, 105), "space_after": 2},
    {"text": "   Full experimental notebooks, checkpoint weights, design decision log (D01-D18), and results tables are openly available at:", "size": 13, "bold": False, "space_after": 3},
    {"text": "   github.com/jflxx/deep-learning", "size": 15.5, "bold": True, "color": (30, 58, 138), "space_after": 6},
    {"text": "• Authors & Contact:", "size": 14.5, "bold": True, "color": (15, 23, 42), "space_after": 2},
    {"text": "   Alfredo Jimenez  |  Jhonatan Davila  |  Italo Perez\n   Emails: {alfredo.jimenez, jhonatan.davila, italo.perez}@pucp.edu.pe\n   IA-PUCP Research Group, San Miguel, Lima, Peru.", "size": 12.5, "bold": False, "space_after": 0}
])

# Add QR code to github repository
# Left = 22.8", top = 34.6", size = 2.0" x 2.0"
add_fitted_picture(slide, FIGS_DIR / "repo_qr.png", 23.4, 34.6, 2.0, 2.0)
# Add QR caption text box
qr_box = slide.shapes.add_textbox(Inches(20.4), Inches(36.7), Inches(8.0), Inches(0.4))
set_tf_text(qr_box.text_frame, [
    {
        "text": "Scan QR for Code, Notebooks & Design Log",
        "size": 11.5,
        "bold": True,
        "color": (71, 85, 105),
        "align": PP_ALIGN.CENTER
    }
])

prs.save(str(OUTPUT_PPTX_WAIMLAP))
prs.save(str(OUTPUT_PPTX_ROOT))
print(f"Successfully generated clean poster: {OUTPUT_PPTX_WAIMLAP}")
