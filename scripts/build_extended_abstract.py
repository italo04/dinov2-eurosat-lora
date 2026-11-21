import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
import os

TEMPLATE_PATH = r"c:\Users\italo\Documents\Ciclo 2026-2\Deep Learning\TA\Entregable_2\templates\template_abstract.docx"
OUTPUT_DOCX = r"c:\Users\italo\Documents\Ciclo 2026-2\Deep Learning\TA\Entregable_2\presentacion_waimlap\Abstract_DINOv2_EuroSAT_LoRA.docx"
ROOT_DOCX = r"c:\Users\italo\Documents\Ciclo 2026-2\Deep Learning\TA\Entregable_2\Abstract_DINOv2_EuroSAT_LoRA.docx"

doc = docx.Document(TEMPLATE_PATH)

# Clear existing body paragraphs while preserving headers/footers
for p in list(doc.paragraphs):
    p._p.getparent().remove(p._p)

# Adjust margins to 0.75 in to guarantee clean 2-page fit
for s in doc.sections:
    s.top_margin = Inches(0.7)
    s.bottom_margin = Inches(0.7)
    s.left_margin = Inches(0.8)
    s.right_margin = Inches(0.8)

FONT_NAME = "Times New Roman"

def add_p(text="", space_before=0, space_after=3.5, align=WD_ALIGN_PARAGRAPH.JUSTIFY, line_spacing=1.12):
    p = doc.add_paragraph()
    p.alignment = align
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = line_spacing
    if text:
        r = p.add_run(text)
        r.font.name = FONT_NAME
        r.font.size = Pt(10)
        r.font.color.rgb = RGBColor(15, 23, 42)
    return p

def add_heading(text, space_before=7, space_after=2.5):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text)
    r.font.name = FONT_NAME
    r.font.size = Pt(10.5)
    r.bold = True
    r.font.color.rgb = RGBColor(15, 23, 42)
    return p

# 1. Title
p_title = doc.add_paragraph()
p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
p_title.paragraph_format.space_before = Pt(0)
p_title.paragraph_format.space_after = Pt(3)
r_title = p_title.add_run("Dissecting and Adapting DINOv2 for Land Cover Classification:\nAn Architectural Teardown and Parameter-Efficient Study on Sentinel-2 Imagery")
r_title.font.name = FONT_NAME
r_title.font.size = Pt(12.5)
r_title.bold = True
r_title.font.color.rgb = RGBColor(15, 23, 42)

# 2. Authors
p_auth = doc.add_paragraph()
p_auth.alignment = WD_ALIGN_PARAGRAPH.CENTER
p_auth.paragraph_format.space_before = Pt(1)
p_auth.paragraph_format.space_after = Pt(1)
r_auth = p_auth.add_run("Alfredo Jimenez, Jhonatan Davila, Italo Perez")
r_auth.font.name = FONT_NAME
r_auth.font.size = Pt(10)
r_auth.bold = True

# 3. Affiliations
p_aff = doc.add_paragraph()
p_aff.alignment = WD_ALIGN_PARAGRAPH.CENTER
p_aff.paragraph_format.space_before = Pt(0)
p_aff.paragraph_format.space_after = Pt(6)
r_aff = p_aff.add_run(
    "Departamento de Ingeniería, Pontificia Universidad Católica del Perú (PUCP), Lima, Peru\n"
    "Grupo de Investigación en Inteligencia Artificial (IA-PUCP)  |  Corresponding email: italo.perez@pucp.edu.pe"
)
r_aff.font.name = FONT_NAME
r_aff.font.size = Pt(8.5)
r_aff.font.color.rgb = RGBColor(71, 85, 105)

# SECTION 1: INTRODUCTION
add_heading("1. INTRODUCTION")
p1 = add_p()
p1.add_run(
    "Foundation vision models pretrained via self-supervised learning, particularly Meta AI's DINOv2 (Vision Transformer architecture), "
    "yield powerful, highly generalizable feature representations on natural imagery. However, transferring these representations to satellite remote sensing "
    "entails navigating a severe domain shift: Sentinel-2 Earth observation imagery features nadir perspectives, repetitive agricultural spatial textures, "
    "absence of canonical foreground objects, and subtle inter-class spectral boundaries. While standard linear probing (training an affine classifier over frozen features) "
    "avoids catastrophic forgetting, it remains constrained by features not explicitly aligned with remote sensing textures. Conversely, full model fine-tuning updates "
    "over 22 million parameters, incurring prohibitive computational costs and risking overfitting on specialized target distributions. "
    "In this study, we present an architectural teardown of DINOv2 (ViT-S/14) and examine a controlled nucleus intervention using Low-Rank Adaptation (LoRA) "
    "restricted to the deepest self-attention layers. We evaluate this strategy across the complete EuroSAT dataset (27,000 images) on an NVIDIA Tesla T4 GPU, "
    "demonstrating statistically significant performance gains with minimal parameter footprint and zero inference overhead."
)

# SECTION 2: METHODOLOGY
add_heading("2. METHODOLOGY")
p2 = add_p()
p2.add_run(
    "Dataset and Protocol: We benchmark on the complete EuroSAT RGB dataset (Sentinel-2, 10 Land Use and Land Cover classes, 27,000 images). "
    "Images are upsampled from 64x64 to 224x224 pixels (bicubic, no crop to preserve peripheral scene context) and normalized with ImageNet statistics. "
    "We employ a deterministic stratified 70/15/15 partition with a fixed global seed (42): 18,900 images for training, 4,050 for validation, and 4,050 for testing. "
    "Following the canonical DINOv2 protocol, the representation fed to the classifier concatenates the [CLS] token and the average of all spatial patch tokens "
    "([CLS ; Mean(Tokens)], 768 dimensions), yielding a linear head of 7,690 parameters.\n"
    "Phase A (Architectural Teardown): DINOv2-Small (ViT-S/14 without registers) comprises 22,056,576 parameters (88.2 MB fp32) distributed across 12 identical transformer "
    "encoder blocks (embedding dimension d=384, 6 heads, MLP expansion 4x to 1536). Total analytical computation is 12.25 GFLOPs per 224px image (1.011 GFLOPs per block, "
    "where quadratic self-attention accounts for 9.9%). Benchmarking on a Tesla T4 GPU reveals a latency of 8.89 ms (112.5 img/s, batch 1, fp32) and 1.17 ms/img "
    "(853.7 img/s, batch 128, fp16). Effective Receptive Field (ERF) analysis confirms that the receptive field spans globally from block 1 onwards (r50 expands from 69.4 px "
    "to 85.0 px, and r90 from 117.2 px to 127.1 px in block 12). Principal Component Analysis (PCA) reveals that while block 4 tokens encode spatial coordinates, "
    "block 12 tokens cluster into clear semantic class representations.\n"
    "Phase B (Nucleus Intervention via LoRA): Hypothesizing that remote sensing domain transfer requires adapting high-level semantic tokens rather than early geometric primitives, "
    "we freeze blocks 0 through 10 and inject low-rank decomposition matrices into only the Query (Q) and Value (V) projections of blocks 11 and 12. "
    "Each adapted weight is parameterized as W = W0 + (alpha / r) * (B * A), where A in R^(8x384) is Kaiming-uniform initialized and B in R^(384x8) is zero-initialized (r=8, alpha=8). "
    "This confines trainable parameters to 32,266 (24,576 LoRA + 7,690 head, representing 0.146% of the backbone). Training is executed under identical budgets "
    "(AdamW, lr=1e-3, weight decay=1e-4, batch 64, 8 epochs) using precomputed frozen embeddings, reducing training time from ~8 minutes to 61 seconds."
)

# SECTION 3: RESULTS AND DISCUSSION
add_heading("3. RESULTS AND DISCUSSION")
p3 = add_p()
p3.add_run(
    "Quantitative Benchmarks: As presented in Table 1, LoRA adaptation achieves superior performance across all evaluation metrics on the 4,050 held-out test images. "
    "Test Macro-F1 increases from 95.80% to 97.14% (+1.34%), and Test Accuracy improves from 96.00% to 97.28% (+1.28%, reducing test misclassifications from 162 to 110). "
    "A two-tailed McNemar test on paired test predictions confirms that this improvement is statistically significant (chi2 = 9.56, p = 0.002)."
)

# Table 1
table = doc.add_table(rows=3, cols=6)
table.alignment = WD_TABLE_ALIGNMENT.CENTER
headers = ["Method / Configuration", "Trainable Params", "% Backbone", "Test Accuracy", "Test Macro-F1", "Test Errors"]
widths = [Inches(2.1), Inches(1.1), Inches(0.8), Inches(1.0), Inches(1.0), Inches(0.8)]

hdr_cells = table.rows[0].cells
for i, title in enumerate(headers):
    hdr_cells[i].text = title
    hdr_cells[i].width = widths[i]
    shading = parse_xml(r'<w:shd {} w:fill="0F172A"/>'.format(nsdecls('w')))
    hdr_cells[i]._tc.get_or_add_tcPr().append(shading)
    p = hdr_cells[i].paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for r in p.runs:
        r.font.name = FONT_NAME
        r.font.size = Pt(8.5)
        r.bold = True
        r.font.color.rgb = RGBColor(255, 255, 255)

rows_data = [
    ["Linear Probing (Frozen DINOv2)", "7,690", "0.035%", "96.00%", "95.80%", "162 / 4,050"],
    ["LoRA (Blocks 11-12, r=8, α=8)", "32,266", "0.146%", "97.28%", "97.14% (+1.34%)", "110 / 4,050"]
]

for row_idx, rdata in enumerate(rows_data):
    row_cells = table.rows[row_idx + 1].cells
    bg_color = "F8FAFC" if row_idx == 0 else "F0FDF4"
    for col_idx, val in enumerate(rdata):
        row_cells[col_idx].text = val
        row_cells[col_idx].width = widths[col_idx]
        shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{bg_color}"/>')
        row_cells[col_idx]._tc.get_or_add_tcPr().append(shd)
        p = row_cells[col_idx].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT if col_idx == 0 else WD_ALIGN_PARAGRAPH.CENTER
        for r in p.runs:
            r.font.name = FONT_NAME
            r.font.size = Pt(8.5)
            if row_idx == 1 and col_idx in [3, 4, 5]:
                r.bold = True

p_caption = doc.add_paragraph()
p_caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
p_caption.paragraph_format.space_before = Pt(2)
p_caption.paragraph_format.space_after = Pt(3)
r_cap = p_caption.add_run("Table 1: Performance comparison on full EuroSAT test partition (4,050 images, seed 42).")
r_cap.font.name = FONT_NAME
r_cap.font.size = Pt(8)
r_cap.font.italic = True

p3_2 = add_p()
p3_2.add_run(
    "Class-Level Dynamics & Error Analysis: Disaggregating F1 performance by class reveals that the largest gains occur in complex linear infrastructure "
    "and hydrology: Highway F1 increases by +5.85% (91.87% to 97.72%) and River improves by +3.71% (93.80% to 97.51%), while Residential (+0.88%) "
    "and Industrial (+0.57%) also advance. Natural categories with high baselines (Forest at 98.89%, SeaLake at 99.44%) maintain peak fidelity. "
    "Residual confusion is predominantly concentrated among subtle spectral boundaries between Pasture and HerbaceousVegetation, where absence of short-wave infrared "
    "bands in standard RGB imagery creates inherent physical ambiguity."
)

# SECTION 4: CONCLUSIONS
add_heading("4. CONCLUSIONS")
p4 = add_p()
p4.add_run(
    "Targeted Low-Rank Adaptation on the Query and Value projections of the final two transformer blocks bridges the domain gap between natural and remote sensing imagery, "
    "delivering a statistically significant gain (+1.34% Macro-F1, p=0.002) while tuning only 0.146% of parameters. Embedding caching accelerates training by over 8x "
    "without numerical deviation, and inference weight merging eliminates runtime latency penalties entirely.\n"
    "Limitations & Phase C Roadmap: We maintain strict epistemological honesty: the random stratified partition evaluates intra-distribution generalization. "
    "In geographic remote sensing, spatial autocorrelation between adjacent tiles can inflate empirical metrics. Ongoing work for Phase C investigates out-of-distribution "
    "robustness via geographically disjoint regional splits using EuroSAT 13-band Sentinel-2 L2A GeoTIFFs and cross-sensor evaluation on Sentinel-1 SAR data."
)

# REFERENCES
add_heading("REFERENCES", space_before=5, space_after=2)
refs = [
    "[1] M. Oquab et al., 'DINOv2: Learning Robust Visual Features without Supervision,' arXiv preprint arXiv:2304.07193, 2023.",
    "[2] E. J. Hu et al., 'LoRA: Low-Rank Adaptation of Large Language Models,' in Proc. Int. Conf. Learn. Represent. (ICLR), 2022.",
    "[3] P. Helber et al., 'EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification,' IEEE JSTARS, 12(7):2217-2226, 2019.",
    "[4] W. Luo et al., 'Understanding the Effective Receptive Field in Deep Convolutional Neural Networks,' in Proc. NeurIPS, 2016."
]
for ref in refs:
    p_ref = add_p(ref, space_before=0, space_after=1.5)
    for r in p_ref.runs:
        r.font.name = FONT_NAME
        r.font.size = Pt(8.0)

doc.save(OUTPUT_DOCX)
doc.save(ROOT_DOCX)
print(f"Successfully generated: {OUTPUT_DOCX}")
