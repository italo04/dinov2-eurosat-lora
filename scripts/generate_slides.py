import pathlib
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT_PPTX_WAIMLAP = ROOT / "presentacion_waimlap" / "Presentacion_E2_DINOv2_EuroSAT.pptx"
OUT_PPTX_ROOT = ROOT / "Presentacion_E2_DINOv2_EuroSAT.pptx"
FIG = ROOT / "outputs" / "figures"

# Palette: Academic Tech High-End
C_BG = RGBColor(248, 250, 252)        # Slate 50
C_NAVY = RGBColor(15, 23, 42)          # Slate 900
C_PRIMARY = RGBColor(30, 58, 138)      # Blue 900
C_ACCENT = RGBColor(2, 132, 199)       # Sky 600
C_SUCCESS = RGBColor(5, 150, 105)      # Emerald 600
C_CARD_BG = RGBColor(255, 255, 255)    # White
C_BORDER = RGBColor(226, 232, 240)     # Slate 200
C_MUTED = RGBColor(100, 116, 139)      # Slate 500

def create_deck():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    def add_bg(slide):
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
        bg.fill.solid()
        bg.fill.fore_color.rgb = C_BG
        bg.line.fill.background()
        return bg

    def add_header(slide, title, category="ENTREGABLE 2 • DEEP LEARNING (1INF52) - PUCP"):
        tb_cat = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.35))
        p_cat = tb_cat.text_frame.paragraphs[0]
        p_cat.text = category.upper()
        p_cat.font.size = Pt(10)
        p_cat.font.bold = True
        p_cat.font.color.rgb = C_ACCENT

        tb_title = slide.shapes.add_textbox(Inches(0.8), Inches(0.75), Inches(11.7), Inches(0.65))
        p_title = tb_title.text_frame.paragraphs[0]
        p_title.text = title
        p_title.font.size = Pt(22)
        p_title.font.bold = True
        p_title.font.color.rgb = C_NAVY

    def add_card(slide, left, top, width, height, bg_color=C_CARD_BG, border_color=C_BORDER):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        card.fill.solid()
        card.fill.fore_color.rgb = bg_color
        if border_color:
            card.line.color.rgb = border_color
            card.line.width = Pt(1.5)
        else:
            card.line.fill.background()
        return card

    # ==========================================
    # SLIDE 1: PORTADA
    # ==========================================
    s1 = prs.slides.add_slide(blank_layout)
    bg1 = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
    bg1.fill.solid()
    bg1.fill.fore_color.rgb = C_NAVY
    bg1.line.fill.background()

    tb1 = s1.shapes.add_textbox(Inches(1.2), Inches(1.2), Inches(11.0), Inches(5.2))
    tf1 = tb1.text_frame
    tf1.word_wrap = True

    p = tf1.paragraphs[0]
    p.text = "PONTIFICIA UNIVERSIDAD CATÓLICA DEL PERÚ • 1INF52 DEEP LEARNING 2026-2"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = RGBColor(147, 197, 253)
    p.space_after = Pt(20)

    p = tf1.add_paragraph()
    p.text = "Disección Instrumentada e Intervención Preliminar con LoRA en DINOv2 (ViT-S/14)"
    p.font.size = Pt(30)
    p.font.bold = True
    p.font.color.rgb = RGBColor(255, 255, 255)
    p.space_after = Pt(12)

    p = tf1.add_paragraph()
    p.text = "Adaptación de Bajo Rango para Clasificación de Cobertura del Suelo sobre EuroSAT Sentinel-2 RGB"
    p.font.size = Pt(18)
    p.font.color.rgb = RGBColor(203, 213, 225)
    p.space_after = Pt(36)

    p = tf1.add_paragraph()
    p.text = "Equipo: Alfredo Jimenez • Jhonatan Davila • Italo Perez\nDocente: Dr. César A. Beltrán Castañón | Grupo de Investigación: IA-PUCP\nDataset: 27,000 imágenes (Colab T4 GPU) | Repositorio: github.com/jflxx/deep-learning"
    p.font.size = Pt(13)
    p.font.color.rgb = RGBColor(148, 163, 184)

    # ==========================================
    # SLIDE 2: MOTIVACIÓN Y CONTEXTO
    # ==========================================
    s2 = prs.slides.add_slide(blank_layout)
    add_bg(s2)
    add_header(s2, "1. Motivación y Planteamiento del Problema")

    add_card(s2, Inches(0.8), Inches(1.5), Inches(5.6), Inches(5.4))
    tb2_l = s2.shapes.add_textbox(Inches(1.0), Inches(1.7), Inches(5.2), Inches(5.0))
    tf2_l = tb2_l.text_frame
    tf2_l.word_wrap = True

    p = tf2_l.paragraphs[0]
    p.text = "El Paradigma del Modelo Fundacional"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = C_PRIMARY
    p.space_after = Pt(10)

    p = tf2_l.add_paragraph()
    p.text = "• DINOv2 (Meta AI) extrae representaciones visuales ricas y generalizables mediante aprendizaje auto-supervisado en millones de imágenes web.\n\n" \
             "• El Reto: Despliegue en Teledetección Satelital\n" \
             "  - Perspectiva cenital (nadir) sin objeto central dominante.\n" \
             "  - Patrones texturales repetitivos y variación multiespectral.\n" \
             "  - Alta similitud inter-clase (p. ej., Highway vs River, vegetación).\n\n" \
             "• ¿Linear Probing o Fine-Tuning Completo?\n" \
             "  - Linear Probing congela la red pero no adapta patrones espectrales.\n" \
             "  - Full Fine-Tuning actualiza 22M+ parámetros, arriesgando olvido catastrófico y exigiendo enorme cómputo."
    p.font.size = Pt(13)
    p.font.color.rgb = C_NAVY

    add_card(s2, Inches(6.7), Inches(1.5), Inches(5.8), Inches(5.4))
    if (FIG / "00_muestras.png").exists():
        s2.shapes.add_picture(str(FIG / "00_muestras.png"), Inches(6.9), Inches(1.7), Inches(5.4), Inches(3.6))
    tb2_rc = s2.shapes.add_textbox(Inches(6.9), Inches(5.4), Inches(5.4), Inches(1.3))
    tf2_rc = tb2_rc.text_frame
    tf2_rc.word_wrap = True
    p = tf2_rc.paragraphs[0]
    p.text = "EuroSAT Sentinel-2 RGB (27,000 imágenes, 10 clases LULC)\n" \
             "Pregunta de Investigación: ¿Añadir adaptadores LoRA (r=8) en capas profundas de atención supera al DINOv2 congelado con mínimo cómputo?"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = C_PRIMARY

    # ==========================================
    # SLIDE 3: HIPÓTESIS Y ESTRATEGIA EXPERIMENTAL
    # ==========================================
    s3 = prs.slides.add_slide(blank_layout)
    add_bg(s3)
    add_header(s3, "2. Hipótesis Científica y Control Experimental")

    add_card(s3, Inches(0.8), Inches(1.5), Inches(11.7), Inches(1.6), bg_color=RGBColor(240, 253, 244), border_color=C_SUCCESS)
    tb3_top = s3.shapes.add_textbox(Inches(1.0), Inches(1.6), Inches(11.3), Inches(1.4))
    tf3_top = tb3_top.text_frame
    tf3_top.word_wrap = True
    p = tf3_top.paragraphs[0]
    p.text = "Hipótesis Formal (Registrada a priori en DESIGN_LOG.md D09):"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = C_SUCCESS
    p.space_after = Pt(4)
    p = tf3_top.add_paragraph()
    p.text = "«Las capas iniciales de DINOv2 codifican primitivas geométricas y texturas universales transferibles. " \
             "La brecha de dominio satelital radica en la agregación semántica de alto nivel. Por tanto, inyectar adaptadores LoRA (r=8, α=8) " \
             "únicamente en las proyecciones Query (Q) y Value (V) de los bloques 11 y 12 será suficiente para superar la línea base congelada " \
             "con significancia estadística (p < 0.05), ajustando solo el 0.146% de los parámetros.»"
    p.font.size = Pt(12.5)
    p.font.color.rgb = C_NAVY

    col_w = Inches(3.7)
    add_card(s3, Inches(0.8), Inches(3.3), col_w, Inches(3.6))
    tb3_c1 = s3.shapes.add_textbox(Inches(0.95), Inches(3.45), Inches(3.4), Inches(3.3))
    tf3_c1 = tb3_c1.text_frame
    tf3_c1.word_wrap = True
    p = tf3_c1.paragraphs[0]
    p.text = "Partición Determinista"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = C_PRIMARY
    p.space_after = Pt(8)
    p = tf3_c1.add_paragraph()
    p.text = "• 27,000 imágenes totales de EuroSAT.\n" \
             "• Split estratificado 70 / 15 / 15 (Semilla 42).\n" \
             "• 18,900 imágenes de entrenamiento.\n" \
             "• 4,050 imágenes de validación.\n" \
             "• 4,050 imágenes de test ciego (error estándar < 0.3%)."
    p.font.size = Pt(12)

    add_card(s3, Inches(4.8), Inches(3.3), col_w, Inches(3.6))
    tb3_c2 = s3.shapes.add_textbox(Inches(4.95), Inches(3.45), Inches(3.4), Inches(3.3))
    tf3_c2 = tb3_c2.text_frame
    tf3_c2.word_wrap = True
    p = tf3_c2.paragraphs[0]
    p.text = "Representación Oficial"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = C_PRIMARY
    p.space_after = Pt(8)
    p = tf3_c2.add_paragraph()
    p.text = "• Protocolo canónico DINOv2 (Oquab et al.):\n" \
             "  [CLS ; Media(tokens_parche)]\n" \
             "• Dimensión = 384 + 384 = 768 dims.\n" \
             "• Cabeza lineal idéntica para ambas condiciones (7,690 parámetros).\n" \
             "• Precomputación y caché de embeddings."
    p.font.size = Pt(12)

    add_card(s3, Inches(8.8), Inches(3.3), col_w, Inches(3.6))
    tb3_c3 = s3.shapes.add_textbox(Inches(8.95), Inches(3.45), Inches(3.4), Inches(3.3))
    tf3_c3 = tb3_c3.text_frame
    tf3_c3.word_wrap = True
    p = tf3_c3.paragraphs[0]
    p.text = "Mismo Presupuesto"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = C_PRIMARY
    p.space_after = Pt(8)
    p = tf3_c3.add_paragraph()
    p.text = "• Optimizador: AdamW (lr = 1e-3, weight decay = 1e-4).\n" \
             "• Batch size: 64 | 8 épocas de entrenamiento.\n" \
             "• Selección de checkpoint por mejor Macro-F1 en validación.\n" \
             "• Control experimental estricto: un cambio, una hipótesis."
    p.font.size = Pt(12)

    # ==========================================
    # SLIDE 4: TEARDOWN ARQUITECTÓNICO
    # ==========================================
    s4 = prs.slides.add_slide(blank_layout)
    add_bg(s4)
    add_header(s4, "3. Fase A: Teardown Instrumentado de DINOv2 (ViT-S/14)")

    add_card(s4, Inches(0.8), Inches(1.5), Inches(5.6), Inches(5.4))
    tb4_l = s4.shapes.add_textbox(Inches(1.0), Inches(1.7), Inches(5.2), Inches(5.0))
    tf4_l = tb4_l.text_frame
    tf4_l.word_wrap = True
    p = tf4_l.paragraphs[0]
    p.text = "Métricas Analíticas y Hardware (Tesla T4)"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = C_PRIMARY
    p.space_after = Pt(8)
    p = tf4_l.add_paragraph()
    p.text = "• Parámetros Totales: 22,056,576 (88.2 MB en fp32).\n" \
             "• 12 bloques idénticos (1.775M params/bloque: MLP 67%, Atención 33%).\n" \
             "• Cómputo Analítico: 12.25 GFLOPs por imagen a 224 px.\n" \
             "  - 1.011 GFLOPs por bloque (atención cuadrática es 9.9%).\n\n" \
             "• Perfilado en GPU NVIDIA Tesla T4:\n" \
             "  - batch 1 (fp32): 8.89 ms (112.5 img/s, pico 187 MB).\n" \
             "  - batch 32 (fp32): 3.80 ms/img (263.4 img/s, pico 344 MB).\n" \
             "  - batch 128 (fp16): 1.17 ms/img (853.7 img/s, pico 640 MB).\n" \
             "• Hallazgo: fp16 con Tensor Cores triplica el throughput (×3.3)."
    p.font.size = Pt(12.5)

    add_card(s4, Inches(6.7), Inches(1.5), Inches(5.8), Inches(5.4))
    if (FIG / "A1_parametros.png").exists():
        s4.shapes.add_picture(str(FIG / "A1_parametros.png"), Inches(6.9), Inches(1.7), Inches(5.4), Inches(2.5))
    if (FIG / "A2_flops.png").exists():
        s4.shapes.add_picture(str(FIG / "A2_flops.png"), Inches(6.9), Inches(4.3), Inches(5.4), Inches(2.4))

    # ==========================================
    # SLIDE 5: REPRESENTACIONES Y CAMPO RECEPTIVO
    # ==========================================
    s5 = prs.slides.add_slide(blank_layout)
    add_bg(s5)
    add_header(s5, "4. Fase A: PCA de Tokens y Campo Receptivo Efectivo (ERF)")

    add_card(s5, Inches(0.8), Inches(1.5), Inches(5.6), Inches(5.4))
    if (FIG / "A4_pca_tokens.png").exists():
        s5.shapes.add_picture(str(FIG / "A4_pca_tokens.png"), Inches(0.95), Inches(1.65), Inches(5.3), Inches(3.2))
    tb5_l = s5.shapes.add_textbox(Inches(0.95), Inches(5.0), Inches(5.3), Inches(1.7))
    tf5_l = tb5_l.text_frame
    tf5_l.word_wrap = True
    p = tf5_l.paragraphs[0]
    p.text = "Evolución Semántica en PCA de Tokens:"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = C_PRIMARY
    p = tf5_l.add_paragraph()
    p.text = "• Bloque 4: El color sigue la coordenada espacial del parche (gradiente posicional idéntico).\n" \
             "• Bloque 12: Emerge color uniforme por clase y regiones semánticas nítidas (carretera vs bosque)."
    p.font.size = Pt(11.5)

    add_card(s5, Inches(6.7), Inches(1.5), Inches(5.8), Inches(5.4))
    if (FIG / "A5_erf.png").exists():
        s5.shapes.add_picture(str(FIG / "A5_erf.png"), Inches(6.9), Inches(1.65), Inches(5.4), Inches(3.2))
    tb5_r = s5.shapes.add_textbox(Inches(6.9), Inches(5.0), Inches(5.4), Inches(1.7))
    tf5_r = tb5_r.text_frame
    tf5_r.word_wrap = True
    p = tf5_r.paragraphs[0]
    p.text = "Dinámica del Campo Receptivo Efectivo (ERF):"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = C_PRIMARY
    p = tf5_r.add_paragraph()
    p.text = "• A diferencia de una CNN, el ERF es global desde el bloque 1 (solo 9% en el parche central).\n" \
             "• Expansión: r50 crece de 69.4 a 85.0 px; r90 crece de 117.2 a 127.1 px en el bloque 12."
    p.font.size = Pt(11.5)

    # ==========================================
    # SLIDE 6: INTERVENCIÓN CON LORA
    # ==========================================
    s6 = prs.slides.add_slide(blank_layout)
    add_bg(s6)
    add_header(s6, "5. Fase B: Intervención en el Núcleo con LoRA")

    add_card(s6, Inches(0.8), Inches(1.5), Inches(5.6), Inches(5.4))
    tb6_l = s6.shapes.add_textbox(Inches(1.0), Inches(1.7), Inches(5.2), Inches(5.0))
    tf6_l = tb6_l.text_frame
    tf6_l.word_wrap = True
    p = tf6_l.paragraphs[0]
    p.text = "Formulación e Implementación Propia"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = C_PRIMARY
    p.space_after = Pt(8)
    p = tf6_l.add_paragraph()
    p.text = "• Implementado a mano en clase LoRALinear (~20 líneas PyTorch) para explicabilidad en la defensa oral:\n\n" \
             "       W_adaptado = W₀ + (α / r) · (B · A)\n\n" \
             "• Localización: Query (Q) y Value (V) en Bloques 11 y 12.\n" \
             "  - W₀ ∈ ℝ^(384×384) congelado.\n" \
             "  - A ∈ ℝ^(8×384) inicializado con Kaiming Uniforme.\n" \
             "  - B ∈ ℝ^(384×8) inicializado en CERO (inicio estricto = DINOv2).\n" \
             "  - Hiperparámetros: r = 8, α = 8 (escala neutra α/r = 1.0).\n\n" \
             "• Parámetros entrenables:\n" \
             "  - 4 capas LoRA × 6,144 = 24,576 parámetros LoRA.\n" \
             "  - Cabeza lineal: 7,690 parámetros.\n" \
             "  - Total: 32,266 parámetros (0.146% del modelo)."
    p.font.size = Pt(12)

    add_card(s6, Inches(6.7), Inches(1.5), Inches(5.8), Inches(5.4))
    tb6_r = s6.shapes.add_textbox(Inches(6.9), Inches(1.7), Inches(5.4), Inches(5.0))
    tf6_r = tb6_r.text_frame
    tf6_r.word_wrap = True
    p = tf6_r.paragraphs[0]
    p.text = "Aceleración con Caché y Fusión de Pesos"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = C_PRIMARY
    p.space_after = Pt(8)
    p = tf6_r.add_paragraph()
    p.text = "• Caching de Embeddings:\n" \
             "  - Como los bloques 0–10 están congelados, las representaciones intermedias se precomputaron una sola vez.\n" \
             "  - Tiempo de extracción: 61 segundos (vs ~8 min de cómputo repetido).\n" \
             "  - Equivalencia numérica exacta comprobada.\n\n" \
             "• Fusión de Pesos en Inferencia (Zero-Overhead):\n" \
             "  - Durante inferencia, los adaptadores se pliegan en la matriz original:\n" \
             "       W_eval = W₀ + (B · A)\n" \
             "  - Verificado: salida fusionada == salida LoRA (diff = 0.00e+00).\n" \
             "  - Costo de inferencia idéntico al DINOv2 original (853.7 img/s)."
    p.font.size = Pt(12.5)

    # ==========================================
    # SLIDE 7: RESULTADOS EXPERIMENTALES
    # ==========================================
    s7 = prs.slides.add_slide(blank_layout)
    add_bg(s7)
    add_header(s7, "6. Resultados en Test (4,050 imágenes, Semilla 42)")

    add_card(s7, Inches(0.8), Inches(1.5), Inches(3.6), Inches(2.2), bg_color=RGBColor(241, 245, 249))
    tb_k1 = s7.shapes.add_textbox(Inches(0.9), Inches(1.6), Inches(3.4), Inches(2.0))
    tf_k1 = tb_k1.text_frame
    tf_k1.word_wrap = True
    p = tf_k1.paragraphs[0]
    p.text = "LINEAL CONGELADO"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = C_MUTED
    p = tf_k1.add_paragraph()
    p.text = "95.80%"
    p.font.size = Pt(28)
    p.font.bold = True
    p.font.color.rgb = C_NAVY
    p = tf_k1.add_paragraph()
    p.text = "Test Accuracy: 96.00% | 162 errores\nParams: 7,690 (0.035%)"
    p.font.size = Pt(11)

    add_card(s7, Inches(4.8), Inches(1.5), Inches(3.6), Inches(2.2), bg_color=RGBColor(240, 253, 244), border_color=C_SUCCESS)
    tb_k2 = s7.shapes.add_textbox(Inches(4.9), Inches(1.6), Inches(3.4), Inches(2.0))
    tf_k2 = tb_k2.text_frame
    tf_k2.word_wrap = True
    p = tf_k2.paragraphs[0]
    p.text = "DINOv2 + LoRA (Q, V)"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = C_SUCCESS
    p = tf_k2.add_paragraph()
    p.text = "97.14%"
    p.font.size = Pt(28)
    p.font.bold = True
    p.font.color.rgb = C_SUCCESS
    p = tf_k2.add_paragraph()
    p.text = "Test Accuracy: 97.28% | 110 errores\nParams: 32,266 (0.146%)"
    p.font.size = Pt(11)

    add_card(s7, Inches(8.8), Inches(1.5), Inches(3.7), Inches(2.2), bg_color=RGBColor(239, 246, 255), border_color=C_ACCENT)
    tb_k3 = s7.shapes.add_textbox(Inches(8.9), Inches(1.6), Inches(3.5), Inches(2.0))
    tf_k3 = tb_k3.text_frame
    tf_k3.word_wrap = True
    p = tf_k3.paragraphs[0]
    p.text = "MEJORA NETA SIGNIFICATIVA"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = C_ACCENT
    p = tf_k3.add_paragraph()
    p.text = "+1.34%"
    p.font.size = Pt(28)
    p.font.bold = True
    p.font.color.rgb = C_PRIMARY
    p = tf_k3.add_paragraph()
    p.text = "Δ Macro-F1 | 52 errores menos\nMcNemar test: p = 0.002 (significativo)"
    p.font.size = Pt(11)

    add_card(s7, Inches(0.8), Inches(3.9), Inches(11.7), Inches(3.1))
    if (FIG / "B1_confusion_y_curvas.png").exists():
        s7.shapes.add_picture(str(FIG / "B1_confusion_y_curvas.png"), Inches(1.0), Inches(4.0), Inches(11.3), Inches(2.8))

    # ==========================================
    # SLIDE 8: DESGLOSE POR CLASE Y FALLOS
    # ==========================================
    s8 = prs.slides.add_slide(blank_layout)
    add_bg(s8)
    add_header(s8, "7. Análisis de Clases y Diagnóstico de Fallos")

    add_card(s8, Inches(0.8), Inches(1.5), Inches(5.6), Inches(5.4))
    tb8_l = s8.shapes.add_textbox(Inches(1.0), Inches(1.7), Inches(5.2), Inches(5.0))
    tf8_l = tb8_l.text_frame
    tf8_l.word_wrap = True
    p = tf8_l.paragraphs[0]
    p.text = "Grandes Ganadores con LoRA"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = C_SUCCESS
    p.space_after = Pt(8)
    p = tf8_l.add_paragraph()
    p.text = "• Highway (Carreteras):\n" \
             "  F1 sube de 91.87% a 97.72% (+5.85 pts)\n" \
             "  Resuelve la confusión con suelo desnudo y pastizal.\n\n" \
             "• River (Ríos):\n" \
             "  F1 sube de 93.80% a 97.51% (+3.71 pts)\n" \
             "  Distingue el cauce lineal de árboles circundantes.\n\n" \
             "• Residential (+0.88 pts) e Industrial (+0.57 pts).\n\n" \
             "• Clases estables sin olvido catastrófico:\n" \
             "  - Forest: 98.78% → 98.89%\n" \
             "  - SeaLake: 99.22% → 99.44%"
    p.font.size = Pt(13)

    add_card(s8, Inches(6.7), Inches(1.5), Inches(5.8), Inches(5.4))
    tb8_r = s8.shapes.add_textbox(Inches(6.9), Inches(1.7), Inches(5.4), Inches(5.0))
    tf8_r = tb8_r.text_frame
    tf8_r.word_wrap = True
    p = tf8_r.paragraphs[0]
    p.text = "Diagnóstico de Confusiones Residuales"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = C_PRIMARY
    p.space_after = Pt(8)
    p = tf8_r.add_paragraph()
    p.text = "• Confusión Residual entre Clases de Vegetación:\n" \
             "  - Pasture (Pastizales) vs HerbaceousVegetation.\n" \
             "  - PermanentCrop vs HerbaceousVegetation.\n\n" \
             "• Causa Física del Error:\n" \
             "  EuroSAT RGB utiliza únicamente los canales visibles (B4, B3, B2).\n" \
             "  Sin bandas infrarrojas (NIR / SWIR), la firma de reflectancia de distintas especies vegetales es físicamente idéntica.\n\n" \
             "• Comportamiento de Optimización:\n" \
             "  La pérdida de entrenamiento LoRA desciende a ~0.0003, lo que indica un ajuste fino muy cercano a la distribución interna."
    p.font.size = Pt(13)

    # ==========================================
    # SLIDE 9: CONCLUSIONES Y HONESTIDAD EPISTÉMICA
    # ==========================================
    s9 = prs.slides.add_slide(blank_layout)
    add_bg(s9)
    add_header(s9, "8. Conclusiones y Compromiso con la Fase C")

    add_card(s9, Inches(0.8), Inches(1.5), Inches(11.7), Inches(2.4), bg_color=RGBColor(240, 253, 244), border_color=C_SUCCESS)
    tb9_top = s9.shapes.add_textbox(Inches(1.0), Inches(1.65), Inches(11.3), Inches(2.1))
    tf9_top = tb9_top.text_frame
    tf9_top.word_wrap = True
    p = tf9_top.paragraphs[0]
    p.text = "Conclusiones Principales del Entregable 2:"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = C_SUCCESS
    p.space_after = Pt(4)
    p = tf9_top.add_paragraph()
    p.text = "1. Viabilidad de LoRA en Visión: Inyectar adaptadores en Q y V de los 2 últimos bloques (0.146% de parámetros) supera significativamente al clasificador lineal (+1.34% F1, p = 0.002) en las 27,000 imágenes de EuroSAT.\n" \
             "2. Eficiencia Extrema: El almacenamiento en caché reduce el entrenamiento a 61 segundos, y la fusión de pesos elimina cualquier sobrecosto de latencia en inferencia (853.7 img/s en GPU T4)."
    p.font.size = Pt(13)

    add_card(s9, Inches(0.8), Inches(4.1), Inches(11.7), Inches(2.8), bg_color=RGBColor(254, 242, 242), border_color=RGBColor(239, 68, 68))
    tb9_bot = s9.shapes.add_textbox(Inches(1.0), Inches(4.25), Inches(11.3), Inches(2.5))
    tf9_bot = tb9_bot.text_frame
    tf9_bot.word_wrap = True
    p = tf9_bot.paragraphs[0]
    p.text = "Honestidad Epistémica y Hoja de Ruta para Fase C (E3):"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = RGBColor(185, 28, 28)
    p.space_after = Pt(4)
    p = tf9_bot.add_paragraph()
    p.text = "• Limitación Declarada: La partición 70/15/15 actual es aleatoria estratificada. Debido a la autocorrelación espacial entre teselas satelitales, este resultado no garantiza robustez ante cambio geográfico real.\n" \
             "• Trabajo en Curso para Fase C (E3):\n" \
             "  - Evaluación bajo Shift Geográfico: Partición espacial por regiones de Europa usando EuroSAT Multiespectral (GeoTIFFs con coordenadas GPS).\n" \
             "  - Comparación con DINOv2 con Registros (dinov2_vits14_reg) para evaluar la eliminación de artefactos de alta norma observados en el Teardown."
    p.font.size = Pt(13)

    prs.save(str(OUT_PPTX_WAIMLAP))
    prs.save(str(OUT_PPTX_ROOT))
    print(f"Successfully generated: {OUT_PPTX_WAIMLAP}")

if __name__ == "__main__":
    create_deck()
