import base64
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT_HTML_WAIMLAP = ROOT / "presentacion_waimlap" / "diapositivas.html"
OUT_HTML_ROOT = ROOT / "diapositivas.html"
FIG = ROOT / "outputs" / "figures"

def b64(path):
    if path.exists():
        return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("utf-8")
    return ""

img_samples = b64(FIG / "00_muestras.png")
img_params = b64(FIG / "A1_parametros.png")
img_flops = b64(FIG / "A2_flops.png")
img_pca = b64(FIG / "A4_pca_tokens.png")
img_erf = b64(FIG / "A5_erf.png")
img_results = b64(FIG / "B1_confusion_y_curvas.png")

# Team's class data from B2_f1_por_clase.csv
class_data = [
    {"name": "Highway", "base_f1": 91.9, "lora_f1": 97.7, "delta": 5.8},
    {"name": "River", "base_f1": 93.8, "lora_f1": 97.5, "delta": 3.7},
    {"name": "Residential", "base_f1": 97.9, "lora_f1": 98.8, "delta": 0.9},
    {"name": "HerbaceousVegetation", "base_f1": 95.4, "lora_f1": 96.1, "delta": 0.7},
    {"name": "PermanentCrop", "base_f1": 93.5, "lora_f1": 94.2, "delta": 0.7},
    {"name": "Industrial", "base_f1": 97.9, "lora_f1": 98.4, "delta": 0.5},
    {"name": "Pasture", "base_f1": 93.3, "lora_f1": 93.8, "delta": 0.5},
    {"name": "SeaLake", "base_f1": 99.2, "lora_f1": 99.4, "delta": 0.2},
    {"name": "Forest", "base_f1": 98.8, "lora_f1": 98.9, "delta": 0.1},
    {"name": "AnnualCrop", "base_f1": 96.5, "lora_f1": 96.5, "delta": 0.0}
]

class_data_json = json.dumps(class_data)

html = f"""<!DOCTYPE html>
<html lang="es" class="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>DINOv2 + LoRA en EuroSAT — Entregable 2 (1INF52 PUCP)</title>
  <!-- Tailwind CSS -->
  <script src="https://cdn.tailwindcss.com"></script>
  <script>
    tailwind.config = {{
      darkMode: 'class',
      theme: {{
        extend: {{
          colors: {{
            brand: {{
              50: '#eff6ff',
              100: '#dbeafe',
              400: '#60a5fa',
              500: '#3b82f6',
              600: '#2563eb',
              700: '#1d4ed8',
              800: '#1e40af',
              900: '#1e3a8a',
            }}
          }}
        }}
      }}
    }}
  </script>
  <!-- KaTeX -->
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js"
          onload="renderMathInElement(document.body, {{delimiters: [{{left: '$$', right: '$$', display: true}}, {{left: '$', right: '$', display: false}}]}});">
  </script>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Fira+Code:wght@400;500&display=swap');
    body {{
      font-family: 'Inter', sans-serif;
    }}
    .math-font {{
      font-family: KaTeX_Main, Times New Roman, serif;
    }}
    .slide-enter {{
      animation: fadeIn 0.3s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }}
    @keyframes fadeIn {{
      from {{ opacity: 0; transform: translateY(6px); }}
      to {{ opacity: 1; transform: translateY(0); }}
    }}
  </style>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen flex flex-col justify-between overflow-x-hidden selection:bg-brand-500 selection:text-white">

  <!-- TOP BAR -->
  <header class="border-b border-slate-800 bg-slate-900/80 backdrop-blur sticky top-0 z-40 px-6 py-3 flex items-center justify-between">
    <div class="flex items-center space-x-3">
      <span class="inline-flex items-center justify-center bg-brand-600 text-white font-bold text-xs px-2.5 py-1 rounded">1INF52</span>
      <span class="text-sm font-semibold text-slate-200 hidden sm:inline">DINOv2 (ViT-S/14) + LoRA • EuroSAT RGB</span>
    </div>
    
    <div class="flex items-center space-x-2 text-xs">
      <button id="overviewBtn" class="bg-slate-800 hover:bg-slate-700 text-slate-300 px-3 py-1.5 rounded transition flex items-center gap-1.5" title="Ver cuadrícula (Tecla O)">
        <span>📑</span> <span>Índice</span>
      </button>
      <button id="notesBtn" class="bg-slate-800 hover:bg-slate-700 text-slate-300 px-3 py-1.5 rounded transition flex items-center gap-1.5" title="Notas del orador (Tecla N)">
        <span>🎙️</span> <span>Notas</span>
      </button>
      <button id="themeBtn" class="bg-slate-800 hover:bg-slate-700 text-slate-300 px-2.5 py-1.5 rounded transition" title="Modo claro/oscuro">
        ☀️
      </button>
      <div class="h-4 w-px bg-slate-700 mx-1"></div>
      <span id="slideCounter" class="font-mono text-brand-400 font-bold px-2 py-1 bg-slate-800/80 rounded border border-slate-700">1 / 9</span>
    </div>
  </header>

  <!-- MAIN SLIDE CONTAINER -->
  <main class="flex-1 max-w-7xl w-full mx-auto p-4 sm:p-6 md:p-8 flex items-center justify-center">
    
    <!-- SLIDE 1: PORTADA -->
    <div class="slide slide-enter w-full space-y-6" data-index="1">
      <div class="bg-gradient-to-br from-slate-900 via-slate-900 to-brand-950/40 border border-slate-800 rounded-2xl p-8 sm:p-12 shadow-2xl relative overflow-hidden">
        <div class="absolute -right-20 -top-20 w-80 h-80 bg-brand-600/10 rounded-full blur-3xl pointer-events-none"></div>
        <div class="inline-block bg-brand-500/10 border border-brand-500/30 text-brand-400 text-xs font-bold tracking-wider uppercase px-3 py-1 rounded-full mb-4">
          Entregable 2 • Deep Learning 2026-2 (PUCP)
        </div>
        <h1 class="text-3xl sm:text-4xl md:text-5xl font-extrabold text-white tracking-tight leading-tight">
          Disección Instrumentada e Intervención Preliminar con LoRA en DINOv2 (ViT-S/14)
        </h1>
        <p class="text-lg sm:text-xl text-slate-300 font-light mt-3">
          Adaptación de Bajo Rango para Clasificación de Uso y Cobertura del Suelo sobre EuroSAT Sentinel-2 RGB
        </p>
        
        <div class="grid grid-cols-1 sm:grid-cols-3 gap-4 mt-8 pt-8 border-t border-slate-800/80 text-sm">
          <div>
            <span class="text-slate-500 block text-xs uppercase font-semibold">Equipo</span>
            <span class="text-slate-200 font-medium">Alfredo Jimenez • Jhonatan Davila • Italo Perez</span>
          </div>
          <div>
            <span class="text-slate-500 block text-xs uppercase font-semibold">Docente e Institución</span>
            <span class="text-slate-200 font-medium">Dr. César Beltrán • IA-PUCP</span>
          </div>
          <div>
            <span class="text-slate-500 block text-xs uppercase font-semibold">Entorno Experimental</span>
            <span class="text-emerald-400 font-medium font-mono">27,000 imágenes • Tesla T4 GPU</span>
          </div>
        </div>
      </div>
    </div>

    <!-- SLIDE 2: MOTIVACIÓN -->
    <div class="slide hidden w-full space-y-6" data-index="2">
      <div class="flex items-center justify-between border-b border-slate-800 pb-3">
        <div>
          <span class="text-xs uppercase tracking-wider text-brand-400 font-bold">1. Planteamiento</span>
          <h2 class="text-2xl font-bold text-white">Motivación y Cambio de Dominio (Domain Shift)</h2>
        </div>
      </div>

      <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div class="space-y-4">
          <div class="bg-slate-900/90 border border-slate-800 rounded-xl p-5 space-y-3">
            <h3 class="text-brand-400 font-bold text-sm uppercase tracking-wide">El Paradigma del Modelo Fundacional</h3>
            <p class="text-sm text-slate-300 leading-relaxed">
              <strong>DINOv2 (Meta AI)</strong> extrae características auto-supervisadas de alta calidad entrenado en 142M de fotos naturales. Sin embargo, transferirlo a satélite implica un <em>domain shift</em> crítico:
            </p>
            <ul class="text-sm text-slate-300 space-y-2 list-disc pl-5">
              <li><strong>Geometría Nadir:</strong> Vista cenital sin horizonte ni perspectivas oblicuas.</li>
              <li><strong>Falta de Sujeto Central:</strong> Escenas completas de texturas y coberturas.</li>
              <li><strong>Ambigüedad Espectral:</strong> Clases lineales (Highway vs River) y vegetación.</li>
            </ul>
          </div>

          <div class="bg-slate-900/90 border border-slate-800 rounded-xl p-5 space-y-3">
            <h3 class="text-brand-400 font-bold text-sm uppercase tracking-wide">El Dilema de Adaptación</h3>
            <p class="text-sm text-slate-300 leading-relaxed">
              • <strong>Linear Probing:</strong> Rápido pero incapaz de re-ponderar la atención espacial.<br>
              • <strong>Full Fine-Tuning:</strong> Modifica 22M+ pesos, arriesga olvido catastrófico y exige GPU pesada.<br>
              • <strong>Solución:</strong> Intervención quirúrgica con LoRA en la atención tardía.
            </p>
          </div>
        </div>

        <div class="bg-slate-900/90 border border-slate-800 rounded-xl p-4 flex flex-col items-center justify-center">
          <img src="{img_samples}" class="rounded-lg shadow-lg max-h-[300px] object-contain cursor-pointer hover:opacity-95 transition" onclick="openLightbox(this.src)" alt="EuroSAT Muestras">
          <span class="text-xs text-slate-400 mt-3 text-center italic">EuroSAT Sentinel-2 RGB (27,000 imágenes, 10 clases de uso de suelo). Haz clic para ampliar.</span>
        </div>
      </div>
    </div>

    <!-- SLIDE 3: HIPÓTESIS Y CONTROL EXPERIMENTAL -->
    <div class="slide hidden w-full space-y-6" data-index="3">
      <div class="flex items-center justify-between border-b border-slate-800 pb-3">
        <div>
          <span class="text-xs uppercase tracking-wider text-brand-400 font-bold">2. Metodología</span>
          <h2 class="text-2xl font-bold text-white">Hipótesis a Priori y Control Experimental</h2>
        </div>
      </div>

      <div class="bg-emerald-950/30 border border-emerald-500/40 rounded-xl p-5 text-slate-200">
        <span class="text-xs uppercase font-bold text-emerald-400 block mb-1">Hipótesis Formal (DESIGN_LOG.md D09):</span>
        <p class="text-sm italic leading-relaxed">
          «Las capas iniciales de DINOv2 codifican filtros de Gabor y frecuencias locales universales. La brecha satelital radica en la síntesis semántica tardía. Inyectar adaptadores LoRA ($r=8, \alpha=8$) únicamente en Query y Value de los bloques 11 y 12 bastará para superar significativamente al linear probing ($p < 0.05$) alterando solo el 0.146% de los parámetros.»
        </p>
      </div>

      <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div class="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-2">
          <span class="text-xs text-brand-400 font-bold uppercase tracking-wider block">Partición 70/15/15</span>
          <p class="text-xs text-slate-400">Semilla fija 42 sobre 27,000 imágenes:</p>
          <div class="text-2xl font-bold text-white">4,050</div>
          <p class="text-xs text-slate-400">Imágenes de test ciego (error estándar &lt; 0.3%).</p>
        </div>

        <div class="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-2">
          <span class="text-xs text-brand-400 font-bold uppercase tracking-wider block">Protocolo Canónico</span>
          <p class="text-xs text-slate-400">Concatenación oficial DINOv2:</p>
          <div class="text-xl font-bold text-white font-mono">[CLS ; Media(Parches)]</div>
          <p class="text-xs text-slate-400">768 dimensiones | 7,690 params en cabeza lineal.</p>
        </div>

        <div class="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-2">
          <span class="text-xs text-brand-400 font-bold uppercase tracking-wider block">Presupuesto Idéntico</span>
          <p class="text-xs text-slate-400">Control experimental riguroso:</p>
          <div class="text-xl font-bold text-white font-mono">AdamW (lr=1e-3)</div>
          <p class="text-xs text-slate-400">8 épocas, batch 64, selección por val Macro-F1.</p>
        </div>
      </div>
    </div>

    <!-- SLIDE 4: TEARDOWN -->
    <div class="slide hidden w-full space-y-6" data-index="4">
      <div class="flex items-center justify-between border-b border-slate-800 pb-3">
        <div>
          <span class="text-xs uppercase tracking-wider text-brand-400 font-bold">3. Fase A</span>
          <h2 class="text-2xl font-bold text-white">Teardown Arquitectónico de DINOv2 (ViT-S/14)</h2>
        </div>
      </div>

      <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div class="space-y-4">
          <div class="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-3">
            <h3 class="text-brand-400 font-bold text-sm uppercase">Disección de Capas y Parámetros</h3>
            <ul class="text-sm text-slate-300 space-y-1.5 list-disc pl-5">
              <li><strong>22,056,576 parámetros</strong> (88.2 MB en fp32).</li>
              <li>12 bloques idénticos (1.775M c/u: MLP 67%, Atención 33%).</li>
              <li><strong>12.25 GFLOPs</strong> analíticos por imagen a 224 px (1.011 GFLOPs/bloque; atención cuadrática = 9.9%).</li>
            </ul>
          </div>

          <div class="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-3">
            <h3 class="text-brand-400 font-bold text-sm uppercase">Perfilado GPU Tesla T4</h3>
            <div class="grid grid-cols-2 gap-3 text-center">
              <div class="bg-slate-800/80 p-2.5 rounded">
                <span class="text-xs text-slate-400 block">Batch 1 (fp32)</span>
                <span class="text-lg font-bold text-white">8.89 ms</span>
                <span class="text-xs text-slate-500 block">112 img/s</span>
              </div>
              <div class="bg-slate-800/80 p-2.5 rounded border border-emerald-500/30">
                <span class="text-xs text-emerald-400 block font-semibold">Batch 128 (fp16)</span>
                <span class="text-lg font-bold text-emerald-400">1.17 ms</span>
                <span class="text-xs text-slate-400 block">853.7 img/s</span>
              </div>
            </div>
            <p class="text-xs text-slate-400 italic">Los Tensor Cores fp16 multiplican el rendimiento por ×3.3.</p>
          </div>
        </div>

        <div class="space-y-4">
          <div class="bg-slate-900 border border-slate-800 rounded-xl p-3 flex flex-col items-center">
            <img src="{img_params}" class="rounded max-h-[160px] object-contain cursor-pointer hover:opacity-95" onclick="openLightbox(this.src)" alt="Parámetros por bloque">
            <span class="text-xs text-slate-500 mt-1">Parámetros por bloque (MLP 67% vs Atención 33%)</span>
          </div>
          <div class="bg-slate-900 border border-slate-800 rounded-xl p-3 flex flex-col items-center">
            <img src="{img_flops}" class="rounded max-h-[160px] object-contain cursor-pointer hover:opacity-95" onclick="openLightbox(this.src)" alt="FLOPs por bloque">
            <span class="text-xs text-slate-500 mt-1">FLOPs por bloque (1.011 GFLOPs c/u)</span>
          </div>
        </div>
      </div>
    </div>

    <!-- SLIDE 5: PCA Y ERF -->
    <div class="slide hidden w-full space-y-6" data-index="5">
      <div class="flex items-center justify-between border-b border-slate-800 pb-3">
        <div>
          <span class="text-xs uppercase tracking-wider text-brand-400 font-bold">4. Fase A</span>
          <h2 class="text-2xl font-bold text-white">PCA de Tokens y Campo Receptivo Efectivo (ERF)</h2>
        </div>
      </div>

      <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div class="bg-slate-900 border border-slate-800 rounded-xl p-4 flex flex-col items-center">
          <img src="{img_pca}" class="rounded max-h-[280px] object-contain cursor-pointer hover:opacity-95" onclick="openLightbox(this.src)" alt="PCA de Tokens">
          <div class="mt-3 text-left w-full text-xs text-slate-300 space-y-1">
            <p class="font-bold text-brand-400">Evolución de Representaciones:</p>
            <p>• <strong>Bloque 4:</strong> El color sigue la coordenada espacial (gradiente posicional).</p>
            <p>• <strong>Bloque 12:</strong> Emerge color uniforme por clase y regiones semánticas claras (carretera vs bosque).</p>
          </div>
        </div>

        <div class="bg-slate-900 border border-slate-800 rounded-xl p-4 flex flex-col items-center">
          <img src="{img_erf}" class="rounded max-h-[280px] object-contain cursor-pointer hover:opacity-95" onclick="openLightbox(this.src)" alt="ERF">
          <div class="mt-3 text-left w-full text-xs text-slate-300 space-y-1">
            <p class="font-bold text-brand-400">Dinámica del ERF (Luo et al.):</p>
            <p>• El ERF es <strong>global desde el bloque 1</strong> (solo 9% en el parche central).</p>
            <p>• Expansión: $r_{{50}}$ crece de 69.4 a 85.0 px; $r_{{90}}$ crece de 117.2 a 127.1 px en el bloque 12.</p>
          </div>
        </div>
      </div>
    </div>

    <!-- SLIDE 6: INTERVENCIÓN LORA -->
    <div class="slide hidden w-full space-y-6" data-index="6">
      <div class="flex items-center justify-between border-b border-slate-800 pb-3">
        <div>
          <span class="text-xs uppercase tracking-wider text-brand-400 font-bold">5. Fase B</span>
          <h2 class="text-2xl font-bold text-white">Intervención en el Núcleo: LoRA en Q y V</h2>
        </div>
      </div>

      <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div class="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-3">
          <h3 class="text-brand-400 font-bold text-sm uppercase">Implementación Propia (~20 líneas)</h3>
          <div class="bg-slate-950 p-3 rounded font-mono text-xs text-emerald-400 border border-slate-800">
            $$W = W_0 + \\frac{{\\alpha}}{{r}}(B \\cdot A)$$
          </div>
          <ul class="text-xs text-slate-300 space-y-1.5 list-disc pl-5">
            <li><strong>Localización:</strong> Query ($Q$) y Value ($V$) en Bloques 11 y 12.</li>
            <li>$W_0 \\in \\mathbb{{R}}^{{384 \\times 384}}$ congelado.</li>
            <li>$A \\in \\mathbb{{R}}^{{8 \\times 384}}$ Kaiming uniforme | $B \\in \\mathbb{{R}}^{{384 \\times 8}}$ en <strong>CERO</strong>.</li>
            <li><strong>Presupuesto:</strong> 4 capas × 6,144 = 24,576 params LoRA + 7,690 cabeza = <strong>32,266 parámetros (0.146%)</strong>.</li>
          </ul>
        </div>

        <div class="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-3">
          <h3 class="text-brand-400 font-bold text-sm uppercase">Caché y Fusión Zero-Overhead</h3>
          <ul class="text-xs text-slate-300 space-y-2 list-disc pl-5">
            <li><strong>Caché de Embeddings:</strong> Bloques 0–10 congelados permiten precomputar representaciones. Tiempo de extracción: <strong>61 segundos</strong> (vs ~8 min recalcular).</li>
            <li><strong>Fusión en Inferencia:</strong> $W_{{eval}} = W_0 + (B \\cdot A)$. Sin latencia extra, idéntica a DINOv2 original (853.7 img/s).</li>
            <li><strong>Verificado:</strong> Salida fusionada == salida LoRA (diferencia estricta = 0.00e+00).</li>
          </ul>
        </div>
      </div>
    </div>

    <!-- SLIDE 7: RESULTADOS GENERALES -->
    <div class="slide hidden w-full space-y-6" data-index="7">
      <div class="flex items-center justify-between border-b border-slate-800 pb-3">
        <div>
          <span class="text-xs uppercase tracking-wider text-brand-400 font-bold">6. Resultados</span>
          <h2 class="text-2xl font-bold text-white">Desempeño en Test (4,050 imágenes)</h2>
        </div>
      </div>

      <div class="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <div class="bg-slate-900 border border-slate-800 rounded-xl p-4 text-center">
          <span class="text-xs text-slate-400 uppercase font-semibold block">Linear Probing</span>
          <div class="text-3xl font-extrabold text-white mt-1">95.80%</div>
          <span class="text-xs text-slate-400 block mt-1">Accuracy: 96.00% (162 errores)</span>
        </div>

        <div class="bg-slate-900 border border-emerald-500/50 rounded-xl p-4 text-center">
          <span class="text-xs text-emerald-400 uppercase font-semibold block">DINOv2 + LoRA</span>
          <div class="text-3xl font-extrabold text-emerald-400 mt-1">97.14%</div>
          <span class="text-xs text-slate-400 block mt-1">Accuracy: 97.28% (110 errores)</span>
        </div>

        <div class="bg-slate-900 border border-brand-500/50 rounded-xl p-4 text-center">
          <span class="text-xs text-brand-400 uppercase font-semibold block">Ganancia Neta</span>
          <div class="text-3xl font-extrabold text-brand-400 mt-1">+1.34%</div>
          <span class="text-xs text-emerald-400 block mt-1">McNemar p = 0.002 (Significativo)</span>
        </div>
      </div>

      <div class="bg-slate-900 border border-slate-800 rounded-xl p-4 flex flex-col items-center">
        <img src="{img_results}" class="rounded max-h-[290px] object-contain cursor-pointer hover:opacity-95" onclick="openLightbox(this.src)" alt="Matriz de Confusión y Curvas">
        <span class="text-xs text-slate-500 mt-2">Matriz de confusión y curvas de aprendizaje en 8 épocas (AdamW, Colab T4). Haz clic para ampliar.</span>
      </div>
    </div>

    <!-- SLIDE 8: DESGLOSE POR CLASE -->
    <div class="slide hidden w-full space-y-6" data-index="8">
      <div class="flex items-center justify-between border-b border-slate-800 pb-3">
        <div>
          <span class="text-xs uppercase tracking-wider text-brand-400 font-bold">7. Análisis Fino</span>
          <h2 class="text-2xl font-bold text-white">Desglose por Clase y Diagnóstico de Fallos</h2>
        </div>
      </div>

      <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div class="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-3">
          <h3 class="text-emerald-400 font-bold text-sm uppercase">Mayores Mejoras con LoRA</h3>
          <ul class="text-xs text-slate-300 space-y-2 list-disc pl-5">
            <li><strong>Highway (Carreteras):</strong> F1 sube de 91.87% a <strong>97.72% (+5.85 pts)</strong>. Resuelve la confusión con suelo desnudo.</li>
            <li><strong>River (Ríos):</strong> F1 sube de 93.80% a <strong>97.51% (+3.71 pts)</strong>. Separa el canal de los árboles circundantes.</li>
            <li><strong>Residential:</strong> 97.89% → 98.77% (+0.88 pts).</li>
            <li><strong>Clases Naturales Estables:</strong> Forest (98.89%) y SeaLake (99.44%) sin olvido catastrófico.</li>
          </ul>
        </div>

        <div class="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-3">
          <h3 class="text-brand-400 font-bold text-sm uppercase">Confusiones Residuales y Causa Física</h3>
          <ul class="text-xs text-slate-300 space-y-2 list-disc pl-5">
            <li><strong>Vegetación Espectralmente Similar:</strong> Confusión entre <em>Pasture</em> y <em>HerbaceousVegetation</em>.</li>
            <li><strong>Causa Física:</strong> EuroSAT RGB solo usa canales visibles (B4, B3, B2). Sin bandas infrarrojas (NIR/SWIR), la firma de reflectancia es casi indistinguible.</li>
            <li><strong>Optimización:</strong> Pérdida de entrenamiento cae a ~0.0003, indicando un ajuste fino muy cercano a la distribución interna.</li>
          </ul>
        </div>
      </div>
    </div>

    <!-- SLIDE 9: CONCLUSIONES -->
    <div class="slide hidden w-full space-y-6" data-index="9">
      <div class="flex items-center justify-between border-b border-slate-800 pb-3">
        <div>
          <span class="text-xs uppercase tracking-wider text-brand-400 font-bold">8. Conclusiones</span>
          <h2 class="text-2xl font-bold text-white">Conclusiones y Hoja de Ruta para Fase C</h2>
        </div>
      </div>

      <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div class="bg-emerald-950/30 border border-emerald-500/40 rounded-xl p-5 space-y-3">
          <h3 class="text-emerald-400 font-bold text-sm uppercase">Conclusiones Clave</h3>
          <ul class="text-xs text-slate-300 space-y-2 list-disc pl-5">
            <li><strong>Eficacia Demostrada:</strong> LoRA en Q y V de bloques 11 y 12 (0.146% de parámetros) supera significativamente al clasificador lineal (<strong>97.14% vs 95.80% Macro-F1, p = 0.002</strong>) en 27,000 imágenes.</li>
            <li><strong>Eficiencia Extrema:</strong> Embedding caching reduce el entrenamiento a 61 segundos y la fusión de pesos garantiza cero sobrecosto en inferencia (853.7 img/s).</li>
          </ul>
        </div>

        <div class="bg-rose-950/30 border border-rose-500/40 rounded-xl p-5 space-y-3">
          <h3 class="text-rose-400 font-bold text-sm uppercase">Honestidad y Fase C (E3)</h3>
          <ul class="text-xs text-slate-300 space-y-2 list-disc pl-5">
            <li><strong>Limitación:</strong> El split 70/15/15 es aleatorio; debido a la autocorrelación espacial, no mide robustez ante cambio geográfico real.</li>
            <li><strong>Plan Fase C:</strong> Validación cruzada espacial en biomas disjuntos de Europa (EuroSAT GeoTIFFs) y evaluación de robustez con Sentinel-1 SAR.</li>
          </ul>
        </div>
      </div>
    </div>

  </main>

  <!-- BOTTOM CONTROLS BAR -->
  <footer class="border-t border-slate-800 bg-slate-900/80 backdrop-blur px-6 py-3 flex items-center justify-between">
    <div class="text-xs text-slate-400 hidden sm:block">
      Navegación: <kbd class="px-1.5 py-0.5 bg-slate-800 border border-slate-700 rounded text-slate-300">←</kbd> <kbd class="px-1.5 py-0.5 bg-slate-800 border border-slate-700 rounded text-slate-300">→</kbd> o <kbd class="px-1.5 py-0.5 bg-slate-800 border border-slate-700 rounded text-slate-300">Espacio</kbd>
    </div>
    <div class="flex items-center space-x-3 mx-auto sm:mx-0">
      <button id="prevBtn" class="bg-slate-800 hover:bg-slate-700 text-white px-4 py-2 rounded-lg text-sm font-semibold transition flex items-center gap-1">
        <span>←</span> <span>Anterior</span>
      </button>
      <button id="nextBtn" class="bg-brand-600 hover:bg-brand-500 text-white px-5 py-2 rounded-lg text-sm font-semibold transition flex items-center gap-1 shadow-lg shadow-brand-600/30">
        <span>Siguiente</span> <span>→</span>
      </button>
    </div>
  </footer>

  <!-- LIGHTBOX MODAL -->
  <div id="lightbox" class="fixed inset-0 bg-black/90 backdrop-blur-sm z-50 hidden flex items-center justify-center p-4" onclick="closeLightbox()">
    <img id="lightboxImg" class="max-w-full max-h-[90vh] rounded-lg shadow-2xl" src="" alt="Enlarged view">
    <button class="absolute top-6 right-6 text-white text-3xl font-bold bg-slate-800/80 hover:bg-slate-700 w-10 h-10 rounded-full flex items-center justify-center" onclick="closeLightbox()">&times;</button>
  </div>

  <script>
    let currentSlide = 1;
    const totalSlides = 9;

    function showSlide(index) {{
      if (index < 1) index = 1;
      if (index > totalSlides) index = totalSlides;
      currentSlide = index;

      document.querySelectorAll('.slide').forEach(s => {{
        s.classList.add('hidden');
        s.classList.remove('slide-enter');
      }});

      const target = document.querySelector(`.slide[data-index="${{index}}"]`);
      if (target) {{
        target.classList.remove('hidden');
        target.classList.add('slide-enter');
      }}

      document.getElementById('slideCounter').innerText = `${{index}} / ${{totalSlides}}`;
      document.getElementById('prevBtn').disabled = (index === 1);
    }}

    document.getElementById('nextBtn').addEventListener('click', () => showSlide(currentSlide + 1));
    document.getElementById('prevBtn').addEventListener('click', () => showSlide(currentSlide - 1));

    document.addEventListener('keydown', (e) => {{
      if (e.key === 'ArrowRight' || e.key === ' ') {{
        showSlide(currentSlide + 1);
      }} else if (e.key === 'ArrowLeft') {{
        showSlide(currentSlide - 1);
      }} else if (e.key === 'Escape') {{
        closeLightbox();
      }}
    }});

    function openLightbox(src) {{
      document.getElementById('lightboxImg').src = src;
      document.getElementById('lightbox').classList.remove('hidden');
    }}

    function closeLightbox() {{
      document.getElementById('lightbox').classList.add('hidden');
    }}

    // Theme toggle
    document.getElementById('themeBtn').addEventListener('click', () => {{
      document.documentElement.classList.toggle('dark');
      document.getElementById('themeBtn').innerText = document.documentElement.classList.contains('dark') ? '☀️' : '🌙';
    }});

    showSlide(1);
  </script>
</body>
</html>
"""

OUT_HTML_WAIMLAP.write_text(html, encoding="utf-8")
OUT_HTML_ROOT.write_text(html, encoding="utf-8")
print(f"Successfully generated: {OUT_HTML_WAIMLAP}")
