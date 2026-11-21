import base64
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT_HTML = ROOT / "diapositivas.html"
FIG = ROOT / "outputs" / "figures"

def b64(path):
    if path.exists():
        return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("utf-8")
    return ""

img_samples = b64(FIG / "dataset_samples.png")
img_teardown = b64(FIG / "teardown.png")
img_features = b64(FIG / "intermediate_features.png")
img_erf = b64(FIG / "effective_receptive_field.png")
img_curves = b64(FIG / "learning_curves.png")
img_cm_lora = b64(FIG / "confusion_lora.png")
img_failure = b64(FIG / "failure_examples.png")

html_content = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Disección Instrumentada e Intervención con LoRA — DINOv2 EuroSAT</title>
  <!-- Tailwind CSS -->
  <script src="https://cdn.tailwindcss.com"></script>
  <!-- KaTeX -->
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js"
          onload="renderMathInElement(document.body);"></script>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
    body {{
      font-family: 'Inter', sans-serif;
      background-color: #0f172a;
      color: #f8fafc;
      overflow: hidden;
      user-select: none;
    }}
    .slide {{
      display: none;
      height: 100vh;
      width: 100vw;
      padding: 3rem 4rem;
      box-sizing: border-box;
      flex-direction: column;
      justify-content: space-between;
      animation: fadeIn 0.25s ease-out forwards;
    }}
    .slide.active {{
      display: flex;
    }}
    @keyframes fadeIn {{
      from {{ opacity: 0; transform: translateY(6px); }}
      to {{ opacity: 1; transform: translateY(0); }}
    }}
    .card {{
      background: rgba(30, 41, 59, 0.7);
      border: 1px solid rgba(51, 65, 85, 0.8);
      backdrop-filter: blur(12px);
      border-radius: 1rem;
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      padding: 0.25rem 0.75rem;
      border-radius: 9999px;
      font-size: 0.75rem;
      font-weight: 700;
      letter-spacing: 0.05em;
      text-transform: uppercase;
    }}
    .stat-pill {{
      background: rgba(15, 23, 42, 0.6);
      border: 1px solid rgba(51, 65, 85, 0.6);
      border-radius: 0.75rem;
      padding: 0.75rem 1rem;
    }}
  </style>
</head>
<body>

  <!-- Slide 1: Portada -->
  <div class="slide active" id="slide-1">
    <div class="flex items-center justify-between border-b border-slate-800 pb-4">
      <div class="flex items-center space-x-3">
        <span class="badge bg-sky-500/10 text-sky-400 border border-sky-500/30">PUCP • 1INF52 Deep Learning</span>
        <span class="text-sm text-slate-400 font-medium">Ciclo 2026-2</span>
      </div>
      <div class="text-sm font-mono text-slate-500">Prof. Dr. César A. Beltrán Castañón</div>
    </div>

    <div class="my-auto max-w-5xl">
      <div class="inline-block px-3 py-1 rounded-full bg-emerald-500/10 text-emerald-400 text-xs font-semibold tracking-wider uppercase mb-4 border border-emerald-500/20">
        Entregable 2 • Proyecto Final
      </div>
      <h1 class="text-5xl font-extrabold tracking-tight text-white mb-6 leading-tight">
        Disección Instrumentada e Intervención Preliminar con LoRA en <span class="text-transparent bg-clip-text bg-gradient-to-r from-sky-400 to-indigo-400">DINOv2 (ViT-S/14)</span>
      </h1>
      <p class="text-xl text-slate-300 font-normal leading-relaxed mb-8 max-w-4xl">
        Adaptación eficiente de representaciones autosupervisadas para clasificación de cobertura del suelo (Sentinel-2 EuroSAT RGB) bajo restricciones estrictas de cómputo en CPU.
      </p>

      <div class="grid grid-cols-4 gap-4 max-w-4xl">
        <div class="stat-pill">
          <div class="text-xs text-slate-400 uppercase tracking-wider mb-1">Backbone Total</div>
          <div class="text-2xl font-bold text-white">22.06 M</div>
          <div class="text-xs text-slate-500">Parámetros (ViT-S/14)</div>
        </div>
        <div class="stat-pill border-emerald-500/30 bg-emerald-500/5">
          <div class="text-xs text-emerald-400 uppercase tracking-wider mb-1">LoRA Adaptado</div>
          <div class="text-2xl font-bold text-emerald-400">0.128%</div>
          <div class="text-xs text-slate-400">28,426 params entrenados</div>
        </div>
        <div class="stat-pill">
          <div class="text-xs text-slate-400 uppercase tracking-wider mb-1">Ganancia Test</div>
          <div class="text-2xl font-bold text-sky-400">+3.89%</div>
          <div class="text-xs text-slate-500">Macro-F1 (90.14% → 94.03%)</div>
        </div>
        <div class="stat-pill">
          <div class="text-xs text-slate-400 uppercase tracking-wider mb-1">Entorno CPU</div>
          <div class="text-2xl font-bold text-white">97 ms</div>
          <div class="text-xs text-slate-500">Latencia forward (i5-1135G7)</div>
        </div>
      </div>
    </div>

    <div class="flex items-center justify-between text-xs text-slate-500 border-t border-slate-800 pt-4">
      <div>Hardware: Intel Core i5-1135G7 (4C/8T) • 8 GB RAM • float32 • Semilla fija 2026</div>
      <div class="flex items-center space-x-2">
        <span>Usa las teclas</span>
        <kbd class="px-2 py-0.5 rounded bg-slate-800 text-slate-300 font-mono">←</kbd>
        <kbd class="px-2 py-0.5 rounded bg-slate-800 text-slate-300 font-mono">→</kbd>
        <span>o Espacio para navegar</span>
      </div>
    </div>
  </div>

  <!-- Slide 2: Problema y Pregunta -->
  <div class="slide" id="slide-2">
    <div class="flex items-center justify-between border-b border-slate-800 pb-3">
      <span class="badge bg-sky-500/10 text-sky-400 border border-sky-500/30">1. Contexto y Problema</span>
      <span class="text-xs text-slate-500">Entregable 2 • Deep Learning</span>
    </div>

    <div class="grid grid-cols-3 gap-6 my-auto">
      <div class="card p-6 flex flex-col justify-between">
        <div>
          <div class="w-10 h-10 rounded-lg bg-sky-500/20 text-sky-400 flex items-center justify-center font-bold text-lg mb-4">1</div>
          <h3 class="text-lg font-bold text-white mb-2">El Desafío de Dominio</h3>
          <p class="text-sm text-slate-300 leading-relaxed mb-4">
            <strong class="text-white">DINOv2:</strong> Preentrenado con 142M de fotos normales de internet (perspectiva horizontal, rostros, iluminación artificial).
          </p>
          <p class="text-sm text-slate-300 leading-relaxed">
            <strong class="text-white">EuroSAT:</strong> Imágenes del satélite Sentinel-2 en vista cenital (100% perpendicular), sin horizonte, con resolución de 10 metros/píxel.
          </p>
        </div>
        <div class="mt-4 p-3 rounded-lg bg-slate-900/60 border border-slate-800 text-xs text-sky-300">
          ¿Cómo transferir este modelo fundacional a un dominio visual tan dispar?
        </div>
      </div>

      <div class="card p-6 flex flex-col justify-between">
        <div>
          <div class="w-10 h-10 rounded-lg bg-amber-500/20 text-amber-400 flex items-center justify-center font-bold text-lg mb-4">2</div>
          <h3 class="text-lg font-bold text-white mb-2">Restricciones y Dilema</h3>
          <ul class="text-sm text-slate-300 space-y-3">
            <li><strong class="text-white">• Hardware Estudiantil:</strong> Laptop estándar con 8 GB RAM y CPU Intel i5, sin acceso a GPU CUDA.</li>
            <li><strong class="text-white">• Riesgo Full Fine-Tuning:</strong> Reentrenar 22M de pesos es inviable en CPU y provoca sobreajuste severo con pocas muestras.</li>
            <li><strong class="text-white">• Línea Base Congelada:</strong> Solo entrena una cabeza lineal (3,850 params), dejando las características internas rígidas.</li>
          </ul>
        </div>
        <div class="mt-4 p-3 rounded-lg bg-slate-900/60 border border-slate-800 text-xs text-amber-300">
          Se requiere una estrategia de ajuste fino eficiente en parámetros (PEFT).
        </div>
      </div>

      <div class="card p-6 flex flex-col justify-between border-sky-500/40 bg-sky-950/20">
        <div>
          <div class="w-10 h-10 rounded-lg bg-emerald-500/20 text-emerald-400 flex items-center justify-center font-bold text-lg mb-4">3</div>
          <h3 class="text-lg font-bold text-white mb-2">La Hipótesis a Priori (D04)</h3>
          <p class="text-sm text-slate-200 leading-relaxed mb-3">
            Inyectar adaptadores de bajo rango (<strong class="text-emerald-400">LoRA, $r=8$</strong>) en las proyecciones Query y Value ($W_q, W_v$) de los <strong>dos últimos bloques (11 y 12)</strong> permitirá especializar el espacio de atención satelital.
          </p>
          <p class="text-sm text-slate-200 leading-relaxed">
            Se conjetura una mejora estadísticamente verificable en Macro-F1 frente a la línea base congelada, con un costo paramétrico marginal (&lt; 0.15%).
          </p>
        </div>
        <div class="mt-4 p-3 rounded-lg bg-emerald-950/40 border border-emerald-500/30 text-xs text-emerald-300 font-mono">
          Registrado formalmente en DECISION_LOG.md antes de ejecutar.
        </div>
      </div>
    </div>

    <div class="text-xs text-slate-500 text-right">Diapositiva 2 / 9</div>
  </div>

  <!-- Slide 3: Auditoría de Datos -->
  <div class="slide" id="slide-3">
    <div class="flex items-center justify-between border-b border-slate-800 pb-3">
      <span class="badge bg-sky-500/10 text-sky-400 border border-sky-500/30">2. Fase A — Auditoría de Integridad y Datos</span>
      <span class="text-xs text-slate-500">Entregable 2 • Deep Learning</span>
    </div>

    <div class="grid grid-cols-12 gap-6 my-auto items-center">
      <div class="col-span-5 space-y-4">
        <div class="card p-5">
          <h3 class="text-base font-bold text-white mb-2 flex items-center space-x-2">
            <span class="w-2.5 h-2.5 rounded-full bg-emerald-400"></span>
            <span>Auditoría de Integridad (SHA-256)</span>
          </h3>
          <ul class="text-xs text-slate-300 space-y-2">
            <li>• <strong>27,000 imágenes</strong> crudas leídas y hasheadas.</li>
            <li>• <strong>27,000 hashes únicos</strong> $\implies$ <strong>0 duplicados exactos</strong>.</li>
            <li>• <strong>10 Clases balanceadas:</strong> AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake.</li>
          </ul>
        </div>

        <div class="card p-5">
          <h3 class="text-base font-bold text-white mb-2">Partición Experimental (Semilla 2026)</h3>
          <div class="grid grid-cols-3 gap-2 text-center text-xs">
            <div class="p-2 bg-slate-900/60 rounded border border-slate-800">
              <div class="text-slate-400">Train</div>
              <div class="text-lg font-bold text-white">1,000</div>
              <div class="text-[10px] text-slate-500">100 / clase</div>
            </div>
            <div class="p-2 bg-slate-900/60 rounded border border-slate-800">
              <div class="text-slate-400">Val</div>
              <div class="text-lg font-bold text-white">300</div>
              <div class="text-[10px] text-slate-500">30 / clase</div>
            </div>
            <div class="p-2 bg-slate-900/60 rounded border border-slate-800">
              <div class="text-slate-400">Test</div>
              <div class="text-lg font-bold text-white">500</div>
              <div class="text-[10px] text-slate-500">50 / clase</div>
            </div>
          </div>
        </div>

        <div class="p-4 rounded-xl bg-amber-500/10 border border-amber-500/30 text-xs text-amber-200 leading-relaxed">
          <strong>Decisión D03 (Honestidad Epistemológica):</strong> Los JPEG de EuroSAT no incluyen coordenadas GPS en EXIF. Declaramos formalmente una partición <em>aleatoria estratificada interna</em>. No se simula división geográfica en E2; la partición espacial estricta se reserva para la Fase C con GeoTIFFs.
        </div>
      </div>

      <div class="col-span-7 card p-4">
        <div class="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Muestras del Subconjunto de Entrenamiento (10 clases)</div>
        <img src="{img_samples}" alt="EuroSAT Samples" class="w-full rounded-lg border border-slate-800">
      </div>
    </div>

    <div class="text-xs text-slate-500 text-right">Diapositiva 3 / 9</div>
  </div>

  <!-- Slide 4: Disección de la Arquitectura (Teardown) -->
  <div class="slide" id="slide-4">
    <div class="flex items-center justify-between border-b border-slate-800 pb-3">
      <span class="badge bg-sky-500/10 text-sky-400 border border-sky-500/30">3. Fase A — Disección de la Arquitectura DINOv2</span>
      <span class="text-xs text-slate-500">Entregable 2 • Deep Learning</span>
    </div>

    <div class="grid grid-cols-12 gap-6 my-auto items-center">
      <div class="col-span-5 space-y-4">
        <div class="card p-5">
          <h3 class="text-base font-bold text-white mb-3">Conteo Analítico de Operaciones</h3>
          <ul class="text-xs text-slate-300 space-y-2.5">
            <li>• <strong>Tokenización:</strong> Imagen $224 \times 224 \to 256$ parches ($14 \times 14$) + 1 CLS = <strong>257 tokens</strong> ($d=384$).</li>
            <li>• <strong>12 Bloques Idénticos:</strong> Multi-Head Attention (6 cabezas) + FFN SwiGLU ($d_{{ffn}}=1536$).</li>
            <li>• <strong>Parámetros por bloque:</strong> 1,775,232 $\implies$ Total: <strong>22,056,576 parámetros</strong>.</li>
            <li>• <strong>FLOPs Analíticos (2 FLOPs/MAC):</strong>
              <div class="ml-3 mt-1 text-slate-400 font-mono text-[11px]">
                - Proyecciones densas: ~909.5 MFLOPs / bloque<br>
                - Matmuls de atención ($QK^T$, $AV$): ~101.5 MFLOPs / bloque<br>
                - Gran Total Forward (batch 1): <strong class="text-sky-400">12.25 GFLOPs</strong>
              </div>
            </li>
          </ul>
        </div>

        <div class="grid grid-cols-2 gap-3">
          <div class="stat-pill">
            <div class="text-[10px] text-slate-400 uppercase">Latencia Mediana CPU</div>
            <div class="text-xl font-bold text-white">97.29 ms</div>
            <div class="text-[10px] text-slate-500">p95: 114.12 ms (batch 1)</div>
          </div>
          <div class="stat-pill">
            <div class="text-[10px] text-slate-400 uppercase">Memoria Residente (RSS)</div>
            <div class="text-xl font-bold text-white">504.7 MiB</div>
            <div class="text-[10px] text-slate-500">Pico medido cada 10ms</div>
          </div>
        </div>
      </div>

      <div class="col-span-7 card p-4">
        <div class="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Desglose de Parámetros y GFLOPs por Bloque</div>
        <img src="{img_teardown}" alt="Teardown DINOv2" class="w-full rounded-lg border border-slate-800">
      </div>
    </div>

    <div class="text-xs text-slate-500 text-right">Diapositiva 4 / 9</div>
  </div>

  <!-- Slide 5: Representaciones y ERF -->
  <div class="slide" id="slide-5">
    <div class="flex items-center justify-between border-b border-slate-800 pb-3">
      <span class="badge bg-sky-500/10 text-sky-400 border border-sky-500/30">4. Fase A — Representaciones y Campo Receptivo Efectivo</span>
      <span class="text-xs text-slate-500">Entregable 2 • Deep Learning</span>
    </div>

    <div class="grid grid-cols-2 gap-6 my-auto">
      <div class="card p-5 flex flex-col justify-between">
        <div>
          <div class="flex items-center justify-between mb-2">
            <h3 class="text-base font-bold text-white">Proyección PC1 de Tokens (SVD)</h3>
            <span class="text-[11px] text-sky-400 font-mono">Bloques 1, 6 y 12</span>
          </div>
          <p class="text-xs text-slate-300 leading-relaxed mb-4">
            Evolución de las activaciones: en el <strong>Bloque 1</strong> dominan texturas y gradientes locales; en el <strong>Bloque 6</strong> surgen estructuras lineales; en el <strong>Bloque 12</strong> se forman clusters semánticos limpios de parcelas y caminos.
          </p>
        </div>
        <img src="{img_features}" alt="Intermediate Features" class="w-full rounded-lg border border-slate-800">
      </div>

      <div class="card p-5 flex flex-col justify-between">
        <div>
          <div class="flex items-center justify-between mb-2">
            <h3 class="text-base font-bold text-white">Campo Receptivo Efectivo (ERF)</h3>
            <span class="text-[11px] text-emerald-400 font-mono">Autogradiente Token Central</span>
          </div>
          <p class="text-xs text-slate-300 leading-relaxed mb-4">
            Fracción de píxeles para cubrir el <strong>90% de la masa de gradiente</strong>:
            <span class="text-white font-bold">Bloque 1: 52.16%</span> $\to$
            <span class="text-emerald-400 font-bold">Bloque 12: 68.03%</span>.
            La autoatención expande la influencia global a través de toda la escena.
          </p>
        </div>
        <img src="{img_erf}" alt="Effective Receptive Field" class="w-full rounded-lg border border-slate-800">
      </div>
    </div>

    <div class="text-xs text-slate-500 text-right">Diapositiva 5 / 9</div>
  </div>

  <!-- Slide 6: LoRA y Prefix Caching -->
  <div class="slide" id="slide-6">
    <div class="flex items-center justify-between border-b border-slate-800 pb-3">
      <span class="badge bg-sky-500/10 text-sky-400 border border-sky-500/30">5. Fase B — Intervención con LoRA y Prefix Caching</span>
      <span class="text-xs text-slate-500">Entregable 2 • Deep Learning</span>
    </div>

    <div class="grid grid-cols-3 gap-6 my-auto">
      <div class="card p-6 flex flex-col justify-between">
        <div>
          <h3 class="text-base font-bold text-white mb-2">Formulación Matemática</h3>
          <div class="p-3 my-3 bg-slate-900/80 rounded-lg border border-slate-800 text-center text-sky-300 font-mono text-sm">
            $W_{{adapt}} = W_0 + \\frac{{\\alpha}}{{r}} (B \\cdot A)$
          </div>
          <ul class="text-xs text-slate-300 space-y-2 mt-4">
            <li>• $W_0 \\in \\mathbb{{R}}^{{384 \\times 384}}$: Congelada.</li>
            <li>• $A \\sim \\mathcal{{N}}(0, \\sigma^2)$, rango $r=8$.</li>
            <li>• $B = 0$ al inicio $\\implies$ <strong>$\Delta W = 0$ en $t=0$</strong>.</li>
            <li>• <strong>Identidad inicial estricta:</strong> Error de equivalencia al inicio = <strong class="text-emerald-400">0.00000</strong>.</li>
          </ul>
        </div>
        <div class="text-[11px] text-slate-400 bg-slate-900/50 p-2.5 rounded border border-slate-800">
          Inicia exactamente en el estado preentrenado de DINOv2 sin desestabilizar la red.
        </div>
      </div>

      <div class="card p-6 flex flex-col justify-between">
        <div>
          <h3 class="text-base font-bold text-white mb-2">Localización y Eficiencia</h3>
          <p class="text-xs text-slate-300 mb-3">
            Inyectado exclusivamente en <strong class="text-white">Query ($W_q$)</strong> y <strong class="text-white">Value ($W_v$)</strong> de los <strong>Bloques 11 y 12</strong>. Key ($K$) permanece congelada.
          </p>
          <div class="space-y-2 text-xs">
            <div class="flex justify-between py-1 border-b border-slate-800">
              <span class="text-slate-400">Params LoRA (Q, V en 2 bloques):</span>
              <span class="font-bold text-white font-mono">24,576</span>
            </div>
            <div class="flex justify-between py-1 border-b border-slate-800">
              <span class="text-slate-400">Cabeza lineal ($384 \\to 10$):</span>
              <span class="font-bold text-white font-mono">3,850</span>
            </div>
            <div class="flex justify-between py-1 text-emerald-400">
              <span class="font-bold">Total entrenable:</span>
              <span class="font-bold font-mono">28,426 (0.128%)</span>
            </div>
          </div>
        </div>
        <div class="text-[11px] text-slate-400 bg-slate-900/50 p-2.5 rounded border border-slate-800">
          Solo se entrena el 0.128% del volumen total del modelo fundacional.
        </div>
      </div>

      <div class="card p-6 flex flex-col justify-between border-emerald-500/40 bg-emerald-950/20">
        <div>
          <h3 class="text-base font-bold text-white mb-2 flex items-center space-x-2">
            <span class="badge bg-emerald-500/20 text-emerald-300 text-[10px]">Aceleración CPU</span>
            <span>Prefix Caching Exacto</span>
          </h3>
          <p class="text-xs text-slate-300 leading-relaxed mb-3">
            Los Bloques 1 a 10 están congelados en modo <code>eval()</code>. Como las imágenes son deterministas, sus salidas intermedias $z_{{10}}$ se precomputan una sola vez.
          </p>
          <ul class="text-xs text-slate-300 space-y-2">
            <li>• Guardado en <code>cache/prefix.npy</code> (float32).</li>
            <li>• Error máximo absoluto: <strong class="text-emerald-400">&lt; $10^{{-5}}$</strong>.</li>
            <li>• <strong>Aceleración: &gt; 80%</strong> en tiempo de época en CPU.</li>
          </ul>
        </div>
        <div class="text-[11px] text-emerald-300 bg-emerald-900/40 p-2.5 rounded border border-emerald-500/30 font-mono">
          Equivalencia matemática demostrada empíricamente.
        </div>
      </div>
    </div>

    <div class="text-xs text-slate-500 text-right">Diapositiva 6 / 9</div>
  </div>

  <!-- Slide 7: Resultados de la Ablación -->
  <div class="slide" id="slide-7">
    <div class="flex items-center justify-between border-b border-slate-800 pb-3">
      <span class="badge bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">6. Fase B — Resultados y Contraste de Hipótesis</span>
      <span class="text-xs text-slate-500">Entregable 2 • Deep Learning</span>
    </div>

    <div class="grid grid-cols-12 gap-6 my-auto items-center">
      <div class="col-span-6 space-y-4">
        <div class="card p-5">
          <div class="text-xs text-emerald-400 font-bold uppercase tracking-wider mb-1">Confirmación de la Hipótesis a Priori</div>
          <div class="text-4xl font-extrabold text-white mb-2">
            +3.89% <span class="text-lg font-medium text-slate-400">en Macro-F1 de Test Ciego</span>
          </div>
          <p class="text-xs text-slate-300 leading-relaxed">
            Evaluado una sola vez sobre 500 imágenes de prueba tras seleccionar el mejor punto de control por validación (Época 3).
          </p>
        </div>

        <div class="card overflow-hidden">
          <table class="w-full text-xs text-left">
            <thead class="bg-slate-900/80 text-slate-400 uppercase font-mono text-[10px]">
              <tr>
                <th class="p-3">Métrica</th>
                <th class="p-3">Línea Base (Linear Probe)</th>
                <th class="p-3 text-emerald-400">Intervención LoRA</th>
              </tr>
            </thead>
            <tbody class="divide-y divide-slate-800 text-slate-200">
              <tr>
                <td class="p-3 font-semibold text-white">Parámetros entrenables</td>
                <td class="p-3 font-mono">3,850 (0.017%)</td>
                <td class="p-3 font-mono text-emerald-400 font-bold">28,426 (0.128%)</td>
              </tr>
              <tr>
                <td class="p-3 font-semibold text-white">Val Macro-F1 (mejor)</td>
                <td class="p-3 font-mono">88.49% (Época 5)</td>
                <td class="p-3 font-mono text-emerald-400 font-bold">93.03% (Época 3)</td>
              </tr>
              <tr class="bg-emerald-500/5">
                <td class="p-3 font-semibold text-white">Test Accuracy</td>
                <td class="p-3 font-mono">90.20%</td>
                <td class="p-3 font-mono text-emerald-400 font-bold">94.00% (+3.80%)</td>
              </tr>
              <tr class="bg-emerald-500/10">
                <td class="p-3 font-semibold text-white">Test Macro-F1</td>
                <td class="p-3 font-mono">90.14%</td>
                <td class="p-3 font-mono text-emerald-400 font-bold">94.03% (+3.89%)</td>
              </tr>
              <tr>
                <td class="p-3 font-semibold text-white">Latencia Despliegue CPU</td>
                <td class="p-3 font-mono">97.29 ms</td>
                <td class="p-3 font-mono text-slate-300">103.58 ms (+6.3 ms)</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <div class="col-span-6 card p-4">
        <div class="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Curvas de Aprendizaje (Pérdida y Macro-F1 Val)</div>
        <img src="{img_curves}" alt="Learning Curves" class="w-full rounded-lg border border-slate-800">
      </div>
    </div>

    <div class="text-xs text-slate-500 text-right">Diapositiva 7 / 9</div>
  </div>

  <!-- Slide 8: Diagnóstico de Fallos -->
  <div class="slide" id="slide-8">
    <div class="flex items-center justify-between border-b border-slate-800 pb-3">
      <span class="badge bg-amber-500/10 text-amber-400 border border-amber-500/30">7. Diagnóstico Cualitativo de Fallos Residuales</span>
      <span class="text-xs text-slate-500">Entregable 2 • Deep Learning</span>
    </div>

    <div class="grid grid-cols-2 gap-6 my-auto items-center">
      <div class="card p-4">
        <div class="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Matriz de Confusión LoRA (Test 500 imágenes)</div>
        <img src="{img_cm_lora}" alt="Confusion Matrix LoRA" class="w-full rounded-lg border border-slate-800">
      </div>

      <div class="space-y-4">
        <div class="card p-4">
          <div class="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Primeros Errores en Orden del Manifiesto</div>
          <img src="{img_failure}" alt="Failure Examples" class="w-full rounded-lg border border-slate-800 mb-3">
        </div>

        <div class="card p-4 bg-amber-500/5 border-amber-500/20">
          <h4 class="text-xs font-bold text-amber-300 uppercase tracking-wider mb-2">Causas Físicas de Confusión Espectral:</h4>
          <ul class="text-xs text-slate-300 space-y-2">
            <li>• <strong class="text-white">Highway vs. River:</strong> Geometría lineal serpenteante similar en 10 m/píxel (64×64 píxeles).</li>
            <li>• <strong class="text-white">Pasture vs. HerbaceousVegetation:</strong> Firmas espectrales casi idénticas en el rango visible (RGB); requieren bandas no visibles (NIR / Red Edge).</li>
          </ul>
        </div>
      </div>
    </div>

    <div class="text-xs text-slate-500 text-right">Diapositiva 8 / 9</div>
  </div>

  <!-- Slide 9: Conclusiones y Hoja de Ruta -->
  <div class="slide" id="slide-9">
    <div class="flex items-center justify-between border-b border-slate-800 pb-3">
      <span class="badge bg-sky-500/10 text-sky-400 border border-sky-500/30">8. Conclusiones y Hoja de Ruta hacia la Fase C</span>
      <span class="text-xs text-slate-500">Entregable 2 • Deep Learning</span>
    </div>

    <div class="grid grid-cols-2 gap-6 my-auto">
      <div class="card p-6 flex flex-col justify-between">
        <div>
          <h3 class="text-lg font-bold text-white mb-4 flex items-center space-x-2">
            <span class="text-emerald-400">✔</span>
            <span>Conclusiones del Entregable 2</span>
          </h3>
          <ul class="text-xs text-slate-300 space-y-3.5">
            <li>
              <strong class="text-white">• Hipótesis Validada:</strong> LoRA en $Q, V$ de bloques 11 y 12 supera a la línea base en <strong class="text-emerald-400">+3.89% Macro-F1</strong> en test ciego.
            </li>
            <li>
              <strong class="text-white">• Eficiencia de Parámetros:</strong> Se especializó el modelo fundacional con apenas <strong class="text-sky-400">28,426 parámetros</strong> (0.128% de la red).
            </li>
            <li>
              <strong class="text-white">• Viabilidad en CPU:</strong> Latencia de despliegue prácticamente inalterada (103 ms vs 97 ms), corriendo 100% en una laptop común.
            </li>
            <li>
              <strong class="text-white">• Rigor de Auditoría:</strong> Verificación estricta de hashes SHA-256, inicialización idéntica y reproducibilidad con semilla fija 2026.
            </li>
          </ul>
        </div>
        <div class="p-3 bg-emerald-500/10 rounded-lg border border-emerald-500/20 text-xs text-emerald-300 font-medium">
          Hito de la Fase B cumplido con métricas y evidencias auditables.
        </div>
      </div>

      <div class="card p-6 flex flex-col justify-between border-sky-500/30">
        <div>
          <h3 class="text-lg font-bold text-white mb-4 flex items-center space-x-2">
            <span class="text-sky-400">➔</span>
            <span>Hoja de Ruta hacia la Fase C (Entregable 3)</span>
          </h3>
          <ul class="text-xs text-slate-300 space-y-3.5">
            <li>
              <strong class="text-white">• 1. Partición Geográfica Estricta:</strong> Vincular las imágenes con metadatos espaciales GeoTIFF para evaluar generalización regional disjunta.
            </li>
            <li>
              <strong class="text-white">• 2. Bandas Multiespectrales:</strong> Incorporar canales Red Edge, NIR y SWIR para resolver la ambigüedad entre pastizales y vegetación.
            </li>
            <li>
              <strong class="text-white">• 3. Ablación de Hiperparámetros LoRA:</strong> Evaluar variación de rango $r \\in \\{{4, 8, 16\\}}$ y adaptación en FFN SwiGLU.
            </li>
            <li>
              <strong class="text-white">• 4. Aumentos Cenitales Invariantes:</strong> Integrar rotaciones de 90°, 180°, 270° y reflexiones espaciales durante el entrenamiento.
            </li>
          </ul>
        </div>
        <div class="p-3 bg-sky-500/10 rounded-lg border border-sky-500/20 text-xs text-sky-300 font-medium">
          Próximo hito: Evaluación de robustez espacial y transferencia multimodal.
        </div>
      </div>
    </div>

    <div class="flex items-center justify-between text-xs text-slate-500 border-t border-slate-800 pt-3">
      <div>PUCP • 1INF52 Deep Learning 2026-2</div>
      <div class="text-slate-400 font-semibold">¡Muchas gracias! Preguntas y Discusión</div>
      <div>Diapositiva 9 / 9</div>
    </div>
  </div>

  <!-- Barra de Navegación Inferior -->
  <div class="fixed bottom-0 left-0 right-0 h-1 bg-slate-800">
    <div id="progress" class="h-full bg-gradient-to-r from-sky-500 to-emerald-400 transition-all duration-300" style="width: 11.11%;"></div>
  </div>

  <script>
    let currentSlide = 1;
    const totalSlides = 9;

    function showSlide(n) {{
      if (n < 1) n = 1;
      if (n > totalSlides) n = totalSlides;
      currentSlide = n;

      document.querySelectorAll('.slide').forEach((s, idx) => {{
        s.classList.toggle('active', idx + 1 === currentSlide);
      }});

      document.getElementById('progress').style.width = ((currentSlide / totalSlides) * 100) + '%';
    }}

    document.addEventListener('keydown', (e) => {{
      if (e.key === 'ArrowRight' || e.key === ' ' || e.key === 'PageDown') {{
        showSlide(currentSlide + 1);
      }} else if (e.key === 'ArrowLeft' || e.key === 'PageUp') {{
        showSlide(currentSlide - 1);
      }} else if (e.key === 'f' || e.key === 'F') {{
        if (!document.fullscreenElement) {{
          document.documentElement.requestFullscreen();
        }} else {{
          document.exitFullscreen();
        }}
      }}
    }});
  </script>
</body>
</html>
"""

OUT_HTML.write_text(html_content, encoding="utf8")
print(f"Interactive HTML presentation generated at: {OUT_HTML}")
