# 🎓 Guía Completa de Presentación y Defensa en WAIMLAp 2026
**Proyecto:** *Dissecting and Adapting DINOv2 for Land Cover Classification (EuroSAT)*  
**Autor:** Italo Perez | **Curso:** Deep Learning (1INF52) - PUCP | **Docente:** Dr. César Beltrán Castañón  
**Evento:** Workshop on Artificial Intelligence and Machine Learning Applications (WAIMLAp 2026)

---

## 📌 1. ¿Cómo Funciona la Sesión de Pósters en WAIMLAp?

En WAIMLAp rige el principio: **"Tu póster es tu entrada"** (*Bring something worth discussing*).
A diferencia de una exposición tradicional en proyector, en una sesión de pósters científicos:
1. **No hay un tiempo rígido de 20 minutos con diapositivas:** Estarás de pie junto a tu panel A0 asignado durante la franja horaria del workshop (18–20 de noviembre en el campus PUCP).
2. **Dinámica abierta y diversa:** Quienes se acerquen a tu póster serán:
   * **Curiosos / Estudiantes:** Querrán un resumen rápido y amigable de 1 a 2 minutos.
   * **Investigadores / Jurado Evaluador (p. ej., Dr. Beltrán o profesores del comité IA-PUCP):** Querrán profundizar en la metodología, cuestionar decisiones arquitectónicas y evaluar tu dominio conceptual de Deep Learning.
3. **El objetivo principal:** Demostrar **ingeniería rigurosa**, **comprensión aplicada del núcleo de la red** y **honestidad científica**.

---

## 🎒 2. Checklist del Expositor: ¿Qué necesitas llevar?

| Elemento | Descripción y Especificación | ¿Dónde está en el proyecto? |
| :--- | :--- | :--- |
| **1. Póster A0 Impreso** | Tamaño $841 \times 1189\text{ mm}$ (A0 vertical), papel couché o banner mate. | [`Poster_DINOv2_EuroSAT_LoRA.pdf`](./Poster_DINOv2_EuroSAT_LoRA.pdf) |
| **2. Material de montaje** | Cinta masking tape o limpia-tipos (los paneles suelen tener chinches, pero es mejor prevenir). | Llevar en mochila |
| **3. Laptop o Tablet (Demo)** | Para mostrar en vivo las activaciones, el Teardown interactivo y la matriz de confusión. | Abrir [`diapositivas.html`](./diapositivas.html) o [`poster.html`](./poster.html) |
| **4. Copias del Abstract (5-10)** | Imprimir 5 a 10 copias grapadas del Extended Abstract a 2 páginas para entregar a jurados interesados. | [`Abstract_DINOv2_EuroSAT_LoRA.pdf`](./Abstract_DINOv2_EuroSAT_LoRA.pdf) |
| **5. Código QR visible** | Código QR enlazando al repositorio de GitHub con el código reproducible y el log de decisiones. | `github.com/italoperez/DINOv2-EuroSAT-LoRA-Teardown` |

---

## 🎙️ 3. Los 3 Niveles de Exposición (Guion de Pitch)

### 🔹 Nivel 1: El Hook de 30 Segundos (Para captar a quien pasa caminando)
> *"¡Hola! ¿Te interesa la visión computacional o la teledetección satelital? En este trabajo abrimos el núcleo de **DINOv2**, el modelo fundacional de Meta AI, para evaluar si sus características auto-supervisadas en fotos naturales sirven para clasificar territorio satelital en Sentinel-2.*  
> *Descubrimos que con solo adaptar el **0.12% de los parámetros** mediante LoRA en las capas tardías de atención, superamos al clasificador lineal en **+3.89% de F1-score**, resolviendo la confusión entre cultivos permanentes y vegetación natural, con cero sobrecosto en inferencia."*

---

### 🔹 Nivel 2: El Pitch de 2 Minutos (Para quien se detiene a escuchar)
*(Señala con la mano las secciones correspondientes del póster)*

1. **Problema (Tarjeta 1):**  
   *"DINOv2 fue preentrenado con millones de fotografías web (objetos centrados, perspectiva horizontal). Pero las imágenes satelitales como Sentinel-2 EuroSAT tienen vista cenital, texturas repetitivas y no tienen un sujeto central. El linear probing tradicional se queda corto y el fine-tuning completo es carísimo y corre riesgo de olvido catastrófico."*
2. **Disección y Teardown (Tarjeta 2):**  
   *"Primero disecamos DINOv2 ViT-S/14: 22 millones de parámetros y 12.25 GFLOPs. Mediante el mapa de campo receptivo efectivo (ERF) comprobamos que la atención global se consolida rápido a partir del bloque 4, mientras que los bloques 11 y 12 sintetizan la semántica en el token [CLS]."*
3. **Nuestra Intervención (Tarjeta 2 - LoRA + Caching):**  
   *"Nuestra hipótesis fue intervenir únicamente la atención tardía: congelamos los bloques 0 al 10 y aplicamos LoRA ($r=8, \alpha=8$) en Query y Value de los bloques 11 y 12. Son solo 28,426 parámetros entrenables (0.128% del total). Además, implementamos Prefix Caching: precomputamos los bloques congelados una sola vez, acelerando el entrenamiento en CPU en más de 80% con equivalencia matemática exacta."*
4. **Resultados y Cierre (Tarjeta 4 y 5):**  
   *"¿El resultado? Subimos el Macro-F1 de 90.14% a 94.03%. El mayor impacto estuvo en clases vegetales difíciles como PermanentCrop (+8.6% F1). Finalmente, somos honestos: la partición fue estratificada aleatoria; para la Fase C evaluaremos el shift geográfico real en distintas regiones de Europa."*

---

### 🔹 Nivel 3: El Walkthrough Técnico de 5 a 7 Minutos (Para Jurados y Docentes)
Utiliza la estructura de la tarjeta central del póster:
* **Detalle del Teardown:** Explica cómo calculaste los $12.25\text{ GFLOPs}$ ($2 \times$ multiply-accumulate en proyecciones QKV, atención escalada $QK^T$, proyección de salida y el MLP de dos capas lineales con expansión $4\times$).
* **Fórmula de LoRA:**
  $$W = W_0 + \frac{\alpha}{r}(B \cdot A), \quad B \in \mathbb{R}^{384 \times 8}, \quad A \in \mathbb{R}^{8 \times 384}$$
  Explica por qué $B$ inicia en 0 (para que en el paso 0 de entrenamiento el modelo sea exactamente idéntico al DINOv2 original) y por qué $\alpha=8$ y $r=8$ mantienen el escalamiento $1.0\times$.
* **Mapeo de Activaciones e Histéresis:** Muestra en tu tablet o laptop la figura [`03_intermediate_features.png`](../outputs/figures/intermediate_features.png) para demostrar cómo evoluciona la representación interna desde bordes de alta frecuencia hasta regiones semánticas globales.

---

## 🛡️ 4. Banco de Respuestas a Preguntas Difíciles del Jurado

### P1: *"¿Por qué aplicaste LoRA solo a los bloques 11 y 12 y no a toda la red?"*
* **Respuesta:**  
  *"Por diseño y economía de parámetros. Los primeros bloques de un Vision Transformer (bloques 0 a 6) actúan como detectores de bordes, frecuencias espaciales y texturas locales de bajo nivel, las cuales son universales y perfectamente transferibles de ImageNet a satélite. El cuello de botella del dominio satelital radica en la agregación semántica de alto nivel que ocurre en las cabezas de atención finales. Al adaptar únicamente los bloques 11 y 12, alteramos solo el 0.128% de la red, preservamos las representaciones primitivas y habilitamos Prefix Caching con costo computacional despreciable."*

### P2: *"¿Por qué pusiste LoRA en Query y Value, y no en Key o en las capas lineales del MLP?"*
* **Respuesta:**  
  *"En la literatura canónica de LoRA (Hu et al., ICLR 2022), intervenir Query y Value ofrece el balance óptimo entre re-ponderar la distribución de atención ($Q$) y adaptar la representación de características agregadas ($V$), manteniendo estática la clave semántica ($K$). Intervenir el MLP cuadruplica el número de parámetros debido a la dimensión oculta de 1536 ($4 \times 384$), violando nuestro presupuesto estricto de parámetros."*

### P3: *"¿Qué es exactamente el Prefix Caching y por qué garantizas que es idéntico?"*
* **Respuesta:**  
  *"Dado que los bloques 0 al 10 y la proyección de parches están completamente congelados (sin dropout activo durante evaluación/entrenamiento), la salida del bloque 10 para una imagen dada $X$ es una función determinista y estática $f_{0:10}(X)$. Precomputamos y guardamos en disco estos tensores intermedios. Al entrenar, alimentamos directamente el tensor precomputado a los bloques 11 y 12. Matemáticamente no hay aproximación: la diferencia máxima absoluta entre pasar la imagen completa o usar el caché es menor a $10^{-5}$ (debido a precisión flotante fp32), pero ahorramos más del 80% de operaciones FLOPs por época."*

### P4: *"Veo que obtuviste 94.03% de Macro-F1. ¿Tu modelo ya está listo para producción en teledetección satelital?"*
* **Respuesta (Honestidad Epistémica - Clave para la nota máxima):**  
  *"No categóricamente. Y es fundamental ser honestos en este punto: nuestro split actual fue aleatorio estratificado a nivel de teselas. En teledetección existe autocorrelación espacial: teselas vecinas capturadas el mismo día comparten condiciones atmosféricas y texturas idénticas. Para validar robustez real en producción, en la Fase C realizaremos validación cruzada espacial (Spatial Cross-Validation) evaluando el modelo en biomas geográficamente distantes (p. ej. Europa del Norte vs. Mediterráneo) y evaluaremos la robustez con imágenes Sentinel-1 SAR."*

### P5: *"¿Por qué la parada temprana fue en la época 3 y no dejaste entrenar hasta la época 5?"*
* **Respuesta:**  
  *"Porque observamos un sobreajuste fino de atención. En la época 3 la validación alcanzó su pico con 93.03% de Macro-F1. En las épocas 4 y 5, la pérdida de entrenamiento continuó descendiendo, pero la Macro-F1 de validación cayó a 91.4% debido a que los pesos adaptados comenzaron a memorizar las peculiaridades del conjunto reducido de 100 imágenes por clase. Siguiendo nuestra regla estricta de parada por validación, congelamos el checkpoint de la época 3, lo que garantizó un 94.03% en el conjunto de prueba nunca antes visto."*

---

## 🎯 5. Mapa de Archivos en tu Carpeta de Presentación (`presentacion_waimlap/`)

```text
presentacion_waimlap/
├── DINOv2_EuroSAT_LoRA.zip         <-- EL ARCHIVO QUE SE SUBE AL GOOGLE FORM DE WAIMLAP
├── Abstract_DINOv2_EuroSAT_LoRA.pdf  <-- 2 páginas oficiales en inglés
├── Abstract_DINOv2_EuroSAT_LoRA.docx <-- Versión Word editable
├── Abstract_DINOv2_EuroSAT_LoRA.md   <-- Resumen para lectura rápida
├── Poster_DINOv2_EuroSAT_LoRA.pdf   <-- PDF A0 (841 x 1189 mm) listo para imprimir
├── Poster_DINOv2_EuroSAT_LoRA.pptx  <-- PowerPoint A0 editable
├── poster.html                      <-- Vista interactiva en navegador del póster
├── Presentacion_E2_DINOv2_EuroSAT.pptx <-- Diapositivas 16:9 por si piden proyectar
├── diapositivas.html                <-- Diapositivas interactivas con KaTeX y Lightbox
└── GUIA_PRESENTACION_WAIMLAP.md     <-- Esta guía de defensa
```
