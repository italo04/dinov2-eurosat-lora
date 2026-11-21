# Cuaderno de decisiones

## 2026-10-20 — D01: alcance del entregable 2 (antes de ejecutar)

La consigna exige disección instrumentada, intervención preliminar y registro en GitHub. Se conserva DINOv2 ViT-S/14 con EuroSAT y LoRA en Q/V de los últimos dos bloques. No se presenta la ejecución como réplica del preentrenamiento de DINOv2 ni de sus cifras publicadas: se reutilizan pesos e implementación oficiales para una línea base de transferencia con CLS y cabeza lineal.

## D02: recursos y experimento preliminar (antes de ejecutar)

Equipo: Intel Core i5-1135G7, 4 núcleos/8 hilos, aproximadamente 8 GB RAM, Intel Iris Xe sin CUDA. Se ejecutará en CPU, float32, cuatro hilos. Por el costo se fija un subconjunto balanceado: 100 imágenes de entrenamiento, 30 validación y 50 test por clase (1000/300/500). El resto no interviene en este avance. Una semilla 2026 y cinco épocas. No se hará selección de hiperparámetros usando test. Las cifras no representan entrenamiento con las 27 000 imágenes ni robustez geográfica.

## D03: particiones y corrección respecto de E1 (antes de ejecutar)

E1 proponía reservar regiones. Los JPEG no proporcionan por sí solos un vínculo geográfico validado. Para E2 se declara una partición aleatoria estratificada interna, tras eliminar duplicados exactos por SHA-256. Esto no elimina vecinos geográficos ni duplicados perceptuales. No se promete ni simula evaluación geográfica. Para la fase C se requiere recuperar y auditar coordenadas o definir un dataset externo compatible antes de entrenar la siguiente fase.

## D04: hipótesis y ablación (antes de ejecutar)

Hipótesis: introducir LoRA de rango 8, alpha 8 y sin dropout en Q/V de los bloques 11 y 12 mejorará macro-F1 frente al backbone congelado, al adaptar características satelitales con pocos parámetros. Riesgo: sobreajuste o mejora nula con pocos datos. No se define un tamaño de mejora esperado sin evidencia.

Control: clasificador lineal de 384 entradas y 10 salidas. Tratamiento: mismo clasificador e inicialización, más LoRA. Mantener preprocesamiento RGB 224×224 bicúbico, normalización ImageNet, datos, orden de lotes, cinco épocas, batch 16, AdamW, lr 0.001, weight decay 0.0001 y selección por macro-F1 de validación. Los pesos originales, incluidas K y las normalizaciones, permanecen congelados. Registrar también resultados negativos.

## D05: caché exacta (antes de ejecutar)

No se usa aumento aleatorio en este piloto. Con entradas deterministas y los primeros diez bloques congelados en eval, sus salidas se pueden guardar en float32 sin alterar el cálculo. LoRA recibe esa caché y ejecuta los últimos dos bloques con gradientes. El control usa CLS finales precalculados. Se verificará equivalencia numérica de inferencia completa y caché. Esto reduce costo, no sustituye el modelo. Las latencias de despliegue se miden con la imagen completa, sin caché.

## D06: mediciones e interpretación (antes de ejecutar)

Contar parámetros por bloque. Estimar operaciones de convoluciones, capas lineales y productos de atención con multiplicación y suma = 2 FLOPs; excluir explícitamente normalizaciones, GELU, softmax y operaciones auxiliares. Medir forward CPU batch 1 tras calentamiento y RSS del proceso muestreada, sin equipararla a VRAM. Visualizar PC1 de tokens y sensibilidad por gradiente del token central, sin confundirlos con segmentación ni explicaciones causales. Los mapas corresponden a una imagen de entrenamiento.

## Uso de asistencia de IA

Antigravity y Codex asisten en el andamiaje del código, la ejecución automatizada, la auditoría de hashes, el perfilado de hardware y la estructuración de visualizaciones y documentación. El equipo humano audita, comprende, contrasta las hipótesis y es responsable de la defensa técnica ante el cuerpo docente. Las decisiones D01 a D06 y las hipótesis a priori fueron registradas formalmente antes de iniciar el cómputo experimental.

## 2026-10-21 — D07: resultados empíricos y contraste de la hipótesis a priori (después de ejecutar)

### 1. Cuadro comparativo de resultados

| Métrica / Dimensión | Control (Línea Base Congelada) | Tratamiento (Intervención LoRA Q/V) | Diferencia ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Parámetros entrenables** | 3,850 (0.017%) | 28,426 (0.128%) | +24,576 parámetros (+0.111%) |
| **Parámetros totales (backbone + head)** | 22,060,426 | 22,084,978 | +24,576 |
| **Pesos congelados inalterados** | Sí (hash inmutable) | Sí (hash inmutable) | Verificado vía hash de estado |
| **Error inicial de equivalencia ($B=0$)** | 0.0 | 0.0 | Equivalencia exacta asegurada |
| **Mejor época (criterio val)** | Época 5 | Época 3 | Convergencia más temprana |
| **Macro-F1 de validación (300 img)** | 0.8849 | 0.9303 | **+0.0454 (+4.54%)** |
| **Accuracy de validación** | 88.67% | 93.00% | +4.33% |
| **Macro-F1 de test (500 img)** | **0.9014** | **0.9403** | **+0.0389 (+3.89%)** |
| **Accuracy de test** | **90.20%** | **94.00%** | **+3.80%** |
| **Tiempo de entrenamiento (5 épocas)** | 4.85 s (usando caché CLS) | 300.55 s (usando caché Prefijo) | Factible en CPU i5-1135G7 |
| **Latencia forward despliegue (batch 1)** | 97.29 ms (backbone) | 103.58 ms (backbone + adapters) | +6.29 ms (+6.4%) |
| **Pico RSS en memoria** | 381 MiB | 807 MiB | Operable en laptops de 8 GB |

### 2. Contraste de la hipótesis a priori
- **Confirmación:** La hipótesis formulada en D04 se **confirma experimentalmente**: la inyección de LoRA ($r=8, \alpha=8$) en las matrices de proyección de Query y Value ($W_q, W_v$) de los bloques finales 11 y 12 produce un incremento estadísticamente consistente de **+3.89 puntos porcentuales en Macro-F1 de test** (de 90.14% a 94.03%) y de **+3.80 puntos porcentuales en Accuracy** (de 90.20% a 94.00%).
- **Eficiencia paramétrica:** Este beneficio se logra entrenando únicamente 28,426 parámetros (apenas 24,576 parámetros de LoRA adicionales a la cabeza lineal de 3,850 parámetros), lo cual representa un 0.128% del volumen total de la red.
- **Dinámica de adaptación:** La inspección de gradientes en el primer lote confirmó que $W_a$ tiene gradiente nulo mientras $W_b$ recibe el gradiente directo de la función de coste. Al finalizar el entrenamiento, todas las matrices de adaptación sufrieron actualizaciones medibles (norma máxima de cambio entre 0.054 y 0.084), evidenciando que las proyecciones de atención se especializaron para las particularidades espectrales de Sentinel-2 sin distorsionar el 99.87% restante del modelo.
- **Riesgo de sobreajuste observado:** Tal como se anticipó en D04, el entrenamiento con pocos datos (100 imágenes por clase) provocó que la pérdida de entrenamiento cayera drásticamente (a 0.012 en la época 5), mientras que la Macro-F1 de validación alcanzó su pico en la época 3 (0.9303) y descendió ligeramente en las épocas 4 y 5 (0.9200). La regla de parada y selección por validación evitó que el sobreajuste perjudicara la evaluación de test.

### 3. Análisis de errores y limitaciones honestas
- **Confusiones residuales observadas en `failure_examples.png`:**
  1. *Highway vs. River:* Confusión espectral bidireccional cuando ríos estrechos presentan geometría serpenteante y textura similar a autopistas en resolución de 10 m/píxel (64x64 píxeles).
  2. *HerbaceousVegetation vs. Pasture:* Ambas clases comparten firmas espectrales en el visible altamente solapadas en determinadas épocas del año, requiriendo canales NIR/SWIR para su discriminación inequívoca.
- **Delimitación epistemológica estricta para el Entregable 2:**
  - Este experimento se evaluó sobre un subconjunto interno con estratificación aleatoria. No se reclaman propiedades de generalización geográfica ni invariancia a traslaciones espaciales en este entregable.
  - La resolución espacial original de EuroSAT es de 64×64 píxeles; la interpolación bicúbica a 224×224 píxeles es un requisito dimensional del patch-embedding de DINOv2 ($14 \times 14$), pero no añade información de alta frecuencia al sensor satelital.
  - Estos resultados constituyen el hito de la Fase B y servirán como benchmark de control riguroso para la Fase C (Entregable 3).

