# Guion de Exposición — Entregable 2 (E2)

**Curso:** 1INF52 - Deep Learning  
**Institución:** Pontificia Universidad Católica del Perú (PUCP)  
**Semestre Académico:** 2026-2  
**Profesor:** Dr. César A. Beltrán Castañón  
**Tema:** Disección Instrumentada e Intervención Preliminar con LoRA en DINOv2 (ViT-S/14) sobre EuroSAT (RGB)  
**Duración Total Sugerida:** 5 a 7 minutos  
**Archivos Disponibles para Presentar:**
* Archivo PowerPoint: `Presentacion_E2_DINOv2_EuroSAT.pptx`
* Presentación Interactiva en Navegador: `diapositivas.html` (abrir en Chrome/Edge y presionar `F` para pantalla completa)

---

### Diapositiva 1: Portada (30 segundos)

* **Visual:** Título del proyecto, metadatos PUCP y tarjetas de métricas clave (22.06 M params, 0.128% LoRA, +3.89% Macro-F1, 97 ms latencia).
* **Guion del orador:**
  > *"Buenas tardes, profesor César Beltrán y compañeros. En este segundo entregable presentamos la caracterización arquitectónica profunda y la intervención preliminar sobre el modelo fundacional de visión autosupervisado **DINOv2 (Vision Transformer Small con parches de 14×14)** adaptado para teledetección en el conjunto de datos satelital **Sentinel-2 EuroSAT RGB**.  
  > Todo nuestro cómputo se realizó de forma estricta y reproducible sobre CPU estándar de 4 núcleos y 8 GB de RAM, con semilla fija 2026."*

---

### Diapositiva 2: Contexto, Problema y Pregunta Científica (45 segundos)

* **Visual:** Tres columnas comparando el desafío de dominio, las restricciones de hardware y la hipótesis formulada.
* **Guion del orador:**
  > *"El problema central que investigamos es una brecha de dominio visual: DINOv2 fue preentrenado con más de 142 millones de fotos comunes de internet, las cuales poseen perspectiva frontal y objetos centrados. En contraste, las imágenes satelitales de Sentinel-2 son vistas cenitales, sin horizonte, tomadas a 10 metros por píxel.  
  > Ante esto, un ajuste fino completo (*Full Fine-Tuning*) de los 22 millones de parámetros es inasumible en CPU y causaría sobreajuste severo con pocas muestras. Por otro lado, una simple sonda lineal (*Linear Probe*) deja las representaciones rígidas.  
  > Por ello, en nuestro Cuaderno de Decisiones formulamos la **Hipótesis D04 a priori**: conjeturamos que inyectar adaptadores de bajo rango (**LoRA**) exclusivamente en las proyecciones de atención Query y Value de los dos últimos bloques nos permitiría especializar el modelo para imágenes satelitales con un incremento de parámetros menor al 0.15%."*

---

### Diapositiva 3: Fase A — Auditoría de Integridad y Partición de Datos (45 segundos)

* **Visual:** Resultados de hashing SHA-256, tabla de división balanceada (1000/300/500) y mosaico visual de las 10 clases (`dataset_samples.png`).
* **Guion del orador:**
  > *"En la Fase A iniciamos con una auditoría rigurosa: calculamos el hash SHA-256 de las 27 000 imágenes de EuroSAT, comprobando que existen 27 000 hashes únicos, es decir, cero duplicados exactos.  
  > Seleccionamos un subconjunto balanceado de 1 800 imágenes: 100 de entrenamiento, 30 de validación y 50 de prueba por cada una de las 10 clases LULC.  
  > **Queremos destacar nuestra Decisión D03:** al auditar los metadatos EXIF comprobamos que los JPEG originales no incluyen coordenadas GPS. Por rigor y honestidad científica, nuestra partición es **aleatoria estratificada interna**, no geográfica. No simulamos regiones arbitrarias; la partición espacial estricta la implementaremos en la Fase C cuando vinculemos los GeoTIFFs."*

---

### Diapositiva 4: Fase A — Disección de la Arquitectura DINOv2 (45 segundos)

* **Visual:** Desglose analítico de parámetros, conteo de FLOPs (2 FLOPs/MAC), latencia empírica y gráfico de barras (`teardown.png`).
* **Guion del orador:**
  > *"Desarmamos matemáticamente el Vision Transformer: la imagen de $224 \times 224$ se divide en 256 parches de $14 \times 14$ píxeles, más un token de clasificación CLS, sumando 257 tokens de dimensión 384.  
  > La red cuenta con 12 bloques idénticos de autoatención con 6 cabezas y FFN SwiGLU. Cada bloque contiene exactamente **1.77 millones de parámetros** y demanda **1.01 GFLOPs**. En total, una pasada hacia adelante requiere **12.25 GFLOPs**.  
  > Al medir el rendimiento empírico en nuestra CPU Intel i5, obtuvimos una latencia mediana de **97 milisegundos por imagen** y un consumo residente pico de memoria de **504 MiB**, lo que valida la factibilidad del modelo en equipos sin GPU."*

---

### Diapositiva 5: Fase A — Representaciones Intermedias y ERF (45 segundos)

* **Visual:** Proyección PC1 de tokens (`intermediate_features.png`) y mapas de sensibilidad del Campo Receptivo Efectivo (`effective_receptive_field.png`).
* **Guion del orador:**
  > *"Para entender qué información retiene el modelo sin sesgo supervisado, proyectamos el primer componente principal (PC1) de los tokens espaciales:  
  > En el Bloque 1 se aprecian texturas y frecuencias locales de alta resolución; mientras que en el Bloque 12 los tokens convergen hacia una segmentación semántica continua de parcelas agrícolas y carreteras.  
  > Cuantificamos además el **Campo Receptivo Efectivo**: para acumular el 90% de la masa de gradiente que influye sobre el token central, el Bloque 1 solo abarca el **52%** de los píxeles de entrada; pero en el Bloque 12 se expande al **68%**. Esto demuestra que la autoatención propaga el contexto global a lo largo de toda la imagen satelital."*

---

### Diapositiva 6: Fase B — Intervención con LoRA y Prefix Caching (1 minuto)

* **Visual:** Ecuación $W = W_0 + \frac{\alpha}{r} (B \cdot A)$, desglose de parámetros (28,426) y esquema de Prefix Caching.
* **Guion del orador:**
  > *"Para la intervención, implementamos LoRA con rango $r=8$ y $\alpha=8$. La matriz $B$ se inicializa en cero exacto, lo que asegura que en el paso cero la red arranque en un estado matemáticamente idéntico al modelo preentrenado, con error de equivalencia de cero.  
  > LoRA se inyectó únicamente en las proyecciones Query y Value de los bloques 11 y 12. De esta forma, de los 22 millones de parámetros, **solo entrenamos 28 426 parámetros** (apenas el 0.128% de la red).  
  > Además, implementamos un aporte de ingeniería clave: **Prefix Caching**. Como los bloques 1 al 10 están congelados y las imágenes son deterministas, almacenamos en memoria mapeada la salida del bloque 10 una sola vez. Con un error numérico comprobado menor a $10^{-5}$, redujimos el tiempo de entrenamiento en CPU en más del 80%."*

---

### Diapositiva 7: Fase B — Resultados y Contraste de Hipótesis (1 minuto)

* **Visual:** Tarjeta de +3.89% Macro-F1, tabla comparativa detallada y curvas de aprendizaje (`learning_curves.png`).
* **Guion del orador:**
  > *"Al evaluar en el subconjunto de prueba ciego de 500 imágenes que nunca intervino en el entrenamiento:  
  > * La **Línea Base congelada** alcanzó **90.14%** de Macro-F1 y 90.20% de Accuracy.  
  > * Nuestra **intervención con LoRA** alcanzó **94.03%** de Macro-F1 y 94.00% de Accuracy.  
  > Esto representa una **ganancia neta de +3.89 puntos porcentuales en Macro-F1**, confirmando formalmente nuestra hipótesis a priori.  
  > Como observan en las curvas de aprendizaje, la validación alcanzó su punto óptimo en la **Época 3 (93.03%)**. En las épocas 4 y 5 la pérdida siguió cayendo hacia cero mientras la validación retrocedió a 92%, detectando sobreajuste incipiente. Nuestra regla de selección por validación guardó el punto de control de la época 3, protegiendo la generalización final."*

---

### Diapositiva 8: Diagnóstico de Confusiones y Análisis de Fallos (45 segundos)

* **Visual:** Matriz de confusión (`confusion_lora.png`) y panel de fallos reales (`failure_examples.png`).
* **Guion del orador:**
  > *"En la matriz de confusión observamos que la diagonal es marcadamente dominante, pero analizamos honestamente los errores residuales:  
  > 1. **Highway vs. River:** Ocurre confusión bidireccional en ríos estrechos que serpentean entre vegetación, pues a una resolución de 10 metros/píxel comparten una geometría lineal y reflectancia similar a carreteras asfaltadas.  
  > 2. **Pasture vs. Herbaceous Vegetation:** Ambas coberturas vegetales presentan curvas de reflectancia casi idénticas en el espectro visible (RGB). Para resolver esta ambigüedad física se hace indispensable incorporar bandas infrarrojas (NIR) y de borde rojo (Red Edge)."*

---

### Diapositiva 9: Conclusiones y Hoja de Ruta hacia la Fase C (45 segundos)

* **Visual:** Conclusiones con checks verdes y los 4 pilares de la hoja de ruta hacia el Entregable 3.
* **Guion del orador:**
  > *"Para concluir este Entregable 2:  
  > 1. Validamos que la adaptación con LoRA en bloques profundos es altamente eficiente: ganamos casi 4% en Macro-F1 modificando apenas el 0.128% de los parámetros.  
  > 2. Demostramos la viabilidad de transferir Vision Transformers en CPU estudiantil sin alterar significativamente la latencia de inferencia (+6 ms).  
  > **Nuestra hoja de ruta hacia la Fase C (Entregable 3):**  
  > * Primero, incorporar las coordenadas espaciales GeoTIFF para realizar una evaluación geográfica estricta frente a traslados de región.  
  > * Segundo, incorporar canales multiespectrales (NIR y SWIR) para eliminar las confusiones de vegetación.  
  > * Tercero, realizar una ablación explorando rangos $r \in \{4, 8, 16\}$ y aumentos rotacionales cenitales.  
  > Muchas gracias. Quedamos atentos a sus preguntas y comentarios."*
