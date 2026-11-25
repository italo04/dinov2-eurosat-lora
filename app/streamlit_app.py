"""app/streamlit_app.py — Aplicación interactiva Streamlit: Base vs LoRA."""

import os
from PIL import Image
import streamlit as st
import pandas as pd

from app.demo_nucleo import (
    RUTA_CKPT,
    checkpoints_disponibles,
    cargar_modelos,
    predecir_imagen,
    CLASES_EUROSAT,
)

st.set_page_config(
    page_title="DINOv2 + LoRA · EuroSAT RGB",
    page_icon="🛰️",
    layout="wide",
)

st.title("🛰️ Clasificación EuroSAT RGB: DINOv2 vs DINOv2 + LoRA")
st.markdown(
    "**Proyecto:** Clasificación de cobertura del suelo con DINOv2 · 1INF52 Deep Learning 2026-2  \n"
    "**Pregunta de investigación:** ¿Añadir adaptadores LoRA mejora la clasificación en regiones geográficas nuevas frente a usar DINOv2 congelado?"
)

if not checkpoints_disponibles(RUTA_CKPT):
    st.info(
        f"⚠️ **Checkpoints no encontrados en `{RUTA_CKPT}`**\n\n"
        "Para ejecutar la demo interactiva se requieren los pesos entrenados de la semilla 42 en la partición geográfica:\n"
        "1. Ejecute el cuaderno `E3_regiones_ablaciones.ipynb` en Google Colab con GPU T4.\n"
        "2. El cuaderno exportará automáticamente `checkpoints/geo_s42/` al terminar la corrida geográfica.\n"
        "3. Descargue dicha carpeta o colóquela en `checkpoints/geo_s42/` en este repositorio."
    )
    st.stop()


@st.cache_resource
def obtener_modelos():
    return cargar_modelos(RUTA_CKPT, device="cpu")


with st.spinner("Cargando modelos en CPU..."):
    modelos = obtener_modelos()

opcion = st.sidebar.radio(
    "Seleccionar entrada:",
    ["Subir imagen (JPG/PNG)", "Ejemplos de prueba (Test fuera)"]
)

imagen_a_predecir = None
etiqueta_real = None

if opcion == "Subir imagen (JPG/PNG)":
    archivo_subido = st.file_uploader("Cargue una imagen satelital (64x64 o similar):", type=["jpg", "jpeg", "png"])
    if archivo_subido is not None:
        imagen_a_predecir = Image.open(archivo_subido)
else:
    ejemplos_disponibles = []
    for clase in CLASES_EUROSAT:
        ruta_clase = os.path.join("data", "EuroSAT_RGB", clase)
        if os.path.exists(ruta_clase):
            archivos = [f for f in os.listdir(ruta_clase) if f.endswith(".jpg")]
            if archivos:
                ejemplos_disponibles.append((clase, os.path.join(ruta_clase, archivos[0])))

    if ejemplos_disponibles:
        seleccion = st.selectbox(
            "Seleccione una imagen de ejemplo:",
            [f"{c}: {p}" for c, p in ejemplos_disponibles]
        )
        clase_sel, ruta_sel = seleccion.split(": ")
        imagen_a_predecir = Image.open(ruta_sel)
        etiqueta_real = clase_sel
    else:
        st.write("Coloque imágenes en `data/EuroSAT_RGB/` para habilitar el selector de ejemplos.")

if imagen_a_predecir is not None:
    col_img, col_base, col_lora = st.columns([1, 1.5, 1.5])

    with col_img:
        st.subheader("Imagen de entrada")
        st.image(imagen_a_predecir, use_container_width=True)
        if etiqueta_real:
            st.info(f"**Etiqueta real:** `{etiqueta_real}`")

    with st.spinner("Calculando predicciones..."):
        res = predecir_imagen(modelos, imagen_a_predecir)

    with col_base:
        st.subheader("Linear Probing (Base)")
        top_base_clase, top_base_prob = res["base"][0]
        if etiqueta_real:
            if top_base_clase == etiqueta_real:
                st.success(f"✅ **Acierto:** `{top_base_clase}` ({top_base_prob*100:.1f}%)")
            else:
                st.error(f"❌ **Error:** `{top_base_clase}` ({top_base_prob*100:.1f}%)")
        else:
            st.metric("Predicción principal", top_base_clase, f"{top_base_prob*100:.1f}%")

        df_base = pd.DataFrame(res["base"], columns=["Clase", "Probabilidad"])
        st.dataframe(df_base.style.format({"Probabilidad": "{:.3f}"}), use_container_width=True)

    with col_lora:
        st.subheader("DINOv2 + LoRA (Fase C)")
        top_lora_clase, top_lora_prob = res["lora"][0]
        if etiqueta_real:
            if top_lora_clase == etiqueta_real:
                st.success(f"✅ **Acierto:** `{top_lora_clase}` ({top_lora_prob*100:.1f}%)")
            else:
                st.error(f"❌ **Error:** `{top_lora_clase}` ({top_lora_prob*100:.1f}%)")
        else:
            st.metric("Predicción principal", top_lora_clase, f"{top_lora_prob*100:.1f}%")

        df_lora = pd.DataFrame(res["lora"], columns=["Clase", "Probabilidad"])
        st.dataframe(df_lora.style.format({"Probabilidad": "{:.3f}"}), use_container_width=True)

if os.path.exists("figures/C0_mapa_regiones.png"):
    st.markdown("---")
    st.subheader("🗺️ Partición Geográfica (Bloques Espaciales)")
    st.image("figures/C0_mapa_regiones.png", use_container_width=True)
