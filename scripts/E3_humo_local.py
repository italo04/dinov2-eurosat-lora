"""scripts/E3_humo_local.py

Prueba de humo local de extremo a extremo:
- Dataset sintético de 10 clases en tempdir
- Coordenadas sintéticas en tempdir
- Partición geográfica y partición interna
- 1 corrida base + 1 corrida LoRA con 1 época
- Verificación de idempotencia (segunda corrida usa cache sin error)
- Verificación de archivos de salida (metricas.json, historial.csv, predicciones)
- Imprime HUMO OK al finalizar
"""

import json
import os
import shutil
import tempfile
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder

from eurosat_lora.config import Config
from eurosat_lora.datos import archivos_de, particion_estratificada
from eurosat_lora.geo import particion_geografica
from eurosat_lora.experimentos import (
    Corrida,
    Contexto,
    ejecutar_corrida,
)


def main():
    print("Iniciando prueba de humo local...")

    with tempfile.TemporaryDirectory() as tmp_dir:
        data_dir = os.path.join(tmp_dir, "data")
        salida_dir = os.path.join(tmp_dir, "salida")
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(salida_dir, exist_ok=True)

        clases = [
            "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
            "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"
        ]

        # 1. Crear imágenes sintéticas (20 por clase)
        print("Generando dataset sintético (10 clases x 20 imágenes)...")
        rng = np.random.default_rng(42)
        filas_coords = []

        n_grupos = 24
        centros_lon = rng.uniform(5, 30, n_grupos)
        centros_lat = rng.uniform(42, 55, n_grupos)

        img_idx = 0
        for c_idx, clase in enumerate(clases):
            clase_dir = os.path.join(data_dir, clase)
            os.makedirs(clase_dir, exist_ok=True)
            for i in range(20):
                img_path = os.path.join(clase_dir, f"{clase}_{i}.jpg")
                color = (int(c_idx * 25), int(i * 12), int((c_idx + i) * 6))
                im = Image.new("RGB", (28, 28), color=color)
                im.save(img_path)

                # Asignar a un grupo espacial
                g = img_idx % n_grupos
                lon = centros_lon[g] + rng.normal(0, 0.2)
                lat = centros_lat[g] + rng.normal(0, 0.2)
                filas_coords.append({
                    "archivo_rgb": f"{clase}/{clase}_{i}.jpg",
                    "clase": clase,
                    "lon": lon,
                    "lat": lat,
                    "crs_epsg": 32632,
                })
                img_idx += 1

        df_coords = pd.DataFrame(filas_coords)
        coords_csv = os.path.join(tmp_dir, "eurosat_coords.csv")
        df_coords.to_csv(coords_csv, index=False)

        # 2. Cargar dataset ImageFolder
        tf = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
        ])
        dataset = ImageFolder(data_dir, transform=tf)
        n_total = len(dataset)
        assert n_total == 200, f"Esperadas 200 imágenes, se obtuvieron {n_total}"

        # 3. Particiones
        print("Generando particiones interna y geográfica...")
        y = np.array(dataset.targets)
        idx_tr, idx_va, idx_te = particion_estratificada(y, frac_val=0.15, frac_test=0.15, seed=42)

        particion_interna = {
            "train": archivos_de(dataset, idx_tr),
            "val": archivos_de(dataset, idx_va),
            "test": archivos_de(dataset, idx_te),
        }

        particion_geo = particion_geografica(df_coords, seed=42)

        particiones = {
            "interna": particion_interna,
            "geo": particion_geo,
        }

        cfg = Config(
            modelo_id="modelo_chico",
            img_size=28,
            batch=8,
            epocas=1,
            max_por_clase=20,
            lora_bloques=(2, 3),
            num_workers=0,
        )

        ctx = Contexto(
            dataset=dataset,
            clases=clases,
            particiones=particiones,
            cfg=cfg,
            device="cpu",
            amp=False,
        )

        # 4. Corrida 1: Base (interna)
        print("Ejecutando corrida base (1 época)...")
        c_base = Corrida.crear("A2", "interna", "base", 42, bloques=(2, 3))
        res_base = ejecutar_corrida(c_base, ctx, salida_dir)

        assert os.path.exists(os.path.join(salida_dir, c_base.id, "metricas.json")), "Falta metricas.json en base"
        assert os.path.exists(os.path.join(salida_dir, c_base.id, "historial.csv")), "Falta historial.csv en base"
        assert os.path.exists(os.path.join(salida_dir, c_base.id, "predicciones_test.csv")), "Falta predicciones_test.csv"

        # 5. Corrida 2: LoRA (geo)
        print("Ejecutando corrida LoRA geográfica (1 época)...")
        c_lora = Corrida.crear("A4", "geo", "lora", 42, bloques=(2, 3))
        res_lora = ejecutar_corrida(c_lora, ctx, salida_dir)

        assert os.path.exists(os.path.join(salida_dir, c_lora.id, "metricas.json")), "Falta metricas.json en lora"
        assert os.path.exists(os.path.join(salida_dir, c_lora.id, "historial.csv")), "Falta historial.csv en lora"
        assert os.path.exists(os.path.join(salida_dir, c_lora.id, "predicciones_test_dentro.csv")), "Falta predicciones dentro"
        assert os.path.exists(os.path.join(salida_dir, c_lora.id, "predicciones_test_fuera.csv")), "Falta predicciones fuera"

        # 6. Comprobar idempotencia
        print("Verificando idempotencia...")
        res_lora_2 = ejecutar_corrida(c_lora, ctx, salida_dir)
        assert res_lora["id"] == res_lora_2["id"]

    print("HUMO OK")


if __name__ == "__main__":
    main()
