"""scripts/E3_particion_geo.py

Lee splits/eurosat_coords.csv, genera la partición geográfica disjunta
y guarda splits/split_geo.json, results/C0_particion_geo.csv y figures/C0_mapa_regiones.png.
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eurosat_lora.geo import particion_geografica, proyectar, distancias_test_train


def main():
    parser = argparse.ArgumentParser(description="Genera partición geográfica por bloques espaciales")
    parser.add_argument("--coords", type=str, default="splits/eurosat_coords.csv", help="Ruta al CSV de coordenadas")
    parser.add_argument("--salida-json", type=str, default="splits/split_geo.json", help="Ruta del JSON con la partición")
    parser.add_argument("--salida-csv", type=str, default="results/C0_particion_geo.csv", help="Ruta del CSV de resumen")
    parser.add_argument("--salida-fig", type=str, default="figures/C0_mapa_regiones.png", help="Ruta de la figura del mapa")
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria")
    args = parser.parse_args()

    if not os.path.exists(args.coords):
        raise FileNotFoundError(f"No se encuentra el archivo de coordenadas: {args.coords}")

    print(f"Cargando coordenadas desde {args.coords}...")
    coords = pd.read_csv(args.coords)
    print(f"Total imágenes: {len(coords)}")

    print("Calculando partición geográfica por bloques espaciales...")
    particion = particion_geografica(coords, seed=args.seed)
    print(f"Partición generada con k={particion['k']}, respaldo={particion['intento_respaldo']}")

    # 1. Guardar JSON
    os.makedirs(os.path.dirname(args.salida_json), exist_ok=True)
    with open(args.salida_json, "w", encoding="utf-8") as f:
        json.dump(particion, f, indent=2)
    print(f"Guardado split en {args.salida_json}")

    # 2. Resumen por clase y conjunto
    archivo_a_conjunto = {}
    for cjto in ["train", "val", "test_dentro", "test_fuera"]:
        for arch in particion[cjto]:
            archivo_a_conjunto[arch] = cjto

    coords["conjunto"] = coords["archivo_rgb"].map(archivo_a_conjunto)
    tabla_resumen = pd.crosstab(coords["clase"], coords["conjunto"])
    columnas_orden = [c for c in ["train", "val", "test_dentro", "test_fuera"] if c in tabla_resumen.columns]
    tabla_resumen = tabla_resumen[columnas_orden]
    tabla_resumen["total"] = tabla_resumen.sum(axis=1)

    os.makedirs(os.path.dirname(args.salida_csv), exist_ok=True)
    tabla_resumen.to_csv(args.salida_csv)
    print(f"Guardado resumen por clase en {args.salida_csv}")
    print("\nResumen de la partición:")
    print(tabla_resumen)

    # 3. Distancias a train
    xy = proyectar(coords["lon"].values, coords["lat"].values)
    coords["x"] = xy[:, 0]
    coords["y"] = xy[:, 1]

    mask_tr = coords["conjunto"] == "train"
    mask_td = coords["conjunto"] == "test_dentro"
    mask_tf = coords["conjunto"] == "test_fuera"

    xy_tr = xy[mask_tr]
    xy_td = xy[mask_td]
    xy_tf = xy[mask_tf]

    dist_td = distancias_test_train(xy_td, xy_tr)
    dist_tf = distancias_test_train(xy_tf, xy_tr)

    print("\nDistancias al conjunto de entrenamiento:")
    print(f"  Test dentro: mín {dist_td['min_km']:.2f} km, mediana {dist_td['mediana_km']:.2f} km")
    print(f"  Test fuera:  mín {dist_tf['min_km']:.2f} km, mediana {dist_tf['mediana_km']:.2f} km")

    # 4. Mapa de regiones
    colores = {
        "train": "#1f77b4",        # azul
        "val": "#ff7f0e",          # naranja
        "test_dentro": "#2ca02c",  # verde
        "test_fuera": "#d62728",   # rojo
    }

    plt.figure(figsize=(12, 8))
    for cjto, color in colores.items():
        subset = coords[coords["conjunto"] == cjto]
        plt.scatter(
            subset["lon"], subset["lat"],
            c=color, label=f"{cjto} (n={len(subset)})",
            alpha=0.6, s=12, edgecolors="none"
        )

    # Anotar centroides de regiones
    coords["region"] = coords["archivo_rgb"].map(particion["region"])
    for r_id in sorted(coords["region"].unique()):
        sub_r = coords[coords["region"] == r_id]
        cx = sub_r["lon"].mean()
        cy = sub_r["lat"].mean()
        plt.annotate(
            str(r_id), (cx, cy),
            fontsize=11, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", alpha=0.85)
        )

    plt.title("Partición geográfica por bloques espaciales (EuroSAT)", fontsize=14, fontweight="bold")
    plt.xlabel("Longitud (°)", fontsize=11)
    plt.ylabel("Latitud (°)", fontsize=11)
    plt.legend(loc="upper right", frameon=True)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.salida_fig), exist_ok=True)
    plt.savefig(args.salida_fig, dpi=150)
    plt.close()
    print(f"Guardado mapa en {args.salida_fig}")


if __name__ == "__main__":
    main()
