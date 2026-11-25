import glob
import json
import os
import zipfile
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import torchvision
from torchvision import datasets, transforms

EUROSAT_RGB_URL = "https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip"
MEDIA = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def encontrar_raiz_clases(base: str) -> str | None:
    for raiz, dirs, _ in os.walk(base):
        if len(dirs) == 10 and all(len(glob.glob(os.path.join(raiz, d, "*.jpg"))) > 0 for d in dirs):
            return raiz
    return None


def preparar_eurosat_rgb(data_dir: str = "data") -> str:
    raiz = encontrar_raiz_clases(data_dir)
    if raiz is not None:
        return raiz

    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, "EuroSAT_RGB.zip")
    if not os.path.exists(zip_path):
        import urllib.request
        urllib.request.urlretrieve(EUROSAT_RGB_URL, zip_path)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(data_dir)

    raiz = encontrar_raiz_clases(data_dir)
    if raiz is None:
        raise RuntimeError(
            f"No se encontró la carpeta con las 10 clases en {data_dir} tras descomprimir {zip_path}"
        )
    return raiz


# Alias para compatibilidad con cuadernos y scripts
descargar_eurosat_rgb = preparar_eurosat_rgb


def obtener_transform(img_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(MEDIA, STD),
    ])


def construir_dataset(raiz: str, img_size: int = 224) -> datasets.ImageFolder:
    return datasets.ImageFolder(raiz, transform=obtener_transform(img_size))


def particion_estratificada(
    y: np.ndarray,
    frac_val: float = 0.15,
    frac_test: float = 0.15,
    seed: int = 42,
    max_por_clase: int | None = None,
) -> tuple[list[int], list[int], list[int]]:
    rng = np.random.default_rng(seed)
    tr, va, te = [], [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        if max_por_clase:
            idx = idx[:max_por_clase]
        n_te = int(round(len(idx) * frac_test))
        n_va = int(round(len(idx) * frac_val))
        te += idx[:n_te].tolist()
        va += idx[n_te:n_te + n_va].tolist()
        tr += idx[n_te + n_va:].tolist()
    return sorted(tr), sorted(va), sorted(te)


def archivos_de(dataset: datasets.ImageFolder, indices: list[int]) -> list[str]:
    raiz = getattr(dataset, "root", "")
    res = []
    for i in indices:
        path = dataset.samples[i][0]
        rel = os.path.relpath(path, raiz) if raiz else path
        res.append(rel.replace("\\", "/"))
    return res


def indices_de(dataset: datasets.ImageFolder, archivos: list[str]) -> list[int]:
    raiz = getattr(dataset, "root", "")
    mapa = {}
    for i, (path, _) in enumerate(dataset.samples):
        rel = os.path.relpath(path, raiz) if raiz else path
        mapa[rel.replace("\\", "/")] = i

    indices = []
    faltantes = []
    for a in archivos:
        normalizado = a.replace("\\", "/")
        if normalizado in mapa:
            indices.append(mapa[normalizado])
        else:
            faltantes.append(normalizado)
    if faltantes:
        raise KeyError(f"No se encontraron {len(faltantes)} archivo(s) en el dataset: {faltantes[:5]}")
    return indices


def cargador(
    dataset,
    indices: list[int],
    batch: int = 64,
    shuffle: bool = False,
    seed: int = 42,
    num_workers: int = 2,
    pin_memory: bool = False,
) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        Subset(dataset, indices),
        batch_size=batch,
        shuffle=shuffle,
        generator=g,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def guardar_particion_interna(
    dataset: datasets.ImageFolder,
    ruta: str,
    seed: int = 42,
    max_por_clase: int | None = None,
) -> dict:
    etiquetas = np.array(dataset.targets)
    idx_tr, idx_va, idx_te = particion_estratificada(
        etiquetas, frac_val=0.15, frac_test=0.15, seed=seed, max_por_clase=max_por_clase
    )
    datos = {
        "train": archivos_de(dataset, idx_tr),
        "val": archivos_de(dataset, idx_va),
        "test": archivos_de(dataset, idx_te),
        "semilla": seed,
        "max_por_clase": max_por_clase,
    }
    os.makedirs(os.path.dirname(os.path.abspath(ruta)), exist_ok=True)
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(datos, f, indent=2)
    return datos


def cargar_particion(ruta: str) -> dict:
    with open(ruta, "r", encoding="utf-8") as f:
        return json.load(f)
