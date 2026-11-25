from dataclasses import dataclass


@dataclass
class Config:
    modelo_id: str = "facebook/dinov2-small"
    img_size: int = 224
    batch: int = 64
    epocas: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-4
    lora_rango: int = 8
    lora_alpha: float = 8.0
    lora_dropout: float = 0.0
    lora_bloques: tuple[int, ...] = (10, 11)
    lora_proyecciones: tuple[str, ...] = ("q", "v")
    max_por_clase: int | None = None
    num_workers: int = 2
