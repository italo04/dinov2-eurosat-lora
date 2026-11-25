import math
import random
import numpy as np
import torch
import torch.nn as nn
from transformers import Dinov2Model


def fijar_semilla(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cargar_dinov2(modelo_id: str = "facebook/dinov2-small", device: str = "cpu") -> Dinov2Model:
    if modelo_id == "modelo_chico":
        from transformers import Dinov2Config
        config = Dinov2Config(
            hidden_size=32,
            num_hidden_layers=4,
            num_attention_heads=2,
            intermediate_size=64,
            image_size=28,
            patch_size=14,
        )
        m = Dinov2Model(config)
    else:
        m = Dinov2Model.from_pretrained(modelo_id)
    m.eval()
    for p in m.parameters():
        p.requires_grad = False
    return m.to(device)


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.r = int(r)
        self.alpha = float(alpha)
        self.escala = float(alpha) / float(r)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        in_features = base.in_features
        out_features = base.out_features
        self.A = nn.Parameter(torch.empty(r, in_features))
        self.B = nn.Parameter(torch.zeros(out_features, r))

        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.base(x)
        lora = (self.dropout(x) @ self.A.T) @ self.B.T
        return res + self.escala * lora

    def fusionar(self) -> nn.Linear:
        linear = nn.Linear(self.base.in_features, self.base.out_features, bias=self.base.bias is not None)
        linear.weight.data = self.base.weight.data + (self.escala * (self.B @ self.A)).data
        if self.base.bias is not None:
            linear.bias.data = self.base.bias.data.clone()
        return linear


def salida_bloque(salida):
    return salida[0] if isinstance(salida, tuple) else salida


def aplicar_lora(
    m: nn.Module,
    bloques: tuple[int, ...],
    proyecciones: tuple[str, ...],
    r: int,
    alpha: float,
    dropout: float = 0.0,
) -> list[str]:
    proy_map = {
        "q": ("query", "q_proj"),
        "v": ("value", "v_proj"),
    }
    reemplazados = []
    layers = m.encoder.layer
    for b in bloques:
        layer = layers[b]
        # transformers ≥5.x usa layer.attention directamente (q_proj/v_proj)
        # versiones antiguas usaban layer.attention.attention (query/value)
        if hasattr(layer.attention, "attention"):
            attn_sub = layer.attention.attention
        else:
            attn_sub = layer.attention
        for p in proyecciones:
            nombres_posibles = proy_map.get(p.lower(), (p,))
            encontrado = False
            for nombre in nombres_posibles:
                if hasattr(attn_sub, nombre):
                    base_lin = getattr(attn_sub, nombre)
                    if isinstance(base_lin, nn.Linear) and not isinstance(base_lin, LoRALinear):
                        setattr(attn_sub, nombre, LoRALinear(base_lin, r=r, alpha=alpha, dropout=dropout))
                        reemplazados.append(f"layer.{b}.attention.{nombre}")
                        encontrado = True
                        break
            if not encontrado:
                raise AttributeError(f"No se encontró proyección '{p}' en el bloque {b}")

    assert len(reemplazados) == len(proyecciones) * len(bloques), (
        f"Esperados {len(proyecciones) * len(bloques)} reemplazos, pero se hicieron {len(reemplazados)}: {reemplazados}"
    )
    return reemplazados


def representacion(ultima: torch.Tensor) -> torch.Tensor:
    cls_token = ultima[:, 0, :]
    media_parches = ultima[:, 1:, :].mean(dim=1)
    return torch.cat([cls_token, media_parches], dim=-1)


def adelante_lora(m: nn.Module, x: torch.Tensor, primer_bloque: int) -> torch.Tensor:
    with torch.no_grad():
        embeddings = m.embeddings(x)
        h = embeddings
        for b in range(primer_bloque):
            h = salida_bloque(m.encoder.layer[b](h))
    for b in range(primer_bloque, len(m.encoder.layer)):
        h = salida_bloque(m.encoder.layer[b](h))
    h = m.layernorm(h)
    return representacion(h)


def fusionar_lora(m: nn.Module) -> int:
    contador = 0
    for name, module in m.named_modules():
        for attr, submod in list(module.named_children()):
            if isinstance(submod, LoRALinear):
                setattr(module, attr, submod.fusionar())
                contador += 1
    return contador


def contar_entrenables(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
