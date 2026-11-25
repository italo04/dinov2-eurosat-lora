import pytest
import torch
from transformers import Dinov2Config, Dinov2Model


@pytest.fixture
def modelo_chico():
    config = Dinov2Config(
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=2,
        intermediate_size=64,
        image_size=28,
        patch_size=14,
    )
    m = Dinov2Model(config)
    m.eval()
    for p in m.parameters():
        p.requires_grad = False
    return m
