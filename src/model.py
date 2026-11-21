"""Modelo oficial fijado por commit y LoRA en Q/V (K permanece intacta)."""
import os,sys,json,pathlib,math,hashlib
os.environ['XFORMERS_DISABLED']='1'
import torch
from torch import nn
ROOT=pathlib.Path(__file__).resolve().parents[1]

def load_backbone():
    lock=json.loads((ROOT/'sources.lock.json').read_text())
    source=ROOT/'vendor'/('dinov2-'+lock['dinov2_commit'])
    sys.path.insert(0,str(source))
    from dinov2.hub.backbones import dinov2_vits14
    model=dinov2_vits14(pretrained=False)
    state=torch.load(ROOT/'weights'/'dinov2_vits14_pretrain.pth',map_location='cpu',weights_only=True)
    model.load_state_dict(state,strict=True)
    model.requires_grad_(False)
    return model.eval()

class LoRAQKV(nn.Module):
    """W_qkv x + [s Bq Aq x, 0, s Bv Av x], s=alpha/r."""
    def __init__(self,base,rank=8,alpha=8):
        super().__init__();self.base=base;self.dim=base.in_features;self.rank=rank;self.scale=alpha/rank
        self.q_a=nn.Linear(self.dim,rank,bias=False);self.q_b=nn.Linear(rank,self.dim,bias=False)
        self.v_a=nn.Linear(self.dim,rank,bias=False);self.v_b=nn.Linear(rank,self.dim,bias=False)
        for a in [self.q_a,self.v_a]:nn.init.kaiming_uniform_(a.weight,a=math.sqrt(5))
        for b in [self.q_b,self.v_b]:nn.init.zeros_(b.weight)
        self.base.requires_grad_(False)
    def forward(self,x):
        q,k,v=self.base(x).chunk(3,dim=-1)
        return torch.cat([q+self.scale*self.q_b(self.q_a(x)),k,v+self.scale*self.v_b(self.v_a(x))],dim=-1)

def inject_lora(model,cfg):
    for i in cfg['lora_blocks']:
        model.blocks[i].attn.qkv=LoRAQKV(model.blocks[i].attn.qkv,cfg['rank'],cfg['alpha'])
    return model

def prefix(model,x):
    z=model.prepare_tokens_with_masks(x)
    for b in model.blocks[:10]:z=b(z)
    return z

def suffix(model,z):
    for b in model.blocks[10:]:z=b(z)
    return model.norm(z)[:,0]

def frozen_hash(model):
    h=hashlib.sha256()
    for n,p in model.named_parameters():
        if not p.requires_grad:h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
