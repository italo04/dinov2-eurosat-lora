"""Pipeline E2: auditoría, teardown, features, ablación y evidencia verificable."""
import argparse,copy,csv,datetime,gc,hashlib,json,os,pathlib,platform,random,subprocess,sys,time,threading
import numpy as np
import torch
from torch import nn
from PIL import Image
from torchvision import transforms
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix,classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import psutil
from .model import ROOT,load_backbone,inject_lora,prefix,suffix,frozen_hash

OUT=ROOT/'outputs'; FIG=OUT/'figures';CACHE=ROOT/'cache';CKPT=OUT/'checkpoints'
for d in [OUT,FIG,CACHE,CKPT,ROOT/'docs']:d.mkdir(parents=True,exist_ok=True)
def save(name,obj): (OUT/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf8')
def rows_csv(path,rows):
    with path.open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def seed_all(seed):random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.use_deterministic_algorithms(True)
def transform(cfg):return transforms.Compose([transforms.Resize((cfg['image_size'],cfg['image_size']),interpolation=transforms.InterpolationMode.BICUBIC),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
def image_batch(rows,cfg):
    t=transform(cfg)
    return torch.stack([t(Image.open(ROOT/r['path']).convert('RGB')) for r in rows])
def config():return json.loads((ROOT/'configs/e2_cpu.json').read_text())

def prepare(cfg):
    """No usar el orden de archivos como división. Eliminar duplicados exactos antes."""
    files=sorted((ROOT/'data/rgb').rglob('*.jpg'))
    assert len(files)==27000,f'Expected 27000, got {len(files)}'
    classes=sorted({p.parent.name for p in files}); assert len(classes)==10
    rng=np.random.default_rng(cfg['seed']); seen={};inventory=[];duplicates=[]
    for p in files:
        h=hashlib.sha256(p.read_bytes()).hexdigest()
        if h in seen:
            duplicates.append({'path':str(p.relative_to(ROOT)),'duplicate_of':seen[h]});continue
        seen[h]=str(p.relative_to(ROOT)).replace('\\','/')
        inventory.append({'path':seen[h],'label':classes.index(p.parent.name),'class':p.parent.name,'sha256':h})
    selected=[]
    for c in classes:
        group=[r for r in inventory if r['class']==c];rng.shuffle(group);offset=0
        for split,key in [('train','train_per_class'),('val','val_per_class'),('test','test_per_class')]:
            for r in group[offset:offset+cfg[key]]:selected.append(dict(r,split=split))
            offset+=cfg[key]
    for i,r in enumerate(selected):r['cache_index']=i
    rows_csv(OUT/'split_manifest.csv',selected)
    sample=Image.open(ROOT/selected[0]['path'])
    audit={'raw_images':len(files),'unique_sha256':len(inventory),'exact_duplicates':duplicates,'classes':classes,'raw_class_counts':{c:sum(p.parent.name==c for p in files) for c in classes},'selected_counts':{s:sum(r['split']==s for r in selected) for s in ['train','val','test']},'split_type':'stratified random balanced subset, NOT geographic','image_size':list(sample.size),'first_image_exif':dict(sample.getexif()),'geographic_status':'JPEGs do not provide a validated coordinate-to-image mapping; phase C deferred. Filename index is not location.','manifest_sha256':hashlib.sha256((OUT/'split_manifest.csv').read_bytes()).hexdigest(),'created_utc':stamp()}
    save('data_audit.json',audit)
    fig,axes=plt.subplots(2,5,figsize=(13,6))
    for ax,c in zip(axes.flat,classes):
        r=next(r for r in selected if r['class']==c and r['split']=='train');ax.imshow(Image.open(ROOT/r['path']));ax.set_title(c,fontsize=10);ax.axis('off')
    fig.suptitle('EuroSAT: ejemplos reales del subconjunto de entrenamiento');fig.tight_layout();fig.savefig(FIG/'dataset_samples.png',dpi=160);plt.close(fig)
    print('DATA AUDIT',json.dumps(audit['selected_counts']),flush=True)

def manifest():
    with (OUT/'split_manifest.csv').open(encoding='utf8') as f:rows=list(csv.DictReader(f))
    for r in rows:r['label']=int(r['label']);r['cache_index']=int(r['cache_index'])
    return rows

def benchmark(model,x,repeats=20):
    process=psutil.Process(); samples=[];stop=threading.Event()
    def monitor():
        while not stop.wait(.01):samples.append(process.memory_info().rss)
    initial=process.memory_info().rss;t=threading.Thread(target=monitor,daemon=True);t.start()
    with torch.no_grad():
        for _ in range(3):model(x)
        times=[]
        for _ in range(repeats):
            t0=time.perf_counter();model(x);times.append((time.perf_counter()-t0)*1000)
    stop.set();t.join()
    return {'batch':len(x),'warmup':3,'repeats':repeats,'median_ms_per_batch':float(np.median(times)),'p95_ms_per_batch':float(np.percentile(times,95)),'median_ms_per_image':float(np.median(times)/len(x)),'rss_initial_mib':initial/2**20,'rss_peak_mib':max(samples+[initial])/2**20,'scope':'CPU forward only, no image decoding/preprocessing; RSS is whole process sampled every 10ms'}

def teardown(cfg):
    model=load_backbone();items=manifest();x=image_batch(items[:1],cfg)
    total=sum(p.numel() for p in model.parameters());parts=[];d=model.embed_dim;n=1+(cfg['image_size']//14)**2
    # Count dense multiply-add as 2 FLOPs. Exclude LayerNorm, GELU, softmax, interpolation and residual adds.
    for i,b in enumerate(model.blocks):
        dense=sum(2*n*m.in_features*m.out_features for m in b.modules() if isinstance(m,nn.Linear))
        attn=4*n*n*d
        parts.append({'component':f'block_{i+1:02d}','parameters':sum(p.numel() for p in b.parameters()),'dense_flops':dense,'attention_matmul_flops':attn,'counted_flops':dense+attn})
    conv=model.patch_embed.proj
    patch_flops=2*(n-1)*conv.out_channels*conv.in_channels*14*14
    other=total-sum(r['parameters'] for r in parts)
    parts.insert(0,{'component':'embedding_tokens_final_norm','parameters':other,'dense_flops':patch_flops,'attention_matmul_flops':0,'counted_flops':patch_flops})
    assert sum(r['parameters'] for r in parts)==total
    rows_csv(OUT/'teardown_by_block.csv',parts)
    latency=benchmark(model,x)
    architecture={'parameters':total,'embed_dim':d,'depth':len(model.blocks),'heads':model.blocks[0].attn.num_heads,'patch_size':14,'input':list(x.shape),'tokens':n,'counted_flops':sum(r['counted_flops'] for r in parts),'flops_convention':'2 FLOPs per multiply-add. Conv, linear and QK^T/AV only. Excludes softmax, LayerNorm, GELU, bias, interpolation and residual operations. Analytical forward estimate, batch 1.','latency':latency,'hardware':{'platform':platform.platform(),'processor':platform.processor(),'logical_cpus':os.cpu_count(),'ram_gib':psutil.virtual_memory().total/2**30,'torch':torch.__version__,'cuda_available':torch.cuda.is_available(),'threads':torch.get_num_threads()},'created_utc':stamp()}
    save('teardown.json',architecture)
    fig,ax=plt.subplots(1,2,figsize=(12,4.6));labels=[r['component'].replace('embedding_tokens_final_norm','otros') for r in parts]
    ax[0].barh(labels,[r['parameters']/1e6 for r in parts]);ax[0].set_xlabel('Millones de parámetros')
    ax[1].barh(labels,[r['counted_flops']/1e9 for r in parts],color='#2c786c');ax[1].set_xlabel('GFLOPs contados (batch 1)')
    fig.tight_layout();fig.savefig(FIG/'teardown.png',dpi=170);plt.close(fig)
    # Feature maps: PC1 is descriptive per-image projection, not a trained classifier.
    hooks=[];activations={}
    for i in [0,5,11]:hooks.append(model.blocks[i].register_forward_hook(lambda m,inp,out,i=i:activations.__setitem__(i,out.detach().clone())))
    with torch.no_grad():model(x)
    for h in hooks:h.remove()
    fig,ax=plt.subplots(1,4,figsize=(12,3.2));ax[0].imshow(Image.open(ROOT/items[0]['path']));ax[0].set_title(items[0]['class'])
    for a,i in zip(ax[1:],[0,5,11]):
        a0=activations[i][0,1:].numpy();u,sv,vh=np.linalg.svd(a0-a0.mean(0),full_matrices=False);im=(u[:,0]*sv[0]).reshape(16,16);a.imshow(im,cmap='viridis');a.set_title(f'Bloque {i+1}: PC1 de tokens')
    for a in ax:a.axis('off')
    fig.tight_layout();fig.savefig(FIG/'intermediate_features.png',dpi=180);plt.close(fig)
    erf_stats=[];fig,ax=plt.subplots(1,3,figsize=(10,3.4));ax[0].imshow(Image.open(ROOT/items[0]['path']));ax[0].set_title('Imagen de entrenamiento')
    for a,layer in zip(ax[1:],[0,11]):
        xin=x.clone().requires_grad_(True);z=model.prepare_tokens_with_masks(xin)
        for b in model.blocks[:layer+1]:z=b(z)
        # Center patch-token pre-final-LN energy sensitivity. Not a segmentation mask.
        score=z[0,1+8*16+8].square().mean();g=torch.autograd.grad(score,xin)[0][0].square().sum(0).sqrt().numpy();g=g/(g.sum()+1e-12)
        order=np.sort(g.ravel())[::-1];coverage=int(np.searchsorted(np.cumsum(order),.9)+1)/g.size
        erf_stats.append({'block':layer+1,'top_pixel_fraction_for_90pct_gradient_mass':coverage,'target':'mean squared activation of central patch token before final norm','image':items[0]['path']})
        a.imshow(g,cmap='inferno',vmin=0,vmax=np.percentile(g,99));a.set_title(f'Bloque {layer+1}: sensibilidad')
    for a in ax:a.axis('off')
    fig.tight_layout();fig.savefig(FIG/'effective_receptive_field.png',dpi=180);plt.close(fig);save('erf.json',erf_stats)
    print('TEARDOWN',json.dumps(architecture),flush=True)

def cache_features(cfg):
    model=load_backbone();items=manifest();N=len(items);start=time.perf_counter()
    dest=CACHE/'prefix.npy';complete=CACHE/'complete.json'
    fingerprint=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()+(OUT/'split_manifest.csv').read_bytes()+(ROOT/'sources.lock.json').read_bytes()).hexdigest()
    if complete.exists() and json.loads(complete.read_text()).get('fingerprint')==fingerprint:
        print('CACHE valid; reuse',flush=True);return
    zcache=np.lib.format.open_memmap(dest,mode='w+',dtype='float32',shape=(N,257,384))
    feats=np.lib.format.open_memmap(CACHE/'cls.npy',mode='w+',dtype='float32',shape=(N,384))
    batch=cfg['feature_batch_size'];check=0
    with torch.no_grad():
        for offset in range(0,N,batch):
            x=image_batch(items[offset:offset+batch],cfg);z=prefix(model,x);f=suffix(model,z)
            if offset==0:
                check=float((model(x)-f).abs().max());assert check<1e-5
            zcache[offset:offset+len(x)]=z.numpy();feats[offset:offset+len(x)]=f.numpy()
            if offset%80==0: print(f'CACHE {offset}/{N} elapsed={time.perf_counter()-start:.1f}s',flush=True)
    zcache.flush();feats.flush()
    info={'fingerprint':fingerprint,'images':N,'elapsed_s':time.perf_counter()-start,'full_vs_cached_max_abs':check,'dtype':'float32','prefix':'patch embedding + blocks 1-10, frozen eval, deterministic input','created_utc':stamp()}
    complete.write_text(json.dumps(info,indent=2));save('cache_info.json',info)

def evaluate_logits(logits,labels):
    pred=logits.argmax(1);return {'accuracy':float(accuracy_score(labels,pred)),'macro_f1':float(f1_score(labels,pred,average='macro',zero_division=0))}

def train(cfg,variant):
    seed_all(cfg['seed']);items=manifest();labels=np.array([r['label'] for r in items]);idx={s:np.array([i for i,r in enumerate(items) if r['split']==s]) for s in ['train','val','test']}
    head=nn.Linear(384,10);initial_head=copy.deepcopy(head.state_dict());model=load_backbone()
    if variant=='lora':inject_lora(model,cfg)
    model.eval();before=frozen_hash(model)
    initial_trainable={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
    params=list(head.parameters())+[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    zcache=np.load(CACHE/'prefix.npy',mmap_mode='r');features=np.load(CACHE/'cls.npy',mmap_mode='r')
    def forward(indices):
        if variant=='baseline':f=torch.from_numpy(np.array(features[indices]))
        else:f=suffix(model,torch.from_numpy(np.array(zcache[indices])))
        return head(f)
    def eval_split(indices):
        logits=[]
        with torch.no_grad():
            for o in range(0,len(indices),cfg['batch_size']):logits.append(forward(indices[o:o+cfg['batch_size']]).numpy())
        return np.concatenate(logits)
    # LoRA B=0 must preserve the baseline initially; K is never adapted.
    with torch.no_grad():
        test_idx=idx['train'][:2];expected=nn.functional.linear(torch.from_numpy(np.array(features[test_idx])),initial_head['weight'],initial_head['bias']);err=float((forward(test_idx)-expected).abs().max());assert err<1e-5
    history=[];best=-1;start=time.perf_counter();rng=np.random.default_rng(cfg['seed']);peak=psutil.Process().memory_info().rss
    first_grad={}
    for epoch in range(cfg['epochs']):
        epoch_start=time.perf_counter();order=rng.permutation(idx['train']);loss_sum=0
        for step,o in enumerate(range(0,len(order),cfg['batch_size'])):
            batch=order[o:o+cfg['batch_size']];optimizer.zero_grad(set_to_none=True);logits=forward(batch);loss=nn.functional.cross_entropy(logits,torch.tensor(labels[batch]));loss.backward()
            if epoch==0 and step==0:
                first_grad={n:float(p.grad.norm()) for n,p in model.named_parameters() if p.requires_grad and p.grad is not None}
                assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
            optimizer.step();loss_sum+=loss.item()*len(batch);peak=max(peak,psutil.Process().memory_info().rss)
        val=eval_split(idx['val']);scores=evaluate_logits(val,labels[idx['val']]);entry={'epoch':epoch+1,'train_loss':loss_sum/len(order),**{f'val_{k}':v for k,v in scores.items()},'epoch_s':time.perf_counter()-epoch_start};history.append(entry)
        print(variant.upper(),json.dumps(entry),flush=True)
        if scores['macro_f1']>best:
            best=scores['macro_f1'];checkpoint={'head':copy.deepcopy(head.state_dict()),'adapters':{n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad},'epoch':epoch+1,'config':cfg,'classes':json.loads((OUT/'data_audit.json').read_text())['classes']};torch.save(checkpoint,CKPT/f'{variant}.pt')
        save(f'{variant}_history.json',history)
    checkpoint=torch.load(CKPT/f'{variant}.pt',weights_only=True);head.load_state_dict(checkpoint['head'])
    for n,p in model.named_parameters():
        if n in checkpoint['adapters']:p.data.copy_(checkpoint['adapters'][n])
    frozen_unchanged=before==frozen_hash(model);assert frozen_unchanged
    test_logits=eval_split(idx['test']);test=evaluate_logits(test_logits,labels[idx['test']]);np.save(OUT/f'{variant}_test_logits.npy',test_logits)
    changed={n:float((p-initial_trainable[n]).abs().max()) for n,p in model.named_parameters() if p.requires_grad}
    if variant=='lora':assert any(v>0 for v in changed.values())
    pred=test_logits.argmax(1);classes=checkpoint['classes'];cm=confusion_matrix(labels[idx['test']],pred,labels=list(range(10)))
    rows_csv(OUT/f'{variant}_predictions.csv',[{'path':items[i]['path'],'true':classes[labels[i]],'predicted':classes[pred[j]]} for j,i in enumerate(idx['test'])])
    save(f'{variant}_classification_report.json',classification_report(labels[idx['test']],pred,target_names=classes,output_dict=True,zero_division=0))
    fig,ax=plt.subplots(figsize=(8,7));im=ax.imshow(cm,cmap='Blues');ax.set_xticks(range(10),classes,rotation=70,ha='right');ax.set_yticks(range(10),classes)
    for a in range(10):
        for b in range(10):ax.text(b,a,str(cm[a,b]),ha='center',va='center',fontsize=8,color='white' if cm[a,b]>cm.max()/2 else 'black')
    ax.set_xlabel('Predicción');ax.set_ylabel('Etiqueta');ax.set_title(f'{variant}: test interno (500 imágenes)');fig.tight_layout();fig.savefig(FIG/f'confusion_{variant}.png',dpi=160);plt.close(fig)
    result={'variant':variant,'best_epoch':checkpoint['epoch'],'best_val_macro_f1':best,'test':test,'elapsed_training_and_validation_s':time.perf_counter()-start,'trainable_parameters':sum(p.numel() for p in params),'backbone_parameters':sum(p.numel() for p in model.parameters()),'rss_sampled_peak_mib':peak/2**20,'initial_equivalence_max_abs':err,'frozen_weights_unchanged':frozen_unchanged,'first_batch_adapter_gradient_norms':first_grad,'adapter_max_abs_changes':changed,'one_seed':cfg['seed'],'selection':'highest validation macro-F1, first epoch on ties; test evaluated once after selection','split':'internal random stratified, NOT geographic','created_utc':stamp()}
    # Full-image inference timing with trained head (cache never used for deployment).
    wrapper=nn.Sequential(model,head).eval();result['inference']=benchmark(wrapper,image_batch(items[:1],cfg),repeats=10)
    save(f'{variant}_results.json',result);print('RESULT',json.dumps(result),flush=True)

def summarize(cfg):
    a=json.loads((OUT/'baseline_results.json').read_text());b=json.loads((OUT/'lora_results.json').read_text())
    save('comparison.json',{'baseline':a['test'],'lora':b['test'],'delta_macro_f1':b['test']['macro_f1']-a['test']['macro_f1'],'status':'preliminary E2, one seed, 1000 train / 300 validation / 500 internal test; no geographic claim'})
    fig,ax=plt.subplots(1,2,figsize=(11,4))
    for var,c in [('baseline','#3765a3'),('lora','#2c786c')]:
        hist=json.loads((OUT/f'{var}_history.json').read_text());ax[0].plot([r['epoch'] for r in hist],[r['train_loss'] for r in hist],'-o',label=var,color=c);ax[1].plot([r['epoch'] for r in hist],[r['val_macro_f1'] for r in hist],'-o',label=var,color=c)
    ax[0].set_ylabel('Cross-entropy de entrenamiento');ax[1].set_ylabel('Macro-F1 de validación')
    for v in ax:v.set_xlabel('Época');v.legend();v.grid(alpha=.2)
    fig.tight_layout();fig.savefig(FIG/'learning_curves.png',dpi=170);plt.close(fig)
    items=[r for r in manifest() if r['split']=='test'];pred=np.load(OUT/'lora_test_logits.npy').argmax(1);wrong=[i for i,r in enumerate(items) if r['label']!=pred[i]][:6]
    classes=json.loads((OUT/'data_audit.json').read_text())['classes'];fig,axes=plt.subplots(2,3,figsize=(10,7))
    for ax in axes.flat:ax.axis('off')
    for ax,i in zip(axes.flat,wrong):ax.imshow(Image.open(ROOT/items[i]['path']));ax.set_title(f'Real: {items[i]["class"]}\nLoRA: {classes[pred[i]]}',fontsize=10)
    fig.suptitle('Primeros errores por orden del manifiesto, sin selección favorable');fig.tight_layout();fig.savefig(FIG/'failure_examples.png',dpi=160);plt.close(fig)
    print('SUMMARY COMPLETE',flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','teardown','cache','baseline','lora','summarize','all']);args=parser.parse_args();cfg=config();torch.set_num_threads(cfg['threads']);seed_all(cfg['seed'])
    stages=['prepare','teardown','cache','baseline','lora','summarize'] if args.stage=='all' else [args.stage]
    for stage in stages:
        print('STAGE',stage,stamp(),flush=True)
        if stage in ['baseline','lora']:train(cfg,stage)
        else:globals()[{'cache':'cache_features'}.get(stage,stage)](cfg)
if __name__=='__main__':main()
