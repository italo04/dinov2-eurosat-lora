"""Descarga fuentes oficiales; conserva hashes y commit de DINOv2."""
import hashlib, json, pathlib, urllib.request, zipfile, datetime
ROOT=pathlib.Path(__file__).resolve().parents[1]
def get(url,path):
    if path.exists(): return
    path.parent.mkdir(parents=True,exist_ok=True)
    print('Descargando',url,flush=True)
    req=urllib.request.Request(url,headers={'User-Agent':'TA-DeepLearning-E2/1.0'})
    with urllib.request.urlopen(req,timeout=120) as r, path.with_suffix(path.suffix+'.part').open('wb') as w:
        while chunk:=r.read(1024*1024):w.write(chunk)
    path.with_suffix(path.suffix+'.part').replace(path)
def digest(path,algorithm='sha256'):
    h=hashlib.new(algorithm)
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def extract(path,folder):
    folder.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(path) as z:
        for n in z.namelist():
            if not (folder/n).resolve().is_relative_to(folder.resolve()):raise ValueError(n)
        z.extractall(folder)
def main():
    meta_path=ROOT/'sources.lock.json'
    meta=json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if 'dinov2_commit' not in meta:
        req=urllib.request.Request('https://api.github.com/repos/facebookresearch/dinov2/commits/main',headers={'User-Agent':'TA-DeepLearning-E2/1.0'})
        with urllib.request.urlopen(req,timeout=60) as r:meta['dinov2_commit']=json.load(r)['sha']
        meta_path.write_text(json.dumps(meta,indent=2))
    sha=meta['dinov2_commit']
    src=ROOT/'vendor'/'dinov2.zip'
    get(f'https://codeload.github.com/facebookresearch/dinov2/zip/{sha}',src)
    if not (ROOT/'vendor'/f'dinov2-{sha}').exists():extract(src,ROOT/'vendor')
    weights=ROOT/'weights'/'dinov2_vits14_pretrain.pth'
    url='https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth'
    get(url,weights)
    data=ROOT/'data'/'EuroSAT_RGB.zip'
    data_url='https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1'
    get(data_url,data)
    assert digest(data,'md5')=='f46e308c4d50d4bf32fedad2d3d62f3b','EuroSAT MD5 incorrecto'
    if not (ROOT/'data'/'rgb'/'EuroSAT_RGB').exists():extract(data,ROOT/'data'/'rgb')
    meta.update({'model_url':url,'model_sha256':digest(weights),'source_sha256':digest(src),'dataset_url':data_url,'dataset_md5':digest(data,'md5'),'dataset_sha256':digest(data),'prepared_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    meta_path.write_text(json.dumps(meta,indent=2),encoding='utf8')
    print(json.dumps(meta,indent=2),flush=True)
if __name__=='__main__':main()
