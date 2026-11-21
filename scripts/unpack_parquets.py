import io
import pathlib
import sys
from PIL import Image
import pyarrow.parquet as pq

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
OUT_DIR = DATA_DIR / 'rgb' / 'EuroSAT_RGB'

CLASSES = [
    'AnnualCrop',
    'Forest',
    'HerbaceousVegetation',
    'Highway',
    'Industrial',
    'Pasture',
    'PermanentCrop',
    'Residential',
    'River',
    'SeaLake'
]

def unpack():
    for c in CLASSES:
        (OUT_DIR / c).mkdir(parents=True, exist_ok=True)
    
    parquet_files = [
        DATA_DIR / 'train.parquet',
        DATA_DIR / 'val.parquet',
        DATA_DIR / 'test.parquet'
    ]
    
    counts = {c: 0 for c in CLASSES}
    total = 0
    
    for p in parquet_files:
        print(f"Reading {p.name}...", flush=True)
        table = pq.read_table(p)
        print(f"Schema: {table.schema.names}", flush=True)
        
        # Determine column names
        cols = table.schema.names
        img_col = 'image' if 'image' in cols else cols[0]
        label_col = 'label' if 'label' in cols else cols[1]
        
        images = table[img_col].to_pylist()
        labels = table[label_col].to_pylist()
        
        for item, lbl in zip(images, labels):
            if isinstance(lbl, int):
                cls_name = CLASSES[lbl]
            else:
                cls_name = str(lbl)
            
            # Extract bytes
            if isinstance(item, dict) and 'bytes' in item and item['bytes'] is not None:
                img_bytes = item['bytes']
            elif isinstance(item, bytes):
                img_bytes = item
            else:
                raise ValueError(f"Unexpected image structure: {type(item)}")
            
            counts[cls_name] += 1
            filename = f"{cls_name}_{counts[cls_name]}.jpg"
            target_path = OUT_DIR / cls_name / filename
            
            with open(target_path, 'wb') as f:
                f.write(img_bytes)
            
            total += 1
            if total % 5000 == 0:
                print(f"Extracted {total} images...", flush=True)

    print(f"\nUnpacking completed! Total images: {total}", flush=True)
    for c in CLASSES:
        print(f"  {c}: {counts[c]}", flush=True)
    assert total == 27000, f"Expected 27000 images, got {total}"

if __name__ == '__main__':
    unpack()
