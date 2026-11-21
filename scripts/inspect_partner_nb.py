import json

with open(r'C:\Users\italo\.gemini\antigravity\brain\33baf598-a1d5-4ede-8b5f-7db59ef93aff\scratch\partner_repo\E2_teardown_lora.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'code':
        print(f"Cell {i:02d}: exec_count={c.get('execution_count')} outputs={len(c.get('outputs', []))}")
        for o in c.get('outputs', []):
            if o.get('output_type') == 'stream':
                txt = ''.join(o.get('text', []))[:80].replace('\n', ' ')
                print(f"   [stream] {txt.encode('ascii', errors='replace').decode('ascii')}")
            elif o.get('output_type') == 'display_data':
                print(f"   [display_data] keys={list(o.get('data', {}).keys())}")
            elif o.get('output_type') == 'execute_result':
                print(f"   [execute_result] keys={list(o.get('data', {}).keys())}")
