# =====================================================================================
# C1 · Ejemplos individuales de predicción + McNemar exacto con predicciones pareadas
# Pegar como ÚLTIMA celda de E2_teardown_lora.ipynb y ejecutar después de "Ejecutar todo"
# (usa: dataset, idx_te, CLASES, cache, cabeza_base, cabeza_lora, modelo_lora, dl_te,
#  p_base, p_lora, y_test, representacion, DEVICE, USAR_AMP).
# Genera: figures/C1_prediction_examples.png, results/C1_predicciones_ejemplo.csv,
#         results/C1_mcnemar_exacto.json
# =====================================================================================
import json, numpy as np, pandas as pd, torch, matplotlib.pyplot as plt
from scipy.stats import binomtest

# 1) Probabilidades en test para ambos modelos (mismo orden que idx_te / y_test)
with torch.no_grad():
    prob_base = torch.softmax(cabeza_base(cache['test'][0].to(DEVICE)).float(), 1).cpu().numpy()
    pl = []
    for x, _ in dl_te:   # modelo_lora ya está fusionado (celda de costo): forward normal
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=USAR_AMP):
            pl.append(torch.softmax(cabeza_lora(representacion(modelo_lora(pixel_values=x.to(DEVICE)).last_hidden_state)).float(), 1).cpu())
    prob_lora = torch.cat(pl).numpy()
assert (prob_lora.argmax(1) == p_lora).mean() > 0.999, 'Las predicciones no coinciden con p_lora: revisar el orden de ejecución'

# 2) McNemar EXACTO con las predicciones pareadas (reemplaza la cota conservadora del póster)
ok_b, ok_l = p_base == y_test, p_lora == y_test
b = int((ok_b & ~ok_l).sum())    # base acierta, LoRA falla
c = int((~ok_b & ok_l).sum())    # base falla, LoRA acierta
p_exacto = binomtest(b, b + c, 0.5).pvalue
chi2_cc = (abs(b - c) - 1) ** 2 / (b + c)
mcn = {'base_ok_lora_mal': b, 'base_mal_lora_ok': c, 'chi2_con_correccion': round(chi2_cc, 2), 'p_exacto_binomial': float(p_exacto)}
json.dump(mcn, open('results/C1_mcnemar_exacto.json', 'w'), indent=2); print('McNemar pareado:', mcn)

# 3) Elegir ejemplos reales: 2 corregidos por LoRA (River→Highway), 1 acierto de ambos, 1 error nuevo de LoRA
C = {n: i for i, n in enumerate(CLASES)}
def elegir(mask, k):
    idx = np.where(mask)[0]
    return idx[np.argsort(-prob_lora[idx].max(1))][:k].tolist()   # los de mayor confianza de LoRA
casos = (
    [('Fixed by LoRA', i) for i in elegir((y_test == C['River']) & (p_base == C['Highway']) & (p_lora == C['River']), 2)] +
    [('Correct in both', i) for i in elegir((y_test == C['Highway']) & ok_b & ok_l, 1)] +
    [('New LoRA error', i) for i in elegir((y_test == C['PermanentCrop']) & ok_b & (p_lora == C['HerbaceousVegetation']), 1)]
)
assert len(casos) == 4, f'Solo se encontraron {len(casos)} casos; cambiar las clases elegidas'

filas = []
fig, axes = plt.subplots(1, len(casos), figsize=(4.2 * len(casos), 5.6))
for ax, (tipo, k) in zip(axes, casos):
    ruta = dataset.samples[idx_te[k]][0]
    pb, pl_ = int(p_base[k]), int(p_lora[k])
    filas.append({'caso': tipo, 'archivo': ruta.split('/')[-1], 'real': CLASES[y_test[k]],
                  'base_pred': CLASES[pb], 'base_prob': round(float(prob_base[k, pb]), 3),
                  'lora_pred': CLASES[pl_], 'lora_prob': round(float(prob_lora[k, pl_]), 3)})
    ax.imshow(plt.imread(ruta)); ax.set_xticks([]); ax.set_yticks([])
    color = {'Fixed by LoRA': '#C8601E', 'Correct in both': '#4B2A82', 'New LoRA error': '#7F7F7F'}[tipo]
    for s in ax.spines.values(): s.set_edgecolor(color); s.set_linewidth(6)
    ax.set_title(tipo, fontsize=18, fontweight='bold', color=color)
    ax.set_xlabel(f"True: {CLASES[y_test[k]]}\nBaseline: {CLASES[pb]} ({prob_base[k, pb]:.2f})\nLoRA: {CLASES[pl_]} ({prob_lora[k, pl_]:.2f})",
                  fontsize=15, linespacing=1.4)
plt.tight_layout(); plt.savefig('figures/C1_prediction_examples.png', dpi=300, bbox_inches='tight'); plt.show()
df_casos = pd.DataFrame(filas); df_casos.to_csv('results/C1_predicciones_ejemplo.csv', index=False); print(df_casos.to_string(index=False))
