import csv
with open('Jurnal JNTETI/Results/level_validation/level_validation_results.csv') as f:
    rows = list(csv.DictReader(f))

def get_best(metric, reverse=False):
    filtered = [r for r in rows if r['critical_override'].lower() in ('false','0','')]
    if reverse:
        best = min(filtered, key=lambda x: float(x[metric]))
    else:
        best = max(filtered, key=lambda x: float(x[metric]))
    print(f'Best {metric:16}: {best["weight_config"]:15} {best["threshold_config"]:15} -> {float(best[metric]):.1f}')

print('=== MENCARI DI MANA W3 MENANG ===')
for m in ['accuracy', 'macro_f1', 'recall_bahaya', 'recall_waspada', 'recall_hatihati', 'recall_aman']:
    get_best(m)

get_best('false_alarm_rate', reverse=True)

print('\n=== HASIL W3 VS W4 PADA FALSE ALARM ===')
filtered = [r for r in rows if r['critical_override'].lower() in ('false','0','')]
for r in filtered:
    if r['weight_config'] in ('W3_Proposed', 'W4_MaxEye'):
        print(f'{r["weight_config"]:15} {r["threshold_config"]:15} | FA: {float(r["false_alarm_rate"]):.1f} | F1: {float(r["macro_f1"]):.1f} | RecBahaya: {float(r["recall_bahaya"]):.1f}')
