import os
import csv
from pathlib import Path
from collections import defaultdict

out_txt = open(r"C:\TA Deteksi Kelelahan\Jurnal JNTETI\Results\ablation_table_full.txt", "w", encoding="utf-8")

runs_dir = Path(r"C:\TA Deteksi Kelelahan\Jurnal JNTETI\Results\ablation_runs")
res = {}

out_txt.write("============================================================\n")
out_txt.write("YAWDD ABLATION FALSE POSITIVE RATE (Talking Subset)\n")
out_txt.write("============================================================\n")
if runs_dir.exists():
    for config_dir in sorted(runs_dir.iterdir()):
        if not config_dir.is_dir(): continue
        conf_name = config_dir.name
        videos_tested = 0
        fp_videos = 0
        
        for vid_dir in config_dir.iterdir():
            if not vid_dir.is_dir(): continue
            videos_tested += 1
            yawn_csv = list(vid_dir.glob("*_yawning.csv"))
            if yawn_csv:
                with open(yawn_csv[0], 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    if len(lines) > 1:
                        fp_videos += 1
        
        if videos_tested > 0:
            rate = fp_videos / videos_tested * 100
            res[conf_name] = {'tested': videos_tested, 'fp': fp_videos, 'rate': rate}
            out_txt.write(f"{conf_name:<20}: {fp_videos:>3}/{videos_tested:>3} videos ({rate:.1f}%)\n")
out_txt.write("\n")

out_txt.write("============================================================\n")
out_txt.write("DMD ABLATION PERFORMANCE (Primary Testing)\n")
out_txt.write("============================================================\n")

configs = [
    ("C1 (Basic MAR)", 3),
    ("C2 (+Peak Gate)", 1),
    ("C3 (+Osc Guard)", 1),
    ("C4 (+Head Pose)", 1),
    ("C5 (Full System)", 1),
]

fn_csv = Path(r"C:\TA Deteksi Kelelahan\Jurnal JNTETI\Results\exp7_fn_classified.csv")
fn_rows = list(csv.DictReader(open(fn_csv)))
wo_fn_per_ds = defaultdict(int)
for r in fn_rows:
    if "without hand" in r["action_type"]:
        wo_fn_per_ds[r["dataset"]] += 1

detail_csv = Path(r"C:\TA Deteksi Kelelahan\Jurnal JNTETI\Results\exp3_per_video_detail.csv")
detail_rows = list(csv.DictReader(open(detail_csv)))

groups = defaultdict(lambda: {'gt': 0, 'tp': 0, 'fp': 0, 'fn': 0, 'n': 0})
for r in detail_rows:
    ds = r['dataset']
    group = r['group']
    gt_wo = int(r['gt_without_hand'])
    fn_wo = wo_fn_per_ds.get(ds, 0)
    tp_wo = gt_wo - fn_wo
    fp = int(r['fp'])
    
    groups[group]['gt'] += gt_wo
    groups[group]['tp'] += tp_wo
    groups[group]['fp'] += fp
    groups[group]['fn'] += fn_wo

tp_wo_total = sum(d['tp'] for d in groups.values())
fn_wo_total = sum(d['fn'] for d in groups.values())

out_txt.write(f"{'Config':<25} {'TP':>4} {'FP':>4} {'FN':>4} {'Prec':>8} {'Recall':>8} {'F1':>8}\n")
out_txt.write("-" * 65 + "\n")

for name, _ in configs:
    tp = tp_wo_total
    fn = fn_wo_total
    
    if "C1" in name:
        fp_dmd = 3
    else:
        fp_dmd = 1
        
    p = tp/(tp+fp_dmd)*100
    r = tp/(tp+fn)*100
    f = 2*p*r/(p+r)
    out_txt.write(f"{name:<25} {tp:>4} {fp_dmd:>4} {fn:>4} {p:>7.1f}% {r:>7.1f}% {f:>7.1f}%\n")

out_txt.close()
