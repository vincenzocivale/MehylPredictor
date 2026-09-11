#!/usr/bin/env python3
"""Collect real ladder histories into TSV and Markdown; never fabricates missing metrics."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path

FIELDS = ["variant", "epochs_run", "params", "peak_vram_bytes", "train_cpg_x_val_sample_mas",
          "val_cpg_x_train_sample_mas", "val_cpg_x_val_sample_mas", "mac", "mse", "mae", "sec_per_epoch",
          "early_stop", "run_dir"]

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("run_root"); p.add_argument("--output-prefix", required=True); a=p.parse_args()
    rows=[]
    for summary_path in sorted(Path(a.run_root).rglob("training/summary.json")):
        summary=json.loads(summary_path.read_text()); run_dir=summary_path.parents[1]
        if "functional-fusion-f" not in run_dir.name: continue
        history=json.loads((run_dir/"training/history.json").read_text()); best=int(summary.get("best_epoch") or len(history))
        row=history[max(0,best-1)]; views=row.get("development", {}); headline=views.get("val_cpg_x_val_sample", {})
        get=lambda view,key: views.get(view,{}).get(key)
        rows.append({"variant":run_dir.name.split("functional-fusion-")[1].split("-")[0], "epochs_run":summary.get("epochs_run"),
            "params":summary.get("trainable_parameters"), "peak_vram_bytes":summary.get("peak_cuda_memory_bytes"),
            "train_cpg_x_val_sample_mas":get("train_cpg_x_val_sample","mas_pcc"), "val_cpg_x_train_sample_mas":get("val_cpg_x_train_sample","mas_pcc"),
            "val_cpg_x_val_sample_mas":headline.get("mas_pcc"), "mac":headline.get("mac_pcc"), "mse":headline.get("mse"), "mae":headline.get("mae"),
            "sec_per_epoch":sum(x["seconds"] for x in history)/len(history), "early_stop":summary.get("epochs_run",0)<summary.get("epochs_planned",0), "run_dir":str(run_dir)})
    prefix=Path(a.output_prefix); prefix.parent.mkdir(parents=True,exist_ok=True)
    with prefix.with_suffix(".tsv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=FIELDS,delimiter="\t"); w.writeheader(); w.writerows(rows)
    lines=["| "+" | ".join(FIELDS)+" |", "|"+"---|"*len(FIELDS)]
    lines += ["| "+" | ".join(str(row.get(k,"")) for k in FIELDS)+" |" for row in rows]
    prefix.with_suffix(".md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"runs":len(rows),"tsv":str(prefix.with_suffix('.tsv')),"markdown":str(prefix.with_suffix('.md'))},indent=2))

if __name__=="__main__": main()
