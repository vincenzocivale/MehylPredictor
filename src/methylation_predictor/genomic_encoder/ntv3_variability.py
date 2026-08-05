#!/usr/bin/env python3
"""Train-only TCGA locus-variability targets and frozen-NTv3 probes.

The methylation matrix is streamed once; only CpGs in ``--input`` and only
sample IDs in the locked training manifest enter target construction.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, subprocess, sys, tempfile
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
if __package__ in {None, ""}: sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from methylation_predictor.genomic_encoder.ntv3_prior_common import sha256
from methylation_predictor.genomic_encoder.ntv3_prior import MLP, _fit_mlp
from methylation_predictor.genomic_encoder.static_prior import _partition

def build_targets(frame, matrix, train_manifest, metadata, minimum):
    print("[targets] loading locked sample manifest and metadata", flush=True)
    train = set(pd.read_parquet(train_manifest).sample_idx.astype(int))
    md = pd.read_parquet(metadata).set_index("sample_idx")
    md = md.loc[md.index.intersection(train)]
    wanted = set(frame.cpg_idx.astype(int)); rows = []
    key_to_cpg = {f'"{chrom}_{int(pos)}",': int(cpg) for cpg, chrom, pos in
                  frame[["cpg_idx", "chromosome", "position"]].itertuples(index=False, name=None)}
    with open(matrix, newline="") as handle:
        print("[targets] reading matrix header", flush=True)
        header = next(csv.reader(handle))
        selected = [(i, md.loc[md.index[md.sample_name.eq(name)][0], "cancer_type"])
                    for i, name in enumerate(header[1:], 1)
                    if (md.sample_name == name).any()]
        if len(selected) != len(md): raise ValueError("matrix header does not cover every locked train sample")
        cols = np.array([x[0] for x in selected]); cancer = np.array([x[1] for x in selected], object)
        print(f"[targets] filtering {matrix} with ripgrep; samples={len(cols)}, CpGs={len(wanted)}", flush=True)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False) as patterns:
            patterns.write("\n".join(key_to_cpg) + "\n"); pattern_path = patterns.name
        proc = subprocess.Popen(["rg", "--no-filename", "-F", "-f", pattern_path, str(matrix)], stdout=subprocess.PIPE, text=True, bufsize=1)
        for line in proc.stdout:
            cpg_idx = key_to_cpg.get(line.split(",", 1)[0] + ",")
            if cpg_idx is None: raise ValueError("unexpected matching matrix row")
            values = line.rstrip("\n").split(",")
            x = pd.to_numeric(pd.Series([values[i] for i in cols]), errors="coerce").to_numpy(float)
            ok = np.isfinite(x); x, c = x[ok], cancer[ok]
            if len(x) < 2: continue
            group = pd.DataFrame({"x": x, "c": c}).groupby("c", sort=True).x
            counts = group.count(); means = group.mean(); eligible = counts[counts >= minimum].index
            n = counts.loc[eligible].to_numpy(float); m = means.loc[eligible].to_numpy(float)
            if len(n) < 2: continue
            grand = np.average(m, weights=n)
            between = float(np.average((m-grand)**2, weights=n))
            within = float(sum(((group.get_group(k)-means[k])**2).sum() for k in eligible) / n.sum())
            rows.append((cpg_idx, len(x), int(n.sum()), int(len(eligible)), float(np.var(x)), between, within))
            if len(rows) % 500 == 0: print(f"[targets] retained {len(rows)}/{len(wanted)} CpGs", flush=True)
        if proc.wait() != 0: raise RuntimeError("ripgrep matrix scan failed")
        Path(pattern_path).unlink(missing_ok=True)
    result = pd.DataFrame(rows, columns=["cpg_idx","n_train_nonmissing","n_eligible","n_cancer_types","total_variance","between_cancer_variance","within_cancer_variance"])
    if result.cpg_idx.duplicated().any() or len(result) != len(wanted): raise ValueError(f"target coverage {len(result)}/{len(wanted)}")
    return result

def metrics(y, p, train_y):
    y, p, train_y = np.asarray(y, dtype=float), np.asarray(p, dtype=float), np.asarray(train_y, dtype=float)
    out={"mse_log_variance":float(np.mean((y-p)**2)),"spearman":float(spearmanr(y,p).statistic),"pearson":float(pearsonr(y,p).statistic)}
    cuts=np.quantile(train_y,[1/3,2/3]); a=np.digitize(y,cuts); b=np.digitize(p,cuts)
    out["tercile_accuracy"]=float(np.mean(a==b)); out["tercile_confusion"]=[[int(((a==i)&(b==j)).sum()) for j in range(3)] for i in range(3)]
    bins=pd.qcut(pd.Series(p),10,duplicates="drop").to_numpy(); out["calibration_deciles"]= [{"n":int(len(z)),"pred_mean_logvar":float(z.p.mean()),"observed_mean_logvar":float(z.y.mean())} for _,z in pd.DataFrame({"y":y,"p":p,"bin":bins}).groupby("bin",observed=True)]
    return out

def run_probe(frame, z, split, device, output, include_test, component, test_model):
    frame=frame.copy(); frame["split"]=_partition(frame,split,"cpg_idx","chromosome","position",5_000_000,20260728)
    tr,va,te=(frame[frame.split.eq(x)] for x in ("train","validation","test")); ycol=f"log_{component}_variance"
    scaler=StandardScaler().fit(z[tr._row]); xtr=scaler.transform(z[tr._row]); ytr=tr[ycol].to_numpy(float)
    xva=scaler.transform(z[va._row]); xte=scaler.transform(z[te._row])
    ridge=Ridge(alpha=10.).fit(xtr,ytr); pred={"ridge":(ridge.predict(xva),ridge.predict(xte) if include_test and test_model == "all" else None)}
    other=xva if not include_test else np.concatenate([xva,xte])
    for seed in (17,29,43):
        q=_fit_mlp(xtr,ytr,other,seed,500,1e-3,1e-4,.1,device); pred[f"mlp_seed_{seed}"]=(q[:len(va)],q[len(va):] if include_test else None)
    names=[x for x in pred if x.startswith("mlp_seed")]; pred["mlp_ensemble"]=(np.mean([pred[x][0] for x in names],0),np.mean([pred[x][1] for x in names],0) if include_test else None)
    report={"split":split,"target":f"log({component}_variance + 1e-8)","selection":"minimum validation MSE log-variance", "rows":{x:int(len(v)) for x,v in (("train",tr),("validation",va),("test",te))},"validation":{k:metrics(va[ycol],v[0],ytr) for k,v in pred.items()}}
    if include_test: report["test"]={k:metrics(te[ycol],v[1],ytr) for k,v in pred.items() if v[1] is not None}
    output.mkdir(parents=True,exist_ok=True); (output/"metrics.json").write_text(json.dumps(report,indent=2)+"\n")

def main():
 p=argparse.ArgumentParser(); p.add_argument("--input",required=True); p.add_argument("--embeddings",required=True); p.add_argument("--matrix",required=True); p.add_argument("--train-manifest",required=True); p.add_argument("--sample-metadata",required=True); p.add_argument("--output-dir",required=True); p.add_argument("--targets", help="existing train-only variance targets parquet"); p.add_argument("--minimum-per-cancer-type",type=int,default=10); p.add_argument("--split",choices=("block","random","fixed"),default="block"); p.add_argument("--variance-component",choices=("total","between_cancer","within_cancer"),default="total"); p.add_argument("--test-model",choices=("all","mlp_ensemble"),default="all"); p.add_argument("--validation-only",action="store_true"); p.add_argument("--device",default="cuda"); a=p.parse_args()
 print("[variability] starting", flush=True); frame=pd.read_parquet(a.input); target_path=Path(a.targets) if a.targets else Path(a.output_dir)/"train_only_variance_targets.parquet"
 if target_path.exists():
  print(f"[variability] reusing existing train-only targets: {target_path}", flush=True); targets=pd.read_parquet(target_path)
 else:
  targets=build_targets(frame,Path(a.matrix),Path(a.train_manifest),Path(a.sample_metadata),a.minimum_per_cancer_type); target_path.parent.mkdir(parents=True,exist_ok=True); targets.to_parquet(target_path,index=False)
 if "split" in targets.columns and "split" in frame.columns:
  # build_genome_wide_targets.py's output doubles as --prior (consumed by
  # static_features.py, which is where --input's own split column came from)
  # and as --targets here; --input's split is authoritative, drop the
  # duplicate to avoid a split_x/split_y merge collision.
  targets=targets.drop(columns="split")
 frame=frame.merge(targets,on="cpg_idx",validate="one_to_one")
 for component in ("total", "between_cancer", "within_cancer"):
  frame[f"log_{component}_variance"] = np.log(frame[f"{component}_variance"] + 1e-8)
 with np.load(a.embeddings) as e: ids=e["cpg_idx"].astype(int); z=e["centre"].astype(np.float32)
 frame=frame.merge(pd.DataFrame({"cpg_idx":ids,"_row":np.arange(len(ids))}),on="cpg_idx",validate="one_to_one")
 run_probe(frame,z,a.split,a.device,Path(a.output_dir),not a.validation_only,a.variance_component,a.test_model)
 manifest={"created_at_utc":datetime.now(timezone.utc).isoformat(),"command":" ".join(sys.argv),"sources":{x:sha256(Path(getattr(a,x))) for x in ("input","embeddings","matrix","train_manifest","sample_metadata")},"target_definitions":{"total":"population variance across all nonmissing locked train samples","between":"sample-count-weighted variance of eligible cancer-type means","within":"sample-count-weighted mean within-cancer population variance","minimum_per_cancer_type":a.minimum_per_cancer_type,"epsilon":1e-8},"n_targets":len(frame)}
 (Path(a.output_dir)/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
if __name__=="__main__": main()
