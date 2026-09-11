#!/usr/bin/env python3
"""Download/extract a BigWig pilot cache at CpGs, with train-only scaling."""
from __future__ import annotations
import argparse, hashlib, json, subprocess
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser(); p.add_argument("--manifest",type=Path,required=True); p.add_argument("--cpg-coordinates",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--download-dir",type=Path,required=True); p.add_argument("--window",type=int,default=250); p.add_argument("--train-mask",type=Path,required=True,help="Boolean mask fitted only on the requested training CpGs"); p.add_argument("--download",action="store_true"); a=p.parse_args()
    try: import pyBigWig
    except ImportError as e: raise SystemExit("Install pyBigWig from requirements-genomics.txt") from e
    manifest=json.loads(a.manifest.read_text()); coords=np.genfromtxt(a.cpg_coordinates,delimiter="\t",names=True,dtype=None,encoding="utf8")
    chrom=np.asarray(coords["chr"]).astype(str); pos1=np.asarray(coords["position"],np.int64)
    # BigWig uses 0-based half-open coordinates.  The atlas position column is
    # 1-based; bed_pos0 is retained as an explicit audit of this conversion.
    pos=np.asarray(coords["bed_pos0"],np.int64) if "bed_pos0" in coords.dtype.names else pos1 - 1
    if np.any(pos < 0) or np.any(pos1 != pos + 1): raise ValueError("invalid 1-based/0-based coordinate conversion")
    train=np.load(a.train_mask).astype(bool)
    if len(train)!=len(pos): raise ValueError("train mask and coordinates have different lengths")
    a.download_dir.mkdir(parents=True,exist_ok=True); a.output.mkdir(parents=True,exist_ok=True)
    tracks=[]; values=[]
    for item in manifest["tracks"]:
        path=a.download_dir/(item["accession"]+".bigWig")
        if a.download and (not path.exists() or path.stat().st_size != int(item["bytes"])):
            cmd=["curl","-L","--fail","--retry","3"]
            if path.exists(): cmd += ["-C","-"]
            cmd += ["-o",str(path),item["url"]]
            subprocess.run(cmd,check=True)
        if not path.exists(): raise FileNotFoundError(path)
        if item.get("md5"):
            h=hashlib.md5(path.read_bytes()).hexdigest()
            if h != item["md5"]: raise ValueError(f"checksum mismatch: {path}")
        bw=pyBigWig.open(str(path)); vals=np.empty((len(pos),5),np.float32)
        # Read chromosome in ~1 Mb blocks. Random per-CpG BigWig calls are
        # prohibitively slow at 2M loci; block reads reduce this to ~250 calls
        # per track while preserving the exact local windows.
        for current_chrom in np.unique(chrom):
            chrom_rows=np.flatnonzero(chrom == current_chrom)
            order=chrom_rows[np.argsort(pos[chrom_rows],kind="stable")]
            sorted_pos=pos[order]
            # CpGs are sparse along chr1; smaller blocks avoid materializing
            # enormous mostly-empty BigWig intervals.
            block_size = 2000
            for block_start in range(0,len(order),block_size):
                block_ids=order[block_start:block_start+block_size]; p0=int(sorted_pos[block_start]); p1=int(sorted_pos[min(len(order)-1,block_start+block_size-1)])
                chrom_len=int(bw.chroms().get(str(current_chrom), 0))
                lo=max(0,p0-a.window); hi=min(chrom_len,p1+a.window+1)
                arr=np.asarray(bw.values(str(current_chrom),lo,hi,numpy=True),np.float32)
                finite=np.isfinite(arr); safe=np.where(finite,arr,0.0)
                for row in block_ids:
                    x=int(pos[row]); l=max(0,x-a.window)-lo; r=min(hi,x+a.window+1)-lo; q=safe[l:r]; f=finite[l:r]
                    q=q[f]
                    vals[row]=[float(q.mean()) if len(q) else 0.,float(q.max()) if len(q) else 0.,float(np.quantile(q,.9)) if len(q) else 0.,float(np.median(q)) if len(q) else 0.,float(np.mean(q>0)) if len(q) else 0.]
        bw.close(); center=np.nanmedian(vals[train],axis=0); scale=np.nanpercentile(vals[train],75,axis=0)-np.nanpercentile(vals[train],25,axis=0)
        # Sparse signal summaries can have zero IQR; a unit fallback avoids
        # turning a harmless zero into a million-scale feature.  Clipping is
        # fitted-feature hygiene for heavy-tailed public signal tracks.
        scale=np.maximum(scale,1.0); normalized=np.clip((vals-center)/scale,-20.0,20.0)
        values.append(normalized.astype(np.float32)); tracks.append({**item,"center":center.tolist(),"scale":scale.tolist(),"clip":[-20.0,20.0]})
    x=np.concatenate(values,axis=1); np.save(a.output/"features.f32.npy",x); np.save(a.output/"cpg_idx.npy",coords["cpg_idx"]); (a.output/"manifest.json").write_text(json.dumps({"schema_version":3,"status":"complete","assembly":"GRCh38","coordinate_system":"0-based center derived from 1-based position","window":a.window,"summary_features":["mean","max","q90","median","positive_fraction"],"tracks":tracks,"fit_mask":"explicit_training_mask","train_mask_sha256":hashlib.sha256(train.tobytes()).hexdigest(),"rows":len(pos),"dimensions":x.shape[1],"normalization":{"kind":"median_iqr","minimum_scale":1.0,"clip":[-20.0,20.0]}},indent=2)+"\n")
    print(json.dumps({"rows":len(pos),"dimensions":x.shape[1],"output":str(a.output)},indent=2))
if __name__=="__main__": main()
