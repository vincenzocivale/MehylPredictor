#!/usr/bin/env python3
"""Select a small, auditable GRCh38 BigWig panel for the first pilot."""
from __future__ import annotations
import argparse, glob, json
from pathlib import Path

def label(x):
    target = x.get("target") or ""
    if isinstance(target, dict):
        target = target.get("label") or target.get("name") or ""
    return str(target).split("/")[-2] if "/" in str(target) else str(target)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--source",type=Path,default=Path("/dune/DATASETS/MethylPredictionData/derived/ntv3_post_track_manifest/methylation_bigwigs/api_cache")); p.add_argument("--output",type=Path,required=True); p.add_argument("--per-class",type=int,default=3)
    a=p.parse_args(); rows=[]
    for path in glob.glob(str(a.source/"*.json")):
        data=json.loads(Path(path).read_text())
        rows.extend(x for x in data.get("@graph",[]) if "File" in x.get("@type",[]) and x.get("assembly")=="GRCh38" and x.get("file_format","").lower() in {"bigwig","bigwig signal"} and x.get("cloud_metadata",{}).get("url"))
    selected=[]; seen=set()
    # Prefer independent assay/target classes, then the smallest valid file.
    # Do not mix p-value tracks with quantitative signal tracks.
    for assay, targets in [("DNase-seq",[""]), ("Histone ChIP-seq",["H3K27ac","H3K4me3","H3K27me3"]), ("TF ChIP-seq",["CTCF","REST","SP1"])]:
        for target in targets:
            candidates=[x for x in rows if x.get("assay_title")==assay and (not target or target.lower() in label(x).lower()) and x.get("accession") not in seen and "encode-private" not in x.get("cloud_metadata",{}).get("url","")]
            allowed = {"read-depth normalized signal"} if assay == "DNase-seq" else {"fold change over control"}
            candidates = [x for x in candidates if x.get("output_type") in allowed]
            candidates.sort(key=lambda x:int(x.get("cloud_metadata",{}).get("file_size",10**30)))
            for x in candidates[:a.per_class if not target else 1]:
                seen.add(x["accession"]); selected.append({"accession":x["accession"],"assay":assay,"target":label(x),"output_type":x.get("output_type"),"dataset":x.get("dataset"),"biological_replicates":x.get("biological_replicates"),"assembly":x["assembly"],"bytes":x["cloud_metadata"]["file_size"],"url":x["cloud_metadata"]["url"],"md5":x.get("md5sum")})
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps({"schema_version":1,"assembly":"GRCh38","selection":"smallest valid file per requested assay/target","tracks":selected},indent=2)+"\n")
    print(json.dumps({"tracks":len(selected),"bytes":sum(x["bytes"] for x in selected),"output":str(a.output)},indent=2))
if __name__=="__main__": main()
