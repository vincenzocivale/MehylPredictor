#!/usr/bin/env python3
"""Extract frozen, locus-centred NTv3 embeddings from validated hg38 CpGs.

The output is an NPZ shard per context length/orientation plus JSON provenance.
It deliberately accepts only one-row-per-CpG input and validates every CG.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from types import SimpleNamespace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM, AutoTokenizer
from huggingface_hub import model_info

if __package__ in {None, ""}:  # allow direct module-file execution in development
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from methylation_predictor.genomic_encoder.ntv3_prior_common import base_to_output, centred_window, load_fasta, reverse_complement, sha256


def _out_vectors(output, length: int) -> dict[str, np.ndarray]:
    final = getattr(output, "embedding", None)
    transformer = getattr(output, "after_transformer_embedding", None)
    if final is None or transformer is None:
        raise RuntimeError("checkpoint did not expose both embedding and after_transformer_embedding")
    # Pool on device: copying full [B,L,D] towers to host dominates long-context
    # inference while the experiment needs only five [B,D] locus vectors.
    def centre(tensor):
        tensor = tensor.float()
        c, g = length // 2 - 1, length // 2
        ci, gi = base_to_output(c, length, tensor.shape[1]), base_to_output(g, length, tensor.shape[1])
        return tensor[:, [ci, gi]].mean(dim=1)
    def pooled(tensor, radius):
        tensor = tensor.float()
        c, g = length // 2 - 1, length // 2
        lo = base_to_output(max(0, c - radius), length, tensor.shape[1])
        hi = base_to_output(min(length - 1, g + radius), length, tensor.shape[1])
        return tensor[:, lo:hi + 1].mean(dim=1)
    values = {"centre": centre(final), "transformer": centre(transformer),
              "pool_32": pooled(final, 32), "pool_128": pooled(final, 128), "pool_512": pooled(final, 512)}
    return {name: value.detach().float().cpu().numpy().astype(np.float32) for name, value in values.items()}


def _model(repo: str, revision: str | None, device: str, bf16: bool):
    kwargs = {"trust_remote_code": True, "revision": revision}
    if bf16:
        kwargs.update({name: "bfloat16" for name in ("stem_compute_dtype", "down_convolution_compute_dtype", "transformer_qkvo_compute_dtype", "transformer_ffn_compute_dtype", "up_convolution_compute_dtype", "modulation_compute_dtype")})
    tokenizer = AutoTokenizer.from_pretrained(repo, **kwargs)
    # The pre-trained NTv3 checkpoints intentionally expose their custom class
    # through AutoModelForMaskedLM only; the post-trained variants expose
    # AutoModel.  Select from the checkpoint's declared auto-map rather than
    # guessing from its name.
    config = AutoConfig.from_pretrained(repo, **kwargs)
    auto_map = getattr(config, "auto_map", {}) or {}
    loader = AutoModel if "AutoModel" in auto_map else AutoModelForMaskedLM
    if loader is AutoModelForMaskedLM and "AutoModelForMaskedLM" not in auto_map:
        raise RuntimeError(f"checkpoint has no supported AutoModel mapping: {auto_map}")
    model = loader.from_pretrained(repo, **kwargs).to(device).eval()
    return tokenizer, model


def _forward_locus_embeddings(model, input_ids: torch.Tensor, species: list[str]):
    """Normalise the two public NTv3 API variants to the common outputs.

    Post-trained NTv3 supplies base-resolution and post-transformer embeddings
    directly and requires a species id.  The masked-language-model checkpoint
    supplies the same stages as hidden states: its final state is the
    base-resolution decoder output and the final state at the minimum sequence
    resolution is the output of the Transformer tower.
    """
    if hasattr(model, "encode_species"):
        species_ids = model.encode_species(species).to(input_ids.device)
        return model(input_ids=input_ids, species_ids=species_ids)
    # Calling the HF masked-LM wrapper with output_hidden_states=True retains
    # every large convolution/deconvolution tensor.  The underlying public
    # Core API can instead retain exactly the final Transformer and decoder
    # states requested by this experiment.
    core = model.core
    core.config.embeddings_layers_to_save = [len(core.transformer_blocks)]
    core.config.deconv_layers_to_save = [len(core.deconv_tower_blocks)]
    output = core(input_ids=input_ids, output_hidden_states=False)
    transformer = output[f"embeddings_{len(core.transformer_blocks)}"]
    final = output[f"embeddings_deconv_{len(core.deconv_tower_blocks)}"].permute(0, 2, 1)
    return SimpleNamespace(embedding=final, after_transformer_embedding=transformer)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True); p.add_argument("--fasta", required=True)
    p.add_argument("--output-dir", required=True); p.add_argument("--checkpoint", required=True)
    p.add_argument("--revision"); p.add_argument("--lengths", default="1024,8192,32768,131072")
    p.add_argument("--batch-size", type=int, default=1); p.add_argument("--device", default="cuda")
    p.add_argument("--bf16", action="store_true"); p.add_argument("--limit", type=int)
    p.add_argument("--progress-every", type=int, default=25,
                   help="emit a nohup-friendly progress bar every N batches (0 disables)")
    a = p.parse_args()
    if not torch.cuda.is_available() and a.device.startswith("cuda"):
        raise RuntimeError("CUDA is required for the complete NTv3 extraction but is unavailable")
    frame = pd.read_parquet(a.input)
    required = {"cpg_idx", "chromosome", "position"}
    if missing := required - set(frame): raise ValueError(f"input lacks {sorted(missing)}")
    if frame.cpg_idx.duplicated().any() or set(frame.chromosome) != {"chr1"}: raise ValueError("input must be unique chr1 CpGs")
    frame = frame.sort_values("cpg_idx").reset_index(drop=True)
    if a.limit: frame = frame.iloc[:a.limit].copy()
    genome = load_fasta(Path(a.fasta)); lengths = [int(x) for x in a.lengths.split(",")]
    for length in lengths:
        if length % 128: raise ValueError(f"{length} is not divisible by 128")
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    info = model_info(a.checkpoint, revision=a.revision)
    tokenizer, model = _model(a.checkpoint, a.revision, a.device, a.bf16)
    manifest = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "command": " ".join(sys.argv),
                "checkpoint": a.checkpoint, "revision_requested": a.revision, "revision_resolved": info.sha,
                "checkpoint_files": [{"path": item.rfilename, "size": item.size} for item in info.siblings], "species": "human", "genome_build": "GRCh38/hg38",
                "coordinate_convention": "1-based C of validated CG", "input": str(a.input), "input_sha256": sha256(Path(a.input)),
                "fasta": str(a.fasta), "fasta_sha256": sha256(Path(a.fasta)), "n_cpg": len(frame), "lengths": lengths,
                "dtype_saved": "float32", "orientations": ["forward", "reverse_complement"]}
    for length in lengths:
        sequences = [centred_window(genome, int(pos), length) for pos in frame.position]
        for orientation in ("forward", "reverse_complement"):
            seqs = sequences if orientation == "forward" else [reverse_complement(s) for s in sequences]
            collected: dict[str, list[np.ndarray]] = {}
            total_batches = (len(seqs) + a.batch_size - 1) // a.batch_size
            started = time.monotonic()
            print(f"[NTv3] {a.checkpoint} L={length:,} {orientation}: 0/{total_batches} batches", flush=True)
            for begin in range(0, len(seqs), a.batch_size):
                batch = seqs[begin:begin + a.batch_size]
                encoded = tokenizer(batch, add_special_tokens=False, padding=True, return_tensors="pt")
                ids = encoded["input_ids"].to(a.device)
                with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=a.bf16):
                    vectors = _out_vectors(_forward_locus_embeddings(model, ids, ["human"] * len(batch)), length)
                for name, value in vectors.items(): collected.setdefault(name, []).append(value)
                done = begin // a.batch_size + 1
                if a.progress_every and (done % a.progress_every == 0 or done == total_batches):
                    elapsed = time.monotonic() - started; fraction = done / total_batches
                    eta = elapsed * (1 / fraction - 1) if fraction else float("nan")
                    filled = round(24 * fraction)
                    bar = "#" * filled + "-" * (24 - filled)
                    print(f"[NTv3] [{bar}] {100 * fraction:5.1f}% {done}/{total_batches} "
                          f"elapsed={elapsed / 60:.1f}m eta={eta / 60:.1f}m", flush=True)
            arrays = {name: np.concatenate(parts) for name, parts in collected.items()}
            arrays["cpg_idx"] = frame.cpg_idx.to_numpy(np.int64)
            arrays["position"] = frame.position.to_numpy(np.int64)
            target = out / f"{a.checkpoint.rsplit('/', 1)[-1]}_L{length}_{orientation}.npz"
            np.savez_compressed(target, **arrays)
            manifest.setdefault("artifacts", []).append({"path": str(target), "sha256": sha256(target), "orientation": orientation,
                                                            "length": length, "shapes": {k: list(v.shape) for k, v in arrays.items()}})
    (out / "embedding_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__": main()
