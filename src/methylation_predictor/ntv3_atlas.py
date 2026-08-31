"""Build/extend the frozen NTv3 CpG embedding atlas (`cpg/ntv3/ntv3_cpg_atlas_v1.h5`).

Restores the NTv3 extraction pipeline that originally produced this repo's atlas
(`created_utc=2026-08-12`, see the file's own HDF5 attrs) -- the code that ran it was
deleted in commit `55ed6c1` (research variant, per-length/orientation NPZ shards) and
`a8ed56c` (the production `full_suite/expansion.py` variant that actually built the
committed atlas's chr1-3 `origin_1` rows) during pre-publication cleanup passes, before
this repo tracked provenance for one-off data-generation scripts. Recovered from git
history (`git show 8966825:src/methylation_predictor/full_suite/expansion.py`) and
adapted to current module layout: uses `storage.SortedIndex` instead of the removed
`full_suite.feature_store.SortedIndex` (identical `.contains`/`.positions_of` contract).

Locked contract, exactly matching the existing atlas so new rows are embedding-identical
to old ones if ever re-run at the same CpG: checkpoint ``InstaDeepAI/NTv3_650M_post``,
``window_length_bp=32768``, ``pooling=central_CG_mean`` (mean of the two output bins
covering the central C/G, taken from the final decoder tensor before the bf16->fp32
cast), ``orientation=forward`` only, ``reference_genome=hg38``, ``embedding_dim=1536``.

Two ways to build the "missing universe" (the shardable, resumable `cpg_idx`/
`chrom_code`/`position` manifest ``extract_ntv3_worker``/``merge_ntv3_shards`` consume):
``prepare_missing_universe_tcga_protocol`` (original use: TCGA canonical registries'
EPIC/WGBS auxiliary CpGs for a `tcga_mix_*` protocol) and the more general
``prepare_missing_universe_from_positions`` (any external CpG chrom/position list, e.g.
ENCODE) for CpGs not already covered by a base atlas.
"""
from __future__ import annotations

import math
import os
import time
from pathlib import Path
from types import SimpleNamespace

# Keep pyarrow ahead of h5py on the target server (libstdc++ symbol ordering).
import pyarrow.dataset as ds
import h5py
import numpy as np
import torch

from .storage import SortedIndex

CHROM_TO_CODE = {f"chr{i}": i for i in range(1, 23)}
CODE_TO_CHROM = {v: k for k, v in CHROM_TO_CODE.items()}


def _load_protocol_aux_ids(root: Path, protocol: str) -> dict[str, np.ndarray]:
    from .tcga_canonical import TCGACanonicalBundle, load_protocol

    with TCGACanonicalBundle.from_root(root) as bundle:
        p = load_protocol(protocol, bundle)
        return {k: np.asarray(v, dtype=np.int64) for k, v in p.auxiliary_cpg_idx.items()}


def _write_universe(output: Path, ids: np.ndarray, chromosome: np.ndarray, position: np.ndarray, *, shard_size: int, attrs: dict) -> dict:
    order = np.lexsort((position, chromosome))
    ids, chromosome, position = ids[order], chromosome[order], position[order]
    n_shards = int(math.ceil(len(ids) / shard_size))
    tmp = output.with_suffix(output.suffix + ".tmp")
    with h5py.File(tmp, "w") as h:
        h.create_dataset("cpg_idx", data=ids, dtype="i8")
        h.create_dataset("chrom_code", data=chromosome, dtype="u1")
        h.create_dataset("position", data=position, dtype="i8")
        h.attrs["shard_size"] = int(shard_size)
        h.attrs["n_shards"] = n_shards
        for k, v in attrs.items():
            h.attrs[k] = v
    os.replace(tmp, output)
    return {
        "missing_cpg": int(len(ids)),
        "shard_size": int(shard_size),
        "shards": n_shards,
        "by_chromosome": {CODE_TO_CHROM[c]: int((chromosome == c).sum()) for c in sorted(set(chromosome.tolist()))},
    }


def prepare_missing_universe_tcga_protocol(
    canonical_root: str | Path,
    base_embeddings_h5: str | Path,
    output_h5: str | Path,
    *,
    protocol: str = "tcga_mix_chr123",
    shard_size: int = 25_000,
) -> dict[str, object]:
    """Build the unique non-Array CpG universe a TCGA `tcga_mix_*` protocol still needs.

    Original use case: this is exactly how this repo's atlas got its chr1-3
    `origin_1` rows (5,314,693 CpGs). Coordinates are resolved by streaming the
    canonical EPIC/WGBS registries -- the 23M-row WGBS registry is never loaded
    whole into pandas.
    """
    root = Path(canonical_root)
    output = Path(output_h5)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_file():
        with h5py.File(output, "r") as h:
            n = int(h["cpg_idx"].shape[0])
            return {"status": "cached", "missing_cpg": n, "shards": int(h.attrs["n_shards"])}

    aux = _load_protocol_aux_ids(root, protocol)
    required = np.unique(np.concatenate([aux["epic"], aux["wgbs"]])).astype(np.int64)
    with h5py.File(base_embeddings_h5, "r") as h:
        base_ids = np.asarray(h["cpg_idx"][...], dtype=np.int64)
    base_index = SortedIndex(base_ids, "base embeddings")
    missing = required[~base_index.contains(required)]
    missing.sort()

    chromosome = np.zeros(len(missing), dtype=np.uint8)
    position = np.zeros(len(missing), dtype=np.int64)
    found = np.zeros(len(missing), dtype=bool)

    for source_name in ("epic", "wgbs"):
        registry = root / "registries" / f"{source_name}_cpg_map.parquet"
        dataset = ds.dataset(str(registry), format="parquet")
        filt = ds.field("chr").isin(["chr1", "chr2", "chr3"])
        scanner = dataset.scanner(columns=["cpg_idx", "chr", "pos"], filter=filt, batch_size=262_144)
        for batch in scanner.to_batches():
            ids = batch.column("cpg_idx").to_numpy(zero_copy_only=False).astype(np.int64)
            slots = np.searchsorted(missing, ids)
            valid = slots < len(missing)
            if not valid.any():
                continue
            slots_v = slots[valid]; ids_v = ids[valid]
            exact = missing[slots_v] == ids_v
            if not exact.any():
                continue
            slots_v = slots_v[exact]
            chrom_values = np.asarray(batch.column("chr").to_pylist(), dtype=object)[valid][exact]
            pos_values = batch.column("pos").to_numpy(zero_copy_only=False).astype(np.int64)[valid][exact]
            chrom_codes = np.asarray([CHROM_TO_CODE.get(str(x), 0) for x in chrom_values], dtype=np.uint8)
            if np.any(chrom_codes == 0):
                raise ValueError("registry contains unsupported chromosome in chr1-3 protocol")
            already = found[slots_v]
            if already.any():
                s = slots_v[already]
                if not np.array_equal(chromosome[s], chrom_codes[already]) or not np.array_equal(position[s], pos_values[already]):
                    raise ValueError(f"coordinate disagreement while merging {source_name} registry")
            chromosome[slots_v] = chrom_codes
            position[slots_v] = pos_values
            found[slots_v] = True

    if not found.all():
        unresolved = missing[~found]
        raise RuntimeError(f"could not resolve coordinates for {len(unresolved)} missing CpGs; examples={unresolved[:10].tolist()}")

    summary = _write_universe(output, missing, chromosome, position, shard_size=shard_size,
                               attrs={"protocol": protocol, "base_embedding_rows": int(len(base_ids))})
    return {"status": "built", "protocol": protocol, "required_aux_unique": int(len(required)), "base_rows": int(len(base_ids)), **summary}


def prepare_missing_universe_from_positions(
    chrom: np.ndarray,
    position: np.ndarray,
    base_embeddings_h5: str | Path,
    output_h5: str | Path,
    *,
    id_offset: int | None = None,
    shard_size: int = 25_000,
    source_label: str = "external",
) -> dict[str, object]:
    """Build the missing-CpG universe for an arbitrary external chrom/position list.

    Unlike ``prepare_missing_universe_tcga_protocol`` this list carries no pre-existing
    global ``cpg_idx`` (e.g. ENCODE's own CpG numbering is not this repo's canonical
    ``cpg_idx`` space): new IDs are minted above the base atlas's max id so they never
    collide with array/EPIC/WGBS ids, and CpGs already covered by ``base_embeddings_h5``
    are matched and dropped **by (chromosome, position)**, not by id.
    """
    output = Path(output_h5)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_file():
        with h5py.File(output, "r") as h:
            n = int(h["cpg_idx"].shape[0])
            return {"status": "cached", "missing_cpg": n, "shards": int(h.attrs["n_shards"])}

    chrom = np.asarray(chrom, dtype=object)
    position = np.asarray(position, dtype=np.int64)
    if len(chrom) != len(position):
        raise ValueError("chrom/position length mismatch")
    chrom_codes = np.asarray([CHROM_TO_CODE.get(str(x), 0) for x in chrom], dtype=np.uint8)
    if np.any(chrom_codes == 0):
        bad = sorted(set(str(x) for x, c in zip(chrom, chrom_codes) if c == 0))
        raise ValueError(f"unsupported chromosome labels (expected chr1..chr22): {bad[:10]}")

    with h5py.File(base_embeddings_h5, "r") as h:
        base_chrom = np.asarray(h["chrom_code"][...], dtype=np.uint8) if "chrom_code" in h else None
        base_position = np.asarray(h["position"][...], dtype=np.int64) if "position" in h else None
        base_ids = np.asarray(h["cpg_idx"][...], dtype=np.int64)
    if base_chrom is None or base_position is None:
        raise RuntimeError(f"{base_embeddings_h5} lacks chrom_code/position -- cannot dedup by coordinate")

    # (chrom_code, position) as one packed int64 key for a vectorized set-membership test.
    def _key(c, p):
        return (c.astype(np.int64) << 40) | p
    base_keys = np.unique(_key(base_chrom, base_position))
    query_keys = _key(chrom_codes, position)
    covered = np.isin(query_keys, base_keys)
    new_mask = ~covered
    if not new_mask.any():
        return {"status": "empty", "missing_cpg": 0, "reason": "every requested CpG already in base atlas"}

    dedup_keys, first_idx = np.unique(query_keys[new_mask], return_index=True)
    keep = np.flatnonzero(new_mask)[first_idx]
    new_chrom = chrom_codes[keep]
    new_position = position[keep]

    start_id = int(id_offset) if id_offset is not None else (int(base_ids.max()) + 1 if len(base_ids) else 0)
    new_ids = np.arange(start_id, start_id + len(new_chrom), dtype=np.int64)

    summary = _write_universe(output, new_ids, new_chrom, new_position, shard_size=shard_size,
                               attrs={"source_label": source_label, "base_embedding_rows": int(len(base_ids)), "id_offset": start_id})
    return {"status": "built", "source_label": source_label, "requested": int(len(position)),
            "already_covered": int(covered.sum()), "id_offset": start_id, **summary}


def _load_ntv3_model(checkpoint: str, device: str, bf16: bool):
    from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM, AutoTokenizer

    kwargs = {"trust_remote_code": True}
    if bf16:
        kwargs.update({
            name: "bfloat16"
            for name in (
                "stem_compute_dtype", "down_convolution_compute_dtype", "transformer_qkvo_compute_dtype",
                "transformer_ffn_compute_dtype", "up_convolution_compute_dtype", "modulation_compute_dtype",
            )
        })
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, **kwargs)
    config = AutoConfig.from_pretrained(checkpoint, **kwargs)
    auto_map = getattr(config, "auto_map", {}) or {}
    loader = AutoModel if "AutoModel" in auto_map else AutoModelForMaskedLM
    model = loader.from_pretrained(checkpoint, **kwargs).to(device).eval()
    return tokenizer, model


def _forward_ntv3(model, input_ids: torch.Tensor, species: list[str]):
    if hasattr(model, "encode_species"):
        species_ids = model.encode_species(species).to(input_ids.device)
        return model(input_ids=input_ids, species_ids=species_ids)
    core = model.core
    core.config.embeddings_layers_to_save = [len(core.transformer_blocks)]
    core.config.deconv_layers_to_save = [len(core.deconv_tower_blocks)]
    output = core(input_ids=input_ids, output_hidden_states=False)
    final = output[f"embeddings_deconv_{len(core.deconv_tower_blocks)}"].permute(0, 2, 1)
    transformer = output[f"embeddings_{len(core.transformer_blocks)}"]
    return SimpleNamespace(embedding=final, after_transformer_embedding=transformer)


def _base_to_output(base_index: int, input_length: int, output_length: int) -> int:
    if not 0 <= base_index < input_length or output_length < 1:
        raise ValueError("invalid base/output coordinate")
    return min(output_length - 1, int(((base_index + 0.5) * output_length) // input_length))


def _centre_embedding(output, length: int) -> torch.Tensor:
    # Only the two output bins covering the central C/G are consumed by the
    # downstream model. Select them before the BF16->FP32 conversion -- verified
    # bit-identical to a historical full-map cast while avoiding a large
    # unnecessary conversion/copy.
    tensor = output.embedding
    c, g = length // 2 - 1, length // 2
    ci = _base_to_output(c, length, tensor.shape[1])
    gi = _base_to_output(g, length, tensor.shape[1])
    return tensor[:, [ci, gi]].float().mean(dim=1)


def _build_fast_char_token_lut(tokenizer) -> np.ndarray:
    """Build and verify the exact NTv3 A/C/G/T/N character-token mapping.

    The released NTv3 tokenizer is character-level for the genomic alphabet.
    Production inference spends substantial CPU time repeatedly invoking the HF
    tokenizer on 32,768-bp strings. Build the mapping once and fail closed unless
    it exactly matches the tokenizer on a deterministic mixed-sequence probe.
    """
    lut = np.full(256, -1, dtype=np.int64)
    for base in "ACGTN":
        ids = tokenizer(base, add_special_tokens=False)["input_ids"]
        if len(ids) != 1:
            raise RuntimeError(f"NTv3 tokenizer is not one-character-per-token for {base!r}: {ids}")
        lut[ord(base)] = int(ids[0])

    probe = ("ACGTN" * 103)[:511]
    expected = np.asarray(tokenizer(probe, add_special_tokens=False)["input_ids"], dtype=np.int64)
    raw = np.frombuffer(probe.encode("ascii"), dtype=np.uint8)
    observed = lut[raw]
    if not np.array_equal(expected, observed):
        raise RuntimeError("fast NTv3 character tokenizer failed equivalence check")
    return lut


def _fast_tokenize_sequences(sequences: list[str], lut: np.ndarray) -> torch.Tensor:
    """Tokenize equal-length genomic strings with the verified ASCII LUT."""
    if not sequences:
        raise ValueError("cannot tokenize an empty NTv3 sequence batch")
    length = len(sequences[0])
    if any(len(sequence) != length for sequence in sequences):
        raise ValueError("NTv3 sequence batch contains unequal sequence lengths")
    raw = np.frombuffer("".join(sequences).encode("ascii"), dtype=np.uint8)
    raw = raw.reshape(len(sequences), length)
    ids = lut[raw]
    if (ids < 0).any():
        unsupported = sorted(chr(int(x)) for x in np.unique(raw[ids < 0]))
        raise ValueError(f"unsupported FASTA characters for NTv3 tokenizer: {unsupported}")
    return torch.from_numpy(np.ascontiguousarray(ids))


def extract_ntv3_worker(
    universe_h5: str | Path,
    fasta_path: str | Path,
    output_dir: str | Path,
    *,
    rank: int,
    world_size: int,
    checkpoint: str = "InstaDeepAI/NTv3_650M_post",
    length: int = 32768,
    batch_size: int = 4,
    device: str = "cuda",
    bf16: bool = True,
    storage_dtype: str = "float16",
    compile_model: bool = False,
    compile_mode: str = "max-autotune-no-cudagraphs",
) -> dict[str, object]:
    """One persistent model process handles every Nth shard (multi-GPU friendly).

    ``compile_model=True`` wraps the model in ``torch.compile``. Default mode is
    ``max-autotune-no-cudagraphs``, NOT plain ``max-autotune``: the latter crashed
    a real production run (`RuntimeError: accessing tensor output of CUDAGraphs
    that has been overwritten by a subsequent run`) -- NTv3's rotary-embedding
    cache (`_ensure_cache` in the remote modeling code) keeps a Python-side
    persistent buffer across calls, which is unsafe under CUDAGraphs replay
    unless the model itself clones/marks step boundaries, which it does not.
    ``max-autotune-no-cudagraphs`` keeps Inductor's kernel fusion/autotuning
    without capturing graphs, and was stress-tested over 200 sequential batches
    of distinct random sequences with no crash. Measured throughput on this
    codebase's RTX PRO 5000 Blackwell: eager 10.7-10.8 cpg/s -> compiled ~26
    cpg/s (~2.4x), output within normal bf16 kernel-fusion noise of eager (max
    relative diff ~0.4% on random sequences, no NaNs). The model's own remote
    code is explicitly written to be torch.compile-friendly (compiler-friendly
    LayerNormFP32, see its docstring) -- just not CUDAGraphs-safe. First-batch
    compilation costs ~2-3 minutes per worker process; worth it for any job
    processing more than a handful of shards.
    """
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for NTv3 expansion but torch.cuda.is_available() is false")
    if length != 32768:
        raise ValueError("the current atlas contract is locked to NTv3 length=32768")
    from pyfaidx import Fasta

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with h5py.File(universe_h5, "r") as h:
        n = int(h["cpg_idx"].shape[0])
        shard_size = int(h.attrs["shard_size"])
        n_shards = int(h.attrs["n_shards"])
    assigned = list(range(rank, n_shards, world_size))
    if not assigned:
        return {"rank": rank, "assigned": 0, "completed": 0}

    # The launcher builds the pyfaidx index exactly once before spawning workers.
    # Workers open it read-only-ish (`rebuild=False`) so multi-GPU startup cannot
    # race while rewriting the same .fai/.gzi files.
    genome = Fasta(str(fasta_path), as_raw=True, sequence_always_upper=True, rebuild=False)
    genome_keys = set(genome.keys())
    tokenizer, model = _load_ntv3_model(checkpoint, device, bf16)
    if compile_model:
        model = torch.compile(model, mode=compile_mode)
    token_lut = _build_fast_char_token_lut(tokenizer)
    save_dtype = np.float16 if storage_dtype == "float16" else np.float32
    completed = 0
    started = time.time()
    log_every_batches = 100

    print(
        f"[ntv3 rank={rank}] START assigned_shards={len(assigned)} "
        f"batch_size={batch_size} storage_dtype={np.dtype(save_dtype).name} "
        f"fast_tokenizer=verified_char_lut compile={compile_model} log_every_batches={log_every_batches}",
        flush=True,
    )

    with h5py.File(universe_h5, "r") as universe:
        for shard_id in assigned:
            target = out / f"shard_{shard_id:05d}.h5"
            done = out / f"shard_{shard_id:05d}.done"
            if done.is_file() and target.is_file():
                with h5py.File(target, "r") as cached:
                    cached_dtype = str(cached["embedding"].dtype)
                    cached_ok = (
                        cached.attrs.get("checkpoint") == checkpoint
                        and int(cached.attrs.get("length", -1)) == length
                        and cached_dtype == np.dtype(save_dtype).name
                    )
                if cached_ok:
                    completed += 1
                    continue
                done.unlink(missing_ok=True)
                target.unlink(missing_ok=True)
            lo = shard_id * shard_size
            hi = min(n, lo + shard_size)
            ids = np.asarray(universe["cpg_idx"][lo:hi], dtype=np.int64)
            chrom = np.asarray(universe["chrom_code"][lo:hi], dtype=np.uint8)
            pos = np.asarray(universe["position"][lo:hi], dtype=np.int64)
            values: list[np.ndarray] = []
            shard_started = time.time()
            n_batches = int(math.ceil(len(ids) / batch_size))

            for batch_index, b0 in enumerate(range(0, len(ids), batch_size), start=1):
                b1 = min(len(ids), b0 + batch_size)
                seqs: list[str] = []
                for ccode, p1 in zip(chrom[b0:b1], pos[b0:b1]):
                    cname = CODE_TO_CHROM[int(ccode)]
                    if cname not in genome_keys:
                        alt = cname.removeprefix("chr")
                        if alt in genome_keys:
                            cname = alt
                        else:
                            raise KeyError(f"FASTA contains neither {CODE_TO_CHROM[int(ccode)]!r} nor {alt!r}")
                    # Exact historical centred_window convention: registry position
                    # is 1-based C of CG; C/G land at L/2-1,L/2.  Near chromosome
                    # boundaries pad with N rather than shifting the locus.
                    start0 = int(p1) - length // 2
                    end0 = start0 + length
                    left = max(0, -start0); right = max(0, end0 - len(genome[cname]))
                    body = str(genome[cname][max(0, start0):min(len(genome[cname]), end0)])
                    seq = "N" * left + body + "N" * right
                    if len(seq) != length or seq[length // 2 - 1:length // 2 + 1] != "CG":
                        centre = seq[length // 2 - 1:length // 2 + 1]
                        raise ValueError(f"hg38 coordinate validation failed for {cname}:{p1}; centre={centre!r}")
                    seqs.append(seq)
                real_n = len(seqs)
                if compile_model and real_n < batch_size:
                    # torch.compile is shape-specialized; a ragged last batch would
                    # trigger a fresh (~minutes-long) recompilation per shard.  Pad
                    # with repeats of an already-validated sequence and discard the
                    # padded rows below instead.
                    seqs = seqs + [seqs[-1]] * (batch_size - real_n)
                input_ids = _fast_tokenize_sequences(seqs, token_lut).to(device)
                with torch.inference_mode(), torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16, enabled=bf16 and device.startswith("cuda")
                ):
                    output = _forward_ntv3(model, input_ids, ["human"] * len(seqs))
                    centre = _centre_embedding(output, length)
                values.append(centre[:real_n].detach().cpu().numpy().astype(save_dtype))

                if batch_index % log_every_batches == 0 or b1 == len(ids):
                    elapsed_shard = max(time.time() - shard_started, 1e-9)
                    rate = b1 / elapsed_shard
                    eta_minutes = (len(ids) - b1) / max(rate, 1e-9) / 60.0
                    print(
                        f"[ntv3 rank={rank}] shard={shard_id+1}/{n_shards} "
                        f"batch={batch_index}/{n_batches} cpg={b1}/{len(ids)} "
                        f"rate={rate:.2f}cpg/s eta_shard={eta_minutes:.1f}m",
                        flush=True,
                    )

            emb = np.concatenate(values, axis=0)
            if emb.shape[0] != len(ids) or emb.shape[1] != 1536:
                raise RuntimeError(f"unexpected NTv3 shard shape {emb.shape}; expected ({len(ids)},1536)")
            tmp = target.with_suffix(".tmp.h5")
            with h5py.File(tmp, "w") as h:
                h.create_dataset("cpg_idx", data=ids, dtype="i8")
                h.create_dataset("embedding", data=emb, chunks=(min(512, len(ids)), 1536))
                h.attrs["checkpoint"] = checkpoint
                h.attrs["length"] = length
                h.attrs["orientation"] = "forward"
                h.attrs["rank"] = rank
                h.attrs["world_size"] = world_size
            os.replace(tmp, target)
            done.write_text("ok\n")
            completed += 1
            elapsed = time.time() - started
            print(
                f"[ntv3 rank={rank}] shard {shard_id+1}/{n_shards}; "
                f"assigned_complete={completed}/{len(assigned)} elapsed={elapsed/3600:.2f}h",
                flush=True,
            )
    return {"rank": rank, "assigned": len(assigned), "completed": completed}


def merge_ntv3_shards(
    universe_h5: str | Path,
    shards_dir: str | Path,
    output_dir: str | Path,
    *,
    storage_dtype: str = "float32",
) -> dict[str, object]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ids_path = out / "expanded_cpg_idx.npy"
    if storage_dtype not in {"float16", "float32"}:
        raise ValueError("storage_dtype must be float16 or float32")
    np_dtype = np.float16 if storage_dtype == "float16" else np.float32
    emb_path = out / f"expanded_embeddings.{'f16' if storage_dtype == 'float16' else 'f32'}.npy"
    done = out / "embeddings.done"
    if done.is_file() and ids_path.is_file() and emb_path.is_file():
        return {"status": "cached", "rows": int(len(np.load(ids_path, mmap_mode="r")))}

    with h5py.File(universe_h5, "r") as universe:
        ids = np.asarray(universe["cpg_idx"][...], dtype=np.int64)
        n_shards = int(universe.attrs["n_shards"])
    for stale in out.glob("expanded_embeddings.*.npy"):
        if stale != emb_path:
            stale.unlink(missing_ok=True)
    mmap = np.lib.format.open_memmap(emb_path, mode="w+", dtype=np_dtype, shape=(len(ids), 1536))
    cursor = 0
    for shard_id in range(n_shards):
        path = Path(shards_dir) / f"shard_{shard_id:05d}.h5"
        marker = Path(shards_dir) / f"shard_{shard_id:05d}.done"
        if not path.is_file() or not marker.is_file():
            raise RuntimeError(f"missing NTv3 shard {shard_id}: {path}")
        with h5py.File(path, "r") as h:
            s_ids = np.asarray(h["cpg_idx"][...], dtype=np.int64)
            emb = np.asarray(h["embedding"][...], dtype=np_dtype)
        expected = ids[cursor:cursor + len(s_ids)]
        if not np.array_equal(s_ids, expected):
            raise RuntimeError(f"shard {shard_id} cpg_idx does not match universe order")
        mmap[cursor:cursor + len(s_ids)] = emb
        cursor += len(s_ids)
    del mmap
    if cursor != len(ids):
        raise RuntimeError(f"merged {cursor} rows but universe has {len(ids)}")
    np.save(ids_path, ids)
    done.write_text("ok\n")
    return {"status": "built", "rows": int(len(ids)), "embedding_shape": [int(len(ids)), 1536]}
