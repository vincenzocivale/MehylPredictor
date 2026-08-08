"""Incremental (chunk-by-chunk) reconstruction of ``metrics.evaluate_predictions``'
metric set, for panels too large to materialize as a full ``[samples, cpgs]``
array in host RAM (e.g. the genome-wide ``in_distribution``/``sample_ood``/
``locus_ood`` panels, which touch the full ~327k-CpG train_cpg pool).

Every chunk fed to ``add_cpg_chunk`` covers ALL samples for a slice of CpGs
(the sample axis is never itself chunked at the accumulator level -- only
GPU-memory chunking of the sample axis happens inside the caller's forward
pass, same as today's ``predict_panel``). This makes almost every metric an
exact, order-independent running sum:

- Locus-centering (``_centre_by_locus`` in metrics.py) is a per-column
  operation -- a CpG's full sample axis is present within one chunk, so
  centering is exact and entirely chunk-local.
- Within-cancer centering (``_within_cancer_centred``) is a per-(cancer,
  column) operation -- same reasoning, exact and chunk-local.
- Global/tertile/cancer/chromosome MSE-family metrics and the dynamic-Pearson/
  amplitude-ratio/calibration-alpha family reduce to running sums of
  (n, Σx, Σy, Σx², Σy², Σxy) over (locus-)centered pairs -- additive across
  chunks, reconstructed exactly at ``finalize()``.
- ``locus_dynamic_pearson_median``/``_spearman_median`` (per-CpG, axis=0) are
  exact and cheap: a CpG's entire sample axis is present in one chunk, so its
  correlation is computed directly, once, per column (same asymptotic cost as
  the existing non-streaming per-column loop in ``metrics._median_axis_correlation``
  -- can be skipped via ``include_locus_median_correlations=False`` for the
  very largest panels).
- ``patient_dynamic_pearson_median`` (per-sample, axis=1) is the one genuine
  cross-chunk case: a sample's values are spread across every CpG chunk, so
  running per-sample (n, Σx, Σy, Σx², Σy², Σxy) accumulators (vectorized numpy
  arrays of length n_samples -- cheap, no Python-level per-sample loop) are
  updated every chunk and finalized once at the end.

Not exact (documented, not silently approximated): ``dynamic_spearman``,
``within_cancer_spearman`` are rank-based and not chunk-additive -- estimated
from a fixed-size seeded reservoir sample (same statistical precedent as
``metrics.evaluate_predictions``' ``correlation_max_n``: standard error of r
shrinks as 1/sqrt(n), so a reservoir of a couple million pairs is
statistically indistinguishable from the exact value). Per-tertile
``dynamic_spearman`` and ``patient_dynamic_spearman_median`` are not computed
at all for streaming panels (would need a full per-sample/per-tertile rank
over the entire CpG axis) -- expensive Spearman diagnostics are dropped at
genome-wide scale by design.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .metrics import _pearson, _safe_skill, _spearman


@dataclass
class _ErrorAccumulator:
    """Running (n, Σerr², Σ|err|) for a scalar error series."""

    n: float = 0.0
    sum_sq: float = 0.0
    sum_abs: float = 0.0

    def update(self, err: np.ndarray) -> None:
        if err.size == 0:
            return
        err = err.astype(np.float64, copy=False)
        self.n += err.size
        self.sum_sq += float((err * err).sum())
        self.sum_abs += float(np.abs(err).sum())

    def mse(self) -> float:
        return self.sum_sq / self.n if self.n else float("nan")

    def mae(self) -> float:
        return self.sum_abs / self.n if self.n else float("nan")


@dataclass
class _PairAccumulator:
    """Running (n, Σx, Σy, Σx², Σy², Σxy) over valid (x, y) pairs (pooled,
    scalar state -- used for the global/tertile/within-cancer dynamic-pair
    accumulators, as opposed to the vectorized per-sample version below)."""

    n: float = 0.0
    sx: float = 0.0
    sy: float = 0.0
    sxx: float = 0.0
    syy: float = 0.0
    sxy: float = 0.0

    def update(self, x: np.ndarray, y: np.ndarray) -> None:
        if x.size == 0:
            return
        x = x.astype(np.float64, copy=False)
        y = y.astype(np.float64, copy=False)
        self.n += x.size
        self.sx += float(x.sum())
        self.sy += float(y.sum())
        self.sxx += float((x * x).sum())
        self.syy += float((y * y).sum())
        self.sxy += float((x * y).sum())

    def _cov_var(self) -> tuple[float, float, float]:
        cov = self.sxy - self.sx * self.sy / self.n
        varx = self.sxx - self.sx * self.sx / self.n
        vary = self.syy - self.sy * self.sy / self.n
        return cov, varx, vary

    def pearson(self) -> float:
        if self.n < 2:
            return float("nan")
        cov, varx, vary = self._cov_var()
        if varx <= 0 or vary <= 0:
            return float("nan")
        return float(cov / np.sqrt(varx * vary))

    def calibration_alpha(self) -> float:
        """dot(x, y) / dot(x, x) on the RAW sums (no additional global-mean
        subtraction beyond the per-locus centering already applied to x, y
        upstream) -- matches metrics.py's `alpha = dot(px, ty) / dot(px, px)`
        exactly; NOT the same as the mean-adjusted covariance used by
        `pearson`/`amplitude_ratio` below."""
        if self.n == 0 or self.sxx <= 0:
            return float("nan")
        return float(self.sxy / self.sxx)

    def amplitude_ratio(self) -> float:
        """std(x) / std(y) -- np.std performs its own global-mean subtraction
        on top of any prior per-locus centering, unlike calibration_alpha."""
        if self.n < 2:
            return float("nan")
        _, varx, vary = self._cov_var()
        if vary <= 0:
            return float("nan")
        return float(np.sqrt(varx / vary))


class _VectorPairAccumulator:
    """Vectorized per-row (Σx, Σy, Σx², Σy², Σxy, n) accumulators over
    `n_rows` independent series, updated one chunk at a time via plain numpy
    reductions (no per-row Python loop) -- used for the exact patient-wise
    (per-sample) median Pearson, which needs a running accumulator per sample
    across CpG chunks."""

    def __init__(self, n_rows: int):
        self.n = np.zeros(n_rows, dtype=np.float64)
        self.sx = np.zeros(n_rows, dtype=np.float64)
        self.sy = np.zeros(n_rows, dtype=np.float64)
        self.sxx = np.zeros(n_rows, dtype=np.float64)
        self.syy = np.zeros(n_rows, dtype=np.float64)
        self.sxy = np.zeros(n_rows, dtype=np.float64)

    def update(self, x: np.ndarray, y: np.ndarray, valid: np.ndarray) -> None:
        """x, y, valid: [n_rows, chunk_cols]."""
        xz = np.where(valid, x, 0.0)
        yz = np.where(valid, y, 0.0)
        self.n += valid.sum(axis=1)
        self.sx += xz.sum(axis=1)
        self.sy += yz.sum(axis=1)
        self.sxx += (xz * xz).sum(axis=1)
        self.syy += (yz * yz).sum(axis=1)
        self.sxy += (xz * yz).sum(axis=1)

    def median_pearson(self) -> float:
        valid_rows = self.n > 1
        cov = self.sxy[valid_rows] - self.sx[valid_rows] * self.sy[valid_rows] / self.n[valid_rows]
        varx = self.sxx[valid_rows] - self.sx[valid_rows] ** 2 / self.n[valid_rows]
        vary = self.syy[valid_rows] - self.sy[valid_rows] ** 2 / self.n[valid_rows]
        denom = varx * vary
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.where(denom > 0, cov / np.sqrt(np.where(denom > 0, denom, 1.0)), np.nan)
        finite = r[np.isfinite(r)]
        return float(np.median(finite)) if finite.size else float("nan")


@dataclass
class _GroupAccumulator:
    model_error: _ErrorAccumulator = field(default_factory=_ErrorAccumulator)
    prior_error: _ErrorAccumulator = field(default_factory=_ErrorAccumulator)
    dynamic_pair: _PairAccumulator = field(default_factory=_PairAccumulator)


class _Reservoir:
    """Fixed-capacity reservoir sample (Algorithm R), seeded, for the
    rank-based metrics that aren't chunk-additive. Operates on already-flat
    1-D arrays passed in per chunk."""

    def __init__(self, capacity: int, seed: int):
        self.capacity = capacity
        self.rng = np.random.default_rng(seed)
        self.x = np.empty(capacity, dtype=np.float64)
        self.y = np.empty(capacity, dtype=np.float64)
        self.filled = 0
        self.seen = 0

    def add(self, x: np.ndarray, y: np.ndarray) -> None:
        """Vectorized -- O(capacity) work per call regardless of chunk size m.
        A per-element Python loop here (the naive textbook reservoir-sampling
        algorithm) was measured to cost >1 hour on a single genome-wide panel
        once the reservoir filled (every subsequent chunk of up to ~3.7M
        elements fell into an unvectorized `for xi, yi in zip(...)` loop).
        Once full, this replaces a batch-appropriate NUMBER of reservoir slots
        with a random subsample of the incoming chunk in one vectorized shot
        -- an approximation of exact sequential reservoir sampling (already
        documented as an approximate estimator, not required to be exactly
        uniform: see module docstring)."""
        m = x.size
        if m == 0:
            return
        if self.filled < self.capacity:
            take = min(self.capacity - self.filled, m)
            idx = self.rng.choice(m, size=take, replace=False) if take < m else np.arange(take)
            self.x[self.filled:self.filled + take] = x[idx]
            self.y[self.filled:self.filled + take] = y[idx]
            self.filled += take
            self.seen += m
            return
        self.seen += m
        expected = self.capacity * m / self.seen
        n_replace = int(expected) + (1 if self.rng.random() < (expected - int(expected)) else 0)
        n_replace = min(n_replace, self.capacity, m)
        if n_replace <= 0:
            return
        replace_slots = self.rng.choice(self.capacity, size=n_replace, replace=False)
        source_idx = self.rng.choice(m, size=n_replace, replace=False)
        self.x[replace_slots] = x[source_idx]
        self.y[replace_slots] = y[source_idx]

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        return self.x[:self.filled], self.y[:self.filled]


class StreamingPanelMetrics:
    """Accumulates sufficient statistics per CpG-axis chunk; `finalize()`
    returns the same key set as `metrics.evaluate_predictions()`, plus
    `per_chromosome` (mse/prior_mse/skill_vs_prior/observations per
    chromosome, mirroring `per_cancer`'s shape rather than the full
    per-tertile diagnostic set -- a documented, deliberately simpler design
    choice since the metric set requested for chromosome breakdowns wasn't
    specified beyond "metriche per cromosoma")."""

    def __init__(
        self,
        n_samples: int,
        cancer_type_names: list[str],
        reservoir_size: int = 2_000_000,
        reservoir_seed: int = 20260803,
        include_locus_median_correlations: bool = True,
    ):
        self.n_samples = n_samples
        self.cancer_type_names = list(cancer_type_names)
        self.include_locus_median_correlations = include_locus_median_correlations

        self._all = _GroupAccumulator()
        self._tertiles: dict[str, _GroupAccumulator] = {
            name: _GroupAccumulator() for name in ("low", "mid", "high")
        }
        self._per_cancer: dict[str, _ErrorAccumulator] = {n: _ErrorAccumulator() for n in self.cancer_type_names}
        self._per_cancer_prior: dict[str, _ErrorAccumulator] = {n: _ErrorAccumulator() for n in self.cancer_type_names}
        self._per_chromosome: dict[str, _ErrorAccumulator] = {}
        self._per_chromosome_prior: dict[str, _ErrorAccumulator] = {}
        self._within_cancer = _PairAccumulator()
        self._within_cancer_spearman = _Reservoir(reservoir_size, reservoir_seed + 1)
        self._dynamic_spearman_reservoir = _Reservoir(reservoir_size, reservoir_seed)

        self._sample_all = _VectorPairAccumulator(n_samples)
        self._sample_tertile = {name: _VectorPairAccumulator(n_samples) for name in ("low", "mid", "high")}

        self._sample_model_sq = np.zeros(n_samples, dtype=np.float64)
        self._sample_prior_sq = np.zeros(n_samples, dtype=np.float64)
        self._sample_valid_count = np.zeros(n_samples, dtype=np.float64)

        self._locus_pearson: list[float] = []
        self._locus_spearman: list[float] = []
        self._locus_tertile: list[str] = []
        self._cpg_win_model: list[float] = []
        self._cpg_win_prior: list[float] = []

    def add_cpg_chunk(
        self,
        target: np.ndarray,
        prediction: np.ndarray,
        prior: np.ndarray,
        cancer_types: np.ndarray,
        chromosome_codes: np.ndarray,
        tertile_codes: np.ndarray | None = None,
    ) -> None:
        """target/prediction: [n_samples, chunk_cpgs]. prior: [chunk_cpgs].
        cancer_types: [n_samples] (constant across calls). chromosome_codes/
        tertile_codes: [chunk_cpgs] (0/1/2 for tertile_codes -> low/mid/high)."""
        target = np.asarray(target, dtype=np.float64)
        prediction = np.asarray(prediction, dtype=np.float64)
        prior_matrix = np.broadcast_to(np.asarray(prior, dtype=np.float64)[None, :], target.shape)
        valid = np.isfinite(target) & np.isfinite(prediction) & np.isfinite(prior_matrix)

        model_err = target - prediction
        prior_err = target - prior_matrix
        true_dynamic = target - prior_matrix
        pred_dynamic = prediction - prior_matrix

        counts = valid.sum(axis=0, keepdims=True)
        true_means = np.divide(np.where(valid, true_dynamic, 0.0).sum(axis=0, keepdims=True), counts,
                                out=np.zeros_like(counts, dtype=np.float64), where=counts > 0)
        pred_means = np.divide(np.where(valid, pred_dynamic, 0.0).sum(axis=0, keepdims=True), counts,
                                out=np.zeros_like(counts, dtype=np.float64), where=counts > 0)
        true_centred = true_dynamic - true_means
        pred_centred = pred_dynamic - pred_means

        n_cpgs = target.shape[1]
        chromosome_codes = np.asarray(chromosome_codes)
        tertile_names = np.full(n_cpgs, "", dtype=object)
        if tertile_codes is not None:
            tertile_codes = np.asarray(tertile_codes, dtype=np.int64)
            for label, name in enumerate(("low", "mid", "high")):
                tertile_names[tertile_codes == label] = name

        self._update_group(
            self._all, model_err, prior_err, valid, pred_centred, true_centred,
            pred_dynamic, true_dynamic, self._sample_all, self._dynamic_spearman_reservoir,
        )
        if tertile_codes is not None:
            for label, name in enumerate(("low", "mid", "high")):
                cols = tertile_codes == label
                if not cols.any():
                    continue
                self._update_group(
                    self._tertiles[name], model_err[:, cols], prior_err[:, cols], valid[:, cols],
                    pred_centred[:, cols], true_centred[:, cols], pred_dynamic[:, cols], true_dynamic[:, cols],
                    self._sample_tertile[name], reservoir=None,
                )

        for cancer_type in self.cancer_type_names:
            rows = cancer_types == cancer_type
            if not rows.any():
                continue
            row_valid = valid[rows]
            self._per_cancer[cancer_type].update(model_err[rows][row_valid])
            self._per_cancer_prior[cancer_type].update(prior_err[rows][row_valid])

        for chromosome in np.unique(chromosome_codes):
            cols = chromosome_codes == chromosome
            key = str(chromosome)
            col_valid = valid[:, cols]
            self._per_chromosome.setdefault(key, _ErrorAccumulator()).update(model_err[:, cols][col_valid])
            self._per_chromosome_prior.setdefault(key, _ErrorAccumulator()).update(prior_err[:, cols][col_valid])

        # Within-cancer centering: per (cancer_type, column) mean, chunk-local.
        within_true = np.full_like(true_dynamic, np.nan)
        within_pred = np.full_like(pred_dynamic, np.nan)
        for cancer_type in np.unique(cancer_types):
            rows = cancer_types == cancer_type
            group_valid = valid[rows]
            group_counts = group_valid.sum(axis=0, keepdims=True)
            true_group_mean = np.divide(
                np.where(group_valid, true_dynamic[rows], 0.0).sum(axis=0, keepdims=True), group_counts,
                out=np.zeros_like(group_counts, dtype=np.float64), where=group_counts > 0,
            )
            pred_group_mean = np.divide(
                np.where(group_valid, pred_dynamic[rows], 0.0).sum(axis=0, keepdims=True), group_counts,
                out=np.zeros_like(group_counts, dtype=np.float64), where=group_counts > 0,
            )
            within_true[rows] = true_dynamic[rows] - true_group_mean
            within_pred[rows] = pred_dynamic[rows] - pred_group_mean
        wx = within_pred[valid]
        wy = within_true[valid]
        self._within_cancer.update(wx, wy)
        self._within_cancer_spearman.add(wx, wy)

        self._sample_model_sq += np.where(valid, model_err ** 2, 0.0).sum(axis=1)
        self._sample_prior_sq += np.where(valid, prior_err ** 2, 0.0).sum(axis=1)
        self._sample_valid_count += valid.sum(axis=1)

        if self.include_locus_median_correlations:
            for j in range(n_cpgs):
                col_valid = valid[:, j]
                if not col_valid.any():
                    self._locus_pearson.append(float("nan"))
                    self._locus_spearman.append(float("nan"))
                else:
                    pd_col = pred_dynamic[col_valid, j]
                    td_col = true_dynamic[col_valid, j]
                    self._locus_pearson.append(_pearson(pd_col, td_col))
                    self._locus_spearman.append(_spearman(pd_col, td_col))
                self._locus_tertile.append(str(tertile_names[j]))

        with np.errstate(invalid="ignore"):
            col_model_mse = np.nanmean(np.where(valid, model_err ** 2, np.nan), axis=0)
            col_prior_mse = np.nanmean(np.where(valid, prior_err ** 2, np.nan), axis=0)
        self._cpg_win_model.extend(col_model_mse.tolist())
        self._cpg_win_prior.extend(col_prior_mse.tolist())

    @staticmethod
    def _update_group(
        group: _GroupAccumulator,
        model_err: np.ndarray, prior_err: np.ndarray, valid: np.ndarray,
        pred_centred: np.ndarray, true_centred: np.ndarray,
        pred_dynamic: np.ndarray, true_dynamic: np.ndarray,
        sample_accumulator: _VectorPairAccumulator,
        reservoir: "_Reservoir | None",
    ) -> None:
        group.model_error.update(model_err[valid])
        group.prior_error.update(prior_err[valid])
        px, ty = pred_centred[valid], true_centred[valid]
        group.dynamic_pair.update(px, ty)
        if reservoir is not None:
            reservoir.add(px, ty)
        sample_accumulator.update(pred_dynamic, true_dynamic, valid)

    def _finalize_group(self, group: _GroupAccumulator, sample_accumulator: _VectorPairAccumulator,
                         spearman: float | None) -> dict[str, object]:
        model_mse = group.model_error.mse()
        prior_mse = group.prior_error.mse()
        p = group.dynamic_pair
        dynamic_baseline_mse = p.syy / p.n if p.n else float("nan")
        dynamic_mse = (p.sxx - 2 * p.sxy + p.syy) / p.n if p.n else float("nan")
        return {
            "mse": model_mse,
            "prior_mse": prior_mse,
            "skill_vs_prior": _safe_skill(model_mse, prior_mse),
            "dynamic_skill": _safe_skill(dynamic_mse, dynamic_baseline_mse),
            "dynamic_pearson": p.pearson(),
            "dynamic_spearman": spearman if spearman is not None else float("nan"),
            "dynamic_calibration_alpha": p.calibration_alpha(),
            "dynamic_amplitude_ratio": p.amplitude_ratio(),
            "patient_dynamic_pearson_median": sample_accumulator.median_pearson(),
        }

    def finalize(self) -> dict[str, object]:
        locus_pearson = np.asarray(self._locus_pearson, dtype=np.float64)
        locus_spearman = np.asarray(self._locus_spearman, dtype=np.float64)
        locus_tertile = np.asarray(self._locus_tertile, dtype=object)

        def _median_finite(values: np.ndarray) -> float:
            finite = values[np.isfinite(values)]
            return float(np.median(finite)) if finite.size else float("nan")

        rx, ry = self._dynamic_spearman_reservoir.arrays()
        wx, wy = self._within_cancer_spearman.arrays()

        result: dict[str, object] = {
            "rows": int(self._sample_valid_count.sum()),
            "samples": self.n_samples,
            "cpgs": len(self._cpg_win_model),
            **self._finalize_group(self._all, self._sample_all, _spearman(rx, ry) if rx.size else None),
            "mae": self._all.model_error.mae(),
            "locus_dynamic_pearson_median": _median_finite(locus_pearson) if self.include_locus_median_correlations else None,
            "locus_dynamic_spearman_median": _median_finite(locus_spearman) if self.include_locus_median_correlations else None,
            "patient_dynamic_spearman_median": None,  # not computed for streaming panels (see module docstring)
            "within_cancer_skill": _safe_skill(
                (self._within_cancer.sxx - 2 * self._within_cancer.sxy + self._within_cancer.syy)
                / self._within_cancer.n if self._within_cancer.n else float("nan"),
                self._within_cancer.syy / self._within_cancer.n if self._within_cancer.n else float("nan"),
            ),
            "within_cancer_pearson": self._within_cancer.pearson(),
            "within_cancer_spearman": _spearman(wx, wy) if wx.size else float("nan"),
            "sample_win_fraction": float(np.nanmean(
                np.where(self._sample_valid_count > 0, self._sample_model_sq < self._sample_prior_sq, np.nan)
            )),
            "cpg_win_fraction": float(np.nanmean(
                np.asarray(self._cpg_win_model) < np.asarray(self._cpg_win_prior)
            )) if self._cpg_win_model else float("nan"),
        }

        per_cancer = {}
        cancer_mses, cancer_skills = [], []
        for name in self.cancer_type_names:
            model_mse = self._per_cancer[name].mse()
            prior_mse = self._per_cancer_prior[name].mse()
            skill = _safe_skill(model_mse, prior_mse)
            per_cancer[name] = {
                "observations": int(self._per_cancer[name].n),
                "mse": model_mse,
                "prior_mse": prior_mse,
                "skill_vs_prior": skill,
            }
            if np.isfinite(model_mse):
                cancer_mses.append(model_mse)
            if np.isfinite(skill):
                cancer_skills.append(skill)
        result["per_cancer"] = per_cancer
        result["macro_cancer_mse"] = float(np.mean(cancer_mses)) if cancer_mses else float("nan")
        result["macro_cancer_skill_vs_prior"] = float(np.mean(cancer_skills)) if cancer_skills else float("nan")

        per_chromosome = {}
        for key in sorted(self._per_chromosome):
            model_mse = self._per_chromosome[key].mse()
            prior_mse = self._per_chromosome_prior[key].mse()
            per_chromosome[key] = {
                "observations": int(self._per_chromosome[key].n),
                "mse": model_mse,
                "prior_mse": prior_mse,
                "skill_vs_prior": _safe_skill(model_mse, prior_mse),
            }
        result["per_chromosome"] = per_chromosome

        per_tertile = {}
        for name, group in self._tertiles.items():
            cols = locus_tertile == name if self.include_locus_median_correlations else None
            if group.model_error.n == 0:
                continue
            block = self._finalize_group(group, self._sample_tertile[name], spearman=None)
            block.pop("dynamic_spearman", None)  # not computed per-tertile in the streaming path
            block["locus_dynamic_pearson_median"] = (
                _median_finite(locus_pearson[cols]) if cols is not None else None
            )
            block["rows"] = int(group.model_error.n)
            per_tertile[name] = block
        result["per_variability_tertile"] = per_tertile

        return result
