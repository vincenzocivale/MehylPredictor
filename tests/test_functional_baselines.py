from __future__ import annotations

from pathlib import Path

import torch

from methylation_predictor.config import EncoderConfig, ModelConfig
from methylation_predictor.modeling import FunctionalBaselinePredictor
from methylation_predictor.modeling.factory import FINAL_VARIANT
from methylation_predictor.rna_training.config import load_rna_recipe


ROOT = Path(__file__).resolve().parents[1]


def _config():
    return ModelConfig(
        encoder=EncoderConfig(
            kind="locus_attention",
            latent_dim=8,
            n_programs=6,
            program_dim=256,
            n_heads=4,
            dropout=0.0,
        )
    )


def _inputs():
    return {
        "functional_track_indices": torch.tensor(
            [1, 2, 5, 9], dtype=torch.int64
        ),
        "functional_offsets": torch.tensor(
            [0, 2, 2, 3, 4], dtype=torch.int64
        ),
        "functional_dense": torch.randn(4, 23),
    }


def _baseline(variant):
    return FunctionalBaselinePredictor(
        48,
        _config(),
        variant=variant,
        final_regressor_dropout=0.15,
        use_mean_proxy=True,
    )


def test_recipes_differ_from_main_only_by_selector_and_tracking():
    main = load_rna_recipe(ROOT / "configs/models/main.yaml").raw
    variants = {
        "functional_global_rna_shift.yaml":
            "functional_baseline_global_shift",
        "functional_mlp_rna_cpg.yaml":
            "functional_baseline_mlp",
        "functional_bilinear_rna_cpg.yaml":
            "functional_baseline_bilinear",
    }

    for filename, selector in variants.items():
        raw = load_rna_recipe(
            ROOT / "configs/models/baselines" / filename
        ).raw
        lhs = dict(raw)
        rhs = dict(main)
        lhs.pop("tracking", None)
        rhs.pop("tracking", None)

        lhs_model = dict(lhs["model"])
        rhs_model = dict(rhs["model"])
        assert lhs_model.pop("functional_fusion_variant") == selector
        assert rhs_model.pop("functional_fusion_variant") == FINAL_VARIANT
        assert lhs_model == rhs_model

        lhs["model"] = lhs_model
        rhs["model"] = rhs_model
        assert lhs == rhs


def test_global_shift_has_no_patient_by_locus_interaction():
    torch.manual_seed(5)
    model = _baseline("functional_baseline_global_shift").eval()
    with torch.no_grad():
        logit = model(
            torch.randn(3, 48), None, **_inputs()
        )["prediction_logit"]
    delta = logit[1] - logit[0]
    torch.testing.assert_close(
        delta, delta[0].expand_as(delta), rtol=0, atol=1e-6
    )


def test_mlp_and_bilinear_depend_on_rna_and_locus():
    inputs = _inputs()
    for variant in (
        "functional_baseline_mlp",
        "functional_baseline_bilinear",
    ):
        torch.manual_seed(9)
        model = _baseline(variant).eval()
        rna = torch.randn(2, 48)
        with torch.no_grad():
            base = model(rna, None, **inputs)["beta"]
            perturbed_rna = rna.clone()
            perturbed_rna[:, 0] += 0.5
            changed_rna = model(
                perturbed_rna, None, **inputs
            )["beta"]

            changed = dict(inputs)
            dense = inputs["functional_dense"].clone()
            dense[:, 0] += 0.5
            changed["functional_dense"] = dense
            changed_locus = model(
                rna, None, **changed
            )["beta"]

        assert not torch.equal(base, changed_rna)
        assert not torch.equal(base, changed_locus)
