from methylation_predictor.rna_branch.config import TrackingConfig
from methylation_predictor.rna_branch.tracking import Tracker


def test_tracking_defaults_are_dependency_free() -> None:
    config = TrackingConfig()
    assert config.backend == "none"
    tracker = Tracker()
    tracker.log({"loss": 1.0}, step=1)
    tracker.set_summary({"best_epoch": 2})
    tracker.finish()
    assert tracker.enabled is False
