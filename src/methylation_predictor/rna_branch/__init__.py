"""RNA-conditioned residual branch for methylation prediction."""

from .config import RunConfig, load_config

__all__ = ["RunConfig", "ResidualMethylationModel", "load_config"]


def __getattr__(name: str):
    """Avoid importing PyTorch for data-only preprocessing entry points."""
    if name == "ResidualMethylationModel":
        from .models import ResidualMethylationModel

        return ResidualMethylationModel
    raise AttributeError(name)
