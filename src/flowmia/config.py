"""Typed configuration for FlowMIA, loadable from a YAML file."""

from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml


@dataclass
class FlowMIAConfig:
    """Typed configuration mirroring the keys accepted by ``FlowMIA(config=...)``.

    Args:
        member_path: Path to the training (member) CSV.
        non_member_path: Path to the held-out (non-member) CSV.
        synth_path: Path to the synthetic data CSV.
        test_path: Path to the utility test CSV.
        save_path: Directory where outputs and checkpoints are saved.
        categorical_cols: Names of categorical feature columns.
        numerical_cols: Names of numerical feature columns.
        ip_cols: Names of IP address columns.
        label_col: Name of the label/target column.
        use_wgan: Whether to use WGAN-GP instead of standard GAN.
        batch_size: Batch size for GAN training.
        num_epochs: Number of GAN training epochs.
        fcheckpoint: Checkpoint save frequency (in epochs).
    """

    member_path: str
    non_member_path: str
    synth_path: str
    test_path: str
    save_path: str
    categorical_cols: list = field(default_factory=list)
    numerical_cols: list = field(default_factory=list)
    ip_cols: list = field(default_factory=list)
    label_col: str = "label"
    use_wgan: bool = True
    batch_size: int = 128
    num_epochs: int = 100
    fcheckpoint: int = 10

    def __post_init__(self):
        required_str_fields = [
            "member_path", "non_member_path", "synth_path", "test_path",
            "save_path", "label_col",
        ]
        for f in required_str_fields:
            if not getattr(self, f):
                raise ValueError(f"FlowMIAConfig.{f} must be a non-empty string")
        if self.batch_size <= 0 or self.num_epochs <= 0 or self.fcheckpoint <= 0:
            raise ValueError("batch_size, num_epochs, and fcheckpoint must be positive")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "FlowMIAConfig":
        known = set(cls.__dataclass_fields__)
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**d)


def load_config(path) -> FlowMIAConfig:
    """Load a FlowMIAConfig from a YAML file.

    Args:
        path: Path to a YAML config file.

    Returns:
        A validated FlowMIAConfig.
    """
    with open(Path(path), "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return FlowMIAConfig.from_dict(raw)
