"""Validated, serializable experiment settings."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class StudyConfig:
    """Fix the split, tuning budget and diagnostic settings before evaluation."""

    profile: str = "full"
    dataset: str = "data/raw/pollution_dataset.csv"
    output_dir: str = "reports"
    expected_sha256: str | None = None
    split_seed: int = 42
    cv_seed: int = 19
    model_seed: int = 42
    bootstrap_seed: int = 2026
    test_size: float = 0.2
    outer_splits: int = 5
    inner_splits: int = 3
    search_iterations: int = 8
    bootstrap_iterations: int = 2000
    permutation_repeats: int = 10
    n_jobs: int = 1
    sample_size: int | None = None
    execute_notebook: bool = True

    def __post_init__(self) -> None:
        """Reject settings that silently compromise stratification or uncertainty estimates."""
        if self.profile not in {"full", "smoke"}:
            raise ValueError("Profile must be 'full' or 'smoke'.")
        if not 0.05 <= self.test_size <= 0.4:
            raise ValueError("test_size must be between 0.05 and 0.4.")
        for name in ("outer_splits", "inner_splits"):
            if getattr(self, name) < 2:
                raise ValueError(f"{name} must be at least 2.")
        for name in ("search_iterations", "bootstrap_iterations", "permutation_repeats"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive.")
        if self.n_jobs not in (1, 2, 3, 4):
            raise ValueError("Use 1–4 parallel search workers to avoid oversubscription.")
        if self.sample_size is not None and self.sample_size < 400:
            raise ValueError("sample_size must be at least 400 for fold-local SMOTE.")
        if self.profile == "full" and self.sample_size is not None:
            raise ValueError("The full study must use the entire fixed data snapshot.")

    def to_dict(self) -> dict:
        """Return JSON-compatible settings for the run manifest."""
        return asdict(self)


def load_config(path: str | Path) -> StudyConfig:
    """Read settings and reject unknown keys through the dataclass constructor."""
    return StudyConfig(**json.loads(Path(path).read_text(encoding="utf-8")))
