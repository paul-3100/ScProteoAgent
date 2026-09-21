"""Portable local/release data-root resolution for the V3 benchmark."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
DATA_ROOT_ENV = "SCPROTEOMICS_DATA_ROOT"


def _is_usable_data_root(path: Path) -> bool:
    if not path.is_dir():
        return False
    return any(
        child.is_dir()
        and (child / "ProteinQuant.csv").is_file()
        and (child / "SampleInfo.csv").is_file()
        and (child / "user_input.txt").is_file()
        for child in path.iterdir()
    )


def resolve_examples_root(cli_value: str | Path | None = None, project_dir: Path = PROJECT_DIR) -> Path:
    """Resolve CLI > environment > packaged examples > workspace canonical data."""
    candidates: list[tuple[str, Path]] = []
    if cli_value:
        candidates.append(("cli", Path(cli_value).expanduser()))
    env_value = os.getenv(DATA_ROOT_ENV, "").strip()
    if env_value:
        candidates.append(("environment", Path(env_value).expanduser()))
    candidates.append(("packaged", project_dir / "examples" / "V3"))
    candidates.append(("workspace_canonical", project_dir.parents[1] / "ScProteomics_DataSets" / "V3"))
    for _, candidate in candidates:
        resolved = candidate.resolve()
        if _is_usable_data_root(resolved):
            return resolved
    checked = "; ".join(f"{source}={path}" for source, path in candidates)
    raise FileNotFoundError(
        f"No usable V3 data root was found. Checked {checked}. "
        f"Pass --examples-root or set {DATA_ROOT_ENV}."
    )


def data_root_resolution_note() -> str:
    return (
        f"Resolution order: --examples-root > {DATA_ROOT_ENV} > packaged examples/V3 "
        "> workspace ScProteomics_DataSets/V3."
    )
