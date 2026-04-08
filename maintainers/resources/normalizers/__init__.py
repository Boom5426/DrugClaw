from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

from .dili import normalize_dili
from .drugcomb import normalize_drugcomb
from .drugcombdb import normalize_drugcombdb
from .gdkd import normalize_gdkd
from .livertox import normalize_livertox
from .rename_only import apply_rename_only_mappings
from .repurposedrugs import normalize_repurposedrugs
from .tarkg import normalize_tarkg

ResourceNormalizer = Callable[[Path, Path], None]

NORMALIZER_BY_RESOURCE_ID: dict[str, ResourceNormalizer] = {
    "gdkd": normalize_gdkd,
    "tarkg": normalize_tarkg,
    "repurposedrugs": normalize_repurposedrugs,
    "drugcombdb": normalize_drugcombdb,
    "drugcomb": normalize_drugcomb,
    "livertox": normalize_livertox,
    "dili": normalize_dili,
}


def normalize_resource(
    *,
    resource_id: str,
    source_path: Path,
    canonical_outputs: Sequence[Path],
) -> None:
    normalizer = NORMALIZER_BY_RESOURCE_ID.get(resource_id)
    if normalizer is None:
        raise KeyError(f"no normalizer registered for resource_id={resource_id!r}")
    if len(canonical_outputs) != 1:
        raise ValueError(
            f"{resource_id}: expected exactly one canonical output, got {len(canonical_outputs)}"
        )
    normalizer(source_path, canonical_outputs[0])


__all__ = [
    "apply_rename_only_mappings",
    "normalize_resource",
]
