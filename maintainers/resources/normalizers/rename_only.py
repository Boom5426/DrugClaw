from __future__ import annotations

import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path


def _resolve_source_file(source_path: Path, *, preferred_name: str) -> Path:
    if source_path.is_file():
        return source_path
    if not source_path.is_dir():
        raise ValueError(f"rename-only source is neither file nor directory: {source_path}")

    preferred = source_path / preferred_name
    if preferred.is_file():
        return preferred

    files = sorted(path for path in source_path.rglob("*") if path.is_file())
    if len(files) == 1:
        return files[0]
    raise ValueError(
        f"unable to infer file mapping for {source_path}: expected one file or {preferred_name}"
    )


def _apply_single_mapping(
    *,
    staging_root: Path,
    resource_id: str,
    source_relative: str,
    canonical_relative: str,
) -> None:
    source_path = staging_root / source_relative
    canonical_path = staging_root / canonical_relative

    if canonical_path.exists() or not source_path.exists():
        return

    canonical_path.parent.mkdir(parents=True, exist_ok=True)
    canonical_is_file = canonical_path.suffix != ""

    if canonical_is_file:
        source_file = _resolve_source_file(source_path, preferred_name=canonical_path.name)
        shutil.copy2(source_file, canonical_path)
        return

    if not source_path.is_dir():
        raise ValueError(
            f"{resource_id}: expected directory source for rename-only mapping at {source_relative}"
        )
    shutil.copytree(source_path, canonical_path, dirs_exist_ok=True)


def apply_rename_only_mappings(
    *,
    staging_root: Path,
    entries: Sequence[Mapping[str, object]],
) -> list[str]:
    errors: list[str] = []
    for entry in entries:
        resource_id = str(entry["resource_id"])
        source_relative = str(entry["source_path"])
        canonical_relative = str(entry["canonical_path"])
        try:
            _apply_single_mapping(
                staging_root=staging_root,
                resource_id=resource_id,
                source_relative=source_relative,
                canonical_relative=canonical_relative,
            )
        except ValueError as exc:
            errors.append(f"{resource_id}: {exc}")
    return errors
