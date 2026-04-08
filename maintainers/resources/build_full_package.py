#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from maintainers.resources.normalizers import (
    apply_rename_only_mappings,
    normalize_resource,
)
from maintainers.resources.validate_full_package import (
    DEFAULT_CONTRACT_PATH,
    load_contract,
    validate_staging_tree,
)


def _resolve_source_tree(source_root: Path, archive_root: str) -> Path:
    if source_root.name == archive_root and source_root.is_dir():
        return source_root

    candidate = source_root / archive_root
    if candidate.is_dir():
        return candidate

    raise ValueError(
        f"source_root must be either {archive_root}/ or contain {archive_root}/ as a child: {source_root}"
    )


def _apply_rename_only_work(
    *,
    staging_root: Path,
    contract: Mapping[str, Any],
) -> list[str]:
    entries = contract.get("resources", {}).get("rename_only", [])
    return apply_rename_only_mappings(staging_root=staging_root, entries=entries)


def _apply_resource_normalizer_work(
    *,
    staging_root: Path,
    contract: Mapping[str, Any],
) -> list[str]:
    errors: list[str] = []
    for entry in contract.get("resources", {}).get("normalized", []):
        outputs = [staging_root / str(item) for item in entry.get("canonical_outputs", [])]
        resource_id = str(entry["resource_id"])
        source_path = staging_root / str(entry["source_path"])
        if outputs and all(path.exists() for path in outputs):
            continue
        if not source_path.exists():
            continue

        try:
            normalize_resource(
                resource_id=resource_id,
                source_path=source_path,
                canonical_outputs=outputs,
            )
        except (KeyError, ValueError) as exc:
            errors.append(
                f"normalized:{resource_id}: {exc}"
            )
    return errors


def _build_tarball(*, staging_root: Path, output_dir: Path, archive_name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / archive_name
    with tarfile.open(archive_path, mode="w:gz") as handle:
        handle.add(staging_root, arcname=staging_root.name)
    return archive_path


def build_full_package(
    *,
    source_root: Path,
    output_dir: Path,
    contract_path: Path | None = None,
) -> Path:
    contract = load_contract(contract_path=contract_path)
    archive_root = str(contract["archive_root"])
    archive_name = str(contract["archive"]["filename"])

    source_tree = _resolve_source_tree(source_root=source_root, archive_root=archive_root)

    with tempfile.TemporaryDirectory(prefix="resources_metadata_full_") as temp_dir:
        staging_root = Path(temp_dir) / archive_root
        shutil.copytree(source_tree, staging_root)

        errors: list[str] = []
        errors.extend(_apply_rename_only_work(staging_root=staging_root, contract=contract))
        errors.extend(
            _apply_resource_normalizer_work(staging_root=staging_root, contract=contract)
        )
        errors.extend(validate_staging_tree(staging_root, contract=contract))

        if errors:
            message = "\n".join(f"- {error}" for error in errors)
            raise ValueError(f"contract validation failed:\n{message}")

        return _build_tarball(
            staging_root=staging_root,
            output_dir=output_dir,
            archive_name=archive_name,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build resources_metadata_full.tar.gz from a canonical staging source.",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help=(
            "Path to canonical resources_metadata tree, or a parent directory containing "
            "resources_metadata/."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where the resulting tar.gz will be written.",
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=DEFAULT_CONTRACT_PATH,
        help="Path to full_package_contract.json.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)

    try:
        archive_path = build_full_package(
            source_root=args.source_root,
            output_dir=args.output_dir,
            contract_path=args.contract,
        )
    except ValueError as exc:
        print(f"FAILED\n{exc}")
        return 1

    print(archive_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
