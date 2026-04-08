#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import tarfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

DEFAULT_CONTRACT_PATH = Path(__file__).with_name("full_package_contract.json")


@dataclass(frozen=True)
class RequiredPath:
    relative_path: str
    kind: str
    expect_file: bool


def load_contract(contract_path: Path | None = None) -> dict[str, Any]:
    path = contract_path or DEFAULT_CONTRACT_PATH
    return json.loads(path.read_text(encoding="utf-8"))


def _is_valid_relative_contract_path(relative_path: str) -> bool:
    if not relative_path:
        return False
    path = PurePosixPath(relative_path)
    if path.is_absolute():
        return False
    if path.parts and path.parts[0] == "resources_metadata":
        return False
    return ".." not in path.parts


def _canonicalize_member_name(member_name: str) -> str:
    normalized = member_name
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return str(PurePosixPath(normalized))


def _iter_required_paths(contract: Mapping[str, Any]) -> list[RequiredPath]:
    resources = contract.get("resources", {})
    required: list[RequiredPath] = []

    for entry in resources.get("direct", []):
        required.append(
            RequiredPath(
                relative_path=str(entry["canonical_path"]),
                kind=f"direct:{entry['resource_id']}",
                expect_file=False,
            )
        )

    for entry in resources.get("rename_only", []):
        relative = str(entry["canonical_path"])
        required.append(
            RequiredPath(
                relative_path=relative,
                kind=f"rename_only:{entry['resource_id']}",
                expect_file=PurePosixPath(relative).suffix != "",
            )
        )

    for entry in resources.get("normalized", []):
        for output in entry.get("canonical_outputs", []):
            required.append(
                RequiredPath(
                    relative_path=str(output),
                    kind=f"normalized:{entry['resource_id']}",
                    expect_file=True,
                )
            )

    return required


def validate_staging_tree(
    staging_root: Path,
    *,
    contract: Mapping[str, Any] | None = None,
    contract_path: Path | None = None,
) -> list[str]:
    payload = contract if contract is not None else load_contract(contract_path=contract_path)
    archive_root = str(payload.get("archive_root", "resources_metadata"))
    errors: list[str] = []

    if not staging_root.exists():
        return [f"staging root does not exist: {staging_root}"]
    if not staging_root.is_dir():
        return [f"staging root is not a directory: {staging_root}"]
    if staging_root.name != archive_root:
        errors.append(
            f"staging root must be named {archive_root}, got {staging_root.name!r}"
        )

    forbidden_roots = payload.get("canonical_root_rules", {}).get(
        "forbidden_runtime_roots", []
    )
    for forbidden in forbidden_roots:
        if (staging_root / Path(forbidden)).exists():
            errors.append(f"forbidden runtime root exists in staging tree: {forbidden}")

    for requirement in _iter_required_paths(payload):
        if not _is_valid_relative_contract_path(requirement.relative_path):
            errors.append(
                f"invalid contract path for {requirement.kind}: {requirement.relative_path!r}"
            )
            continue
        target = staging_root / Path(requirement.relative_path)
        if not target.exists():
            errors.append(
                f"missing required {requirement.kind} path: {requirement.relative_path}"
            )
            continue
        if requirement.expect_file and not target.is_file():
            errors.append(
                f"required {requirement.kind} output must be a file: {requirement.relative_path}"
            )

    return errors


def validate_archive(
    archive_path: Path,
    *,
    contract: Mapping[str, Any] | None = None,
    contract_path: Path | None = None,
) -> list[str]:
    payload = contract if contract is not None else load_contract(contract_path=contract_path)
    archive_root = str(payload.get("archive_root", "resources_metadata"))
    errors: list[str] = []

    if not archive_path.exists():
        return [f"archive does not exist: {archive_path}"]
    if not archive_path.is_file():
        return [f"archive path is not a file: {archive_path}"]

    try:
        with tarfile.open(archive_path, mode="r:*") as handle:
            member_names = [
                _canonicalize_member_name(member.name)
                for member in handle.getmembers()
                if _canonicalize_member_name(member.name) not in ("", ".")
            ]
    except tarfile.TarError as exc:
        return [f"failed to read archive {archive_path}: {exc}"]

    if not member_names:
        return [f"archive is empty: {archive_path}"]

    top_level_roots = {name.split("/", 1)[0] for name in member_names}
    if top_level_roots != {archive_root}:
        errors.append(
            f"archive top-level root must be {archive_root}; found {sorted(top_level_roots)}"
        )

    archive_root_posix = str(PurePosixPath(archive_root))
    archive_root_prefix = f"{archive_root_posix}/"
    member_paths_under_archive_root = [
        name[len(archive_root_prefix) :]
        for name in member_names
        if name.startswith(archive_root_prefix)
    ]

    forbidden_roots = payload.get("canonical_root_rules", {}).get(
        "forbidden_runtime_roots", []
    )
    for forbidden in forbidden_roots:
        forbidden_posix = str(PurePosixPath(forbidden))
        if forbidden_posix in top_level_roots:
            errors.append(f"archive includes forbidden runtime root: {forbidden}")
        if any(
            name == forbidden_posix or name.startswith(f"{forbidden_posix}/")
            for name in member_names
        ):
            errors.append(f"archive includes forbidden path segment: {forbidden}")
        if any(
            rel_path == forbidden_posix
            or rel_path.startswith(f"{forbidden_posix}/")
            for rel_path in member_paths_under_archive_root
        ):
            errors.append(
                f"archive includes forbidden runtime root nested under {archive_root}: {forbidden}"
            )

    member_set = set(member_names)
    for requirement in _iter_required_paths(payload):
        if not _is_valid_relative_contract_path(requirement.relative_path):
            errors.append(
                f"invalid contract path for {requirement.kind}: {requirement.relative_path!r}"
            )
            continue
        expected = str(PurePosixPath(archive_root) / PurePosixPath(requirement.relative_path))
        if requirement.expect_file:
            if expected not in member_set:
                errors.append(
                    f"archive missing required {requirement.kind} file: {requirement.relative_path}"
                )
            continue
        if not any(name == expected or name.startswith(f"{expected}/") for name in member_set):
            errors.append(
                f"archive missing required {requirement.kind} path: {requirement.relative_path}"
            )

    return errors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate a full-package staging tree or archive against the contract.",
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=DEFAULT_CONTRACT_PATH,
        help="Path to full_package_contract.json.",
    )
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        "--staging-root",
        type=Path,
        help="Path to a staging tree whose directory name should be resources_metadata.",
    )
    target_group.add_argument(
        "--archive",
        type=Path,
        help="Path to resources_metadata_full.tar.gz.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    contract_path: Path = args.contract

    if args.staging_root is not None:
        errors = validate_staging_tree(args.staging_root, contract_path=contract_path)
    else:
        errors = validate_archive(args.archive, contract_path=contract_path)

    if errors:
        print("FAILED")
        for error in errors:
            print(f"- {error}")
        return 1

    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
