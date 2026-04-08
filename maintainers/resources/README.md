# Full Package Maintainer Notes

This directory defines the contract for producing `resources_metadata_full.tar.gz`.
The runtime stays single-root: all packaged paths must resolve under `resources_metadata/`.

## Raw Inputs

Use raw source resources from maintainers' import mirrors (for example Hugging Face snapshots, upstream downloads, or local rebuild artifacts). Do not copy raw imports directly into the repository tree.

## Staging and Validation Intent

Build a staging tree that already matches the runtime contract:

- root directory name is exactly `resources_metadata`
- direct resources are copied to canonical paths as-is
- rename-only resources are remapped from source path to canonical path
- normalized resources are transformed first, then written to their canonical output files

Validate staging against `full_package_contract.json` before archiving.
For `drug_review` resources, keep canonical target naming aligned with runtime usage
in skills/examples (for example `drug_review/WebMDDrugReviews/webmd.csv`).

## Final Archive Guarantees

The output tarball must guarantee:

- top-level archive root is `resources_metadata/`
- extraction into a repository root overlays only `resources_metadata/`
- no second runtime root (for example `resources_metadata_full/`) is introduced
- required canonical outputs for normalized resources are present
