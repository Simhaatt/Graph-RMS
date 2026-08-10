# Supplied workspace inventory summary

`WORKSPACE_INVENTORY.csv` records every file discovered during the release
audit using paths relative to the supplied workspace. It deliberately contains
no absolute user profile path. The scan classified 12,045 files:

| Category | Files | Release decision |
|---|---:|---|
| Included release | 189 | curated code, outputs, metadata, figures, and documentation |
| Supplied code or output | 2,030 | reviewed through the frozen registry and evidence hierarchy; superseded/exploratory items excluded |
| Historical archive | 53 | excluded after extracting authoritative evidence |
| Historical notebook | 62 | excluded from canonical entry points; development transport artifacts |
| Large excluded | 31 | excluded for size, duplication, or third-party-data reasons |
| Third-party or raw data | 277 | excluded; original provider terms apply |
| Temporary or build product | 8,188 | excluded caches, dependencies, logs, and build outputs |
| Other excluded | 1,215 | not required for the curated reproducibility release |

The high temporary/build count is dominated by local dependency trees and
compiled caches. The inventory is an audit trail, not a recommendation to
publish every supplied artifact. `docs/LARGE_FILES.md` gives the main large
exclusions, while `results/provenance/release_file_manifest.csv` provides
SHA-256 checksums for the curated release itself.

