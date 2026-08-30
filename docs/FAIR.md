# FAIR and open-science record

## Findable

- `CITATION.cff` provides software citation metadata.
- `.zenodo.json` prepares the software archive metadata.
- `docs/RESULTS_MAP.md` maps every manuscript table and figure to a concrete
  repository artifact.
- The published release tag is `v1.0.0`.

The repository URL, Simhaa T. T.'s verified ORCID, version DOI
`10.5281/zenodo.21876448`, and concept DOI `10.5281/zenodo.21876447` are
recorded from the published records. No funding identifier is invented.

## Accessible

The repository includes compact derived outputs and excludes raw third-party
hyperspectral cubes. `data/README.md` documents exact expected filenames and
the acquisition endpoints used by the loader. Users must accept the original
provider terms.

## Interoperable

- Configurations: YAML.
- Tabular results: CSV.
- Structured records and locks: JSON.
- Partitions: NumPy NPY arrays with documented scene shapes.
- Figures: PNG.
- Code: Python 3.12.

Dataset identifiers are stable across configs, results, and scripts.

## Reusable

The package includes frozen settings, source implementations, per-scene
partitions, raw diagnostic rows, environment capture, checksums, automated
audits, and explicit limitations. The authors approved the Creative Commons
Attribution 4.0 International licence (CC BY 4.0) for this package, covering
the software, documentation, and curated results. Third-party datasets remain
under their original provider terms, are not redistributed, and are not
covered by that grant.

## Remaining manual FAIR tasks

1. verify original-provider licences and preferred citations for all datasets;
2. archive the final manuscript-linked release checksum and DOI citation;
3. update the repository metadata only when a later verified release exists.
