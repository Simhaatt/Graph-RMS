# FAIR and open-science record

## Findable

- `CITATION.cff` provides software citation metadata.
- `.zenodo.json` prepares the software archive metadata.
- `docs/RESULTS_MAP.md` maps every manuscript table and figure to a concrete
  repository artifact.
- The planned release tag is `v1.0.0`.

The planned repository URL and Simhaa T. T.'s verified ORCID are recorded.
No Zenodo DOI or funding identifier is invented; add those only after the
corresponding public records exist and have been verified.

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
audits, and explicit limitations. The authors approved the MIT License for the
software. Third-party datasets remain under their original provider terms and
are not redistributed. No separate Creative Commons grant is asserted for
third-party materials.

## Remaining manual FAIR tasks

1. verify original-provider licences and preferred citations for all datasets;
2. create the public GitHub repository and `v1.0.0` release;
3. connect the release to Zenodo and publish the DOI;
4. insert the verified DOI into the final manuscript and metadata;
5. archive the final manuscript-linked release checksum.
