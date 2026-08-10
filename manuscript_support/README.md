# Manuscript support files

- `manuscript_supplied.tex` is the audited LaTeX source. With author approval,
  its duplicate Data Availability section and fictitious DOI were replaced by
  one release-safe statement naming the GitHub repository; the verified Zenodo
  DOI was inserted after verifying the published Zenodo record.
- `references.bib` is the author-supplied bibliography. A byte-identical copy
  is also stored at the repository root so `\bibliography{references}` resolves
  when LaTeX is run from the repository root.
- `DATA_CODE_AVAILABILITY.tex` is a deduplicated replacement fragment with
  explicit non-public placeholders. Replace them only with verified records.
- `RELEASE_NOTES_v1.0.0.md` is the proposed GitHub release text.

The publication workflow image is stored at `figures/graph.pdf`, with
`figures/graph.png` retained as a preview. The asset and bibliography checks
pass. The source cites the verified `v1.0.0` archive DOI,
`10.5281/zenodo.21876448`.
