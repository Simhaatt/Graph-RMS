# GitHub and Zenodo release checklist

This checklist was prepared from the current official documentation on
10 August 2026. GitHub release `v1.0.0` and its Zenodo archive have now been
published.

Official references:

- GitHub: https://docs.github.com/en/repositories/releasing-projects-on-github
- Zenodo GitHub integration: https://help.zenodo.org/docs/github/
- Enable a Zenodo-connected repository:
  https://help.zenodo.org/docs/github/enable-repository/
- Zenodo API and deposition model: https://developers.zenodo.org/

## 1. Resolve blockers before creating a public repository

- [x] Authors approve the CC BY 4.0 licence for the package.
- [x] Authors decided that the code, documentation, and results are all
      released under CC BY 4.0.
- [x] Replace the `LICENSE` file with the approved CC BY 4.0 text.
- [x] Add the approved SPDX licence identifier to `CITATION.cff` and
      `.zenodo.json`.
- [x] Preserve the publication-ready `figures/graph.pdf` workflow figure and PNG preview.
- [x] Add and validate the manuscript `references.bib`.
- [x] Remove the duplicate Data Availability section and fictitious DOI from
      the manuscript.
- [x] Replace the fictitious DOI only after the real Zenodo DOI was registered.
- [ ] Verify the original provider, preferred citation, and terms for all nine
      third-party datasets.
- [ ] Run `python scripts/audit_results.py` and `pytest -q`.
- [ ] Review the archive produced by `python scripts/package_release.py`.

## 2. Create the GitHub repository

1. Create an empty repository named `Graph-RMS` under the author-approved
   owner or organization.
2. Commit the contents of this directory, not the surrounding research
   workspace.
3. Confirm that raw `.mat`, `.bsq`, `.hdr`, ZIP, cache, and log files are not
   tracked.
4. Add the real repository URL to `CITATION.cff` only after it exists.
5. Run the audit from a clean clone on Linux or Colab if practical.

Suggested local commands, to be run only after repository ownership is agreed:

```bash
git init
git add .
git status
git commit -m "Prepare Graph-RMS reproducibility release"
git branch -M main
git remote add origin <REAL_GITHUB_URL>
git push -u origin main
```

## 3. Enable Zenodo before the release

Zenodo's current GitHub integration requires a connected GitHub account, a
repository sync, and enabling the selected repository. Once connected, new
GitHub releases are automatically ingested and archived.

1. Sign in to Zenodo using the author-approved account.
2. Connect the GitHub account if it is not already connected.
3. Open the Zenodo GitHub page and choose **Sync now**.
4. Find the real Graph-RMS repository and enable its integration toggle.
5. Validate `.zenodo.json` as JSON and review creators/title/description.

Do not insert guessed ORCIDs, grants, or a manuscript DOI.

## 4. Create GitHub release v1.0.0

GitHub releases are based on tags and GitHub automatically provides source ZIP
and tar archives for the tagged commit. Use:

- Tag: `v1.0.0`
- Title: `Graph-RMS v1.0.0 — reproducibility package for training-free and class-count-free HSI region discovery`
- Notes: start from `manuscript_support/RELEASE_NOTES_v1.0.0.md`.

Before pressing **Publish release**, verify the tag points to the audited
commit and that Zenodo integration is enabled. Publishing is an external,
irreversible-enough action and remains an author task.

## 5. Verify the Zenodo draft or record

After GitHub release publication:

1. Confirm Zenodo ingested the correct tag and files.
2. Check title, creator order, affiliations, description, keywords, resource
   type, version, and approved licence.
3. Confirm raw third-party datasets are absent.
4. Record the release archive checksum.
5. If Zenodo presents a draft, verify it before publication. Zenodo states that
   a published deposition can no longer be deleted; do not publish casually.
6. Copy the real version DOI and concept DOI only after Zenodo supplies them.

The Zenodo API exposes a pre-reserved DOI in draft deposition metadata, but a
test or sandbox DOI is not a publication DOI. The sandbox uses the 10.5072
prefix, whereas normal Zenodo records use 10.5281. Never place a sandbox DOI in
the manuscript.

## 6. Final metadata update

After a real DOI exists:

- update `CITATION.cff` with `repository-code`, `doi`, and approved `license`;
- update `.zenodo.json` only with verified identifiers;
- replace the manuscript availability placeholder with the real repository
  and DOI;
- make a documented metadata-only commit if needed;
- preserve the DOI-bearing released archive and checksum.

## 7. Final readiness decision

The scientific evidence package, public GitHub release, and Zenodo archive are
available. The verified version DOI is `10.5281/zenodo.21876448` and the
all-versions concept DOI is `10.5281/zenodo.21876447`.
