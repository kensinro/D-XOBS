# GitHub release checklist

- [ ] Replace `REPLACE_WITH_GITHUB_REPOSITORY_URL` in `CITATION.cff`.
- [ ] Add the final journal citation and DOI when available.
- [ ] Choose and add an open-source license, or intentionally keep the repository closed-source.
- [ ] Confirm that no controlled clinical data, patient identifiers, credentials, or local output files are committed.
- [ ] Review `configs/example_manifest.json` and ensure it contains only portable example paths.
- [ ] Run `python -m compileall src scripts legacy`.
- [ ] Run `pytest -q`.
- [ ] Run a quick mean-z scoring test on a small local dataset.
- [ ] Tag the release, for example `v1.0.0`.
