# Data directory

Raw and processed study data are not included in this repository.

Expected example layout:

```text
data/
├── gene_sets/
│   └── c5.go.bp.symbols.gmt
├── tcga_brca/
│   ├── expression.tsv
│   └── clinical.tsv
└── metabric/
    ├── expression.tsv
    └── clinical.tsv
```

Update `configs/example_manifest.json` to match local filenames and columns.
Do not commit controlled, private, or patient-identifiable data.
