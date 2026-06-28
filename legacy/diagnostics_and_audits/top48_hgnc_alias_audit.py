from pathlib import Path
import ast
import re

import numpy as np
import pandas as pd
from IPython.display import display


# ============================================================
# PATHS
# ============================================================

HGNC_FILE = Path(
    r"D:\AIDO-Data\HGNC\hgnc_complete_set.txt"
)

TOP48_FILE = Path(
    r"D:\AIDO-Temp\XOBS_ORIGINAL_BP_ISLAND_AUDIT"
    r"\top_2p5pct_original_5065_islands.csv"
)

OUTPUT_DIR = Path(
    r"D:\AIDO-Temp\XOBS_ORIGINAL_BP_ISLAND_AUDIT"
    r"\HGNC_ALIAS_AUDIT_TOP48"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

def normalize_symbol(value):
    if pd.isna(value):
        return None

    text = str(value).strip().strip("\"'")

    if not text or text.lower() in {
        "nan", "none", "null", "na", "n/a"
    }:
        return None

    return text.upper()


def split_multi_value(value):
    """
    HGNC multi-value fields may appear as:
    - pipe-delimited text
    - comma-delimited text
    - Python-like list strings
    - single values
    """
    if pd.isna(value):
        return []

    text = str(value).strip()

    if not text:
        return []

    # Handle Python-like list representation.
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)

            if isinstance(parsed, (list, tuple, set)):
                values = list(parsed)
            else:
                values = [parsed]

            return [
                normalize_symbol(item)
                for item in values
                if normalize_symbol(item) is not None
            ]
        except Exception:
            pass

    # Remove surrounding brackets/quotes.
    text = text.strip("[]")
    text = text.replace('"', "").replace("'", "")

    # HGNC commonly uses |, but some exports may use commas.
    parts = re.split(r"\s*[|,;]\s*", text)

    return [
        normalize_symbol(part)
        for part in parts
        if normalize_symbol(part) is not None
    ]


def find_column(df, candidates, required=True):
    lower_map = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for candidate in candidates:
        candidate_lower = candidate.lower()

        if candidate_lower in lower_map:
            return lower_map[candidate_lower]

    if required:
        raise KeyError(
            "Required HGNC column not found. Tried: "
            + ", ".join(candidates)
            + "\nAvailable columns:\n"
            + "\n".join(map(str, df.columns))
        )

    return None


# ============================================================
# VALIDATE FILES
# ============================================================

if not HGNC_FILE.exists():
    raise FileNotFoundError(
        f"HGNC file not found:\n{HGNC_FILE}"
    )

if not TOP48_FILE.exists():
    raise FileNotFoundError(
        f"Top-48 gene file not found:\n{TOP48_FILE}"
    )


# ============================================================
# LOAD 48 GENES
# ============================================================

top48 = pd.read_csv(
    TOP48_FILE,
    low_memory=False
)

gene_column_candidates = [
    column
    for column in top48.columns
    if str(column).strip().lower() in {
        "gene",
        "gene_symbol",
        "symbol"
    }
]

if not gene_column_candidates:
    raise KeyError(
        "No gene column found in the Top-48 file.\n"
        f"Available columns: {list(top48.columns)}"
    )

gene_column = gene_column_candidates[0]

input_genes = (
    top48[gene_column]
    .map(normalize_symbol)
    .dropna()
    .drop_duplicates()
    .tolist()
)

print(f"Input genes loaded: {len(input_genes):,}")

if len(input_genes) != 48:
    print(
        "WARNING: the input file does not contain exactly "
        f"48 unique genes; it contains {len(input_genes):,}."
    )


# ============================================================
# LOAD HGNC COMPLETE SET
# ============================================================

hgnc = pd.read_csv(
    HGNC_FILE,
    sep="\t",
    dtype=str,
    low_memory=False
)

print(f"HGNC records loaded: {len(hgnc):,}")

symbol_col = find_column(
    hgnc,
    ["symbol"]
)

hgnc_id_col = find_column(
    hgnc,
    ["hgnc_id"]
)

name_col = find_column(
    hgnc,
    ["name"],
    required=False
)

status_col = find_column(
    hgnc,
    ["status"],
    required=False
)

alias_col = find_column(
    hgnc,
    ["alias_symbol"],
    required=False
)

prev_col = find_column(
    hgnc,
    ["prev_symbol"],
    required=False
)

entrez_col = find_column(
    hgnc,
    ["entrez_id"],
    required=False
)

ensembl_col = find_column(
    hgnc,
    ["ensembl_gene_id"],
    required=False
)

locus_type_col = find_column(
    hgnc,
    ["locus_type"],
    required=False
)


# ============================================================
# BUILD HGNC LOOKUP TABLES
# ============================================================

approved_lookup = {}
previous_lookup = {}
alias_lookup = {}

for _, row in hgnc.iterrows():
    approved_symbol = normalize_symbol(
        row[symbol_col]
    )

    if approved_symbol is None:
        continue

    record = {
        "approved_symbol": approved_symbol,
        "hgnc_id": (
            row[hgnc_id_col]
            if hgnc_id_col is not None
            else np.nan
        ),
        "gene_name": (
            row[name_col]
            if name_col is not None
            else np.nan
        ),
        "status": (
            row[status_col]
            if status_col is not None
            else np.nan
        ),
        "alias_symbol_raw": (
            row[alias_col]
            if alias_col is not None
            else np.nan
        ),
        "prev_symbol_raw": (
            row[prev_col]
            if prev_col is not None
            else np.nan
        ),
        "entrez_id": (
            row[entrez_col]
            if entrez_col is not None
            else np.nan
        ),
        "ensembl_gene_id": (
            row[ensembl_col]
            if ensembl_col is not None
            else np.nan
        ),
        "locus_type": (
            row[locus_type_col]
            if locus_type_col is not None
            else np.nan
        ),
    }

    approved_lookup.setdefault(
        approved_symbol,
        []
    ).append(record)

    if prev_col is not None:
        for previous_symbol in split_multi_value(
            row[prev_col]
        ):
            previous_lookup.setdefault(
                previous_symbol,
                []
            ).append(record)

    if alias_col is not None:
        for alias_symbol in split_multi_value(
            row[alias_col]
        ):
            alias_lookup.setdefault(
                alias_symbol,
                []
            ).append(record)


# ============================================================
# MAP EACH INPUT GENE
# ============================================================

audit_rows = []

for input_symbol in input_genes:
    approved_matches = approved_lookup.get(
        input_symbol,
        []
    )

    previous_matches = previous_lookup.get(
        input_symbol,
        []
    )

    alias_matches = alias_lookup.get(
        input_symbol,
        []
    )

    all_matches = (
        approved_matches
        + previous_matches
        + alias_matches
    )

    # Deduplicate by HGNC ID + approved symbol.
    unique_matches = {}

    for record in all_matches:
        key = (
            record["hgnc_id"],
            record["approved_symbol"]
        )
        unique_matches[key] = record

    unique_matches = list(
        unique_matches.values()
    )

    if approved_matches:
        mapping_type = "APPROVED_SYMBOL"
    elif previous_matches:
        mapping_type = "PREVIOUS_SYMBOL"
    elif alias_matches:
        mapping_type = "ALIAS_SYMBOL"
    else:
        mapping_type = "UNMATCHED"

    if len(unique_matches) > 1:
        mapping_status = "AMBIGUOUS"
    elif len(unique_matches) == 1:
        mapping_status = "UNIQUE"
    else:
        mapping_status = "UNMATCHED"

    if unique_matches:
        approved_symbols = sorted(
            {
                record["approved_symbol"]
                for record in unique_matches
            }
        )

        hgnc_ids = sorted(
            {
                str(record["hgnc_id"])
                for record in unique_matches
                if pd.notna(record["hgnc_id"])
            }
        )

        gene_names = sorted(
            {
                str(record["gene_name"])
                for record in unique_matches
                if pd.notna(record["gene_name"])
            }
        )

        statuses = sorted(
            {
                str(record["status"])
                for record in unique_matches
                if pd.notna(record["status"])
            }
        )

        alias_values = sorted(
            {
                alias
                for record in unique_matches
                for alias in split_multi_value(
                    record["alias_symbol_raw"]
                )
            }
        )

        previous_values = sorted(
            {
                previous
                for record in unique_matches
                for previous in split_multi_value(
                    record["prev_symbol_raw"]
                )
            }
        )

        entrez_ids = sorted(
            {
                str(record["entrez_id"])
                for record in unique_matches
                if pd.notna(record["entrez_id"])
            }
        )

        ensembl_ids = sorted(
            {
                str(record["ensembl_gene_id"])
                for record in unique_matches
                if pd.notna(record["ensembl_gene_id"])
            }
        )

        locus_types = sorted(
            {
                str(record["locus_type"])
                for record in unique_matches
                if pd.notna(record["locus_type"])
            }
        )
    else:
        approved_symbols = []
        hgnc_ids = []
        gene_names = []
        statuses = []
        alias_values = []
        previous_values = []
        entrez_ids = []
        ensembl_ids = []
        locus_types = []

    audit_rows.append(
        {
            "input_symbol": input_symbol,
            "mapping_type": mapping_type,
            "mapping_status": mapping_status,
            "number_of_HGNC_matches": len(
                unique_matches
            ),
            "approved_symbol": " | ".join(
                approved_symbols
            ),
            "symbol_changed": (
                len(approved_symbols) == 1
                and approved_symbols[0] != input_symbol
            ),
            "hgnc_id": " | ".join(
                hgnc_ids
            ),
            "gene_name": " | ".join(
                gene_names
            ),
            "HGNC_status": " | ".join(
                statuses
            ),
            "alias_symbols": " | ".join(
                alias_values
            ),
            "previous_symbols": " | ".join(
                previous_values
            ),
            "entrez_id": " | ".join(
                entrez_ids
            ),
            "ensembl_gene_id": " | ".join(
                ensembl_ids
            ),
            "locus_type": " | ".join(
                locus_types
            ),
            "has_alias_names": len(
                alias_values
            ) > 0,
            "has_previous_symbols": len(
                previous_values
            ) > 0,
        }
    )

audit = pd.DataFrame(
    audit_rows
)

audit = audit.sort_values(
    [
        "mapping_status",
        "mapping_type",
        "input_symbol"
    ]
).reset_index(drop=True)


# ============================================================
# SUMMARY
# ============================================================

summary = pd.DataFrame(
    {
        "metric": [
            "Input genes",
            "Current approved symbols",
            "Mapped through previous symbol",
            "Mapped through alias symbol",
            "Ambiguous mappings",
            "Unmatched symbols",
            "Symbols changed after HGNC mapping",
            "Genes with listed alias names",
            "Genes with listed previous symbols"
        ],
        "count": [
            len(audit),
            int(
                (
                    audit["mapping_type"]
                    == "APPROVED_SYMBOL"
                ).sum()
            ),
            int(
                (
                    audit["mapping_type"]
                    == "PREVIOUS_SYMBOL"
                ).sum()
            ),
            int(
                (
                    audit["mapping_type"]
                    == "ALIAS_SYMBOL"
                ).sum()
            ),
            int(
                (
                    audit["mapping_status"]
                    == "AMBIGUOUS"
                ).sum()
            ),
            int(
                (
                    audit["mapping_status"]
                    == "UNMATCHED"
                ).sum()
            ),
            int(
                audit["symbol_changed"].sum()
            ),
            int(
                audit["has_alias_names"].sum()
            ),
            int(
                audit["has_previous_symbols"].sum()
            ),
        ]
    }
)


# ============================================================
# SAVE
# ============================================================

audit.to_csv(
    OUTPUT_DIR
    / "Top48_HGNC_alias_previous_symbol_audit.csv",
    index=False
)

summary.to_csv(
    OUTPUT_DIR
    / "Top48_HGNC_alias_audit_summary.csv",
    index=False
)

audit.loc[
    audit["symbol_changed"]
    | (audit["mapping_type"] != "APPROVED_SYMBOL")
    | (audit["mapping_status"] != "UNIQUE")
].to_csv(
    OUTPUT_DIR
    / "Top48_HGNC_symbols_requiring_attention.csv",
    index=False
)


# ============================================================
# DISPLAY
# ============================================================

print()
print("=" * 82)
print("TOP-48 HGNC ALIAS AUDIT COMPLETE")
print("=" * 82)

print("\nSummary")
display(summary)

print("\nAll 48 genes")
display(
    audit[
        [
            "input_symbol",
            "mapping_type",
            "mapping_status",
            "approved_symbol",
            "symbol_changed",
            "alias_symbols",
            "previous_symbols",
            "HGNC_status"
        ]
    ]
)

print("\nGenes requiring attention")
display(
    audit.loc[
        audit["symbol_changed"]
        | (audit["mapping_type"] != "APPROVED_SYMBOL")
        | (audit["mapping_status"] != "UNIQUE"),
        [
            "input_symbol",
            "mapping_type",
            "mapping_status",
            "approved_symbol",
            "alias_symbols",
            "previous_symbols",
            "HGNC_status"
        ]
    ]
)

print("\nOutputs saved to:")
print(OUTPUT_DIR)
