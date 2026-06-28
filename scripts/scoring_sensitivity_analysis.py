#!/usr/bin/env python3
"""AIDO-D-XOBS scoring-formulation sensitivity analysis.

Compares mean-z, GSVA and ssGSEA process scores across TCGA-BRCA and
METABRIC endpoints. TCGA analyses are restricted to primary solid-tumor
samples (TCGA sample-type code 01) before clinical alignment.
"""
from __future__ import annotations

import argparse, json, logging, math, re, sys, warnings
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

try:
    import gseapy as gp
except ImportError as exc:
    raise ImportError(
        'GSEApy is not installed in the current Python environment. '
        'Run `%pip install --upgrade gseapy` in Jupyter, restart the kernel, '
        'and rerun this script.'
    ) from exc

DEFAULT_K_VALUES = (1, 3, 5, 10, 15, 20, 30, 40, 50)
SUPPORTED_METHODS = ('mean_z', 'gsva', 'ssgsea')
EPS = 1e-300

# ---------------------------------------------------------------------
# NOTEBOOK QUICK-RUN CONFIGURATION
# ---------------------------------------------------------------------
# These defaults are used only when the complete script is pasted directly
# into a Jupyter notebook cell without command-line arguments.
#
# QUICK TEST:
#   NOTEBOOK_METHODS = ["mean_z"]
#   NOTEBOOK_CV_SPLITS = 3
#   NOTEBOOK_CV_REPEATS = 1
#
# FULL ANALYSIS:
#   NOTEBOOK_METHODS = ["mean_z", "gsva", "ssgsea"]
#   NOTEBOOK_CV_SPLITS = 5
#   NOTEBOOK_CV_REPEATS = 20

NOTEBOOK_MANIFEST = Path("configs/example_manifest.json")
NOTEBOOK_OUTPUT_DIR = Path("results")
NOTEBOOK_METHODS = ["mean_z"]
NOTEBOOK_K_VALUES = [1, 10, 50]
NOTEBOOK_CV_SPLITS = 3
NOTEBOOK_CV_REPEATS = 1
NOTEBOOK_THREADS = 4
NOTEBOOK_GSVA_KCDF = "Gaussian"
NOTEBOOK_SAVE_SCORE_MATRICES = False


def notebook_default_argv() -> list[str]:
    """Construct explicit arguments for direct execution inside Jupyter."""
    argv = [
        "--manifest", str(NOTEBOOK_MANIFEST),
        "--output-dir", str(NOTEBOOK_OUTPUT_DIR),
        "--methods", *NOTEBOOK_METHODS,
        "--k-values", *[str(k) for k in NOTEBOOK_K_VALUES],
        "--cv-splits", str(NOTEBOOK_CV_SPLITS),
        "--cv-repeats", str(NOTEBOOK_CV_REPEATS),
        "--threads", str(NOTEBOOK_THREADS),
        "--gsva-kcdf", NOTEBOOK_GSVA_KCDF,
    ]
    if NOTEBOOK_SAVE_SCORE_MATRICES:
        argv.append("--save-score-matrices")
    return argv


@dataclass(frozen=True)
class AnalysisSpec:
    name: str
    cohort: str
    endpoint: str
    expression_file: str
    clinical_file: str
    gmt_file: str
    expression_format: str = 'generic'
    clinical_format: str = 'generic'
    sample_id_column: str | None = None
    label_column: str | None = None
    positive_values: tuple[str, ...] = ()
    negative_values: tuple[str, ...] = ()
    positive_label: str = 'positive'
    negative_label: str = 'negative'
    expression_gene_column: str | None = None
    expression_sample_prefix_remove: str | None = None
    min_genes: int = 10
    max_genes: int = 500
    exclude_genes: tuple[str, ...] = ()


def configure_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(output_dir/'run.log', mode='w', encoding='utf-8')],
        force=True,
    )


def norm_token(v: Any) -> str:
    if pd.isna(v): return ''
    return re.sub(r'\s+', ' ', str(v).strip().upper())


def canonical_gene(v: Any) -> str:
    t = str(v).strip().upper()
    if '|' in t:
        parts = [p.strip() for p in t.split('|') if p.strip()]
        nn = [p for p in parts if not p.isdigit()]
        if nn: t = nn[0]
    return t


def read_table(path: Path) -> pd.DataFrame:
    """Read delimited tables, including extensionless UCSC Xena matrices.

    Known TSV/TXT/GCT and CSV suffixes use an explicit separator. For files
    without a useful suffix, the first non-empty lines are inspected and the
    delimiter is inferred. A pandas/Python-engine fallback is used only when
    the simple inspection is inconclusive.
    """
    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()
    if suffix in {'.xlsx', '.xls'}:
        return pd.read_excel(path)
    if suffix in {'.tsv', '.txt', '.gct'}:
        return pd.read_csv(path, sep='\t', low_memory=False)
    if suffix == '.csv':
        return pd.read_csv(path, sep=',', low_memory=False)

    # UCSC Xena clinical matrices commonly have no filename extension but are
    # tab-delimited. Inspect several lines rather than trusting the suffix.
    sample_lines = []
    with path.open('r', encoding='utf-8-sig', errors='replace') as handle:
        for raw in handle:
            line = raw.rstrip('\r\n')
            if line.strip():
                sample_lines.append(line)
            if len(sample_lines) >= 20:
                break

    if not sample_lines:
        raise ValueError(f'Input table is empty: {path}')

    tab_score = sum(line.count('\t') for line in sample_lines)
    comma_score = sum(line.count(',') for line in sample_lines)
    semicolon_score = sum(line.count(';') for line in sample_lines)

    if tab_score >= max(comma_score, semicolon_score) and tab_score > 0:
        return pd.read_csv(path, sep='\t', low_memory=False)
    if comma_score >= semicolon_score and comma_score > 0:
        return pd.read_csv(path, sep=',', low_memory=False)
    if semicolon_score > 0:
        return pd.read_csv(path, sep=';', low_memory=False)

    # Final fallback: pandas delegates delimiter sniffing to Python's parser.
    return pd.read_csv(path, sep=None, engine='python')



def tcga_sample_type_code(sample_id: Any) -> str | None:
    """Return the two-digit TCGA sample-type code when it can be resolved.

    Examples
    --------
    TCGA-XX-XXXX-01A-... -> "01"
    TCGA-XX-XXXX-01      -> "01"
    """
    text = str(sample_id).strip()
    parts = text.split("-")
    if len(parts) >= 4 and parts[0].upper() == "TCGA":
        token = parts[3]
        if len(token) >= 2 and token[:2].isdigit():
            return token[:2]
    return None


def apply_cohort_sample_filter(
    expr: pd.DataFrame,
    spec: AnalysisSpec,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply cohort-specific sample restrictions and return an audit table.

    TCGA-BRCA is restricted to sample-type code 01 (Primary Solid Tumor).
    METABRIC is unchanged.
    """
    rows = []
    for sample in expr.columns:
        code = tcga_sample_type_code(sample)
        rows.append({
            "sample_id": str(sample),
            "tcga_sample_type_code": code if code is not None else "",
            "retained": True,
            "filter_reason": "not_applicable",
        })

    audit = pd.DataFrame(rows)

    if spec.cohort.upper().startswith("TCGA"):
        keep = []
        for sample in expr.columns:
            code = tcga_sample_type_code(sample)
            if code == "01":
                keep.append(sample)

        if not keep:
            examples = list(map(str, expr.columns[:10]))
            raise ValueError(
                "No TCGA primary solid-tumor samples (sample-type code 01) "
                f"were detected. Example expression columns: {examples}"
            )

        audit["retained"] = audit["tcga_sample_type_code"].eq("01")
        audit["filter_reason"] = np.where(
            audit["retained"],
            "TCGA_primary_solid_tumor_code_01",
            "excluded_nonprimary_TCGA_sample",
        )
        before = expr.shape[1]
        expr = expr.loc[:, keep]
        logging.info(
            "%s sample-type filter: retained %d/%d TCGA code-01 primary tumors.",
            spec.name,
            expr.shape[1],
            before,
        )
    else:
        logging.info(
            "%s sample-type filter: not applied to cohort %s (%d samples).",
            spec.name,
            spec.cohort,
            expr.shape[1],
        )

    return expr, audit


def load_expression(spec: AnalysisSpec) -> pd.DataFrame:
    df = read_table(Path(spec.expression_file))
    gene_col = spec.expression_gene_column
    if gene_col is None:
        cands = ['Hugo_Symbol','GENE_SYMBOL','gene_symbol','Gene Symbol','gene','Gene','symbol','Symbol',df.columns[0]]
        gene_col = next((c for c in cands if c in df.columns), None)
    if gene_col is None:
        raise ValueError('Could not resolve gene column')
    genes = df[gene_col].map(canonical_gene)
    drop = {gene_col,'Entrez_Gene_Id','ENTREZ_GENE_ID','Entrez ID','Cytoband','Locus ID'}
    sample_cols = [c for c in df.columns if c not in drop]
    expr = df[sample_cols].apply(pd.to_numeric, errors='coerce')
    expr = expr.loc[:, expr.notna().any(axis=0)]
    expr.index = genes.loc[expr.index]
    if spec.expression_sample_prefix_remove:
        rgx = re.compile(spec.expression_sample_prefix_remove)
        expr.columns = [rgx.sub('', str(c).strip()) for c in expr.columns]
    else:
        expr.columns = [str(c).strip() for c in expr.columns]
    expr = expr.loc[expr.index != '']
    if expr.index.duplicated().any():
        expr = expr.groupby(level=0, sort=False).mean()
    return expr


def read_cbio_clinical(path: Path) -> pd.DataFrame:
    from io import StringIO
    lines = path.read_text(encoding='utf-8', errors='replace').splitlines()
    content = [x for x in lines if x.strip() and not x.startswith('#')]
    return pd.read_csv(StringIO('\n'.join(content)), sep='\t', low_memory=False)


def load_labels(spec: AnalysisSpec) -> pd.Series:
    df = read_cbio_clinical(Path(spec.clinical_file)) if spec.clinical_format.lower() in {'cbioportal','metabric'} else read_table(Path(spec.clinical_file))
    if not spec.sample_id_column or not spec.label_column:
        raise ValueError('sample_id_column and label_column are required')
    if spec.sample_id_column not in df.columns or spec.label_column not in df.columns:
        raise KeyError(f'Clinical columns missing. Available: {list(df.columns)[:40]}')
    pos = {norm_token(x) for x in spec.positive_values}
    neg = {norm_token(x) for x in spec.negative_values}
    vals = []
    for v in df[spec.label_column]:
        t = norm_token(v)
        vals.append(1.0 if t in pos else 0.0 if t in neg else np.nan)
    s = pd.Series(vals, index=df[spec.sample_id_column].astype(str).str.strip(), name='label')
    return s[~s.index.duplicated(keep='first')].dropna().astype(int)


def load_gmt(path: Path) -> dict[str, list[str]]:
    gs = {}
    with path.open('r', encoding='utf-8', errors='replace') as fh:
        for raw in fh:
            p = raw.rstrip('\n').split('\t')
            if len(p) < 3: continue
            name = p[0].strip()
            genes = list(dict.fromkeys(canonical_gene(x) for x in p[2:] if str(x).strip()))
            if name and genes and name not in gs:
                gs[name] = genes
    if not gs: raise ValueError('No gene sets parsed from GMT')
    return gs


def harmonize(expr: pd.DataFrame, labels: pd.Series, gene_sets: Mapping[str, Sequence[str]], spec: AnalysisSpec):
    expr = expr.copy(); labels = labels.copy()
    expr.columns = expr.columns.astype(str).str.strip(); labels.index = labels.index.astype(str).str.strip()
    shared = [s for s in expr.columns if s in labels.index]
    if len(shared) < 20: raise ValueError(f'Only {len(shared)} shared samples')
    expr = expr.loc[:, shared]; labels = labels.loc[shared]
    excl = {canonical_gene(x) for x in spec.exclude_genes}
    if excl: expr = expr.loc[~expr.index.isin(excl)]
    measured = set(expr.index)
    filt, rows = {}, []
    for term, genes in gene_sets.items():
        obs = list(dict.fromkeys(g for g in genes if g in measured and g not in excl))
        ok = spec.min_genes <= len(obs) <= spec.max_genes
        rows.append({'term':term,'original_gene_count':len(genes),'matched_gene_count':len(obs),'eligible':ok})
        if ok: filt[term] = obs
    if len(filt) < 20: raise ValueError('Too few eligible gene sets')
    return expr, labels, filt, pd.DataFrame(rows)


def score_mean_z(expr: pd.DataFrame, gene_sets: Mapping[str, Sequence[str]]) -> pd.DataFrame:
    mu = expr.mean(axis=1); sd = expr.std(axis=1, ddof=0).replace(0, np.nan)
    z = expr.sub(mu, axis=0).div(sd, axis=0).fillna(0.0)
    return pd.DataFrame.from_dict({t:z.loc[list(g)].mean(axis=0) for t,g in gene_sets.items()}, orient='index')


def extract_gseapy_matrix(result: Any, samples: Sequence[str], terms: Sequence[str], score_cols: Sequence[str]) -> pd.DataFrame:
    candidates = []
    for attr in ('res2d','results','data'):
        obj = getattr(result, attr, None)
        if isinstance(obj, pd.DataFrame): candidates.append(obj.copy())
    if isinstance(result, pd.DataFrame): candidates.append(result.copy())
    for df in candidates:
        lower = {str(c).lower():c for c in df.columns}
        name_col = next((lower[k] for k in ('name','sample','sample_name') if k in lower), None)
        term_col = next((lower[k] for k in ('term','gene_set','geneset') if k in lower), None)
        val_col = next((lower[k.lower()] for k in score_cols if k.lower() in lower), None)
        if name_col is not None and term_col is not None and val_col is not None:
            w = df.pivot_table(index=term_col, columns=name_col, values=val_col, aggfunc='first')
            w.index = w.index.astype(str); w.columns = w.columns.astype(str)
            st = [t for t in terms if t in w.index]; ss = [s for s in samples if s in w.columns]
            if st and ss: return w.loc[st, ss].astype(float)
        df.index = df.index.astype(str); df.columns = df.columns.astype(str)
        if set(terms)&set(df.index) and set(samples)&set(df.columns):
            return df.loc[[t for t in terms if t in df.index],[s for s in samples if s in df.columns]].astype(float)
        if set(terms)&set(df.columns) and set(samples)&set(df.index):
            tr = df.T
            return tr.loc[[t for t in terms if t in tr.index],[s for s in samples if s in tr.columns]].astype(float)
    raise RuntimeError('Could not parse GSEApy result matrix')


def compute_scores(method: str, expr: pd.DataFrame, gene_sets: Mapping[str, Sequence[str]], spec: AnalysisSpec, threads: int, kcdf: str) -> pd.DataFrame:
    if method == 'mean_z':
        out = score_mean_z(expr, gene_sets)
    elif method == 'gsva':
        res = gp.gsva(data=expr, gene_sets=dict(gene_sets), outdir=None, min_size=spec.min_genes, max_size=spec.max_genes, kcdf=kcdf, threads=threads, verbose=True)
        out = extract_gseapy_matrix(res, expr.columns, list(gene_sets), ('ES','NES','score'))
    elif method == 'ssgsea':
        res = gp.ssgsea(data=expr, gene_sets=dict(gene_sets), outdir=None, min_size=spec.min_genes, max_size=spec.max_genes, sample_norm_method='rank', permutation_num=0, no_plot=True, threads=threads, verbose=True)
        out = extract_gseapy_matrix(res, expr.columns, list(gene_sets), ('NES','ES','score'))
    else:
        raise ValueError(method)
    out = out.replace([np.inf,-np.inf], np.nan).dropna(axis=0, how='all')
    out = out.loc[:, [s for s in expr.columns if s in out.columns]]
    out = out.apply(lambda r:r.fillna(r.median()), axis=1)
    return out.astype(float)


def process_stats(scores: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    y = labels.loc[scores.columns].to_numpy(int)
    rows=[]
    for term, row in scores.iterrows():
        x = row.to_numpy(float); neg=x[y==0]; pos=x[y==1]
        try:
            _,p = mannwhitneyu(pos, neg, alternative='two-sided', method='auto')
            raw = roc_auc_score(y,x); auc=max(raw,1-raw)
        except ValueError:
            p=1.0; raw=0.5; auc=0.5
        rows.append({'term':term,'n_negative':int((y==0).sum()),'n_positive':int((y==1).sum()),'mean_negative':float(np.mean(neg)),'mean_positive':float(np.mean(pos)),'direction_positive_minus_negative':float(np.mean(pos)-np.mean(neg)),'p_value':float(p),'D':float(-math.log10(max(float(p),EPS))),'auc_raw_positive_vs_negative':float(raw),'auc_oriented':float(auc)})
    out=pd.DataFrame(rows); out['q_value']=multipletests(out['p_value'], method='fdr_bh')[1]
    out=out.sort_values(['D','auc_oriented','term'], ascending=[False,False,True]).reset_index(drop=True)
    out['rank_D']=np.arange(1,len(out)+1)
    return out


def rank_training(X: pd.DataFrame, y: np.ndarray) -> list[str]:
    ranked=[]
    for term in X.columns:
        x=X[term].to_numpy(float); neg=x[y==0]; pos=x[y==1]
        try:
            _,p=mannwhitneyu(pos,neg,alternative='two-sided',method='auto'); a=roc_auc_score(y,x); a=max(a,1-a)
        except ValueError:
            p=1.0; a=0.5
        ranked.append((term,-math.log10(max(float(p),EPS)),a))
    ranked.sort(key=lambda z:(z[1],z[2],z[0]), reverse=True)
    return [z[0] for z in ranked]


def orient_features(Xtr: pd.DataFrame, Xte: pd.DataFrame, ytr: np.ndarray):
    Xtr=Xtr.copy(); Xte=Xte.copy()
    for c in Xtr.columns:
        if Xtr.loc[ytr==1,c].mean() < Xtr.loc[ytr==0,c].mean():
            Xtr[c] = -Xtr[c]; Xte[c] = -Xte[c]
    return Xtr, Xte


def repeated_cv(scores: pd.DataFrame, labels: pd.Series, k_values: Sequence[int], n_splits: int, n_repeats: int, seed: int):
    samples=[s for s in scores.columns if s in labels.index]
    X=scores.loc[:,samples].T; y=labels.loc[samples].to_numpy(int)
    splits=min(n_splits, int(np.bincount(y).min()))
    if splits<2: raise ValueError('Not enough cases per class')
    cv=RepeatedStratifiedKFold(n_splits=splits,n_repeats=n_repeats,random_state=seed)
    metrics=[]; selections=[]
    total_splits = splits * n_repeats
    for sid,(tri,tei) in enumerate(cv.split(X,y), start=1):
        logging.info('CV split %d/%d', sid, total_splits)
        Xtr0=X.iloc[tri]; Xte0=X.iloc[tei]; ytr=y[tri]; yte=y[tei]
        ranked=rank_training(Xtr0,ytr)
        for k in k_values:
            sel=ranked[:min(k,len(ranked))]
            Xtr,Xte=orient_features(Xtr0[sel],Xte0[sel],ytr)
            model=Pipeline([('scale',StandardScaler()),('model',LogisticRegression(class_weight='balanced',solver='liblinear',max_iter=5000,random_state=seed))])
            model.fit(Xtr,ytr); pr=model.predict_proba(Xte)[:,1]; pred=(pr>=0.5).astype(int)
            metrics.append({'split_id':sid,'K':k,'actual_K':len(sel),'auc':roc_auc_score(yte,pr),'balanced_accuracy':balanced_accuracy_score(yte,pred)})
            selections.extend({'split_id':sid,'K':k,'rank_within_fold':r,'term':t} for r,t in enumerate(sel,start=1))
    m=pd.DataFrame(metrics); s=pd.DataFrame(selections)
    summ=m.groupby('K',as_index=False).agg(cv_auc_mean=('auc','mean'),cv_auc_sd=('auc','std'),cv_auc_q025=('auc',lambda x:np.quantile(x,0.025)),cv_auc_q975=('auc',lambda x:np.quantile(x,0.975)),balanced_accuracy_mean=('balanced_accuracy','mean'),n_evaluations=('auc','size'))
    return summ,s


def compare_methods(stats_map: Mapping[str,pd.DataFrame], top_k: int):
    rr=[]; oo=[]; methods=list(stats_map)
    for i,m1 in enumerate(methods):
        for m2 in methods[i+1:]:
            a=stats_map[m1][['term','D','auc_oriented']]; b=stats_map[m2][['term','D','auc_oriented']]
            z=a.merge(b,on='term',suffixes=(f'_{m1}',f'_{m2}'))
            rd,pd_=spearmanr(z[f'D_{m1}'],z[f'D_{m2}']); ra,pa=spearmanr(z[f'auc_oriented_{m1}'],z[f'auc_oriented_{m2}'])
            rr.append({'method_1':m1,'method_2':m2,'n_shared_processes':len(z),'spearman_D':rd,'spearman_D_p':pd_,'spearman_auc':ra,'spearman_auc_p':pa})
            t1=set(stats_map[m1].head(top_k)['term']); t2=set(stats_map[m2].head(top_k)['term']); uni=t1|t2
            oo.append({'method_1':m1,'method_2':m2,'top_k':top_k,'intersection_count':len(t1&t2),'jaccard':len(t1&t2)/len(uni) if uni else np.nan})
    return pd.DataFrame(rr),pd.DataFrame(oo)


def load_manifest(path: Path) -> list[AnalysisSpec]:
    data=json.loads(path.read_text(encoding='utf-8')); rows=data.get('analyses',data)
    specs=[]
    for r in rows:
        r=dict(r)
        for k in ('positive_values','negative_values','exclude_genes'):
            v=r.get(k,[]); r[k]=(v,) if isinstance(v,str) else tuple(str(x) for x in v)
        specs.append(AnalysisSpec(**r))
    return specs


def run_analysis(spec: AnalysisSpec, root: Path, methods: Sequence[str], ks: Sequence[int], splits: int, repeats: int, seed: int, threads: int, kcdf: str, save_scores: bool):
    adir = root / spec.name
    adir.mkdir(parents=True, exist_ok=True)

    logging.info("===== Starting analysis: %s =====", spec.name)
    logging.info("Loading expression: %s", spec.expression_file)
    expr = load_expression(spec)
    logging.info("Expression loaded: %d genes x %d samples", expr.shape[0], expr.shape[1])

    expr, sample_filter_audit = apply_cohort_sample_filter(expr, spec)
    sample_filter_audit.to_csv(adir / "00_sample_type_filter_audit.csv", index=False)

    logging.info("Loading clinical labels: %s", spec.clinical_file)
    labels = load_labels(spec)
    logging.info(
        "Clinical labels loaded: %d records (%d negative, %d positive before alignment)",
        len(labels), int((labels == 0).sum()), int((labels == 1).sum())
    )

    logging.info("Loading GMT: %s", spec.gmt_file)
    gene_sets = load_gmt(Path(spec.gmt_file))
    logging.info("GMT loaded: %d gene sets", len(gene_sets))

    expr, labels, gene_sets, audit = harmonize(expr, labels, gene_sets, spec)
    logging.info(
        "After harmonization: %d genes, %d samples, %d eligible processes; "
        "class counts=%d/%d",
        expr.shape[0], expr.shape[1], len(gene_sets),
        int((labels == 0).sum()), int((labels == 1).sum())
    )

    audit.to_csv(adir / "00_gene_set_mapping_audit.csv", index=False)
    pd.DataFrame({
        "sample_id": labels.index,
        "label": labels.values,
        "label_name": np.where(
            labels.values == 1, spec.positive_label, spec.negative_label
        ),
    }).to_csv(adir / "00_sample_labels.csv", index=False)

    input_summary = {
        "analysis_name": spec.name,
        "cohort": spec.cohort,
        "endpoint": spec.endpoint,
        "n_expression_genes": int(expr.shape[0]),
        "n_aligned_samples": int(expr.shape[1]),
        "n_negative": int((labels == 0).sum()),
        "n_positive": int((labels == 1).sum()),
        "n_eligible_processes": int(len(gene_sets)),
        "tcga_primary_tumor_filter": bool(spec.cohort.upper().startswith("TCGA")),
    }
    (adir / "00_input_summary.json").write_text(
        json.dumps(input_summary, indent=2), encoding="utf-8"
    )

    stats_map = {}
    summaries = []

    for method in methods:
        logging.info("[%s] Computing %s process scores...", spec.name, method)
        mdir = adir / method
        mdir.mkdir(parents=True, exist_ok=True)

        scores = compute_scores(method, expr, gene_sets, spec, threads, kcdf)
        logging.info(
            "[%s/%s] Score matrix: %d processes x %d samples",
            spec.name, method, scores.shape[0], scores.shape[1]
        )
        if save_scores:
            scores.to_csv(
                mdir / "01_process_score_matrix.tsv.gz",
                sep="\t",
                compression="gzip",
            )

        logging.info("[%s/%s] Computing process-level statistics...", spec.name, method)
        st = process_stats(scores, labels)
        st.to_csv(mdir / "02_process_statistics.csv", index=False)
        stats_map[method] = st

        logging.info(
            "[%s/%s] Running repeated CV: %d folds x %d repeats, K=%s",
            spec.name, method, splits, repeats, list(ks)
        )
        cv, sel = repeated_cv(scores, labels, ks, splits, repeats, seed)
        cv.to_csv(mdir / "03_cv_summary.csv", index=False)
        sel.to_csv(
            mdir / "04_cv_feature_selections.csv.gz",
            index=False,
            compression="gzip",
        )

        max_k = max(ks)
        krow = cv.loc[cv["K"] == max_k].iloc[0]
        best_k_row = cv.sort_values(
            ["cv_auc_mean", "K"], ascending=[False, True]
        ).iloc[0]
        top = st.iloc[0]
        n_high = max(1, int(math.floor(len(st) * 0.025)))

        summary = {
            "analysis_name": spec.name,
            "cohort": spec.cohort,
            "endpoint": spec.endpoint,
            "method": method,
            "n_samples": len(labels),
            "n_negative": int((labels == 0).sum()),
            "n_positive": int((labels == 1).sum()),
            "n_processes": len(st),
            "best_process": top["term"],
            "best_individual_auc": top["auc_oriented"],
            "best_process_D": top["D"],
            "best_process_q": top["q_value"],
            "high_D_mean_auc": float(st.head(n_high)["auc_oriented"].mean()),
            "fdr_significant_count": int((st["q_value"] < 0.05).sum()),
            "best_predefined_K": int(best_k_row["K"]),
            "best_predefined_K_cv_auc_mean": float(best_k_row["cv_auc_mean"]),
            "best_predefined_K_cv_auc_sd": float(best_k_row["cv_auc_sd"]),
            "top50_cv_auc_mean": krow["cv_auc_mean"],
            "top50_cv_auc_sd": krow["cv_auc_sd"],
            "top50_cv_auc_q025": krow["cv_auc_q025"],
            "top50_cv_auc_q975": krow["cv_auc_q975"],
            "top50_balanced_accuracy_mean": krow["balanced_accuracy_mean"],
        }
        summaries.append(summary)
        (mdir / "SUMMARY.json").write_text(
            json.dumps(summary, indent=2, default=float), encoding="utf-8"
        )
        logging.info(
            "[%s/%s] Done: best BP AUC=%.4f; best predefined K=%d (AUC=%.4f); "
            "K=%d AUC=%.4f",
            spec.name, method,
            float(summary["best_individual_auc"]),
            int(summary["best_predefined_K"]),
            float(summary["best_predefined_K_cv_auc_mean"]),
            max_k,
            float(summary["top50_cv_auc_mean"]),
        )

    rc, ov = compare_methods(stats_map, max(ks))
    rc.to_csv(adir / "06_method_rank_correlations.csv", index=False)
    ov.to_csv(adir / "07_method_topK_overlap.csv", index=False)
    pd.DataFrame(summaries).to_csv(
        adir / "08_method_comparison_summary.csv", index=False
    )
    logging.info("===== Completed analysis: %s =====", spec.name)
    return summaries


def main(argv=None) -> int:
    p=argparse.ArgumentParser(description='Compare mean-z, GSVA and ssGSEA process scoring.')
    p.add_argument('--manifest',required=True,type=Path); p.add_argument('--output-dir',type=Path,default=Path('scoring_sensitivity_output'))
    p.add_argument('--methods',nargs='+',choices=SUPPORTED_METHODS,default=list(SUPPORTED_METHODS)); p.add_argument('--k-values',nargs='+',type=int,default=list(DEFAULT_K_VALUES))
    p.add_argument('--cv-splits',type=int,default=5); p.add_argument('--cv-repeats',type=int,default=20); p.add_argument('--seed',type=int,default=20260622); p.add_argument('--threads',type=int,default=4)
    p.add_argument('--gsva-kcdf',choices=('Gaussian','Poisson'),default='Gaussian'); p.add_argument('--save-score-matrices',action='store_true')
    a=p.parse_args(argv); out=a.output_dir/f'SCORING_SENSITIVITY_{datetime.now().strftime("%Y%m%d_%H%M%S")}'; configure_logging(out)
    allsum=[]; fails=[]
    for spec in load_manifest(a.manifest):
        try:
            allsum.extend(run_analysis(spec,out,a.methods,a.k_values,a.cv_splits,a.cv_repeats,a.seed,a.threads,a.gsva_kcdf,a.save_score_matrices))
        except Exception as exc:
            logging.exception('Analysis failed: %s',spec.name); fails.append({'analysis_name':spec.name,'error_type':type(exc).__name__,'error':str(exc)})
    pd.DataFrame(allsum).to_csv(out/'ALL_ANALYSES_METHOD_SUMMARY.csv',index=False); pd.DataFrame(fails).to_csv(out/'ALL_ANALYSES_FAILURES.csv',index=False)
    (out/'RUN_METADATA.json').write_text(json.dumps({
        'manifest': str(a.manifest), 'methods': list(a.methods),
        'k_values': list(a.k_values), 'cv_splits': a.cv_splits,
        'cv_repeats': a.cv_repeats, 'seed': a.seed,
        'threads': a.threads, 'gsva_kcdf': a.gsva_kcdf,
        'successful_method_runs': len(allsum),
        'failed_analyses': len(fails)
    }, indent=2), encoding='utf-8')
    if allsum:
        df=pd.DataFrame(allsum); df.pivot_table(index=['analysis_name','cohort','endpoint'],columns='method',values=['best_individual_auc','high_D_mean_auc','top50_cv_auc_mean','fdr_significant_count']).to_csv(out/'ALL_ANALYSES_WIDE_COMPARISON.csv')
    return 1 if fails else 0

if __name__ == "__main__":
    running_in_jupyter = "ipykernel" in sys.modules
    command_line_has_manifest = "--manifest" in sys.argv

    if running_in_jupyter and not command_line_has_manifest:
        print(
            "Jupyter detected: using NOTEBOOK QUICK-RUN CONFIGURATION.\n"
            f"Manifest: {NOTEBOOK_MANIFEST}\n"
            f"Output: {NOTEBOOK_OUTPUT_DIR}\n"
            f"Methods: {NOTEBOOK_METHODS}\n"
            f"K values: {NOTEBOOK_K_VALUES}\n"
            f"CV: {NOTEBOOK_CV_SPLITS} folds x "
            f"{NOTEBOOK_CV_REPEATS} repeats\n"
        )
        notebook_exit_code = main(notebook_default_argv())
        print(f"Analysis finished with exit code {notebook_exit_code}.")
    else:
        raise SystemExit(main())
