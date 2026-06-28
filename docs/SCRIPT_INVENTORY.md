# Script inventory

The repository separates portable manuscript-facing code from historical internal scripts.

| File | Status | Hard-coded local paths |
|---|---|---|
| `legacy/core_pipeline/interpretation_readiness_v5.py` | Legacy/internal | Yes |
| `legacy/core_pipeline/observation_scale_experiment_v4.py` | Legacy/internal | Yes |
| `legacy/core_pipeline/observation_scale_experiment_v5.py` | Legacy/internal | Yes |
| `legacy/diagnostics_and_audits/compare_feature_spaces_v1.py` | Legacy/internal | Yes |
| `legacy/diagnostics_and_audits/high_bp_diagnostic_v1.py` | Legacy/internal | Yes |
| `legacy/diagnostics_and_audits/serious_filter_v1.py` | Legacy/internal | Yes |
| `legacy/diagnostics_and_audits/top48_hgnc_alias_audit.py` | Legacy/internal | Yes |
| `legacy/dual_d_and_reconstruction/bp_enrichment_state_reconstruction_v1_0.py` | Legacy/internal | Yes |
| `legacy/dual_d_and_reconstruction/brca_state_validation_v2_1_complete.py` | Legacy/internal | Yes |
| `legacy/dual_d_and_reconstruction/brca_state_validation_v2_2_fast_resume.py` | Legacy/internal | Yes |
| `legacy/dual_d_and_reconstruction/dual_d_bp_enrichment_v1.py` | Legacy/internal | Yes |
| `legacy/dual_d_and_reconstruction/dual_d_from_scratch_v1_0_complete.py` | Legacy/internal | Yes |
| `legacy/dual_d_and_reconstruction/dual_d_full_pipeline_v1_0.py` | Legacy/internal | Yes |
| `legacy/dual_d_and_reconstruction/dual_d_full_pipeline_v1_1_fixed.py` | Legacy/internal | Yes |
| `legacy/dual_d_and_reconstruction/endpoint_bp_enrichment_v1_0.py` | Legacy/internal | Yes |
| `legacy/endpoint_stress_tests/gse96058_external_v2.py` | Legacy/internal | Yes |
| `legacy/endpoint_stress_tests/metabric_external_v2.py` | Legacy/internal | Yes |
| `legacy/endpoint_stress_tests/tcga_brca_er_status_esr1_excluded_v2.py` | Legacy/internal | Yes |
| `legacy/endpoint_stress_tests/tcga_brca_er_status_v1.py` | Legacy/internal | Yes |
| `legacy/endpoint_stress_tests/tcga_new_target_v1.py` | Legacy/internal | Yes |
| `scripts/run_core_analysis.py` | Production | No |
| `scripts/scoring_sensitivity_analysis.py` | Production | No |
| `src/aido_d_xobs/__init__.py` | Production | No |
| `src/aido_d_xobs/core.py` | Production | No |

## Status definitions

- **Production:** portable, documented entry points intended for reproducibility use.
- **Legacy/internal:** retained for provenance. These scripts may contain workstation-specific paths, older terminology, duplicated logic, or assumptions tied to the original analysis environment.

Legacy scripts should be reviewed and configured before execution. Their presence documents analysis development; it does not imply that every historical script is required to reproduce the final manuscript results.