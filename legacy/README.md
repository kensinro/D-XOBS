# Legacy and internal analysis scripts

This directory preserves historical scripts supplied during manuscript development.
They are grouped by purpose and renamed only for filesystem clarity.

Important:

- Several scripts contain Windows-specific paths such as `D:/AIDO-Temp`.
- Some use earlier internal names such as D-PHY and D-Clinical.
- Some are superseded by later versions or duplicate functions now exposed through the portable core pipeline.
- These files are retained for provenance and auditability, not as the primary public API.

Use `src/aido_d_xobs/core.py` and `scripts/scoring_sensitivity_analysis.py` for the documented manuscript-facing workflows.
See `docs/SCRIPT_INVENTORY.md` for the complete list.
