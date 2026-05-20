# Joint Dark Energy Inference

Research pipeline scaffold for DESI BAO + Pantheon+/SH0ES supernovae + compressed CMB constraints.

The package starts with deterministic, ordinary-PC smoke models: flat LCDM, wCDM, and CPL distances; Gaussian likelihood blocks; and grid-search fitting. Replace the toy data in `data/data_manifest.json` with downloaded public products before using it for publication-grade claims.

## Reproducible Shape

- `data/data_manifest.json`: versioned public data-source checklist.
- `scripts/download_data.py`: dry-run downloader that records URLs and local targets.
- `src/joint_dark_energy_inference`: theory, likelihood, and fitting primitives.
- `tests`: smoke tests that run without external data.
- `results/`, `figures/`, `notebooks/`: reserved publication artifact directories.

## Smoke Test

```powershell
python -m pip install -e . pytest
python -m pytest tests -q
```
