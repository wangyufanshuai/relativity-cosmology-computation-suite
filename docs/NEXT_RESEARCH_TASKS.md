# Next Research Tasks

This file converts the roadmap into concrete engineering tasks. Complete these in order; later tasks depend on earlier data provenance and baseline reproduction. Version 1 now includes local loaders, toy baseline inputs, and reproducible `results/*.json` outputs.

## Done In V1

- `cosmology/joint-dark-energy-inference`: JSON/CSV loaders, LCDM/wCDM/CPL grids, AIC/BIC reports, and `baseline_summary.json`.
- `cosmology/h0-tension`: constraint JSON schema and local/early/standard-siren tension reports.
- `gravitational-waves/pta-background`: binned-spectrum loader and source-ranking interface.
- `gravitational-waves/standard-sirens`: posterior-summary loader and H0 baseline output.

## Phase 1: Replace Toy Inputs With Pinned Public Data

- [ ] `joint-dark-energy-inference`: add exact DESI DR2 BAO tables or chain summaries.
- [ ] `joint-dark-energy-inference`: add Pantheon+/SH0ES SN summary data with covariance notes.
- [ ] `cmb-compressed-likelihoods`: replace placeholder CMB priors with cited Planck/ACT compressed values.
- [ ] `h0-tension`: record source DOI/URL, date accessed, and covariance treatment for each H0 input.
- [ ] `pta-background`: pin NANOGrav 15-year spectral summaries or posterior products.
- [ ] `standard-sirens`: pin GWOSC/LVK event metadata and posterior-summary files.

## Phase 2: Baseline Reproduction

- [ ] Reproduce LCDM distance predictions before fitting extended models.
- [ ] Match published or release-note baseline constraints within documented tolerance.
- [ ] Add `results/baseline_summary.json` for every completed public-data run.
- [ ] Save command line, input files, and checksum metadata beside each baseline result.

## Phase 3: Research Extensions

- [x] Add wCDM and CPL model grids to `joint-dark-energy-inference`.
- [x] Add AIC/BIC model-selection reports.
- [ ] Add delta chi2 and optional nested-sampling evidence.
- [x] Add PTA source comparison for SMBH-like, cosmic-string-like, and phase-transition-like sources.
- [ ] Add full standard-siren posterior combination once posterior adapters are stable.

## Phase 4: Manuscript Assets

- [ ] Generate all figures from scripts or notebooks.
- [x] Store baseline numbers in `results/*.json`.
- [ ] Keep final figure files in `figures/` and include the generation command in manuscript notes.

## Cross-Repository Contract

QFT-side phase-transition outputs should be exported as JSON:

```json
{
  "model": "model-card-name",
  "alpha": 0.1,
  "beta_over_h": 100.0,
  "temperature_gev": 100.0,
  "frequency_hz": [1e-9, 1e-8],
  "omega_gw": [1e-16, 1e-14]
}
```

The PTA pipeline should consume this JSON as an external source model, not import QFT package internals.
