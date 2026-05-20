# 12-Month Data+Theory Research Roadmap

This roadmap turns the suite into a publication-oriented research workspace. The goal is not to promise a Nature paper, but to raise the codebase to a standard where claims are reproducible, data-linked, and suitable for serious manuscript development.

## Primary Track: Dark Energy Joint Inference

- Build around `cosmology/joint-dark-energy-inference`, `cosmology/h0-tension`, and `cosmology/cmb-compressed-likelihoods`.
- Pin exact DESI DR2 BAO, Pantheon+/SH0ES, Planck, and ACT products in each `data/data_manifest.json`.
- Reproduce LCDM and wCDM baseline constraints before adding CPL, early dark energy, or interacting dark energy.
- Treat every figure as generated output from scripts or notebooks, never as a manually edited final.

## Secondary Track: PTA Backgrounds

- Build around `gravitational-waves/pta-background` and `modified-gravity-dark-sector/cosmic-string-constraints`.
- Compare SMBH-binary-like power laws against cosmic-string and early-universe source proxies.
- Exchange QFT phase-transition spectra with the QFT suite through plain JSON result summaries.

## QFT Bridge

- Use `phase-transition-gw` in the QFT suite as the source of first-order phase-transition spectra.
- Keep QFT modules model-driven: parameter cards in, spectra and summary statistics out.

## Acceptance Criteria

- Every project has a no-data smoke test.
- Publication runs must record source URL, file version, checksum, and command line.
- Manuscript claims should cite public data releases and include a reproduction command.
