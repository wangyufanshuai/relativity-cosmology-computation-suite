# Relativity Cosmology Computation Suite

A monorepo of Python computation projects for general relativity, black-hole physics, cosmology, gravitational-wave numerics, modified gravity, and dark-sector models.

This repository currently contains 52 independently installable Python projects. Each project keeps its own `pyproject.toml`, `src/` package layout, and pytest suite.

## Repository Layout

```text
relativity-black-hole/              # GR tests, black holes, geodesics, Kerr/Schwarzschild tools
cosmology/                          # background cosmology, CMB, large-scale structure, inflation
gravitational-waves/                # numerical relativity, waveforms, compact objects, GRMHD
modified-gravity-dark-sector/       # modified gravity, dark matter, dark energy, fifth-force models
docs/                               # project index and testing notes
scripts/                            # local test helpers
```

## Quick Start

Install and test one project:

```bash
cd relativity-black-hole/mercury-precession
python -m pip install pytest
python -m pip install -e .
python -m pytest tests -q
```

From the repository root on Windows, run a project by path:

```powershell
.\scripts\test-project.ps1 relativity-black-hole\mercury-precession
```

Run the smoke test set:

```powershell
.\scripts\test-smoke.ps1
```

## CI Policy

CI runs a smoke matrix instead of all 52 projects. This keeps the first public repository stable and fast while still validating representative projects from each domain.

Representative smoke projects:

- `relativity-black-hole/mercury-precession`
- `relativity-black-hole/kerr-qnm`
- `cosmology/friedmann-solver`
- `cosmology/cmb-power-spectrum`
- `gravitational-waves/pn-waveform`
- `gravitational-waves/bssn-solver`
- `modified-gravity-dark-sector/fR-gravity`
- `modified-gravity-dark-sector/quintessence`

## Project Index

See [docs/PROJECT_INDEX.md](docs/PROJECT_INDEX.md).

## Deferred Projects

The following planned projects were intentionally not included in this first repository state:

- `reheating-simulator`: directory existed locally but had no source files, tests, or `pyproject.toml`.
- `h0-tension`: directory existed locally but had no source files, tests, or `pyproject.toml`.
- `spectral-distortions`: directory was not present in the local workspace.

## License

MIT
