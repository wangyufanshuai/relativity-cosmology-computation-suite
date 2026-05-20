# Testing

Each project is independently installable and testable.

## Test One Project

```powershell
.\scripts\test-project.ps1 relativity-black-hole\mercury-precession
```

The script installs `pytest`, installs the project in editable mode, then runs `pytest tests -q`.

## Smoke Test

```powershell
.\scripts\test-smoke.ps1
```

The smoke suite covers representative projects from all four top-level domains.

## Full Test Strategy

The repository contains many independent scientific packages. Full test runs should be staged rather than forced into every push:

1. Smoke tests on every push.
2. Full matrix manually or on release branches.
3. Numerical-warning cleanup tracked as normal issues.

Known current warning:

- `cosmology/cmb-power-spectrum`: Fisher error calculation can emit `RuntimeWarning: invalid value encountered in sqrt` in one test path.
