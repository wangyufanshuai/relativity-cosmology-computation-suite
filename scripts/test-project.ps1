param(
  [Parameter(Mandatory = $true)]
  [string]$ProjectPath
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$project = Join-Path $root $ProjectPath

if (!(Test-Path (Join-Path $project "pyproject.toml"))) {
  throw "No pyproject.toml found at $project"
}

Push-Location $project
try {
  python -m pip install pytest
  python -m pip install -e .
  python -m pytest tests -q
}
finally {
  Pop-Location
}
