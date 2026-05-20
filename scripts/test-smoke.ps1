Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projects = @(
  "relativity-black-hole\mercury-precession",
  "relativity-black-hole\kerr-qnm",
  "cosmology\friedmann-solver",
  "cosmology\cmb-power-spectrum",
  "gravitational-waves\pn-waveform",
  "gravitational-waves\bssn-solver",
  "modified-gravity-dark-sector\fR-gravity",
  "modified-gravity-dark-sector\quintessence",
  "cosmology\joint-dark-energy-inference",
  "cosmology\h0-tension",
  "cosmology\cmb-compressed-likelihoods",
  "gravitational-waves\pta-background",
  "gravitational-waves\standard-sirens",
  "modified-gravity-dark-sector\cosmic-string-constraints"
)

foreach ($project in $projects) {
  Write-Host "==> $project"
  & "$PSScriptRoot\test-project.ps1" $project
}
