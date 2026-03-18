# package_release.ps1 - Package FlareSim for release
#
# Creates one zip per Nuke version under release_packages\:
#   FlareSim_v<version>_Nuke<N>.zip
#
# Each zip is ready to unpack into %USERPROFILE%\.nuke\plugins\
#
# Usage:
#   .\package_release.ps1
#   .\package_release.ps1 -Version "1.0.0"
#   .\package_release.ps1 -NukeVersions 16,17

param(
    [string] $Version      = "1.0.0",
    [int[]]  $NukeVersions = @(14, 15, 16, 17),
    [string] $DistDir      = "$PSScriptRoot\dist",
    [string] $OutDir       = "$PSScriptRoot\release_packages"
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
Write-Host ""
Write-Host "FlareSim release packager  v$Version" -ForegroundColor Cyan
Write-Host "Output: $OutDir"
Write-Host ""

foreach ($nv in $NukeVersions) {

    $dll = Join-Path $DistDir "nuke$nv\FlareSim.dll"
    if (-not (Test-Path $dll)) {
        Write-Host "  Nuke $nv : FlareSim.dll not found at $dll - skipping." -ForegroundColor Yellow
        continue
    }

    $zipName = "FlareSim_v${Version}_Nuke${nv}.zip"
    $zipPath = Join-Path $OutDir $zipName

    # Build a temp staging folder
    $stage = Join-Path $env:TEMP "flaresim_stage_nuke$nv"
    if (Test-Path $stage) { Remove-Item $stage -Recurse -Force }
    New-Item -ItemType Directory -Force -Path $stage | Out-Null

    # --- Core plugin files ---
    Copy-Item $dll                               (Join-Path $stage "FlareSim.dll")
    Copy-Item (Join-Path $root "menu.py")        (Join-Path $stage "menu.py")
    Copy-Item (Join-Path $root "FlareSim_LensBrowser.py") (Join-Path $stage "FlareSim_LensBrowser.py")

    # --- Lens library ---
    $lensOut = Join-Path $stage "lenses"
    New-Item -ItemType Directory -Force -Path $lensOut | Out-Null
    Copy-Item (Join-Path $root "lenses\lens_files") $lensOut -Recurse

    # --- Converters (useful for advanced users) ---
    Copy-Item (Join-Path $root "lenses\convert_ob.py")  $lensOut
    Copy-Item (Join-Path $root "lenses\convert_p2p.py") $lensOut

    # --- Zip it ---
    if (Test-Path $zipPath) { Remove-Item $zipPath -Force }
    Compress-Archive -Path "$stage\*" -DestinationPath $zipPath

    $sizeMB = [math]::Round((Get-Item $zipPath).Length / 1MB, 1)
    Write-Host "  Nuke $nv : $zipName  ($sizeMB MB)" -ForegroundColor Green

    Remove-Item $stage -Recurse -Force
}

Write-Host ""
Write-Host "Done. Upload the zips from $OutDir as GitHub Release assets." -ForegroundColor Cyan
Write-Host ""
