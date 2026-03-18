# build_all.ps1 - Build FlareSim.dll for multiple Nuke versions
#
# Usage:
#   .\build_all.ps1
#   .\build_all.ps1 -Versions 15,16       # build specific versions only
#   .\build_all.ps1 -NukeRoot "D:\Nuke"   # override install root
#   .\build_all.ps1 -DistDir "C:\deploy"  # override output directory
#
# Output: dist\nuke14\FlareSim.dll, dist\nuke15\FlareSim.dll, etc.

param(
    [int[]]  $Versions = @(14, 15, 16, 17),
    [string] $NukeRoot = "C:\Program Files",
    [string] $DistDir  = "$PSScriptRoot\dist"
)

$ErrorActionPreference = "Continue"
$scriptDir = $PSScriptRoot
$succeeded = @()
$failed    = @()
$skipped   = @()

Write-Host ""
Write-Host "FlareSim multi-version build" -ForegroundColor Cyan
Write-Host "Output: $DistDir"
Write-Host ""

foreach ($version in $Versions) {

    Write-Host "--- Nuke $version ---" -ForegroundColor Cyan

    # Find Nuke installation (e.g. "Nuke16.0v1") - pick newest patch if multiple
    $nukeDir = Get-ChildItem $NukeRoot -Directory -Filter "Nuke${version}.*" |
               Sort-Object Name -Descending |
               Select-Object -First 1

    if (-not $nukeDir) {
        Write-Host "  Nuke $version not found under $NukeRoot - skipping." -ForegroundColor Yellow
        $skipped += $version
        continue
    }

    $nukePath   = $nukeDir.FullName
    $ndkRoot    = Join-Path $nukePath "include"
    $nukeLibDir = $nukePath

    if (-not (Test-Path (Join-Path $ndkRoot "DDImage\Iop.h"))) {
        Write-Host "  NDK headers not found at $ndkRoot - skipping." -ForegroundColor Yellow
        $skipped += $version
        continue
    }

    Write-Host "  Found: $nukePath"

    # CMake configure
    $buildDir = Join-Path $scriptDir "build_nuke$version"

    Write-Host "  Configuring..." -ForegroundColor Gray

    $cmakeArgs = @(
        "-G", "Visual Studio 17 2022",
        "-A", "x64",
        "-DNDK_ROOT=$ndkRoot",
        "-DNUKE_LIB_DIR=$nukeLibDir",
        "-S", $scriptDir,
        "-B", $buildDir
    )
    cmake @cmakeArgs | Out-Null

    if ($LASTEXITCODE -ne 0) {
        Write-Host "  CMake configure FAILED (exit $LASTEXITCODE)" -ForegroundColor Red
        $failed += $version
        continue
    }

    # Build
    Write-Host "  Building..." -ForegroundColor Gray
    cmake --build $buildDir --config Release

    if ($LASTEXITCODE -ne 0) {
        Write-Host "  Build FAILED (exit $LASTEXITCODE)" -ForegroundColor Red
        $failed += $version
        continue
    }

    # Copy output
    $dllSrc = Join-Path $buildDir "Release\FlareSim.dll"

    if (-not (Test-Path $dllSrc)) {
        Write-Host "  FlareSim.dll not found - build may have failed." -ForegroundColor Red
        $failed += $version
        continue
    }

    $outDir = Join-Path $DistDir "nuke$version"
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null
    Copy-Item $dllSrc $outDir -Force

    Write-Host "  OK -> $outDir\FlareSim.dll" -ForegroundColor Green
    $succeeded += $version
}

# Summary
Write-Host ""
Write-Host "--- Summary ---" -ForegroundColor Cyan
if ($succeeded.Count -gt 0) { Write-Host "  Built:   Nuke $($succeeded -join ', ')" -ForegroundColor Green  }
if ($skipped.Count   -gt 0) { Write-Host "  Skipped: Nuke $($skipped   -join ', ')" -ForegroundColor Yellow }
if ($failed.Count    -gt 0) { Write-Host "  Failed:  Nuke $($failed    -join ', ')" -ForegroundColor Red    }
Write-Host ""

if ($failed.Count -gt 0) { exit 1 } else { exit 0 }
