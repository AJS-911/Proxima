#!/usr/bin/env pwsh
# =============================================================================
# Proxima Agent - PowerShell Build Script for Windows
# =============================================================================
#
# Usage:
#   .\scripts\build.ps1 [Command] [-Coverage] [-Verbose] [-Version <version>]
#
# Commands:
#   clean      - Clean build artifacts
#   lint       - Run linting checks
#   format     - Format code
#   typecheck  - Run type checking
#   test       - Run tests
#   build      - Build the package
#   docker     - Build Docker image
#   docs       - Build documentation
#   release    - Prepare a release
#   all        - Run all checks and build
#
# Examples:
#   .\scripts\build.ps1 build
#   .\scripts\build.ps1 test -Coverage
#   .\scripts\build.ps1 release -Version "0.1.0"
#
# =============================================================================

param(
    [Parameter(Position = 0)]
    [ValidateSet("clean", "lint", "format", "typecheck", "test", "build", "docker", "docs", "release", "all")]
    [string]$Command = "help",
    
    [switch]$Coverage,
    [switch]$Push,
    [string]$Tag = "latest",
    [string]$Version,
    [switch]$NoDryRun
)

# Configuration
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$SrcDir = Join-Path $ProjectRoot "src"
$TestsDir = Join-Path $ProjectRoot "tests"
$DistDir = Join-Path $ProjectRoot "dist"
$BuildDir = Join-Path $ProjectRoot "build"

# =============================================================================
# Helper Functions
# =============================================================================

function Write-Step {
    param([string]$Message)
    Write-Host "`n=== $Message ===`n" -ForegroundColor Cyan
}

function Write-SubStep {
    param([string]$Message)
    Write-Host "`n--- $Message ---`n" -ForegroundColor Yellow
}

function Invoke-SafeCommand {
    param(
        [scriptblock]$ScriptBlock,
        [string]$ErrorMessage = "Command failed"
    )
    
    try {
        & $ScriptBlock
        if ($LASTEXITCODE -and $LASTEXITCODE -ne 0) {
            throw $ErrorMessage
        }
    }
    catch {
        Write-Host "Error: $_" -ForegroundColor Red
        exit 1
    }
}

# =============================================================================
# Commands
# =============================================================================

function Invoke-Clean {
    Write-Step "Cleaning build artifacts"
    
    $dirsToRemove = @(
        $DistDir,
        $BuildDir,
        (Join-Path $ProjectRoot ".pytest_cache"),
        (Join-Path $ProjectRoot ".mypy_cache"),
        (Join-Path $ProjectRoot ".ruff_cache"),
        (Join-Path $ProjectRoot "htmlcov"),
        (Join-Path $ProjectRoot ".coverage"),
        (Join-Path $ProjectRoot "site")
    )
    
    foreach ($dir in $dirsToRemove) {
        if (Test-Path $dir) {
            Write-Host "Removing: $dir"
            Remove-Item -Recurse -Force $dir
        }
    }
    
    # Clean egg-info directories
    Get-ChildItem -Path $ProjectRoot -Filter "*.egg-info" -Directory -Recurse | ForEach-Object {
        Write-Host "Removing: $($_.FullName)"
        Remove-Item -Recurse -Force $_.FullName
    }
    
    # Clean __pycache__ directories
    Get-ChildItem -Path $ProjectRoot -Filter "__pycache__" -Directory -Recurse | ForEach-Object {
        Write-Host "Removing: $($_.FullName)"
        Remove-Item -Recurse -Force $_.FullName
    }
    
    Write-Host "`nClean complete!" -ForegroundColor Green
}

function Invoke-Lint {
    Write-Step "Running linting"
    
    Write-SubStep "Ruff"
    Invoke-SafeCommand { ruff check src/ tests/ }
    
    Write-SubStep "Black"
    Invoke-SafeCommand { black --check src/ tests/ }
    
    Write-Host "`nLinting complete!" -ForegroundColor Green
}

function Invoke-Format {
    Write-Step "Formatting code"
    
    Write-SubStep "Black"
    Invoke-SafeCommand { black src/ tests/ }
    
    Write-SubStep "Ruff fix"
    Invoke-SafeCommand { ruff check --fix src/ tests/ }
    
    Write-Host "`nFormatting complete!" -ForegroundColor Green
}

function Invoke-TypeCheck {
    Write-Step "Running type checking"
    
    Invoke-SafeCommand { mypy src/ --ignore-missing-imports }
    
    Write-Host "`nType checking complete!" -ForegroundColor Green
}

function Invoke-Test {
    Write-Step "Running tests"
    
    $testArgs = @("tests/")
    
    if ($VerbosePreference) {
        $testArgs += "-v"
    }
    
    if ($Coverage) {
        $testArgs += @(
            "--cov=proxima",
            "--cov-report=term-missing",
            "--cov-report=html"
        )
    }
    
    Invoke-SafeCommand { pytest @testArgs }
    
    Write-Host "`nTests complete!" -ForegroundColor Green
}

function Invoke-Build {
    Write-Step "Building package"
    
    # Clean first
    Invoke-Clean
    
    # Install build dependencies
    Invoke-SafeCommand { python -m pip install build twine }
    
    # Build
    Invoke-SafeCommand { python -m build }
    
    # Check
    Write-SubStep "Checking package"
    Invoke-SafeCommand { twine check dist/* }
    
    Write-Host "`nBuild complete! Packages in: $DistDir" -ForegroundColor Green
}

function Invoke-Docker {
    Write-Step "Building Docker image"
    
    $imageName = "proxima-agent:$Tag"
    
    Invoke-SafeCommand {
        docker build -t $imageName --target runtime .
    }
    
    Write-Host "`nDocker image built: $imageName" -ForegroundColor Green
    
    if ($Push) {
        Write-SubStep "Pushing image"
        Invoke-SafeCommand { docker push $imageName }
    }
}

function Invoke-Docs {
    Write-Step "Building documentation"
    
    Invoke-SafeCommand { mkdocs build --strict }
    
    Write-Host "`nDocumentation built in: $(Join-Path $ProjectRoot 'site')" -ForegroundColor Green
}

function Invoke-Release {
    if (-not $Version) {
        Write-Host "Error: Version is required. Use -Version <version>" -ForegroundColor Red
        exit 1
    }
    
    Write-Step "Preparing release v$Version"
    
    $dryRun = -not $NoDryRun
    if ($dryRun) {
        Write-Host "DRY RUN - No changes will be made`n" -ForegroundColor Yellow
    }
    
    # 1. Run all checks
    Write-Host "Step 1: Running quality checks..."
    Invoke-Lint
    Invoke-TypeCheck
    $script:Coverage = $true
    Invoke-Test
    
    # 2. Update version
    Write-Host "`nStep 2: Updating version to $Version..."
    $pyprojectPath = Join-Path $ProjectRoot "pyproject.toml"
    
    if (-not $dryRun) {
        $content = Get-Content $pyprojectPath -Raw
        $newContent = $content -replace 'version = "[^"]*"', "version = `"$Version`""
        Set-Content -Path $pyprojectPath -Value $newContent
        Write-Host "Updated pyproject.toml"
    }
    else {
        Write-Host "Would update pyproject.toml"
    }
    
    # 3. Build package
    Write-Host "`nStep 3: Building package..."
    if (-not $dryRun) {
        Invoke-Build
    }
    else {
        Write-Host "Would build package"
    }
    
    # 4. Create git tag
    Write-Host "`nStep 4: Creating git tag v$Version..."
    if (-not $dryRun) {
        git add pyproject.toml
        git commit -m "Release v$Version"
        git tag -a "v$Version" -m "Release v$Version"
        Write-Host "Created tag v$Version"
        Write-Host "`nTo push the release:"
        Write-Host "  git push origin main"
        Write-Host "  git push origin v$Version"
    }
    else {
        Write-Host "Would create tag v$Version"
    }
    
    Write-Host "`nRelease preparation complete!" -ForegroundColor Green
}

function Invoke-All {
    Write-Step "Running all"
    
    Invoke-Clean
    Invoke-Lint
    Invoke-TypeCheck
    $script:Coverage = $true
    Invoke-Test
    Invoke-Build
    Invoke-Docs
    
    Write-Host "`nAll complete!" -ForegroundColor Green
}

function Show-Help {
    Write-Host @"

Proxima Agent Build Script
==========================

Usage:
  .\scripts\build.ps1 [Command] [Options]

Commands:
  clean      Clean build artifacts
  lint       Run linting checks
  format     Format code with black and ruff
  typecheck  Run type checking with mypy
  test       Run tests with pytest
  build      Build the Python package
  docker     Build Docker image
  docs       Build documentation
  release    Prepare a release
  all        Run all checks and build

Options:
  -Coverage     Enable coverage for tests
  -Push         Push Docker image after building
  -Tag          Docker image tag (default: latest)
  -Version      Version for release
  -NoDryRun     Actually make changes for release

Examples:
  .\scripts\build.ps1 build
  .\scripts\build.ps1 test -Coverage
  .\scripts\build.ps1 docker -Tag "0.1.0" -Push
  .\scripts\build.ps1 release -Version "0.1.0" -NoDryRun

"@
}

# =============================================================================
# Main
# =============================================================================

Push-Location $ProjectRoot

try {
    switch ($Command) {
        "clean"     { Invoke-Clean }
        "lint"      { Invoke-Lint }
        "format"    { Invoke-Format }
        "typecheck" { Invoke-TypeCheck }
        "test"      { Invoke-Test }
        "build"     { Invoke-Build }
        "docker"    { Invoke-Docker }
        "docs"      { Invoke-Docs }
        "release"   { Invoke-Release }
        "all"       { Invoke-All }
        default     { Show-Help }
    }
}
finally {
    Pop-Location
}
