<#
.SYNOPSIS
    Proxima Installation Script for Windows
    
.DESCRIPTION
    This script handles the installation of Proxima on Windows systems,
    automatically detecting and configuring for any version of Visual Studio.
    
    It solves the common "Visual Studio 15 2017" compatibility issue by:
    1. Detecting installed Visual Studio versions
    2. Configuring the build environment appropriately
    3. Using pre-built wheels when available
    4. Falling back to source builds only when necessary
    
.PARAMETER InstallMode
    The installation mode:
    - "full" (default): Install all dependencies including quantum backends
    - "minimal": Install core dependencies only
    - "dev": Install with development dependencies
    
.PARAMETER PreferBinary
    If specified, strongly prefer pre-built wheels over source builds.
    This avoids most Visual Studio compatibility issues.
    
.EXAMPLE
    .\install-windows.ps1
    
.EXAMPLE
    .\install-windows.ps1 -InstallMode dev -PreferBinary
#>

param(
    [ValidateSet("full", "minimal", "dev")]
    [string]$InstallMode = "full",
    
    [switch]$PreferBinary = $false,
    
    [switch]$SkipVSCheck = $false
)

$ErrorActionPreference = "Stop"

# Colors for output
function Write-Header {
    param([string]$Message)
    Write-Host ""
    Write-Host ("=" * 70) -ForegroundColor Cyan
    Write-Host $Message -ForegroundColor Cyan
    Write-Host ("=" * 70) -ForegroundColor Cyan
}

function Write-Step {
    param([string]$Message)
    Write-Host "[>] $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[!] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[X] $Message" -ForegroundColor Red
}

function Write-Info {
    param([string]$Message)
    Write-Host "    $Message" -ForegroundColor Gray
}

# Detect Visual Studio installations
function Get-VisualStudioVersions {
    $vsInstalls = @()
    
    # Try vswhere.exe first
    $vswhereLocations = @(
        "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe",
        "${env:ProgramFiles}\Microsoft Visual Studio\Installer\vswhere.exe"
    )
    
    foreach ($vswhere in $vswhereLocations) {
        if (Test-Path $vswhere) {
            try {
                $result = & $vswhere -all -prerelease -format json 2>$null
                if ($result) {
                    $installations = $result | ConvertFrom-Json
                    foreach ($install in $installations) {
                        $vsInstalls += [PSCustomObject]@{
                            Version = $install.displayVersion
                            Path = $install.installationPath
                            Year = switch -Regex ($install.displayVersion) {
                                "^17\." { 2022 }
                                "^16\." { 2019 }
                                "^15\." { 2017 }
                                "^14\." { 2015 }
                                default { "Unknown" }
                            }
                        }
                    }
                }
            } catch {
                # vswhere failed, continue with other detection methods
            }
            break
        }
    }
    
    # Also check common paths
    $commonPaths = @(
        @{ Year = 2022; Editions = @("Enterprise", "Professional", "Community", "BuildTools") },
        @{ Year = 2019; Editions = @("Enterprise", "Professional", "Community", "BuildTools") },
        @{ Year = 2017; Editions = @("Enterprise", "Professional", "Community", "BuildTools") }
    )
    
    foreach ($vsInfo in $commonPaths) {
        foreach ($edition in $vsInfo.Editions) {
            $path = "${env:ProgramFiles}\Microsoft Visual Studio\$($vsInfo.Year)\$edition"
            $path86 = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\$($vsInfo.Year)\$edition"
            
            foreach ($p in @($path, $path86)) {
                if ((Test-Path $p) -and ($vsInstalls | Where-Object { $_.Path -eq $p }).Count -eq 0) {
                    $vsInstalls += [PSCustomObject]@{
                        Version = "$($vsInfo.Year) $edition"
                        Path = $p
                        Year = $vsInfo.Year
                    }
                }
            }
        }
    }
    
    return $vsInstalls
}

# Configure environment for Visual Studio
function Set-VSBuildEnvironment {
    param([object]$VSInstall)
    
    Write-Step "Configuring build environment for Visual Studio $($VSInstall.Year)..."
    
    # Set environment variables for setuptools/distutils
    $env:DISTUTILS_USE_SDK = "1"
    $env:MSSdk = "1"
    
    # Set CMAKE generator based on VS version
    $generators = @{
        2022 = "Visual Studio 17 2022"
        2019 = "Visual Studio 16 2019"
        2017 = "Visual Studio 15 2017"
        2015 = "Visual Studio 14 2015"
    }
    
    if ($generators.ContainsKey($VSInstall.Year)) {
        $env:CMAKE_GENERATOR = $generators[$VSInstall.Year]
        Write-Info "CMAKE_GENERATOR = $($generators[$VSInstall.Year])"
    } else {
        # For future VS versions, use the latest known generator
        $env:CMAKE_GENERATOR = "Visual Studio 17 2022"
        Write-Info "Using Visual Studio 17 2022 generator for VS $($VSInstall.Year)"
    }
    
    # Prefer binary wheels
    $env:PIP_PREFER_BINARY = "1"
}

# Main installation logic
function Install-Proxima {
    Write-Header "Proxima Windows Installation Script"
    
    Write-Host "Installation Mode: $InstallMode" -ForegroundColor White
    Write-Host "Prefer Binary: $PreferBinary" -ForegroundColor White
    Write-Host ""
    
    # Step 1: Check Python
    Write-Step "Checking Python installation..."
    try {
        $pythonVersion = python --version 2>&1
        Write-Info "Found: $pythonVersion"
        
        # Check version is 3.11+
        if ($pythonVersion -match "Python (\d+)\.(\d+)") {
            $major = [int]$Matches[1]
            $minor = [int]$Matches[2]
            if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 11)) {
                Write-Error "Python 3.11+ is required. Found: $pythonVersion"
                exit 1
            }
        }
    } catch {
        Write-Error "Python not found. Please install Python 3.11+ and add it to PATH."
        exit 1
    }
    
    # Step 2: Check Visual Studio (unless skipped)
    if (-not $SkipVSCheck) {
        Write-Step "Detecting Visual Studio installations..."
        $vsInstalls = Get-VisualStudioVersions
        
        if ($vsInstalls.Count -eq 0) {
            Write-Warning "No Visual Studio installation detected!"
            Write-Info "Using pre-built wheels strategy..."
            $env:PIP_ONLY_BINARY = ":all:"
            $PreferBinary = $true
        } else {
            Write-Info "Found $($vsInstalls.Count) Visual Studio installation(s):"
            foreach ($vs in $vsInstalls) {
                Write-Info "  - VS $($vs.Year): $($vs.Path)"
            }
            
            # Use the newest version
            $newestVS = $vsInstalls | Sort-Object Year -Descending | Select-Object -First 1
            Set-VSBuildEnvironment -VSInstall $newestVS
        }
    }
    
    # Step 3: Upgrade pip and setuptools
    Write-Step "Upgrading pip and setuptools..."
    python -m pip install --upgrade pip setuptools wheel --quiet
    
    # Step 4: Run the Python configuration script
    Write-Step "Running build configuration..."
    $scriptPath = Join-Path $PSScriptRoot "configure_build.py"
    if (Test-Path $scriptPath) {
        python $scriptPath
    }
    
    # Step 5: Install Proxima
    Write-Step "Installing Proxima..."
    
    $installCommand = @("python", "-m", "pip", "install", "-e")
    
    switch ($InstallMode) {
        "full" { $installCommand += ".[all]" }
        "minimal" { $installCommand += "." }
        "dev" { $installCommand += ".[dev]" }
    }
    
    if ($PreferBinary) {
        $installCommand += "--prefer-binary"
    }
    
    Write-Info "Running: $($installCommand -join ' ')"
    
    try {
        & $installCommand[0] $installCommand[1..($installCommand.Count-1)]
        
        if ($LASTEXITCODE -eq 0) {
            Write-Header "Installation Complete!"
            Write-Host ""
            Write-Host "Proxima has been installed successfully." -ForegroundColor Green
            Write-Host ""
            Write-Host "Try it out:" -ForegroundColor White
            Write-Host "  proxima --version" -ForegroundColor Gray
            Write-Host "  proxima backends list" -ForegroundColor Gray
            Write-Host "  proxima run --backend cirq 'bell state'" -ForegroundColor Gray
            Write-Host ""
        } else {
            throw "pip install failed with exit code $LASTEXITCODE"
        }
    } catch {
        Write-Error "Installation failed: $_"
        Write-Host ""
        Write-Warning "Attempting fallback installation with binary-only packages..."
        
        # Fallback: Install with binary-only
        $env:PIP_ONLY_BINARY = ":all:"
        try {
            python -m pip install -e ".$([char]91)all$([char]93)" --prefer-binary --only-binary :all:
        } catch {
            Write-Error "Fallback installation also failed."
            Write-Host ""
            Write-Host "Please try:" -ForegroundColor Yellow
            Write-Host "  1. Install Visual Studio with C++ workload" -ForegroundColor Gray
            Write-Host "  2. Run this script again" -ForegroundColor Gray
            Write-Host "  3. Or manually install: pip install -e . --no-binary qiskit-aer" -ForegroundColor Gray
            exit 1
        }
    }
}

# Run the installation
Install-Proxima
