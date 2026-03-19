param(
  [string]$RepoUrl = "https://github.com/kstecrypto-hub/gratsiasUi.git",
  [string]$Branch = "codex/create-ui-with-dark-theme-and-buttons",
  [string]$TargetDir = "$HOME\Downloads\gratsiasUi"
)

$ErrorActionPreference = "Stop"

function Test-Command($name) {
  return [bool](Get-Command $name -ErrorAction SilentlyContinue)
}

function Install-WithWinget($id, $displayName) {
  Write-Host "Installing $displayName with winget..."
  winget install --id $id -e --accept-package-agreements --accept-source-agreements
}

function Ensure-Git {
  if (Test-Command "git") {
    Write-Host "Git already installed: $(git --version)"
    return
  }

  if (Test-Command "winget") {
    Install-WithWinget -id "Git.Git" -displayName "Git"
  } else {
    throw "Git is missing and winget is unavailable. Install Git manually: https://git-scm.com/download/win"
  }

  if (-not (Test-Command "git")) {
    throw "Git installation completed but command not found in current shell. Close and re-open PowerShell, then rerun this script."
  }
}

function Ensure-Node {
  if (Test-Command "npm") {
    Write-Host "npm already installed: $(npm -v)"
    return
  }

  if (Test-Command "winget") {
    Install-WithWinget -id "OpenJS.NodeJS.LTS" -displayName "Node.js LTS"
  } else {
    throw "Node.js is missing and winget is unavailable. Install Node LTS manually: https://nodejs.org"
  }

  if (-not (Test-Command "npm")) {
    throw "Node installation completed but npm not found in current shell. Close and re-open PowerShell, then rerun this script."
  }
}

Write-Host "=== Step 1: Ensure Git and Node/npm are installed ==="
Ensure-Git
Ensure-Node

Write-Host "=== Step 2: Download/update repository ==="
if (Test-Path $TargetDir) {
  Write-Host "Repository directory exists: $TargetDir"
  Set-Location $TargetDir
  git fetch origin
  git checkout $Branch
  git pull origin $Branch
} else {
  $parent = Split-Path -Parent $TargetDir
  if (-not (Test-Path $parent)) {
    New-Item -ItemType Directory -Path $parent | Out-Null
  }
  Set-Location $parent
  git clone -b $Branch $RepoUrl gratsiasUi
  Set-Location $TargetDir
}

Write-Host "=== Step 3: Install dependencies ==="
npm install

Write-Host "=== Step 4: Launch UI ==="
Write-Host "Starting Vite dev server and opening browser..."
npm run dev -- --open
