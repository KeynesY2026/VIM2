param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot

& $Python -c "import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] < (3, 14) else 1)"
if ($LASTEXITCODE -ne 0) {
    throw "Python 3.10 through 3.13 is required."
}

& $Python -c "import bitsandbytes, numpy, PySide6, qwen_asr, sherpa_onnx, sounddevice, soundfile, torch, transformers"
if ($LASTEXITCODE -ne 0) {
    throw "Required global Python packages are missing. See requirements.lock."
}

function Test-ModelComplete {
    param([string]$Path)
    $Metadata = @(
        "config.json",
        "preprocessor_config.json",
        "tokenizer_config.json"
    )
    foreach ($Name in $Metadata) {
        if (-not (Test-Path -LiteralPath (Join-Path $Path $Name))) {
            return $false
        }
    }
    $SingleWeight = Join-Path $Path "model.safetensors"
    $IndexPath = Join-Path $Path "model.safetensors.index.json"
    if (Test-Path -LiteralPath $SingleWeight) {
        return $true
    }
    if (-not (Test-Path -LiteralPath $IndexPath)) {
        return $false
    }
    $Index = Get-Content -LiteralPath $IndexPath -Raw | ConvertFrom-Json
    foreach ($Weight in $Index.weight_map.PSObject.Properties.Value |
        Select-Object -Unique) {
        if (-not (Test-Path -LiteralPath (Join-Path $Path $Weight))) {
            return $false
        }
    }
    return $true
}

function Copy-ModelSnapshot {
    param(
        [string]$CacheName,
        [string]$Revision,
        [string]$DestinationName
    )
    $Destination = Join-Path $Root ".models\$DestinationName"
    if (Test-ModelComplete -Path $Destination) {
        Write-Host "$DestinationName already exists and is complete; skipping."
        return
    }
    if (Test-Path -LiteralPath $Destination) {
        throw "Model directory exists but is incomplete: $Destination"
    }
    $Source = Join-Path $env:USERPROFILE `
        ".cache\huggingface\hub\$CacheName\snapshots\$Revision"
    if (-not (Test-ModelComplete -Path $Source)) {
        throw "Complete cached model snapshot was not found: $Source"
    }
    Copy-Item -LiteralPath $Source -Destination $Destination -Recurse
    if (-not (Test-ModelComplete -Path $Destination)) {
        throw "Copied model failed completeness validation: $Destination"
    }
}

function Test-SherpaModelComplete {
    param([string]$Path)
    $Files = @(
        "conv_frontend.onnx",
        "encoder.int8.onnx",
        "decoder.int8.onnx",
        "tokenizer\merges.txt",
        "tokenizer\tokenizer_config.json",
        "tokenizer\vocab.json"
    )
    foreach ($Name in $Files) {
        if (-not (Test-Path -LiteralPath (Join-Path $Path $Name))) {
            return $false
        }
    }
    return $true
}

New-Item -ItemType Directory -Force -Path (Join-Path $Root ".models") |
    Out-Null
Copy-ModelSnapshot `
    -CacheName "models--Qwen--Qwen3-ASR-0.6B" `
    -Revision "5eb144179a02acc5e5ba31e748d22b0cf3e303b0" `
    -DestinationName "Qwen3-ASR-0.6B"
Copy-ModelSnapshot `
    -CacheName "models--Qwen--Qwen3-ASR-1.7B" `
    -Revision "7278e1e70fe206f11671096ffdd38061171dd6e5" `
    -DestinationName "Qwen3-ASR-1.7B-INT8"
$SherpaModel = Join-Path $Root `
    ".models\sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25"
if (-not (Test-SherpaModelComplete -Path $SherpaModel)) {
    throw "Complete sherpa-onnx CPU model was not found: $SherpaModel"
}

$FilesToHash = Get-ChildItem -LiteralPath (Join-Path $Root ".models") `
    -File -Recurse
$Hashes = [ordered]@{}
foreach ($File in $FilesToHash) {
    $Relative = $File.FullName.Substring($Root.Length).TrimStart("\")
    $Hashes[$Relative] = (Get-FileHash -LiteralPath $File.FullName `
        -Algorithm SHA256).Hash.ToLowerInvariant()
}
$HashPath = Join-Path $Root "release-files.sha256.json"
$Utf8NoBom = New-Object System.Text.UTF8Encoding($false)
$HashJson = ($Hashes | ConvertTo-Json) + [Environment]::NewLine
[IO.File]::WriteAllText($HashPath, $HashJson, $Utf8NoBom)
Write-Host "Release preparation completed."
