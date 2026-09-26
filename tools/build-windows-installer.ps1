# Build the unsigned Windows x64 CPU installer.
# Bind the complete argv with -Arguments. Do not splat an array into Invoke-Python.
[CmdletBinding()]
param(
    [switch] $SelfTest
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$env:PYTHONPATH = Join-Path $Root 'app'
$env:QT_QPA_PLATFORM = 'offscreen'
$env:PYTHONDONTWRITEBYTECODE = '1'

$PythonCommands = @(
    ,@('-m','pip','install','-r','requirements-windows-cpu.lock','-r','requirements-build.lock','pytest==9.1.1')
    ,@('tools/download-cpu-model.py')
    ,@('tools/verify-installer-layout.py','--source')
    ,@('-m','pytest','-q','-p','no:cacheprovider')
    ,@('-m','vim2','--root',$Root,'--check')
    ,@('-m','PyInstaller','--noconfirm','--clean','--distpath','dist/build','--workpath','build/pyinstaller','packaging/vim2.spec')
    ,@('tools/verify-installer-layout.py','dist/build/VIM2')
)

function Invoke-Python {
    param(
        [Parameter(Mandatory = $true)]
        [string[]] $Arguments
    )
    if ($null -eq $Arguments -or $Arguments.Count -lt 1) {
        throw 'Python argv was not bound'
    }
    & python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Python failed: $($Arguments -join ' ')" }
}

function Test-BoundArguments {
    param(
        [Parameter(Mandatory = $true)]
        [string[]] $Arguments
    )
    if ($Arguments.Count -lt 1) { throw 'Python argv was not bound' }
    $script:LastBound = $Arguments -join '|'
}

if ($SelfTest) {
    foreach ($argv in $PythonCommands) {
        $expected = (@($argv) -join '|')
        Test-BoundArguments -Arguments @($argv)
        if ($script:LastBound -ne $expected) {
            throw "Argument binding mismatch: [$($script:LastBound)] != [$expected]"
        }
    }
    Write-Host 'Windows installer argv self-test passed.'
    exit 0
}

foreach ($argv in $PythonCommands) {
    Invoke-Python -Arguments @($argv)
    $rendered = @($argv) -join ' '
    if ($rendered.EndsWith('packaging/vim2.spec')) {
        $exe = Join-Path $Root 'dist/build/VIM2/VIM2.exe'
        if (!(Test-Path -LiteralPath $exe)) { throw "PyInstaller executable missing: $exe" }
        $data = Join-Path $env:TEMP ('vim2-installer-check-' + [guid]::NewGuid().ToString())
        try {
            & $exe --import-smoke --data-root $data
            if ($LASTEXITCODE -ne 0) { throw 'Frozen import smoke failed' }
            & $exe --check --data-root $data
            if ($LASTEXITCODE -ne 0) { throw 'Frozen preflight failed' }
        } finally {
            if (Test-Path -LiteralPath $data) { Remove-Item -LiteralPath $data -Recurse -Force }
        }
    }
}

$iscc = Join-Path ${env:ProgramFiles(x86)} 'Inno Setup 6/ISCC.exe'
if (!(Test-Path -LiteralPath $iscc)) { throw 'Inno Setup 6 ISCC.exe is required on the Windows runner' }
& $iscc 'packaging/vim2.iss'
if ($LASTEXITCODE -ne 0) { throw 'Inno Setup failed' }
$setup = Join-Path $Root 'dist/installers/VIM2-0.1.0-windows-x64-Setup.exe'
if (!(Test-Path -LiteralPath $setup)) { throw 'Setup.exe missing' }
$sha = (Get-FileHash -LiteralPath $setup -Algorithm SHA256).Hash.ToLowerInvariant()
"$sha  VIM2-0.1.0-windows-x64-Setup.exe" | Set-Content -LiteralPath "$setup.sha256" -Encoding ascii
Write-Host "Windows installer: $setup SHA-256 $sha"
