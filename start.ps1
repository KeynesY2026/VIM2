$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
Set-Location -LiteralPath $Root

function Show-StartupError {
    param([string]$Message)
    Add-Type -AssemblyName PresentationFramework
    [System.Windows.MessageBox]::Show(
        $Message,
        "VIM2",
        [System.Windows.MessageBoxButton]::OK,
        [System.Windows.MessageBoxImage]::Error
    ) | Out-Null
}

$Python = Get-Command "python.exe" -ErrorAction SilentlyContinue
if ($null -eq $Python) {
    Show-StartupError "Python 3.10 through 3.13 is required but was not found."
    exit 1
}

& $Python.Source -c "import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] < (3, 14) else 1)"
if ($LASTEXITCODE -ne 0) {
    Show-StartupError "Unsupported Python version. Install Python 3.10 through 3.13."
    exit 1
}

$Pythonw = Join-Path (Split-Path -Parent $Python.Source) "pythonw.exe"
if (-not (Test-Path -LiteralPath $Pythonw)) {
    $PythonwCommand = Get-Command "pythonw.exe" -ErrorAction SilentlyContinue
    if ($null -eq $PythonwCommand) {
        Show-StartupError "pythonw.exe was not found beside the configured Python installation."
        exit 1
    }
    $Pythonw = $PythonwCommand.Source
}

$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:HF_HOME = Join-Path $Root "temp\huggingface"
$env:PYTHONPATH = Join-Path $Root "app"

$Arguments = @("-m", "vim2", "--windowed")
& $Pythonw $Arguments
