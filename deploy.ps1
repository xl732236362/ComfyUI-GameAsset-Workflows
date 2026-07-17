param(
    [Parameter(Mandatory = $true)]
    [string]$ComfyRoot,
    [string]$BaseUrl = 'http://127.0.0.1:8188'
)

$ErrorActionPreference = 'Stop'
$python = Join-Path $ComfyRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "ComfyUI Python not found: $python"
}
& $python (Join-Path $PSScriptRoot 'scripts\deploy.py') `
    --comfy-root $ComfyRoot --base-url $BaseUrl
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
