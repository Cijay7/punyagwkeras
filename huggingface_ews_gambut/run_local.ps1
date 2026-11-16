param([int]$Port=7860, [switch]$Example)
Set-Location -LiteralPath $PSScriptRoot
if ($Example) { $env:DATA_SOURCE='example' }
python -m uvicorn app:app --host 127.0.0.1 --port $Port --workers 1
