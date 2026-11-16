param([int]$Port = 8091)
Set-Location -LiteralPath $PSScriptRoot
python -m uvicorn api_keras:app --host 127.0.0.1 --port $Port
