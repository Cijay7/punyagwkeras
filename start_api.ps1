Set-Location -LiteralPath $PSScriptRoot
python -m uvicorn api:app --host 127.0.0.1 --port 8090
