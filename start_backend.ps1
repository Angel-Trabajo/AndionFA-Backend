Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
& "$PSScriptRoot\env\Scripts\Activate.ps1"
Set-Location $PSScriptRoot
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
