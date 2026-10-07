# One training round, start to finish, meant to run detached so it survives an interrupted session:
#   label new templates -> build train/val -> free the GPU -> train -> evaluate (our val + DecideBench) -> resume generation.
#   Start-Process powershell -ArgumentList '-NoProfile -ExecutionPolicy Bypass -File tools\train_round.ps1 -Version v0' -WindowStyle Hidden
# Progress: models\<Version>\round.log
param([Parameter(Mandatory = $true)][string]$Version, [double]$Epochs = 1, [int]$Micro = 8, [string]$Extra = "",
      [switch]$NoResume)
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$out = "models\$Version"
New-Item -ItemType Directory -Force $out | Out-Null
$log = "$root\$out\round.log"
$py = "$root\.venv\Scripts\python.exe"
$env:PYTHONUTF8 = "1"; $env:PYTHONUNBUFFERED = "1"; $env:HF_HUB_OFFLINE = "1"; $env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
function Step($name, $cmd) {
  Add-Content $log "[$(Get-Date -Format s)] $name"
  cmd /c "$cmd >> `"$log`" 2>&1"
  Add-Content $log "[$(Get-Date -Format s)] $name exit=$LASTEXITCODE"
  return $LASTEXITCODE
}

# 1. stop generation, keep the teacher up for labelling
Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like "*gen.py*generate*" } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
if ((Step "label" "`"$py`" data\gen.py label --inp data\generated\raw.jsonl --out data\generated\labeled.jsonl --concurrency 4") -ne 0) { exit 1 }
if ((Step "build" "`"$py`" data\gen.py build --inp data\generated\labeled.jsonl --out data\generated") -ne 0) { exit 1 }
Copy-Item data\generated\build_stats.json "$out\build_stats.json"

# 2. free the GPU: stop the teacher
Get-CimInstance Win32_Process -Filter "Name='llama-server.exe'" | Where-Object { $_.CommandLine -like "*--port 8080*" } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
Start-Sleep 5

# 3. train, then evaluate
$train = "`"$py`" train\train.py --out $out --epochs $Epochs --micro $Micro --accum 16 --chat data\generated\chat_distill.jsonl --p-chat 0.1 $Extra"
if ((Step "train" $train) -eq 0) {
  Step "val" "`"$py`" train\eval_val.py --model $out\merged --out $out\val.json" | Out-Null
  Step "decidebench" "`"$py`" bench\run_local.py --model $out\merged --device cuda --format compact --name unee-$Version-compact" | Out-Null
  if (-not (Test-Path models\chat_base.json)) {
    Step "chat answers (base)" "`"$py`" train\eval_chat.py answer --model Qwen/Qwen3.5-0.8B --out models\chat_base.json" | Out-Null
  }
  Step "chat answers" "`"$py`" train\eval_chat.py answer --model $out\merged --out $out\chat.json" | Out-Null
}

# 4. resume generation
if (-not $NoResume) {
  # Launched detached: run through Step, the teacher and generator would inherit Step's output pipe and the round
  # would wait on it forever. Poll the teacher's health instead.
  Add-Content $log "[$(Get-Date -Format s)] resume generation"
  Start-Process powershell -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File tools\restart_gen.ps1" -WorkingDirectory $root -WindowStyle Hidden
  for ($i = 0; $i -lt 120; $i++) {
    Start-Sleep 3
    try { if ((Invoke-RestMethod -Uri http://127.0.0.1:8080/health -TimeoutSec 2).status -eq "ok") { break } } catch {}
  }
  if (Test-Path $out\chat.json) {  # the teacher is back up: judge chat quality against the base model
    Step "chat judge" "`"$py`" train\eval_chat.py judge --a models\chat_base.json --b $out\chat.json --out $out\chat_judge.json" | Out-Null
  }
}
Add-Content $log "[$(Get-Date -Format s)] round done"
