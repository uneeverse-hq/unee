# Train one variant on an existing train file and evaluate it (our val + DecideBench), detached.
#   Start-Process powershell -ArgumentList '-NoProfile -ExecutionPolicy Bypass -File tools\experiment.ps1 -Name e1 -Train data\generated\clean85\train.jsonl -Val data\generated\clean85\val.jsonl -Extra "--alpha-teacher 1.0"' -WindowStyle Hidden
# Progress: models\<Name>\round.log. Does not touch the teacher or generation (stop them first: the GPU is shared).
param([Parameter(Mandatory = $true)][string]$Name, [Parameter(Mandatory = $true)][string]$Train, [string]$Val = "",
      [double]$Epochs = 2, [int]$Micro = 8, [string]$Extra = "", [string]$Model = "Qwen/Qwen3.5-0.8B",
      [string]$Chat = "data\generated\chat_distill.jsonl", [double]$PChat = 0.1, [int]$ChatTail = 128,
      [switch]$Eval4bit)
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$out = "models\$Name"
New-Item -ItemType Directory -Force $out | Out-Null
$log = "$root\$out\round.log"
$py = "$root\.venv\Scripts\python.exe"
$env:PYTHONUTF8 = "1"; $env:PYTHONUNBUFFERED = "1"; $env:HF_HUB_OFFLINE = "1"; $env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
function Step($label, $cmd) {
  Add-Content $log "[$(Get-Date -Format s)] $label"
  cmd /c "$cmd >> `"$log`" 2>&1"
  Add-Content $log "[$(Get-Date -Format s)] $label exit=$LASTEXITCODE"
  return $LASTEXITCODE
}
$t = "`"$py`" train\train.py --model $Model --train $Train --out $out --epochs $Epochs --micro $Micro --accum 16 --chat $Chat --p-chat $PChat --chat-max-tail $ChatTail $Extra"
if ((Step "train" $t) -eq 0) {
  $q4 = if ($Eval4bit) { "--load-4bit" } else { "" }  # big models: evaluate in 4-bit so they fit beside other GPU users
  if ($Val) { Step "val" "`"$py`" train\eval_val.py --model $out\merged --val $Val --out $out\val.json $q4" | Out-Null }
  Step "decidebench" "`"$py`" bench\run_local.py --model $out\merged --device cuda --format compact --name unee-$Name-compact $q4" | Out-Null
}
Add-Content $log "[$(Get-Date -Format s)] experiment done"
