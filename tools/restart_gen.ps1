# Safely (re)start the teacher llama-server on :8080 and the resumable generation run, both detached.
#   powershell -File tools/restart_gen.ps1              # restart server + generation
#   powershell -File tools/restart_gen.ps1 -NoGen       # only (re)start the teacher server
# Templates that failed only because the server was down are removed from the rejects file, so they are retried.
param([switch]$NoGen, [int]$N = 2000, [int]$Seed = 11,
      [string]$Focus = "compliance_check,numeric_claim_check,bug_severity,multi_condition_eligibility,support_intent,account_security",
      [string]$ServerArgs = "", [int]$Ctx = 24576, [int]$Ngl = 99, [double]$FocusWeight = 3)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like "*gen.py*generate*" } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force; "stopped generation $($_.ProcessId)" }
Get-CimInstance Win32_Process -Filter "Name='llama-server.exe'" | Where-Object { $_.CommandLine -like "*--port 8080*" } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force; "stopped teacher $($_.ProcessId)" }
Start-Sleep 3
$a = @("-m", "`"$root\models\gguf\Qwen3.5-9B-Q4_K_M.gguf`"", "--jinja", "-ngl", "$Ngl", "-fa", "on", "-np", "4", "-c", "$Ctx",
       "--port", "8080", "--host", "127.0.0.1", "--no-webui") + ($ServerArgs -split " " | Where-Object { $_ })
$s = Start-Process -FilePath "$root\tools\llama.cpp\cuda\llama-server.exe" -ArgumentList $a -WindowStyle Hidden -WorkingDirectory $root `
  -RedirectStandardError "$root\models\teacher_server.log" -RedirectStandardOutput "$root\models\teacher_server.out" -PassThru
"teacher pid $($s.Id)"
for ($i = 0; $i -lt 90; $i++) {
  try { if ((Invoke-RestMethod -Uri http://127.0.0.1:8080/health -TimeoutSec 2).status -eq "ok") { "teacher healthy"; break } } catch {}
  Start-Sleep 2
}
if ($NoGen) { return }
$rej = "$root\data\generated\raw.rejects.jsonl"
if (Test-Path $rej) {
  $lines = Get-Content $rej -Encoding UTF8 | Where-Object { $_ -and ($_ -notmatch "Error|Connect|Timeout|ReadError|RemoteProtocol") }
  [IO.File]::WriteAllLines($rej, [string[]]$lines, (New-Object Text.UTF8Encoding $false))  # no BOM
}
if (Test-Path "$root\data\generated\gen.log") { Add-Content "$root\data\generated\gen.history.log" (Get-Content "$root\data\generated\gen.log") }
$env:PYTHONUTF8 = "1"; $env:PYTHONUNBUFFERED = "1"
$g = Start-Process -FilePath "$root\.venv\Scripts\python.exe" -WorkingDirectory $root -WindowStyle Hidden `
  -ArgumentList @("data\gen.py", "generate", "--n", "$N", "--seed", "$Seed", "--out", "data\generated\raw.jsonl", "--concurrency", "4", "--focus", "$Focus", "--focus-weight", "$FocusWeight") `
  -RedirectStandardOutput "$root\data\generated\gen.log" -RedirectStandardError "$root\data\generated\gen.err" -PassThru
"generation pid $($g.Id)"
