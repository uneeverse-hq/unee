# After the release pass (tools/release_r3.py): processor-only speed of the released GGUFs, the website's data module
# and the Hugging Face staging folders. Nothing is published.
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\post_release.ps1 [-Web <uneeverse-web checkout>]
# Speed caveat (as before): this laptop's cores are much faster than an old dual-core's, so 2 threads here is an
# optimistic stand-in for a potato PC.
param([string]$Web = "..\..\..\uneeverse-web", [switch]$NoSpeed, [string]$Tag = "u3", [string]$Version = "0.3")
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONUTF8 = '1'; $env:PYTHONUNBUFFERED = '1'; $env:HF_HUB_OFFLINE = '1'
$py = '.venv\Scripts\python.exe'
$rel = Get-Content "models\release_$Tag.json" -Raw | ConvertFrom-Json
if (-not $NoSpeed) {
  foreach ($size in '0.8b', '2b') {
    foreach ($t in 2, 4) {
      $name = "potato-$size-t$t"
      "measuring $name"
      $s = Start-Process -FilePath 'tools\llama.cpp\cpu\llama-server.exe' -WindowStyle Hidden -PassThru `
        -ArgumentList @('-m', $rel.gguf.$size, '--jinja', '-t', "$t", '-np', '1', '-c', '8192', '--port', '8095',
                        '--host', '127.0.0.1', '--no-webui') -RedirectStandardError 'models\potato_server.log'
      for ($i = 0; $i -lt 120; $i++) {
        try { if ((Invoke-RestMethod -Uri http://127.0.0.1:8095/health -TimeoutSec 2).status -eq 'ok') { break } } catch {}
        Start-Sleep 2
      }
      & $py bench\speed.py --url http://127.0.0.1:8095/v1 --name $name --n 20
      Stop-Process -Id $s.Id -Force
      Start-Sleep 3
    }
  }
}
& $py tools\report_data.py --version $Version --tag $Tag --decidebench "unee-$Tag-0.8b" "unee-$Tag-2b" `
  --served "unee-$Tag-0.8b-q4km-cuda-c4" "unee-$Tag-2b-q4km-cuda-c4" --speed "unee-$Tag-0.8b-cuda" "unee-$Tag-2b-cuda" `
  --web "$Web\src\content\unee-report.ts"
& $py tools\stage_hf.py --tag $Tag
