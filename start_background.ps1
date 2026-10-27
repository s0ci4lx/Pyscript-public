$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
if (-not (Test-Path -LiteralPath (Join-Path $root '.venv\Scripts\python.exe'))) {
    throw 'Run install_windows.bat once before starting the background services.'
}
$watch = Join-Path $root 'server-runner\watch-servers.ps1'
if (-not (Test-Path -LiteralPath $watch)) { throw 'The local service watchdog is missing.' }
$env:PYTHONUTF8 = '1'
$env:CCTV_PROFILE = 'vehicle'
Remove-Item Env:PSModulePath -ErrorAction SilentlyContinue
Start-Process -FilePath 'powershell.exe' -ArgumentList '-NoProfile','-NonInteractive','-WindowStyle','Hidden','-ExecutionPolicy','Bypass','-File',('"' + $watch + '"') -WindowStyle Hidden
Write-Output 'Background services started. Closing the browser does not stop detection.'
Write-Output 'Use open_dashboard.bat to view the dashboard and sign in as owner.'
