$ErrorActionPreference = 'Stop'
$runnerRoot = $PSScriptRoot
$projectRoot = Split-Path -Parent $runnerRoot
$satunRoot = if ($env:SATUN_PROJECT_DIR) { $env:SATUN_PROJECT_DIR } else { Join-Path $env:SystemDrive "Satun-AI-Cam-main\Satun-AI-Cam-main" }
$mutex = New-Object System.Threading.Mutex($false, 'Local\NeopolisCCTVServerWatch')
if (-not $mutex.WaitOne(0)) { exit 0 }
$services = @(
    @{ Name='satun'; Port=8765; Root=$satunRoot; Python='.venv-win\Scripts\python.exe'; Arguments='-u -m src.run' },
    @{ Name='cctv-backend'; Port=8081; Root=$projectRoot; Python='.venv\Scripts\python.exe'; Arguments='-u rtsp_yolo_backend.py' },
    @{ Name='cctv-web'; Port=8000; Root=$projectRoot; Python='.venv\Scripts\python.exe'; Arguments='-u serve_frontend.py' }
)
$env:PYTHONUTF8='1'
$env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' + [Environment]::GetEnvironmentVariable('Path', 'User')
$env:CCTV_PROFILE='vehicle'
try {
    while ($true) {
        foreach ($service in $services) {
            try {
                if (Test-Path -LiteralPath (Join-Path $satunRoot ('output\' + $service.Name + '-maintenance.flag'))) { continue }
                $listener = Get-NetTCPConnection -LocalPort $service.Port -State Listen -ErrorAction SilentlyContinue
                if ($listener) { continue }
                if ($service.Process -and -not $service.Process.HasExited) { continue }
                # Keep one previous run of each log and avoid unlimited log growth.
                foreach ($suffix in @('stdout','stderr')) {
                    $log = Join-Path $runnerRoot ($service.Name + '.' + $suffix + '.log')
                    if (Test-Path -LiteralPath $log) { Move-Item -LiteralPath $log -Destination ($log + '.previous') -Force }
                }
                $service.Process = Start-Process -FilePath (Join-Path $service.Root $service.Python) -ArgumentList $service.Arguments -WorkingDirectory $service.Root -WindowStyle Hidden -RedirectStandardOutput (Join-Path $runnerRoot ($service.Name + '.stdout.log')) -RedirectStandardError (Join-Path $runnerRoot ($service.Name + '.stderr.log')) -PassThru
                Add-Content -LiteralPath (Join-Path $runnerRoot 'watch.log') -Value "$(Get-Date -Format s) Started $($service.Name), launcher PID $($service.Process.Id)"
            } catch {
                Add-Content -LiteralPath (Join-Path $runnerRoot 'watch.log') -Value "$(Get-Date -Format s) $($service.Name): $($_.Exception.Message)"
            }
        }
        Start-Sleep -Seconds 15
    }
} finally {
    $mutex.ReleaseMutex()
    $mutex.Dispose()
}
