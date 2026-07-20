$ErrorActionPreference = "Continue"

$RflyRoot = "D:\PX4PSP"
$ExampleRoot = Join-Path $RflyRoot "RflySimAPIs\8.RflySimVision\2.AdvExps\e8_SingleObjTracking"
$Python = Join-Path $RflyRoot "Python38\python.exe"
$Failed = 0

function Test-RequiredPath {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Label
    )

    if (Test-Path -LiteralPath $Path) {
        Write-Host "[OK]   $Label" -ForegroundColor Green
    }
    else {
        Write-Host "[FAIL] $Label - missing: $Path" -ForegroundColor Red
        $script:Failed++
    }
}

Write-Host "RflySim vision simulation environment check" -ForegroundColor Cyan
Write-Host "Root: $RflyRoot"

Test-RequiredPath $RflyRoot "RflySim installation"
Test-RequiredPath (Join-Path $RflyRoot "RflySim3D\RflySim3D.exe") "RflySim3D"
Test-RequiredPath (Join-Path $RflyRoot "CopterSim\CopterSim.exe") "CopterSim"
Test-RequiredPath (Join-Path $RflyRoot "QGroundControl\QGroundControl.exe") "QGroundControl"
Test-RequiredPath $Python "Bundled Python"
Test-RequiredPath (Join-Path $ExampleRoot "Tracking.bat") "Two-UAV startup script"
Test-RequiredPath (Join-Path $ExampleRoot "tracking.py") "KCF tracking baseline"
Test-RequiredPath (Join-Path $ExampleRoot "Config.json") "Vision sensor configuration"

if (Test-Path -LiteralPath $Python) {
    Write-Host "`nPython and vision packages:" -ForegroundColor Cyan
    & $Python -c "import sys, cv2, numpy; print('[OK]   Python', sys.version.split()[0]); print('[OK]   OpenCV', cv2.__version__); print('[OK]   NumPy', numpy.__version__); assert hasattr(cv2, 'TrackerKCF_create') or (hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerKCF_create')); print('[OK]   KCF tracker')"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[FAIL] Python vision dependency check" -ForegroundColor Red
        $Failed++
    }
}

Write-Host "`nWSL/PX4 SITL:" -ForegroundColor Cyan
$WslNames = @()
try {
    $WslNames = @(wsl.exe --list --quiet 2>$null) | ForEach-Object { ($_ -replace "`0", "").Trim() } | Where-Object { $_ }
}
catch {
    $WslNames = @()
}

if ($WslNames -contains "RflySim-20.04") {
    Write-Host "[OK]   WSL distribution RflySim-20.04" -ForegroundColor Green
}
else {
    Write-Host "[FAIL] WSL distribution RflySim-20.04 is not registered" -ForegroundColor Red
    Write-Host "       Re-run OnekeyScript.exe from the original RflySim installer ISO to repair WinWSL."
    $Failed++
}

Write-Host "`nResult:" -ForegroundColor Cyan
if ($Failed -eq 0) {
    Write-Host "PASS - ready to run the official e8 baseline." -ForegroundColor Green
    exit 0
}

Write-Host "FAIL - $Failed required check(s) need attention." -ForegroundColor Red
exit 1
