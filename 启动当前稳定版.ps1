$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = "D:\PX4PSP\Python38\python.exe"

if (-not (Test-Path -LiteralPath $python)) {
    throw "未找到 D 盘 RflySim Python 环境：$python"
}

Set-Location -LiteralPath $here
$arguments = @(
    ".\run_tracking_only.py",
    "--weights", ".\runs\rfly_uav_yolo11n_mountain\weights\best.pt",
    "--gimbal", "--scenario", "easy",
    "--config", ".\Config_gimbal.json",
    "--imgsz", "960",
    "--conf", "0.18",
    "--acquire-conf", "0.30",
    "--deadband", "0.03",
    "--gimbal-yaw-kp", "82",
    "--gimbal-pitch-kp", "52",
    "--max-gimbal-yaw-rate", "45",
    "--max-gimbal-pitch-rate", "30",
    "--log-csv", ".\runs\gimbal_saved_version.csv"
)
& $python $arguments

