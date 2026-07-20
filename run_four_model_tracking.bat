@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo Starting four-model UAV gimbal tracking...
echo Weights: runs\rfly_uav_yolo11n_four_models_hardneg\weights\best.pt

"D:\PX4PSP\Python38\python.exe" "%~dp0run_tracking_only.py" ^
  --weights "%~dp0runs\rfly_uav_yolo11n_four_models_hardneg\weights\best.pt" ^
  --config "%~dp0Config_gimbal.json" ^
  --imgsz 960 ^
  --conf 0.14 ^
  --acquire-conf 0.25 ^
  --gimbal ^
  --scenario easy ^
  --gimbal-yaw-kp 75 ^
  --gimbal-pitch-kp 55 ^
  --max-gimbal-yaw-rate 50 ^
  --max-gimbal-pitch-rate 35 ^
  --deadband 0.04 ^
  --log-csv "%~dp0runs\four_models_hardneg_live.csv" %*

echo.
echo Tracking process ended.
pause
