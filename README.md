# 无人机检测与追踪开发

官方 `e8_SingleObjTracking` 的红框来自仿真真值，蓝框来自 KCF；它验证了图像和跟踪链路，但并不是自主识别。本目录的第一步是使用仿真真值自动生成 YOLO 数据集。真值仅用于训练标签和评测，最终推理不得读取目标状态。

## 1. 采集第一批数据

1. 用官方 `e8_SingleObjTracking\Tracking.bat` 启动两架无人机。
2. 等待 QGroundControl 已连接。
3. 不要运行官方 `tracking.py`；如果正在运行，先按 `Ctrl+C` 停止它。
4. 新开 PowerShell，执行：

```powershell
cd "C:\Users\23102\Documents\无人机反制\uav_detection"
& "D:\PX4PSP\Python38\python.exe" .\collect_dataset.py --frames 600 --clear
```

采集窗口中的绿色框是训练标签。按 `q` 可提前结束。程序默认每5张取1张作为验证集，输出到 `datasets\rfly_uav`。`--clear` 会先删除上一轮生成的数据，避免无效样本残留。

## 2. 数据验收

第一批600张仅用于跑通流程。至少检查：

- 近、中、远三种尺度均有样本；
- 框覆盖完整无人机，不能长期偏移；
- 目标不能全部位于画面中心；
- 训练集和验证集均非空；
- 没有黑图、冻结的重复帧或目标完全在画面外的标签。

后续会增加不同地图、光照、目标外观、相机FOV和运动模糊，目标总量建议先达到3000～5000张。

当前默认配置使用45度FOV和3.5～30米航迹，生成近中距离子集。90度FOV采集结果应保留为远距离小目标子集，后续合并训练。

## 3. 合并远近距离数据

```powershell
& "D:\PX4PSP\Python38\python.exe" .\prepare_combined_dataset.py
```

合并程序会保留远距离小目标，并过滤近距离子集中宽度小于20像素或高度小于10像素的边缘碎片。训练配置为 `dataset_combined.yaml`。

## 4. 训练检测器

训练使用项目内 `.venv`，不会修改 RflySim 自带 Python。正式训练命令：

```powershell
& ".\.venv\Scripts\python.exe" .\train_detector.py --epochs 80 --imgsz 960 --batch 8
```

最佳权重输出到 `runs\rfly_uav_yolo11n\weights\best.pt`。

## 5. 在 RflySim 中实时识别与追踪

先用官方 `e8_SingleObjTracking\Tracking.bat` 启动两架飞机和 RflySim3D，但不要运行官方
`tracking.py`，也不要同时运行数据采集脚本。然后在本目录执行：

```powershell
& ".\.venv\Scripts\python.exe" .\run_detect_track.py
```

脚本会自动控制两架飞机起飞，并让目标机在近、中、远距离之间运动。绿色框表示当前帧由 YOLO
检出，橙色框表示检测短暂丢失后使用运动趋势预测。黄色十字是画面中心，
`ex`、`ey` 是归一化中心偏差，下一步将用它们闭环控制观察机。按 `q` 或 `Esc` 退出。

需要降低误检时提高置信度，例如 `--conf 0.4`；需要增强远距离召回时可尝试 `--conf 0.15`。
实时推理阶段不读取 UE 目标真值。

## 6. 视觉闭环对准

确认普通识别追踪运行正常后，重新启动仿真并执行：

```powershell
& ".\.venv\Scripts\python.exe" .\run_detect_track.py --closed-loop --conf 0.25
```

闭环模式使用水平误差控制观察机偏航，使用垂直误差控制升降。默认设有 0.06 归一化死区、
0.35 rad/s 最大偏航角速度、1 m/s 最大升降速度，以及 10～30 m 高度保护。只有绿色检测框
参与控制；橙色预测或完全丢失目标时观察机悬停。画面底部会显示实际控制命令。

这一阶段只验证目标居中，不控制前进距离。按 `q` 退出后再调整增益；若出现左右振荡，可使用
`--yaw-kp 0.5`，若上下振荡可使用 `--vertical-kp 0.8`。

## 7. 稳定追踪版本与困难负样本

`run_tracking_only.py` 是固定的追踪入口，不包含前进/后退控制：

```powershell
& ".\.venv\Scripts\python.exe" .\run_tracking_only.py --closed-loop --conf 0.25
```

实物云台方案对应的仿真模式使用相机二轴云台，不再用飞机升降代替相机俯仰：

```powershell
& ".\.venv\Scripts\python.exe" .\run_tracking_only.py --gimbal --conf 0.25
```

`ex` 控制云台偏航，`ey` 控制云台俯仰。默认俯仰范围为 -60°～+30°、偏航范围为 ±80°；
绿色检测时更新角度，橙色预测或目标丢失时保持当前角度。`--gimbal` 与 `--closed-loop` 不能同时使用。

进行约两分钟的标准测试并记录遥测：

```powershell
& ".\.venv\Scripts\python.exe" .\run_tracking_only.py --gimbal --conf 0.25 `
  --log-csv .\runs\gimbal_test_01.csv
```

测试完成后分析检测占比、中心误差、FPS和最长丢失时间：

```powershell
& ".\.venv\Scripts\python.exe" .\analyze_tracking_log.py .\runs\gimbal_test_01.csv
```

橙色预测默认只在前5个漏检帧以0.35倍增益继续控制云台，随后保持角度。可通过
`--gimbal-predict-control-frames` 和 `--gimbal-predict-gain` 调整。

目标运动分为 `easy`、`medium`、`stress` 三档。默认 `easy` 将速度限制为约1.2 m/s、距离约
7～14 m，并在航点停留1秒；先在这一档完成控制参数整定，再逐级测试。`Config_gimbal.json`
提供960×720原生图像，避免把640×480图像单纯插值放大。

```powershell
& ".\.venv\Scripts\python.exe" .\run_tracking_only.py --gimbal --scenario easy `
  --config .\Config_gimbal.json --imgsz 960 --conf 0.25
```

当模型把山体黑斑、石块或阴影误识别成无人机时，只有在人工确认原始画面内完全没有真实无人机
的情况下按 `n`。程序会把原始画面及空标签保存到 `datasets\rfly_hard_negatives`。不要在真实
无人机仍在画面内时按 `n`，否则会产生错误负标签。

建议采集至少 100～300 张，覆盖不同山体位置、视角、距离和光照。采集完成后重新合并数据：

也可以关闭追踪程序、重新启动仿真后自动采集 200 张无无人机背景：

```powershell
& ".\.venv\Scripts\python.exe" .\collect_background_negatives.py --frames 200
```

程序会先把目标机移到相机后方，再在前方约 ±55° 范围和安全高度内改变观察视角。确认预览中
没有无人机；如有异常立即按 `q`，不要使用这一批数据。

采集完成后重新合并数据：

```powershell
& "D:\PX4PSP\Python38\python.exe" .\prepare_combined_dataset.py
```

合并日志中的 `background negatives` 应大于零，随后再使用新的实验名称重新训练。困难负样本是
降低山体误报的主要手段，不能只靠提高置信度阈值替代。

```powershell
& ".\.venv\Scripts\python.exe" .\train_detector.py `
  --model .\runs\rfly_uav_yolo11n\weights\best.pt `
  --epochs 30 --imgsz 960 --batch 8 --workers 2 `
  --name rfly_uav_yolo11n_hardneg
```

分别评估旧模型和新模型在纯背景验证集上的误报率：

```powershell
& ".\.venv\Scripts\python.exe" .\evaluate_background_false_positives.py
& ".\.venv\Scripts\python.exe" .\evaluate_background_false_positives.py `
  --weights .\runs\rfly_uav_yolo11n_hardneg\weights\best.pt
```

新模型的 `false_alarm_image_rate` 应明显低于旧模型，同时还要检查正常验证集召回率没有严重下降。

## 8. 山体穿越困难正样本

当无人机由下向上穿越山体时检测置信度下降，可使用仿真真值采集专用困难正样本。关闭其他取图、
追踪和采集程序，重新启动双机仿真后执行：

```powershell
& "D:\PX4PSP\Python38\python.exe" .\collect_mountain_crossing.py --frames 200 --frame-stride 4 --clear
```

目标机会在8～14米范围内以约1 m/s反复上升。程序不保存向下复位阶段，只保存目标中心位于
画面纵向20%～68%区域的上升帧，并通过仿真真值绘制绿色标签框。输出目录为
`datasets\rfly_mountain_crossing`。前三种穿越视角进入训练集，第四种视角单独作为验证集，避免
相邻帧泄漏。采集预览中绿色框必须完整覆盖无人机。

采集完成后合并原始远近距离数据与山体困难正样本：

```powershell
& "D:\PX4PSP\Python38\python.exe" .\prepare_combined_dataset.py
```

然后从当前最佳权重继续微调30轮：

```powershell
& ".\.venv\Scripts\python.exe" .\train_detector.py `
  --model .\runs\rfly_uav_yolo11n\weights\best.pt `
  --epochs 30 --imgsz 960 --batch 8 --workers 2 `
  --name rfly_uav_yolo11n_mountain
```

新模型测试时通过 `--weights .\runs\rfly_uav_yolo11n_mountain\weights\best.pt` 显式加载，
保留原始 `best.pt` 作为基线，不覆盖旧模型。
