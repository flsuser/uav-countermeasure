# UAV Countermeasure Vision Prototype

基于 RflySimAdv3Free、PX4 SITL 和 YOLO11n 的无人机视觉检测与二自由度云台追踪仿真原型。本仓库用于防御性研究与仿真验证，不包含任何物理干扰或毁伤功能。

## 当前能力

- RflySim3D 双机仿真与自动起飞。
- 单类别 `uav` 实时检测，约 33–36 FPS（RTX 4060 Laptop，`imgsz=960`）。
- 单目标检测、短时运动预测和丢失后重新捕获。
- 虚拟云台水平与俯仰控制，不控制观察机前进或后退。
- 支持 AR Drone、DJI Phantom、Quad Fold 和 Droneyee QuadX330 等多种外观。
- 山体穿越困难正样本增强和山体/石头困难负样本抑制。
- CSV 追踪遥测与离线指标分析。

当前发布权重：

```text
runs/rfly_uav_yolo11n_four_models_hardneg/weights/best.pt
```

正式运行参数：检测阈值 `0.14`，新目标锁定阈值 `0.25`，输入尺寸 `960`。

## 环境要求

- Windows 10/11
- RflySimAdv3Free，默认安装在 `D:\PX4PSP`
- WSL2 发行版 `RflySim-20.04`
- RflySim3D、CopterSim 和 PX4 SITL
- NVIDIA GPU（推荐）

RflySim SDK 模块 `VisionCaptureApi`、`PX4MavCtrlV4` 和 `UE4CtrlAPI` 来自本地 RflySim 安装，不由 PyPI 提供。其余 Python 依赖见 `requirements.txt`。

## 快速运行

1. 启动官方双机仿真：

```text
D:\PX4PSP\RflySimAPIs\8.RflySimVision\2.AdvExps\e8_SingleObjTracking\Tracking.bat
```

2. 等待两个 PX4 实例、两套 CopterSim 和 RflySim3D 完全启动。

3. 双击 `run_four_model_tracking.bat`。

程序确认两架飞机高度超过 15 m 后开始检测和目标追踪。识别窗口按 `q` 或 `Esc` 退出。

PowerShell 也可直接执行：

```powershell
& "D:\PX4PSP\Python38\python.exe" .\run_tracking_only.py `
  --weights .\runs\rfly_uav_yolo11n_four_models_hardneg\weights\best.pt `
  --config .\Config_gimbal.json --imgsz 960 --conf 0.14 `
  --acquire-conf 0.25 --gimbal --scenario easy
```

不要同时运行多个取图、采集或追踪程序，否则会产生 UDP 图像端口冲突（WinError 10048）。

## 已验证结果

| 外观 | 动态检测率 | 搜索比例 | 最长丢失 |
|---|---:|---:|---:|
| AR_Drone_BlackGreen | 94.02% | 0.88% | 0.485 s |
| DJI_Phantom_White | 98.70% | 0% | 0.075 s |
| Quad_Fold_Gray | 92.51% | 0.55% | 0.431 s |
| Droneyee_QuadX330 | 100% | 0% | 0 s |

困难负样本验证集在阈值 `0.14` 时为 `0/40` 误检。山体穿越增强测试实现 100% 检测且无丢失事件。

这些结果来自仿真环境，不能直接代表真实摄像头、真实天空或极远距离目标性能。

## 主要文件

- `run_detect_track.py`：检测、单目标追踪、云台控制和日志。
- `run_tracking_only.py`：稳定的云台追踪入口。
- `run_four_model_tracking.bat`：正式四外观启动脚本。
- `collect_dataset.py`：基础数据采集与真值标注。
- `collect_mountain_crossing.py`：山体穿越困难正样本。
- `collect_background_negatives.py`：无无人机困难负样本。
- `collect_model_dataset.py`：指定外观数据采集。
- `prepare_combined_dataset.py`、`prepare_multimodel_dataset.py`：数据集合并。
- `train_detector.py`：YOLO11n 训练入口。
- `evaluate_multimodel.py`、`evaluate_background_false_positives.py`：模型评估。
- `analyze_tracking_log.py`：动态追踪日志分析。

数据集、训练缓存、运行日志和历史权重没有提交到仓库；只保留当前正式权重。

## 限制与下一步

当前模型只有一个 `uav` 类别，能够检测无人机，但不会判断具体型号；目前也不支持多目标追踪或可靠的单目距离估计。下一阶段建议接入 USB 摄像头、采集真实正负样本，并在安全限位与急停机制完善后接入真实二自由度云台。