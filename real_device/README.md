# SA100 真机同步采集

迁移Ubuntu、接入RealMan或建设完整UI，请先阅读[仓库总入口](../README.md)、[Ubuntu实施计划](../docs/UBUNTU_IMPLEMENTATION_PLAN.md)和[交接手册](../docs/HANDOFF.md)。本文下面的快速开始采用Windows PowerShell；Linux命令见根文档。

这个目录是独立的真机验证工具，不修改 `../sa100/` 厂家资料，也不修改仿真工程。它完成 SA100 PWM → ESP32-S3-N16R8 → PC 的原始角度采集、USB RGB 视频采集、双向时钟同步、离线对齐、彩色标记视觉角和角度指标。

先运行无需硬件的完整演示：

```powershell
cd G:\PROJECT\CVPR_Real_device_test\real_device
python -m pip install -e .
$env:PYTHONPATH = "$PWD\pc"
$out = "artifacts\demo_free_fall_$(Get-Date -Format yyyyMMdd_HHmmss)"
python -m real_device demo --output $out --category free_fall
python -m unittest discover -s tests -v
```

演示会生成 `rgb.mp4`、`sensor.csv`、`frames.csv`、`sync.csv`、`events.csv`、`aligned.csv`、`metadata.json` 和 `metrics.json`。输出目录不会覆盖；仓库内已有样例可直接查看 `artifacts/demo_free_fall/`。真机操作前必须阅读 [操作与接线手册](docs/OPERATION_MANUAL_ZH.md)。系统不会从按键推断人的手真的离开，`r` 只记录“松手指令时刻”，需人工复核视频。

固件位于 `firmware/`，目标板配置为 16 MB Flash、8 MB Octal PSRAM，默认 GPIO4、921600 baud、约 200 Hz 上报。构建说明及证据见 [构建与测试](docs/BUILD_AND_TEST.md)。工程结构和适配边界见 [项目概览](docs/PROJECT_OVERVIEW.md)，协议和结果字段见 [协议与结果](docs/PROTOCOL_AND_RESULTS.md)，与现有仿真的差异见 [仿真适配审计](docs/SIMULATION_ADAPTER_AUDIT.md)。

真机命令示例（不会自动烧录）：

```powershell
python -m real_device capture --output artifacts\trial_001 --port COM5 `
  --camera 0 --fps 30 --width 640 --height 480 `
  --category free_fall --zero-deg 123.4 --direction 1 `
  --confirm-fixed-housing
```

`--confirm-fixed-housing` 是几何事实确认：SA100 测量轴对外壳的相对角。外壳可以平移，但朝向必须相对世界恒定；若外壳随机械臂旋转，必须另采机械臂姿态并合成为世界竖直角，当前命令不适用。
