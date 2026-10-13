# RealMan 倒立摆真机实验平台

把现有 JointGV 仿真研究迁移到真实装置：RGB 观测摆杆，SA100 + ESP32-S3 提供独立角度验证，统一时间轴采集、导出并计算误差。下一阶段在 Ubuntu 上接入 RealMan 机械臂，并建设可视化实验控制平台。

**当前已有可编译固件和命令行采集工具；RealMan 控制适配器、完整 UI、深度相机原生接口仍是计划，尚未实现。软件测试通过不代表完成实物联调。**

用户另一系统已部署并成功使用过真机库。迁移先盘点/复用该SDK与旧脚本，保留原环境；计划中的新SDK/ROS安装仅在确有缺项时采用。

## 从哪里开始

| 目的 | 入口 |
|---|---|
| 在 Ubuntu 上继续开发 | [Ubuntu 实施计划书](docs/UBUNTU_IMPLEMENTATION_PLAN.md) |
| 下一位开发者接手 | [交接手册](docs/HANDOFF.md)、[当前状态](docs/PROJECT_STATUS.md) |
| 理解现有接口、文件与限制 | [技术文档](docs/TECHNICAL_DOCUMENTATION.md) |
| 烧录、接线、零点标定、采集 | [现有真机工具 README](real_device/README.md)、[操作手册](real_device/docs/OPERATION_MANUAL_ZH.md) |
| 对照仿真与真机实验条件 | [适配审计](real_device/docs/SIMULATION_ADAPTER_AUDIT.md) |
| 查看公开内容与依赖来源 | [公开发布说明](docs/PUBLICATION.md) |

## 当前功能

- ESP32-S3-N16R8：MCPWM 捕获 SA100 PWM，串口 / 可选 Wi-Fi UDP 输出角度、原始计数、序号和设备时间。
- PC：USB RGB 采集、双向时钟拟合、帧级圆周角插值、断连/重启/陈旧数据检查。
- 实验：自由倒落、人工辅助平衡两类；起始角 ±10°，默认 6 秒或绝对角度超过 120°停止记录。
- 分析：铰点和杆尖彩色标记测角、角度 MAE/RMSE、预测 CSV 评估、验证 NPZ 导出。
- 验证：25 项自动化测试；串口和 Wi-Fi 编译分支已在 Windows 构建通过；附两段纯合成演示。

采集停止不等于机械臂安全停止。现有固件上电后持续测量，PC 决定实验记录窗口；尚无 ESP32 `ARM/START/STOP` 会话控制协议，也没有机械臂运动命令。120°是数据集截断条件，不能直接作为机器人安全阈值。

## 快速验证，不连接硬件

需 Python 3.11 或以上。Ubuntu 22.04 的系统 Python 通常为 3.10，请按计划书创建独立 3.11 环境，不替换系统 Python。

```bash
git clone https://github.com/Pan-cyb/CVPR_Real_device_test.git
cd CVPR_Real_device_test
# 激活已创建的 Python 3.11 虚拟环境后：
python -m pip install -e ./real_device
python -m unittest discover -s real_device/tests -v
python -m real_device demo --output "runs/demo_$(date +%Y%m%d_%H%M%S)" --category free_fall
```

Windows PowerShell 用法见 [真机工具 README](real_device/README.md)。已有 [自由倒落演示](real_device/artifacts/demo_free_fall/metadata.json) 和 [6 秒人工辅助类别演示](real_device/artifacts/demo_assisted_balance/metadata.json) 都是 `source=synthetic`，不是实测精度证据。

## 目录

```text
real_device/        已实现的 ESP32 固件、PC CLI、测试和合成样例
docs/               Ubuntu/UI/RealMan 计划、交接、技术与发布说明
Joint_Graph_Vedio_Latent_Dynamic_World_Model-main/
  Joint_Graph_Vedio_Latent_Dynamic_World_Model-main/
                    原有 JointGV 源码快照，保留原结构与许可证
```

`sa100/` 厂家手册、CAD、STM32 原始包仅在原 Windows 工作目录保留，没有上传；新固件的接线和换算说明已整理进本仓库。

## 重要边界

SA100 测量轴相对外壳的角度。外壳平移允许，外壳朝向变化则需要同步姿态补偿。普通 OpenCV 时间戳是主机读取完成时刻，不能假称曝光时刻；深度相机型号确定后接其 SDK 并校准时钟。人工辅助数据不等于自主平衡成功，未知手扶动作不得填成零动作训练样本。

原仿真默认 ±7°、90°判倒、20 秒；当前采集协议为 ±10°、120°、6 秒，比较时需要统一条件。当前没有目标 JointGV / RM65-B 实验 checkpoint 和数据集；旧 WorldModels 基线包含历史 JSON 模型资源，它们不能替代目标权重。三个原上游子模块缺少内容和 Git gitlink 信息，克隆本仓库不会自动补齐。
