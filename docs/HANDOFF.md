# 交接手册

更新日期：2026-10-13。目标仓库：[Pan-cyb/CVPR_Real_device_test](https://github.com/Pan-cyb/CVPR_Real_device_test)。当前发布提交以 `git log -1 --oneline` 为准，本文不写循环依赖自身的commit SHA。

## 接手顺序

1. 阅读根 `AGENTS.md`、[PROJECT_STATUS](PROJECT_STATUS.md)。
2. 阅读 [Ubuntu实施计划书](UBUNTU_IMPLEMENTATION_PLAN.md) 的M0–M3，先做环境与设备只读诊断。
3. 用 [现有接口文档](TECHNICAL_DOCUMENTATION.md) 跑测试/合成demo；有设备后再按接线手册标定。
4. 将实际硬件信息记录到本地站点配置，回填不含敏感信息的兼容性结论。

## 已有内容

- 原 Windows 工作区为 `G:\PROJECT\CVPR_Real_device_test`，最初没有 `.git`。本次在根目录建立Git仓库，保留原源码目录，不将原仿真目录伪装为完整上游Git克隆。
- `real_device/firmware/` 是ESP32-S3-N16R8的PlatformIO工程；`pc/real_device/` 是Python采集/分析；`tests/test_core.py` 含25项测试。
- 前一阶段已经完成Windows构建、串口/WiFi分支编译、模拟设备集成测试、自由倒落和6秒辅助类别的synthetic demo。**没有实物烧录/联调、Ubuntu验证或RealMan控制实现。**
- 本轮新增根README、Ubuntu计划、项目规则、状态、技术文档、交接和公开发布规则；不扩展机械臂执行代码。
- [构建与测试证据](../real_device/docs/BUILD_AND_TEST.md) 链接到公开的日志。历史日志保留当时的Windows路径，不意味着Ubuntu存在这些路径。

## 继续工作时的重要事实

- 摄像头型号未定，用户可能采用深度相机。`DeviceTimestampCamera` 目前只是扩展契约，无具体深度相机实现。
- 用户确认另一系统已部署真机库并成功使用过，型号/版本暂时记不清。优先恢复旧SDK、环境、控制脚本及私有配置；本轮没有核验那个系统。不要直接升级或重装破坏可用环境，RM_API2只是已查验的候选适配路线。
- 真实机械臂型号/固件、Ubuntu版本、ESP32开发板引脚、SA100外壳朝向是否恒定尚未确认。不能把仿真RM65-B名称当作实物核验结果。
- 现有采集`r`键只标记松手指令，不自动检测真实离手；要求离线视频复核。
- 现有ESP协议v1没有ARM/START/STOP；UI和RobotAdapter均待做，不能从计划书直接调用并不存在的API。
- SA100是相对编码器角，外壳旋转要姿态补偿；图像时间戳目前是read-end代理，需实测曝光/传输延迟。
- 原仿真为±7°/90°/20s，真机采集为±10°/>120°/6s。动作是铰点水平速度，不是角度；手扶未知动作不能填零。
- 原JointGV快照含MIT许可证；目标训练数据/权重缺失。旧WorldModels目录存在历史JSON模型，不能作为本研究checkpoint。三个上游子模块目录在源工作区为空，Git不会追踪空目录；没有原gitlink SHA，不支持直接`git submodule update`恢复。
- 原README列出的上游地址在本次登录状态下API返回404，无法确认是私有、迁移或已删除。不要编造上游commit，需向项目作者取得可用仓库和准确版本。

## 验证命令

在仓库根目录、Python3.11+独立环境中：

```bash
python -m pip install -e ./real_device
python -m unittest discover -s real_device/tests -v
python -m real_device --help
python -m real_device demo --output "runs/check_$(date +%Y%m%d_%H%M%S)"
pio run -d real_device/firmware -e esp32-s3-n16r8
git status --short
```

不要复用已存在的trial目录，程序会拒绝覆盖。先让mock/只读路径通过，再开启真实运动。新机器上的首次pip/PlatformIO下载需要联网，Windows编译产物和缓存没有上传。

## 公开与本地状态

本地 `sa100/`、`.work/`、`.pio/`、私人采集目录和凭据被忽略，没有删除。若Ubuntu仍需厂家PDF/CAD/原STM32代码，单独通过授权私有渠道迁移，见 [PUBLICATION](PUBLICATION.md)。真实实验数据不会因`git add .`自动加入已约定的采集输出路径，但仍需检查新路径和暂存区。

接手后以`git status`核实是否有新的用户改动；保留它们，不回滚、不覆盖。完成每项任务后更新`PROJECT_STATUS.md`的验证证据和未完成项，而不是把阶段进度写入全局规则。

## 推荐的技能与下一步任务

后续可使用 `build-platformio`、`serial-monitor`、`project-context-handoff`；只有实际需要烧录/调试时才启用对应硬件技能。下一工作单：只读盘点另一系统已成功运行的真机环境，保留旧SDK和脚本；另建采集环境跑25项测试与demo，检测串口/相机但不驱动机械臂，提交版本清单与失败日志。之后按实际SDK实现并验收RealMan只读适配器和mock服务，再接UI。
