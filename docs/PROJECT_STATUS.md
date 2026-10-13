# 项目状态

更新：2026-10-13。阶段：Windows采集软件已验证，正在迁移至Ubuntu；本轮形成公开源码与实施交接。

## 已完成

- 完成SA100厂家协议/原例程与JointGV仿真流程审查；新增独立ESP32-S3固件、Python采集/对齐/指标工具。
- 串口/WiFi分支在Windows构建通过；25项自动化测试通过。
- 合成自由倒落85帧、6s辅助类别181帧，视频与CSV一致；均明确标记synthetic。
- 本轮完成根README、Ubuntu/UI/RealMan实施计划、接口优先技术文档、交接和公开内容规则。
- 本轮已核对官方RM_API2接口名及ROS2 Humble与Ubuntu22.04的对应关系；本轮未安装SDK或连接实物。
- 用户随后确认另一系统已安装真机库且成功操作过机械臂；本轮未核验该系统。优先盘点/复用既有环境，禁止直接重装升级，RM_API2路线待版本确认。

## 未完成

1. Ubuntu版本、RealMan型号/固件、相机型号和机械结构确认。
2. Ubuntu依赖、PlatformIO/串口权限/相机环境部署及测试复跑。
3. 实物ESP32刷写、SA100零点/方向/PWM精度、实际串口/WiFi稳定性和相机同步标定。
4. RealMan只读适配器、控制进程、安全看门狗和现场停止验收。
5. Qt UI、ESP32 ARM/START/STOP协议、协调采集、图形化回放与导出。
6. 恢复准确的原上游子模块版本与目标checkpoint，定义真实动作/标签后进行公平真机评估。

## 本次证据与仓库状态

测试命令：`python -m unittest discover -s real_device/tests -v`。固件：`pio run -d real_device/firmware -e esp32-s3-n16r8`。历史构建日志及25项测试记录见[BUILD_AND_TEST](../real_device/docs/BUILD_AND_TEST.md)。本次文档发布检查不等于新增硬件验收。

Git根位于当前工作目录，新增根规则和文档；原源码保留。发布commit与后续脏状态用`git log -1 --oneline`和`git status --short`查询。公开地址为[Pan-cyb/CVPR_Real_device_test](https://github.com/Pan-cyb/CVPR_Real_device_test)；实际推送结果由交付消息和远端commit校验确认。

## 下一步

执行[Ubuntu计划](UBUNTU_IMPLEMENTATION_PLAN.md) M0：先只读盘点原已成功使用的SDK/环境/旧控制脚本，再建立采集用Python3.11隔离环境、运行25项测试与两类demo、记录相机/串口枚举结果，不发送机器人运动命令。完成后更新本文件的实测结果，而不是提前将计划改写为完成。
