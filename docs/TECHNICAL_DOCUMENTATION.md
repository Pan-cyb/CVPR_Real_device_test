# 项目技术文档：接口与数据契约

## 1. 操作接口（当前已实现）

先在Python3.11+环境安装`python -m pip install -e ./real_device`。模块入口为`python -m real_device`，也可使用安装后的`real-device`命令。

| 命令 | 输入 | 输出 / 边界 |
|---|---|---|
| `demo --output DIR --category free_fall/assisted_balance` | 不需要硬件，新目录 | 纯synthetic视频、CSV、manifest及角度指标 |
| `probe --port PORT` 或 `--udp-device IP:PORT` | 已烧录ESP/SA100，默认3s | 原始角统计、建议零点；UDP先握手 |
| `capture --output DIR ...` | 串口/UDP、相机、类别、零点、方向、外壳几何确认 | 预览、按r记录、自动/人工停止；无机器人运动 |
| `align DIR --max-gap-ms 25` | sensor.csv、frames.csv | 圆周插值aligned.csv，禁止外推/跨大gap |
| `vision DIR` | RGB视频、aligned.csv、两组HSV范围 | 独立视觉角及对传感器的误差；缺标记为无效 |
| `evaluate DIR predictions.csv` | 每行angle_deg及frame_id或t_pc_ns之一 | wrapped角度MAE/RMSE；拒绝重复键，不外推 |
| `export-validation INPUT.csv OUTPUT.npz` | 传感器/对齐格式数据 | 验证数据，不能直接作为JointGV训练episode |
| `export-training DIR` | 当前缺实测动作与所需标签 | 主动拒绝；不是已可用的训练转换器 |

完整参数见`--help`及[操作手册](../real_device/docs/OPERATION_MANUAL_ZH.md)。`capture`支持`--fps`、`--width/height`、`--camera-latency-ms`；就绪等待超时当前是Python函数参数，尚未暴露CLI选项。现有界面是OpenCV预览窗口，不是完整图形控制平台。

## 2. 硬件与通信接口

SA100 VCC→3V3、GND→GND、Signal→默认GPIO4；实际开发板引脚需核验。MCPWM逐周期捕获高电平/周期计数，ESP默认约200Hz抽取最新完整周期发送。厂家典型PWM约976Hz，不意味着PC每秒保存976个样本。

固件目标是ESP32-S3、16MB Flash、8MB OPI PSRAM，PlatformIO固定espressif32 6.5.0。默认UART0/USB-UART桥921600 baud；可选Wi-Fi UDP 4210。串口每行一个JSON，UDP每包一个JSON。Wi-Fi配置文件必须本地保管。

v1样本：`v, type=sample, boot_id, seq, t_device_us, high_ticks, period_ticks, raw_angle_deg, valid, capture_drops, tx_drops`。高电平占比按`position=high/period*4098-1`换算，再按4096增量/圈转为角度；原始ticks保留用于复核。

v1同步：PC发`sync_req`（id、t0_pc_ns），ESP返回`sync_resp`（原id/t0、boot_id、t1_device_us、t2_device_us），PC补t3。选低RTT交换拟合设备时钟到PC单调时钟，保存漂移与残差。测量时刻是完成PWM周期的设备捕获处理时间，仍含PWM/中断路径的不确定度；没有硬件实验不能承诺纳秒级或某一绝对精度。

当前ESP持续运行；PC停止的是记录会话。未来显式ARM/START/STOP及能力协商见计划书，不能把这些命令发给v1后当成功。

## 3. 时间、文件和指标

`time.monotonic_ns()`与`esp_timer_get_time()`都不是UTC；两台主机的monotonic数不能直接比较。需要跨进程时统一同一主机时基；跨主机需另建映射。文件名日期只用于索引，不能替代测量时间。

| 文件 | 数据含义 |
|---|---|
| `sensor.csv` | 原始ticks、设备时间、映射PC时间、接收时间、校准角和有效性 |
| `frames.csv` | 视频帧号、read start/end、对齐时间、时间戳类型 |
| `sync.csv` | 四时间戳交换与RTT |
| `events.csv` | 就绪、r键、开始、停止/修正/异常事件 |
| `aligned.csv` | 每帧传感器角与valid；vision分析后增加视觉角/有效性 |
| `metadata.json` | 类别、开始结束、停止原因、几何、时钟、相机/传输质量 |
| `metrics.json` | 角度误差结果；真实capture后需执行vision分析生成 |
| `rgb.mp4` | 视频；时间以frames.csv为准，不能仅凭容器固定FPS恢复真实采样间隔 |

正方向/零点在配置中明确。unwrap后插值，再计算圆周差，避免359°与1°误差被算成358°。无效值不填零。合成demo与实测不同schema细节可见原始metadata，消费者应检查schema/字段，不依赖演示专用字段。

现有JPEG缓存仅持有压缩字节和元数据，最终统一裁剪写MP4/CSV。视频有损压缩应纳入图像评估协议，不能拿其误差推算传感器精度。角度MAE/RMSE与原JointGV的图像MSE/局部SSIM/图节点像素误差不是同一指标；本工具的辅助`ssim_global`不能当作原仿真局部SSIM。

## 4. 内部模块

| 路径（位于real_device） | 责任 |
|---|---|
| `firmware/src/main.cpp` | PWM捕获、有界队列、串口/UDP发送与同步 |
| `firmware/platformio.ini` / `boards/esp32-s3-n16r8.json` | 构建环境/板卡存储参数 |
| `pc/real_device/protocol.py` | 解析、有效性、序号与boot跟踪 |
| `sync.py` / `angles.py` | 时钟拟合、圆周角与插值 |
| `session.py` | 起始门禁、记录状态、期限/角度/陈旧数据停止 |
| `camera.py` | USB相机适配、时间戳语义、深度相机扩展契约 |
| `pipeline.py` | 接收线程、采集协调、缓冲/存储、probe/demo/vision |
| `metrics.py` / `export.py` / `cli.py` | 误差、验证导出、命令行入口 |
| `tests/test_core.py` | 单元与fake设备/本地UDP集成回归 |

原仿真总体结构见[项目概览](../real_device/docs/PROJECT_OVERVIEW.md)，具体差异见[适配审计](../real_device/docs/SIMULATION_ADAPTER_AUDIT.md)。新增RobotAdapter、Qt UI、安全看门狗与ESP会话协议尚不存在。

## 5. 验证与限制

Windows历史证据：25项测试通过，串口/WiFi两个分支编译通过，85帧自由倒落与181帧辅助类别synthetic视频分别与CSV一致。用户设备尚未接入验证。Ubuntu需独立重跑；GUI显示、udev、SDK二进制兼容性不被Windows测试覆盖。

当前最重要的限制：目标模型权重和原子模块缺失；相机型号/曝光时钟未定；SA100安装几何未定；未知人工动作没有仿真action语义；没有机器人安全控制回路。下一阶段按M0–M7推进，不直接把现有采集进程当作机械臂闭环控制器。
