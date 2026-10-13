# 协议与结果说明

串口和 UDP 都使用 UTF-8 JSON Lines/单个 JSON 数据报，版本 `v=1`。ESP 的 `t_device_us` 是开机单调时钟；PC 的 `t_pc_ns` 是 `time.monotonic_ns()`，都不是 UTC。

样本字段：`boot_id, seq, t_device_us, high_ticks, period_ticks, raw_angle_deg, valid, capture_drops, tx_drops`。`t_device_us` 表示完成该 PWM 周期捕获的测量时刻；`t_receive_ns` 另存为到达时刻，绝不替代测量时刻。新 boot_id 表示设备重启；seq 空洞、重复和乱序分别统计。非法布尔、非有限角度或负计数会被拒绝。

PC 发送 `{"v":1,"type":"sync_req","id":N,"t0_pc_ns":...}`，ESP 返回接收/发送设备时刻 `t1_device_us/t2_device_us`。PC 加上 `t3_pc_ns` 形成四时间戳交换，只用低 RTT 子集拟合 `PC_ns = slope × ESP_ns + intercept`。原始交换、RTT、拟合残差和 drift 都保存；高 RTT、残差过大或漂移异常不能通过开始门禁。

结果文件：

- `sensor.csv`：每个接受的传感器包，原始 ticks、设备/PC测量时刻、接收时刻和标定角。
- `frames.csv`：帧号、read start/end、对齐时间戳及其语义。
- `sync.csv`：全部四时间戳交换和 RTT。
- `events.csv`：readiness、release key、recording start/stop；release 不是自动离手检测。
- `aligned.csv`：帧时刻的圆周插值角和 valid mask。区间外不外推，跨大 gap 不插值。
- `metadata.json`：类别、几何/相机语义、停止原因、实际频率、时钟和传输统计。
- `metrics.json`：视觉角对传感器角的 wrapped MAE/RMSE；不把角度指标称作图像指标。

预测角 CSV 必须含 `angle_deg`，并且恰好含 `frame_id` 或 `t_pc_ns` 之一：

```powershell
python -m real_device evaluate artifacts\trial_001 predictions.csv
```

仅供真机角度验证的 NPZ：

```powershell
python -m real_device export-validation artifacts\trial_001\sensor.csv artifacts\trial_001\angles.npz
```

输出目录不覆盖已有 trial。数据集切分以完整 trial 为单位，并按日期、硬件配置和初始倾斜侧分组，避免同一连续视频的近邻帧跨 train/val/test 泄漏。
