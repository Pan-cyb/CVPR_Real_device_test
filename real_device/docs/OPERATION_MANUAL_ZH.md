# SA100—ESP32-S3 接线与操作手册

## 接线和安装

断电接线。SA100 允许 3.0–5.5 V，推荐与 ESP32 使用 3.3 V：

| SA100 | ESP32-S3-N16R8 | 说明 |
|---|---|---|
| VCC | 3V3 | 不接 5V 可减少电平风险 |
| GND | GND | 必须共地 |
| Signal | GPIO4 | PWM，不是 I²C；可在编译宏更改 |

GPIO4 避开原生 USB GPIO19/20 和板载 Octal PSRAM/Flash 管脚。若具体开发板把 GPIO4占作其他功能，先查其原理图再换脚。

SA100 输出约 976 Hz PWM，但频率会漂移。文档端点示意在 0 位只有约 0.25 μs 高电平、4095 位约 0.5 μs 低电平，因此固件使用 MCPWM/APB 硬件双边沿捕获，不用微秒软件中断。硬件逐周期捕获完整 high/period ticks；默认约200 Hz抽取最新完整周期上报并落盘，同时保存测量时刻、seq、boot_id 和丢失计数，因此CSV不是976 Hz逐周期全集。

厂家公式是 `position = ton × 4098 / period - 1`，位置标称 4096 增量。厂家 STM32 示例又用 `/4098 × 360`，两者不一致；本固件先按厂家公式求 position，再按 4096 counts/rev 转角并保留原始 ticks，便于复算。示例中的固定 310°零点不适用于本装置，也未采用。

机械安装是硬门禁：传感器输出是轴相对外壳角。如果外壳朝向相对世界恒定且杆只在相机正视平面内运动，标定后可作世界竖直角；外壳随机构平移没有影响。如果外壳发生旋转，必须同步机械臂基座/末端姿态并做坐标合成，本版本不能直接得到世界竖直角。模块型号本身不能证明具体开发板的引脚已引出，接线前仍需核对板卡原理图。

## 标定与相机选择

1. 固定外壳，确认转轴方向。把杆置于真实竖直，记录 SA100 原始角为 `--zero-deg`。
2. 向画面右侧轻推，确认角度应增还是减，据此设 `--direction 1` 或 `-1`。
3. 正视运动平面，镜头光轴尽量垂直。铰点贴绿色标记，杆尖贴红色标记；按现场光照调整 HSV。
4. USB RGB 至少选择稳定 30 fps、手动曝光可控、低运动模糊的相机。命令支持 `--fps 30/60 --width --height`，实际协商值写入 metadata。默认640×480/30 fps；60 fps 有利于视觉复核，但 SA100 原始捕获仍约 976 Hz、默认上报约 200 Hz。帧以JPEG暂存且有按时长/FPS计算的帧数上限，视频写失败会令trial无效。
5. RealSense 等深度相机适合需要设备曝光时间戳、全局快门或深度遮挡分析的场景；摆角本身不需要深度。接入时需实现设备时钟到 PC monotonic 的映射，不能沿用 OpenCV read-end 当曝光时刻。
6. OpenCV 时间戳是 read 完成代理。只有独立测得相机延迟后才设置 `--camera-latency-ms`。无硬触发时不能保证相机与传感器物理同步。

## 构建、烧录和串口选择

本工程验证的构建命令是技能脚本（见 `BUILD_AND_TEST.md`）。需要烧录时可用：

```powershell
$env:PATH = 'C:\Users\0\.platformio\penv\Scripts;' + $env:PATH
pio run -d firmware -e esp32-s3-n16r8 -t upload --upload-port COM5
```

固件默认 `ARDUINO_USB_CDC_ON_BOOT=0`，`Serial` 走板卡 UART0 的 USB-UART 桥，数据口通常是外接 CH340/CP210x 对应的 COM 口。原生 USB 接在 GPIO19/20；若板卡只有原生 USB 或希望改为 USB CDC，需要修改该宏和板卡 USB 配置，并重新确认 COM 口。不要同时让串口监视器和采集程序占用同一 COM 口。

把杆保持竖直，用下面命令读取三秒有效原始角并获得建议零点：

```powershell
python -m real_device probe --port COM5 --duration 3
```

再向画面右侧小幅转动重复 probe，依据原始角变化方向选择 `--direction 1` 或 `-1`。

可选 WiFi UDP：复制 `firmware/include/wifi_secrets.example.h` 为不公开的 `wifi_secrets.h`，填写 SSID/密码并重编译。ESP监听默认 UDP 4210；PC用 `probe --udp-device 192.168.1.50:4210` 或把 capture 的 `--port COM5` 换成 `--udp-device 192.168.1.50:4210`。WiFi掉线不会阻塞 MCPWM采样；恢复连接后PC必须重新同步，重启/断连中的 trial 判无效。

## 一次采集

1. 用 PlatformIO 构建后自行选择合适方式烧录；本仓库测试过程不烧录。打开设备后确认串口没有其他程序占用。
2. 运行 `capture`，每次使用新的输出目录。类别必须选 `free_fall`（按键后不再施力）或 `assisted_balance`（人尽量维持平衡，允许接触）。后者只是“人工辅助”类别，不等于自主控制成功；类别含义写入 metadata。
3. 预览状态达到 ready 后，把杆保持在竖直 ±10°内。按 `r` 的同时松手。按键只表示指令事件，采后必须从视频人工确认真正离手，并另行记录复核结论。
4. 系统在 6 秒或传感器测量角绝对值首次超过 120°时停止。传感器/相机 stale、重启或断连会使 trial 无效。
5. 检查 `metadata.json` 的 `valid_trial`、停止原因、实际 sensor/camera Hz、丢包/重复/乱序/重启数、时钟 drift/RTT/residual，再查看视频、CSV 和 metrics。

真机视频采完后运行视觉标记分析；六个HSV数依次为 low H/S/V 和 high H/S/V：

```powershell
python -m real_device vision artifacts\trial_001 `
  --pivot-hsv 45 100 100 85 255 255 --tip-hsv 0 150 120 10 255 255
```

建立数据集时以完整 trial 为最小单位，按采集日期、装置配置和初始倾斜侧分组后再切 train/val/test。同一视频的帧不得随机散入多个集合，否则会发生近重复帧泄漏。

安全上应设置机械限位/缓冲并让人可立即扶住杆；本工具不控制机械臂，也没有安全控制回路。
