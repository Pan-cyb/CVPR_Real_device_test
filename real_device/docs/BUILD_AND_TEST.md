# 构建与测试

PlatformIO 环境固定为 `espressif32@6.5.0`、Arduino-ESP32 2.0.14、`esp32-s3-n16r8` 自定义板卡描述；Flash 16 MB QIO，PSRAM 8 MB OPI。构建不等于实物板卡、串口或PSRAM已验证。

```powershell
$env:PATH = 'C:\Users\0\.platformio\penv\Scripts;' + $env:PATH
python C:\Users\0\.codex\skills\embed-ai-tool\skills\build-platformio\scripts\platformio_builder.py --detect
python C:\Users\0\.codex\skills\embed-ai-tool\skills\build-platformio\scripts\platformio_builder.py --list-envs --project-dir firmware
python C:\Users\0\.codex\skills\embed-ai-tool\skills\build-platformio\scripts\platformio_builder.py --project-dir firmware --env esp32-s3-n16r8
```

PC测试和无硬件演示：

```powershell
$env:PYTHONPATH = "$PWD\pc"
python -m unittest discover -s tests -v
python -m real_device demo --output artifacts\demo_regression_local --category free_fall
```

本次实际结果写入 `artifacts/test_log.txt`（25项通过，含本地UDP模拟固件握手、JPEG缓存不持有原图、无有效帧失效门禁，以及可注入假设备的触发、重启、失联/阻塞、初始化失败和视频/CSV一致性）、`artifacts/build_log.txt`（默认串口固件成功生成 ELF/BIN）、`artifacts/wifi_build_log.txt`（临时假凭据、`ENABLE_WIFI_UDP=1` 分支编译成功）、`artifacts/demo_validation_log.txt`（85帧视觉角指标）和 `artifacts/assisted_demo_log.txt`。后者对应 `artifacts/demo_assisted_balance/` 的纯synthetic六秒演示：1201个200 Hz传感器样本、181个30 Hz帧，停止原因为duration。WiFi验证后已删除临时凭据头并重新构建默认串口固件。没有运行 upload，也没有打开或驱动任何实体设备。
