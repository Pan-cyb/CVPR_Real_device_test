# 项目概览

## 原 JointGV 工程

原工程位于 `../Joint_Graph_Vedio_Latent_Dynamic_World_Model-main/Joint_Graph_Vedio_Latent_Dynamic_World_Model-main/`。它有四种可运行骨干：`david_ha`（VAE + MDN-LSTM）、`dreamerv3`（原生CNN视觉token + discrete RSSM）、`dreamerv3_vae`（JointGV VAE token + DreamerV3 RSSM）和 `dino_wm`（冻结DINOv2 patch token + causal predictor）。每种骨干都可开启JointGV图条件，或用 `--disable-graph-conditioning` 运行匹配的graph-off基线。

总体数据流是：RGB先由骨干原生视觉编码器形成视觉latent/token；显式图观察器从RGB提取节点、关系和可见性，图动力学预测下一步结构，graph plugin把图校正量注入视觉动力学；骨干再依据当前latent和动作预测未来latent/图像，rollout评估计算图像、运动、joint pixel和可用的结构角指标。组合入口主要在 `script/jointgv/jointgv/world_model.py`，编码器、图构建/动力学/plugin分别位于 `jointgv/backbones/` 和 `jointgv/blocks/`。

RM65-B cart-pole仿真环境入口是 `jointgv/envs/rm65b_cartpole.py`；`utils/collect_dmc.py` 采集 `T+1` 图像、`T` 动作及状态/图标签；`train.py` 做离线世界模型训练，`train_online_dmc.py` 负责DreamerV3在线控制实验，`evaluation/evaluate_rollout.py` 和 `evaluation/evaluate_ablation.py` 做rollout及graph-on/off比较。当前快照没有 `.npz` 数据或 `.pt/.pth/.ckpt` 权重，三个benchmark子模块目录为空，且外层没有Git元数据；README命令是能力说明，不能据此声称已有可运行checkpoint或真机模型。

## 新增真机工具

数据路径为 `SA100 PWM → ESP32 MCPWM硬件捕获 → JSON Lines串口/可选UDP → PC接收线程 → PC monotonic_ns → 原始CSV → 圆周插值 → 帧级对齐与指标`。传感器监听、相机读取和停止看门狗相互独立，快速越过 ±120° 不依赖低帧率视频循环发现。

PC 端状态为 `wait_ready → ready → released → recording → stopped/invalid`。开始门禁要求传感器和相机已有稳定且仍新鲜的数据、时钟拟合合格、竖直零位已标定、且操作者确认外壳朝向固定/单平面几何；初始角绝对值超过 10°、NaN或陈旧数据会拒绝。停止取真实事件的最早测量时刻：±120°或开始后 6 秒。迟到但测量时刻早于deadline的越界样本会回溯修正裁剪点；传感器/相机 stale、重启或断连的 trial 标为无效。

相机接口 `camera.py` 的 `CameraFrame` 同时保留 `read_start_ns`、`read_end_ns`、用于对齐的时间戳和 `timestamp_kind`。OpenCV USB RGB 当前使用 `host_monotonic_read_end_proxy`，这只是主机读完成时刻，不能当曝光时刻。`DeviceTimestampCamera` 是深度相机/RealSense 扩展契约：适配器必须把设备曝光时钟映射到 PC monotonic 并明确命名映射。目前没有声称支持任意深度相机原生时间戳。

`vision.py` 用不同颜色的铰点和杆尖标记独立从 RGB 求世界竖直角；缺标记、遮挡或两点过近均输出 invalid/NaN，不生成 0。导入模型预测 CSV 时必须二选一提供 `frame_id` 或 `t_pc_ns`。角度 MAE/RMSE 使用圆周误差。图像预测的 MSE/SSIM/LPIPS 必须有真实和预测 RGB 图像；单个角度不能替代图像指标，当前只提供成对图像的 MSE 和全局 SSIM，未捆绑 LPIPS 模型。

现有快照没有本次JointGV/RM65-B目标checkpoint或训练数据，不能假称模型已能在真机上运行。旧WorldModels基线另有历史JSON模型参数，不是目标实验权重。本工具只输出角度验证 NPZ；训练导出默认拒绝，直到有明确定义且实际测量的动作和所需图标签。
