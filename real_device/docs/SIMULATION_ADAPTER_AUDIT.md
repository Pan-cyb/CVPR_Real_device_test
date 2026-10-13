# 真机与现有仿真的适配审计

审计对象是 `script/jointgv/jointgv/{data.py,rollout_metrics.py,envs/rm65b_cartpole.py}` 和 `script/jointgv/utils/collect_dmc.py`，未修改这些文件。

| 项目 | 现有仿真 | 本真机协议 | 公平比较建议 |
|---|---|---|---|
| 角度 | `sum(joint2,joint3,joint5)+pole`，世界竖直角 | SA100轴相对外壳角 | 外壳朝向恒定（允许平移）才可直接标定；旋转时同步基座姿态后合成 |
| 初始范围 | 默认 ±7° | 门禁 ±10° | 比较时取共同子集 ±7°，或显式重跑仿真 ±10° |
| 终止角 | 90° | 120° | 主要结果统一按 90°离线裁剪；120°数据只作扩展 |
| 时长 | 20 s | 6 s | 统一比较首 6 s；不能用20 s成功率对6 s成功率 |
| 控制步长 | 0.02 s；collect默认 repeat=2，即25 Hz | 传感器约976 Hz捕获、200 Hz上报；RGB 30/60 Hz | 在真实测量时间轴上比较，按目标25 Hz下采样时保留mask |
| 动作 | 水平 pivot 速度 / 0.45 m/s | 手扶动作未知，未测量 | 不可填0；增加机器人命令/实测pivot速度后才能训练导出 |
| 数据形状 | T+1 image、T action | 连续RGB+角度 | 只能导出验证数据；动作和图标签齐全后另写明确转换器 |
| 终止语义 | fall 后可padding，`is_last/is_terminal`有专门语义 | 在测量越界时立即裁剪 | 评估时先去除仿真padding，按同一角阈值/时长重建mask |

`data.py` 训练窗口要求 `T+1` 图像和 `T` 动作，图监督还要求 node state/mask/relation（部分流程也要pixel/visible）。真机目前没有可定义的手扶 action 和 graph labels，故 `export-training` 主动报错；不能把未知手扶写成零动作，也不能伪造图标签。

`rollout_metrics.py` 的 SSIM、motion MSE、joint pixel和结构角指标需要预测帧/图结构。真实角度只能验证角预测，不能代替真实图像与预测图像的 MSE/SSIM/LPIPS。

仿真的 `arm_pose_for_cart_x` 采用 fixed-height/fixed-pitch 路径，`joint5 = TOOL_PITCH - joint2 - joint3`，因此理想命令下末端总 pitch 恒定；实际机器人仍可能存在跟踪误差。若真机也能证实外壳朝向恒定，SA100相对角可经零位/方向标定后比较；不能仅凭仿真约束假定真机姿态恒定。

当前代码快照中没有任何实际 `.npz/.pt/.pth/.ckpt` 数据或权重，README 的路径只是运行示例；三个 `.gitmodules` 目录存在但为空，且此快照没有 `.git` 元数据。需要先取得原完整 Git 仓库及正确版本，再初始化子模块和准备匹配 checkpoint。现状不能声称模型已训练或可直接真机推理。
