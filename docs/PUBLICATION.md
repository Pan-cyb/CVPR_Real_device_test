# 公开发布与迁移范围

本仓库由用户明确授权公开发布，用于从Windows迁移到Ubuntu并继续真机实验开发。公开可见性不代表全部内容已完成硬件验证。

## 包含

- `real_device/` 的固件、PC源码、测试、文档、构建与测试日志。
- 两个明确标记synthetic的演示目录，用于检查文件格式和回放，不含真实现场人员录像。
- 原JointGV源码快照、已有benchmark参考资源、原MIT许可证和第三方声明；保留其目录结构和作者标注。
- Ubuntu实施计划、UI与RealMan接口设计、技术文档和交接资料。

## 保留本地，不发布

- `sa100/` 厂家PDF/CAD/STM32资料包：原文件保留本机；公开最小工作集不需要复制整个厂家包，新固件必需的接线/协议说明已在操作手册。若Ubuntu需查看原件，单独进行授权的私有迁移。
- `.work/` 提取/渲染的中间资料、`.pio/` 工具链产物、Python/IDE缓存。
- Wi-Fi密码、站点配置、私钥、`.env`、设备网络详情等。本仓库只有`wifi_secrets.example.h`模板。
- 真实实验录像、传感器数据、机器人轨迹、模型输出、私人配置。建议始终写入根`runs/`、`data/`、`recordings/`、`exports/`等被忽略目录。

`.gitignore`不是保密工具，也不会保护任意其他路径或已经被追踪的文件。每次发布前检查暂存文件清单、敏感信息和数据来源；不要对整机文件夹执行无检查的强制添加。不要自动把GUI导出目录同步到GitHub。

## 原源码与许可证

JointGV的MIT许可证在原快照根目录，版权为2026 JasonLyu；benchmark/dreamerv3自带Danijar Hafner的许可文本。它们保持原样，本仓库不替换作者或把所有第三方代码统一改签新许可证。其他参考组件的具体许可与版本应在恢复上游依赖时进一步核对；本次公开不为新增代码擅自追加整仓库授权条款。

原README所述上游为`https://github.com/JasonLvernex/Joint_Graph_Vedio_Latent_Dynamic_World_Model`，本次GitHub API返回404。无法由本地ZIP式快照还原其Git历史。原`.gitmodules`被保留为来源线索，但空目录没有gitlink，不能把它当可恢复的完整submodule配置。请向作者取得正确访问方式/版本。

原WorldModels2018目录含历史JSON模型参数，区别于本次缺失的JointGV/RM65-B目标checkpoint；旧报告中“没有.pt/.pth/.ckpt”不应解读成“任何格式的模型文件都没有”。

## Ubuntu恢复检查

克隆后先对照根README确认源码、测试和两类synthetic样例存在；重新构建固件，不能期待被忽略的Windows二进制随Git恢复。另行拷贝需要的私人配置/厂家资料，重新识别串口、相机与IP，重新标定，不复用Windows的COM名或单调时间戳。
