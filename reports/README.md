# 本次运行证据

这里只保存精简、可公开的检查结果。**所有生成状态仍为待运行。**

| 文件 | 实际含义 |
| --- | --- |
| [runtime-checks.json](runtime-checks.json) | 官方锁文件检查通过、既有隔离环境的版本、CLI help 返回 0 |
| [bimanual-official-preflight.json](bimanual-official-preflight.json) | 官方工具验证 10 行 / 7 个独立样例；没有执行视频生成 |
| [bimanual-boundaries.json](bimanual-boundaries.json) | 官方动作校验器接受有效输入、拒绝 9 类异常输入 |
| [input-videos.json](input-videos.json) | 7 个输入视频的完整解码、尺寸、帧数、FPS 和 SHA |
| [preflight.json](preflight.json) | 当时硬件占用与完整权重分片数量；不代表未来实时状态 |

`media/` 中只有这些输入的预览或动作数据图。原始下载 receipt、控制请求、日志、官方训练 RGB 与权重留在本地 `storage/`、`artifacts/`；后续生成结果进入 `outputs/`。
