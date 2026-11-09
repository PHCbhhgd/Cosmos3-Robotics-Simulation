# 复现指南

本版本完成 CPU 输入测试与脚本准备，GPU 生成尚未运行。下列推理命令是待验证流程，不是成功运行记录。固定模型和源码提交见 [versions.json](../configs/versions.json)。

## 1. 获取项目与轻量依赖

```bash
git clone https://github.com/PHCbhhgd/Cosmos3-Robotics-Simulation.git
cd Cosmos3-Robotics-Simulation
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-cpu.txt
python -m unittest discover -s tests -v
python scripts/audit_public_repo.py
```

需要 Linux、Git、curl。视频检查脚本使用 OpenCV；原生 Framework 还需要其文档指定的 FFmpeg 共享库、CUDA toolkit、NPP 和匹配驱动，不能用仅能打开 MP4 代替运行时依赖检查。

## 2. 下载固定版本

```bash
# 只下载输入与元数据
python scripts/download_assets.py bimanual
python scripts/download_assets.py transfer

# 完整模型；权重绝不加入 Git
python scripts/download_assets.py bimanual --weights
python scripts/download_assets.py transfer --weights

# 可选长序列 Transfer example1
python scripts/download_assets.py transfer --long
```

脚本将文件写入 `storage/models/<模型名称>/`，并记录逐文件 SHA-256。未完成文件使用 `.partial`，不会作为完整分片使用。重复执行跳过已完成文件；同一目标目录不要并发启动两个下载进程。

当前服务器官方站点连接失败，已采用 `--endpoint https://hf-mirror.com` 获取公开模型和样例。这个脚本**不读取或发送 HF token**，不修改系统代理、全局环境或现有登录；需要 gated 模型访问时应使用官方 HF 端点和官方登录流程。不要把 token 放进 URL、配置或 Git。

## 3. 安装固定 Framework

```bash
# Bimanual 推理依赖
bash scripts/setup_framework.sh bimanual

# 若要运行 Transfer，使用其官方 training-group 环境要求
bash scripts/setup_framework.sh transfer

# 切换到 Framework 的 Python 3.13 环境
source repos/cosmos-framework/.venv/bin/activate
```

脚本使用固定的 `uv==0.11.29`，先 `uv lock --check`，再 `uv sync --locked`。锁检查失败即停止，不会重新生成锁文件。默认 CUDA 13.0；其他 CUDA 版本应按同一提交的官方指南选匹配依赖组。本机已验证锁检查，未重新安装整个 Transfer 训练依赖组。

本次复用了原有隔离的 Python 3.13 / PyTorch 2.10 环境进行官方 CPU 校验和原生 `--help` 检查；具体版本见 [runtime-checks.json](../reports/runtime-checks.json)。该隔离启动器包含本机路径，保留在忽略的 `artifacts/` 中，公开步骤不依赖旧驾驶仓库。

## 4. 准备并校验 Bimanual

```bash
python repos/cosmos-framework/tools/check_bimanual_mixed_inputs.py \
  --checkpoint-path storage/models/Cosmos3-Nano-Sim-Bimanual

python scripts/check_bimanual_boundaries.py

# 33 帧短 smoke；只截取前 32 个既有动作，不合成“停止动作”
python scripts/prepare_inputs.py bimanual --action-count 32 \
  --output artifacts/bimanual-smoke

# 完整官方 AgiBot 001：900 动作、901 帧
python scripts/prepare_inputs.py bimanual \
  --output artifacts/bimanual-agibot001

# 另一机器人 / 混合域 / 长序列：--sample 使用完整官方 name
python scripts/prepare_inputs.py bimanual \
  --sample agibot_mixed_007_bulk_seed2026 \
  --output artifacts/bimanual-mixed007
```

本地请求将官方 URL 映射到已下载的相同素材，不修改原始 prompt 或动作数值；短 smoke 同时裁剪逐行动作域、`num_frames` 和 `action_chunk_size`。原始 hosted 清单完整保存在模型目录。测试 60 秒时选择 `agibot_original_long60_019_bulk_seed2026`，不要把它缩短后仍宣称测了 60 秒。

## 5. 准备 Transfer

```bash
python scripts/prepare_inputs.py transfer \
  --output artifacts/transfer-example2

# 机器人真实 RGB 前 121 帧，生成 edge / blur 两个请求
python scripts/prepare_robot_transfer.py \
  --output artifacts/robot-transfer
```

官方通用请求位于 `artifacts/transfer-example2/requests/{depth,seg,edge,blur}.json`。机器人专项请求位于 `artifacts/robot-transfer/{edge,blur}.json`。

Transfer 使用固定 Nano tokenizer 与 Wan VAE。官方 HF CLI 下载方式如下；若已有相同 revision 的完整缓存，可以复用，不复制到 Git。

```bash
hf download nvidia/Cosmos3-Nano \
  --revision e59a53c25979a090fa8706c9acc0c254a6e89b92 \
  --include 'text_tokenizer/*' --local-dir storage/resources/Nano
hf download Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth \
  --revision 921dbaf3f1674a56f47e83fb80a34bac8a8f203e \
  --local-dir storage/resources/Wan
```

## 6. 分配 GPU 后先探测，再推理

`--gpus` / `--gpu` 必须指定已分配给你的物理编号；下面的 0、1 仅为示例。脚本会拒绝已有计算进程、显存占用过高或繁忙的卡，不会结束现有任务。占用检查不是资源预约，需先由管理员或调度器分配。

```bash
# 只查询，不分配 CUDA 张量
python scripts/check_gpus.py --gpus 0,1

# 小规模 native NATTEN 探测；不加载 30 GB 模型
python scripts/probe_attention.py --gpu 0 \
  --output artifacts/attention-probe.json

# 先看启动命令；dry-run 不代表 GPU 兼容
python scripts/run_inference.py bimanual --gpus 0,1 \
  --input artifacts/bimanual-smoke/input.jsonl \
  --output outputs/bimanual-smoke-001 --dry-run

# 正式运行；使用全新的输出目录
python scripts/run_inference.py bimanual --gpus 0,1 \
  --input artifacts/bimanual-smoke/input.jsonl \
  --output outputs/bimanual-smoke-001

python scripts/run_inference.py transfer --gpus 0,1 \
  --input artifacts/robot-transfer/edge.json \
  --vae storage/resources/Wan/Wan2.2_VAE.pth \
  --text-tokenizer storage/resources/Nano/text_tokenizer \
  --output outputs/robot-transfer-edge-001
```

启动器调用官方 `cosmos_framework.scripts.inference`。Bimanual 保持四步 SDE、KV 64、sink 8，并遵循官方模型验证命令中的 `--no-guardrails`；这是明确记录的验证设置，部署时应使用自己的 guardrail 策略。Transfer 保留默认 guardrails，首次需要对应授权及已缓存资源。

两张及以上卡时使用 `torch.distributed.run`、CP=1、DP shard=所选卡数。这是当前显存条件下的**待验证分片配置**，不是已经跑通的多卡方案。单卡示例对应官方进程形式，但每个 BF16 主模型约 30.3 GB；显存还要容纳 VAE、KV 和激活，不可把 30.3 GB 当作完整运行显存要求。

Transfer `example1` 使用其长序列请求并追加 `--long`，对应 KV 30 / sink 3。不要把 example2 的缓存设置直接当成长序列结论。

每次运行保留 `launch.json` 与 `console.log`。启动进程返回 0 仅记录为 `process_exited_zero`；还需核对 `sample_outputs.json` 并检查生成文件。

## 7. 核查输出并更新报告

```bash
# 实际 MP4 路径以官方 sample_outputs.json 为准
python scripts/review_video.py outputs/your-run/path/to/vision.mp4 \
  --output artifacts/your-run-review --expected-frames 121
```

工具完整解码并输出 SHA、帧数、FPS、尺寸、逐秒抽帧与像素变化统计。Bimanual full baseline 应为 901 帧；33 帧 smoke 和 Transfer 121 帧不要混用预期。随后人工检查动作跟随、接触、物体身份、遮挡后恢复、域切换和长时漂移。

像素变化大可能是正常运动，像素变化小可能是冻结；这些统计没有自动物理质量及格线。只有实际生成、完整解码和人工审核后，才把 [测试矩阵](../configs/test_matrix.json) 中对应项目从 planned 改为已测试。

## 8. 发布前检查

```bash
python scripts/audit_public_repo.py
python -m unittest discover -s tests -v
```

`storage/`、`repos/`、`artifacts/`、`outputs/` 均被 Git 忽略。大视频留在本地；公开一个带时间戳的代表图片即可。当前没有模型生成视频或 Release 大附件。
