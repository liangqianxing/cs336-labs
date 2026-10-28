# CS336 学习与实验

跟随 Stanford CS336，从零构建标准 Transformer 语言模型（LM）的全套核心组件与训练闭环。这里保存个人学习代码、官方测试验证结果和维护记录；配套学习记录在 [CS336 专栏](https://liangqianxing.github.io/learning/cs336)。

截至 2026-10-28，Assignment 1（Building a Transformer LM）的全部核心模块均已从零实现并通过官方测试套件验证：涵盖 Byte-Level BPE 分词器训练与流式编码（`train_bpe`、`Tokenizer`）、基础张量层（`Linear`、`Embedding`、`RMSNorm`、`SiLU`、`SwiGLU`、`RotaryPositionalEmbedding`）、注意力与模型主干（`softmax`、`scaled_dot_product_attention`、因果 `MultiHeadSelfAttention`、Pre-Norm `TransformerBlock`、完整 `TransformerLM`）、优化与损失（数值稳定 `cross_entropy`、`gradient_clipping`、解耦权重衰减 `AdamW`、Warmup + 余弦退火 `get_lr_cosine_schedule`）以及数据采样、Checkpoint 序列化与 Temperature / Top-$p$ 解码（`get_batch`、`save_checkpoint`、`load_checkpoint`、`generate`、`train`）。官方测试集共 **48 项测试（macOS 下 46 passed, 2 skipped；2 项跳过为仅限 Linux `RLIMIT_AS` 的内存上限装饰器测试）** 全部通过。完整进度与验证细节见 [docs/progress.md](docs/progress.md)。

## 来源与目录

官方脚手架来自 [stanford-cs336/assignment1-basics](https://github.com/stanford-cs336/assignment1-basics)，基础提交为 [`a158843b20107949f1a8d7df1b05cd33b9166712`](https://github.com/stanford-cs336/assignment1-basics/commit/a158843b20107949f1a8d7df1b05cd33b9166712)。该版本保留了上游 README 的课程年份原文；本仓库以提交号记录来源。保留官方 LICENSE、README、作业说明、教学约束及测试 fixture。

```text
assignment1-basics/        官方脚手架 + 完整 Assignment 1 实现快照
  cs336_basics/           从零实现的 BPE、Transformer LM、AdamW 与训练/解码组件
  tests/                 官方适配器、全量单元测试、参考快照和小型 fixture
docs/progress.md          当前完成度与全量测试结果
docs/snapshot.json        各同步文件的 SHA-256 与基础提交
docs/maintenance.md       后续同步与发布方法
scripts/sync_snapshot.py  从学习目录单向复制白名单文件
scripts/check_completed.sh  执行全部 7 个官方测试模块的验证脚本
```

## 复现验证

安装 [uv](https://docs.astral.sh/uv/getting-started/installation/) 后，在仓库根目录运行：

```sh
bash scripts/check_completed.sh
```

脚本使用 Python 3.12 和官方 `uv.lock`，执行全部 7 个官方测试文件（`test_data.py`、`test_model.py`、`test_nn_utils.py`、`test_optimizer.py`、`test_serialization.py`、`test_tokenizer.py`、`test_train_bpe.py`）。也可以通过 `CS336_PYTHON=/absolute/path/to/python` 使用已经安装依赖的 Python 3.12 环境。此外可运行最小训练与采样闭环自检：

```sh
cd assignment1-basics
python -m cs336_basics.train --max-iters 16 --batch-size 4 --context-length 16 --d-model 32 --num-layers 2 --num-heads 4 --d-ff 64 --vocab-size 128
```

## 学习目录与同步

日常学习在独立的本地课程目录进行，例如 `$HOME/projects/ai/cs336/assignment1-basics`。本仓库是用于维护和公开记录的独立快照，不改变学习目录的 Git 状态。

```sh
python3 scripts/sync_snapshot.py --source "$HOME/projects/ai/cs336/assignment1-basics" --dry-run
python3 scripts/sync_snapshot.py --source "$HOME/projects/ai/cs336/assignment1-basics"
bash scripts/check_completed.sh
git diff --stat
git diff
```

同步工具只接收显式 `--source`，只复制已列出的白名单文件，不复制 `.git`、虚拟环境、缓存、凭据或大型训练数据，也不删除或修改源文件。详见 [维护说明](docs/maintenance.md)。

官方课程文件遵循 [原始 MIT 许可证](assignment1-basics/LICENSE)；新增维护文件同样采用 MIT 许可证。本仓库遵循保留的 [教学约束](assignment1-basics/AGENTS.md)，用于记录自己完成的学习成果。
