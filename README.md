# CS336 学习与实验

跟随 Stanford CS336，从基础张量运算开始实现语言模型组件。这里保存个人学习代码、验证结果和维护记录；配套文章在 [CS336 专栏](https://liangqianxing.github.io/learning/cs336)。

截至 2026-10-20，Assignment 1 已实现 Linear、Embedding、RMSNorm、SiLU、SwiGLU、Softmax、Scaled Dot-Product Attention。对应的 **8 个官方测试通过**，其中 Attention 包含三维、四维输入两个用例。多头注意力、RoPE、Transformer、BPE、优化器和训练流程尚未完成。完整进度与验证边界见 [docs/progress.md](docs/progress.md)。

## 来源与目录

官方脚手架来自 [stanford-cs336/assignment1-basics](https://github.com/stanford-cs336/assignment1-basics)，基础提交为 [`a158843b20107949f1a8d7df1b05cd33b9166712`](https://github.com/stanford-cs336/assignment1-basics/commit/a158843b20107949f1a8d7df1b05cd33b9166712)。该版本保留了上游 README 的课程年份原文；本仓库以提交号记录来源。保留官方 LICENSE、README、作业说明、教学约束及测试 fixture。

```text
assignment1-basics/        官方脚手架 + 当前学习代码快照
  cs336_basics/           已实现的基础组件
  tests/                 官方适配器、测试、参考快照和小型 fixture
docs/progress.md          当前进度与验证结果
docs/snapshot.json        各同步文件的 SHA-256 与基础提交
docs/maintenance.md       后续同步与发布方法
scripts/sync_snapshot.py  从学习目录单向复制白名单文件
scripts/check_completed.sh  仅验证当前已实现的组件
```

## 复现验证

安装 [uv](https://docs.astral.sh/uv/getting-started/installation/) 后，在仓库根目录运行：

```sh
bash scripts/check_completed.sh
```

脚本使用 Python 3.12 和官方 `uv.lock`，只执行当前完成组件的 8 个测试。也可以通过 `CS336_PYTHON=/absolute/path/to/python` 使用已经安装依赖的 Python 3.12 环境。首次运行需要下载 PyTorch 等依赖。GitHub Actions 使用同一份测试清单；测试通过仅代表这些用例通过。

## 学习目录与同步

日常学习继续在独立的本地课程目录进行，例如 `$HOME/projects/ai/cs336/assignment1-basics`。本仓库是用于维护和公开记录的独立快照，不改变学习目录的 Git 状态。

```sh
python3 scripts/sync_snapshot.py --source "$HOME/projects/ai/cs336/assignment1-basics" --dry-run
python3 scripts/sync_snapshot.py --source "$HOME/projects/ai/cs336/assignment1-basics"
bash scripts/check_completed.sh
git diff --stat
git diff
```

同步工具只接收显式 `--source`，只复制已列出的文件，不复制 `.git`、虚拟环境、缓存、凭据或训练数据，也不删除或修改源文件。新增学习模块时，需要先人工审查并更新同步白名单及验证范围，详见 [维护说明](docs/maintenance.md)。

官方课程文件遵循 [原始 MIT 许可证](assignment1-basics/LICENSE)；新增维护文件同样采用 MIT 许可证。本仓库遵循保留的 [教学约束](assignment1-basics/AGENTS.md)，用于记录自己完成的学习成果。
