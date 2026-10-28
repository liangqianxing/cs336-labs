# 维护说明

## 记录方式

学习目录与发布仓库分开：在原课程目录继续学习，在这里同步已经完成的代码和保留的官方测试。不要把整段聊天、个人文件路径、浏览器信息或 API 配置提交到公开仓库。

当前官方基础提交固定为 `a158843b20107949f1a8d7df1b05cd33b9166712`。`docs/snapshot.json` 保存每个同步文件的 SHA-256、文件大小、同步日期和源 Git 提交，不保存源路径。本版本已同步 `cs336_basics/` 下的全部 Assignment 1 核心模块（`linear.py`、`embedding.py`、`rmsnorm.py`、`activations.py`、`swiglu.py`、`rope.py`、`attention.py`、`transformer.py`、`bpe.py`、`tokenizer.py`、`nn_utils.py`、`optimizer.py`、`data.py`、`serialization.py`、`decoding.py`、`train.py`）以及 `tests/adapters.py`。

## 每次更新

1. 在学习目录完成自己的实现，确认需要公开的内容。
2. 先使用 `--dry-run` 审阅白名单文件，随后运行同步。同步前会把所有文件读入内存并检查，拒绝符号链接、超过 8 MiB 的单文件和常见凭据特征。工具不会递归复制任意新文件，也不会操作源目录的 Git、删除源文件或自动提交。
3. 运行 `bash scripts/check_completed.sh`，根据实际结果更新 `docs/progress.md`。
4. 检查 `git diff --stat`、`git diff --check`、`git diff`，确认没有凭据或无关个人信息，再自行提交并推送。

凭据扫描是一层辅助检查，不能替代人工审阅。对于官方二进制 fixture 和作业 PDF，同步工具还要求它们与固定官方基础提交的内容相同；学习代码按原样复制。

## 测试范围

`scripts/check_completed.sh` 与 GitHub Actions CI 现已覆盖全部 7 个官方测试模块（`test_data.py`、`test_model.py`、`test_nn_utils.py`、`test_optimizer.py`、`test_serialization.py`、`test_tokenizer.py`、`test_train_bpe.py`），共 48 项测试用例。

## 体积与数据

仓库保留约 12 MiB 官方测试 fixture，包括测试所需的 `model.pt`（约 5.4 MiB）和 `tinystories_sample_5M.txt`。它们属于官方小型验证材料；大型训练数据集、训练检查点、虚拟环境和日志保持在本地，并由根 `.gitignore` 排除。
