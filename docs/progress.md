# 学习进度

记录日期：2026-10-28。Assignment 1（Building a Transformer LM）全部核心模块与官方适配器已实现并通过验证。

## 已完成组件与官方测试矩阵

| 模块分类 | 组件与实现文件 | 对应官方测试文件 / 用例 | 结果 |
| --- | --- | --- | --- |
| 基础层与激活 | `Linear` (`linear.py`)、`Embedding` (`embedding.py`)、`RMSNorm` (`rmsnorm.py`)、`silu` (`activations.py`)、`SwiGLU` (`swiglu.py`) | `test_linear`、`test_embedding`、`test_rmsnorm`、`test_silu_matches_pytorch`、`test_swiglu` | 全部通过 |
| 位置编码与注意力 | `RotaryPositionalEmbedding` (`rope.py`)、`softmax` (`activations.py`)、`scaled_dot_product_attention`、`MultiHeadSelfAttention` (`attention.py`) | `test_rope`、`test_softmax_matches_pytorch`、`test_scaled_dot_product_attention`、`test_4d_scaled_dot_product_attention`、`test_multihead_self_attention`、`test_multihead_self_attention_with_rope` | 全部通过 |
| Transformer 主干与整网 | `TransformerBlock`、`TransformerLM` (`transformer.py`) | `test_transformer_block`、`test_transformer_lm`、`test_transformer_lm_truncated_input` | 全部通过 |
| Byte-Level BPE 训练与分词器 | `train_bpe` (`bpe.py`)、`Tokenizer` (`tokenizer.py`) | `test_train_bpe.py`（3 项：含速度测试 `< 1.5s`、参考词表/merges 对齐、特殊 token 隔离）与 `test_tokenizer.py`（25 项：含 tiktoken 对齐、重叠特殊 token、流式 `encode_iterable`） | 全部通过（Linux 专属 `RLIMIT_AS` 2 项在 macOS 自动 skip） |
| 损失函数与梯度裁剪 | `cross_entropy`、`gradient_clipping` (`nn_utils.py`) | `test_cross_entropy`（含 `1000x` 大 logit 数值防溢出）、`test_gradient_clipping`（含冻结参数过滤） | 全部通过 |
| 优化器与学习率调度 | `AdamW`、`get_lr_cosine_schedule` (`optimizer.py`) | `test_adamw`、`test_get_lr_cosine_schedule` | 全部通过 |
| 数据加载、检查点与解码训练 | `get_batch` (`data.py`)、`save_checkpoint`、`load_checkpoint` (`serialization.py`)、`generate` (`decoding.py`)、`run_training_loop` (`train.py`) | `test_get_batch`（含均匀采样 $5\sigma$ 统计检验与非法设备检查）、`test_checkpointing`（模型与优化器状态精确恢复） | 全部通过 |

对独立快照执行 `scripts/check_completed.sh` 中的全量测试套件，共收集 **48 个官方测试用例**，在 macOS（Python 3.12.14、PyTorch 2.11.0、pytest 9.0.2）下结果为 **46 passed, 2 skipped in 4.66s**（跳过的 2 个用例为 `test_tokenizer.py` 中标记 `@pytest.mark.skipif(not sys.platform.startswith("linux"))` 的 `resource.RLIMIT_AS` 虚拟内存上限包装器测试）。

## 追溯与历史版本

官方基础提交：`a158843b20107949f1a8d7df1b05cd33b9166712`。本次发布快照保留该版本的官方脚手架，并包含 `cs336_basics/` 下全部 15 个模块文件以及完整对接的 `tests/adapters.py`。逐文件 SHA-256 校验值见 [snapshot.json](snapshot.json)。

- [`a39eb684710817f7f389626423fcb26ca0f6d629`](https://github.com/liangqianxing/cs336-labs/commit/a39eb684710817f7f389626423fcb26ca0f6d629)：阶段一（最初 7 类组件、8 项测试，对应学习记录 01–03）。
- [`ef5d204fa63a2a1fc32287ca641e27b4372d971b`](https://github.com/liangqianxing/cs336-labs/commit/ef5d204fa63a2a1fc32287ca641e27b4372d971b)：阶段二（增加不含 RoPE 的因果多头自注意力，共 8 类组件、9 项测试，对应学习记录 04）。
- 当前版本：完成 RoPE、含 RoPE 的多头自注意力、Pre-Norm Transformer Block、完整 Transformer LM、Byte-Level BPE 训练与分词器、数值稳定交叉熵、全局梯度裁剪、AdamW 优化器、余弦退火调度、随机窗口采样、检查点序列化及解码/训练脚本（对应学习记录 05–08）。
