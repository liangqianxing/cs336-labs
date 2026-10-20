# 学习进度

记录日期：2026-10-20。Assignment 1 尚在进行中。

## 本次已验证

| 组件 | 对应官方测试 | 结果 |
| --- | --- | --- |
| Linear | `test_linear` | 通过 |
| Embedding | `test_embedding` | 通过 |
| RMSNorm | `test_rmsnorm` | 通过 |
| SiLU | `test_silu_matches_pytorch` | 通过 |
| SwiGLU | `test_swiglu` | 通过 |
| Softmax | `test_softmax_matches_pytorch` | 通过 |
| Scaled Dot-Product Attention | `test_scaled_dot_product_attention`、`test_4d_scaled_dot_product_attention` | 均通过 |
| 因果 Multi-Head Self-Attention（不含 RoPE） | `test_multihead_self_attention` | 通过 |

对独立快照执行 `scripts/check_completed.sh` 中的显式测试清单，本轮增加多头自注意力后共验证 **9 个用例、8 类组件**，结果为 **9 passed in 0.33s**。本次复用了本地现有依赖环境：Python 3.12.14、PyTorch 2.11.0、pytest 9.0.2。验证通过测试适配器调用本仓库中的组件；官方参考快照及 fixture 保持原样。

结果说明：覆盖官方用例的前向数值比较、SiLU 与 PyTorch 的一致性、Softmax 的数值稳定性、缩放点积 Attention 的掩码及三维/四维输入，以及多头自注意力的 Q/K/V 投影、拆头、因果注意力、合并和输出投影。不据此声明完整梯度检查、所有 dtype/device、性能或额外边界条件已经验证。

## 尚未完成

当前适配器中，RoPE、含 RoPE 的多头自注意力、Transformer Block、完整 Transformer LM、BPE 训练与 tokenizer、交叉熵、梯度裁剪、优化器、学习率调度、数据加载和训练检查点等接口仍保留上游的 `NotImplementedError`。没有运行完整测试集，也没有完成语言模型训练。

下一步继续在课程学习目录学习并实现 RoPE。目前只介绍了其概念，尚未实现或验证；完成后再同步实现并扩展此处的验证范围。当前 CI 只验证上表中的组件。

## 追溯

官方基础提交：`a158843b20107949f1a8d7df1b05cd33b9166712`。本次发布快照保留该版本的官方脚手架，加上当前已完成的六个实现文件以及 `tests/adapters.py` 的既有改动。多头自注意力位于现有 `attention.py` 中，没有新增文件。逐文件校验值见 [snapshot.json](snapshot.json)。

前一版 [`a39eb684710817f7f389626423fcb26ca0f6d629`](https://github.com/liangqianxing/cs336-labs/commit/a39eb684710817f7f389626423fcb26ca0f6d629) 对应最初 7 类组件、8 项测试的阶段；前三篇学习文章保留引用该历史版本。本轮仅同步既有学习实现，并增加多头自注意力的验证清单。
