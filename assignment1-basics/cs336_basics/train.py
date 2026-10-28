from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time

import numpy as np
import numpy.typing as npt
import torch

from cs336_basics.data import get_batch
from cs336_basics.decoding import generate
from cs336_basics.nn_utils import cross_entropy, gradient_clipping
from cs336_basics.optimizer import AdamW, get_lr_cosine_schedule
from cs336_basics.serialization import load_checkpoint, save_checkpoint
from cs336_basics.transformer import TransformerLM


@dataclass
class TrainConfig:
    vocab_size: int = 10_000
    context_length: int = 256
    d_model: int = 512
    num_layers: int = 4
    num_heads: int = 16
    d_ff: int = 1344
    rope_theta: float = 10000.0
    batch_size: int = 32
    max_iters: int = 100
    warmup_iters: int = 10
    cosine_cycle_iters: int = 100
    max_learning_rate: float = 5e-4
    min_learning_rate: float = 5e-5
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    grad_clip: float = 1.0
    eval_interval: int = 20
    eval_batches: int = 8
    device: str = "cpu"
    seed: int = 42


def load_token_dataset(path: str | Path, dtype: npt.DTypeLike = np.uint16) -> npt.NDArray:
    path_obj = Path(path)
    if path_obj.suffix == ".npy":
        return np.load(path_obj, mmap_mode="r")
    return np.memmap(path_obj, dtype=dtype, mode="r")


@torch.no_grad()
def evaluate_loss(
    model: TransformerLM,
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    num_batches: int = 8,
    device: str = "cpu",
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    losses: list[float] = []

    for _ in range(num_batches):
        x, y = get_batch(dataset, batch_size=batch_size, context_length=context_length, device=device)
        logits = model(x)
        loss = cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        losses.append(float(loss.detach().cpu().item()))

    if was_training:
        model.train()

    mean_loss = float(np.mean(losses))
    return {
        "loss": mean_loss,
        "perplexity": float(math.exp(min(mean_loss, 20.0))),
    }


def run_training_loop(
    config: TrainConfig,
    train_dataset: npt.NDArray,
    val_dataset: npt.NDArray | None = None,
    checkpoint_path: str | Path | None = None,
    resume_from: str | Path | None = None,
) -> dict:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    model = TransformerLM(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        rope_theta=config.rope_theta,
        device=config.device,
    )
    optimizer = AdamW(
        model.parameters(),
        lr=config.max_learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.eps,
        weight_decay=config.weight_decay,
    )

    start_iter = 0
    if resume_from is not None:
        start_iter = load_checkpoint(resume_from, model=model, optimizer=optimizer)

    history: list[dict[str, float]] = []
    t0 = time.perf_counter()

    for it in range(start_iter, config.max_iters):
        lr = get_lr_cosine_schedule(
            it=it,
            max_learning_rate=config.max_learning_rate,
            min_learning_rate=config.min_learning_rate,
            warmup_iters=config.warmup_iters,
            cosine_cycle_iters=config.cosine_cycle_iters,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad()
        x, y = get_batch(
            train_dataset,
            batch_size=config.batch_size,
            context_length=config.context_length,
            device=config.device,
        )
        logits = model(x)
        loss = cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        loss.backward()
        grad_norm = gradient_clipping(model.parameters(), max_l2_norm=config.grad_clip)
        optimizer.step()

        if it % config.eval_interval == 0 or it == config.max_iters - 1:
            record: dict[str, float] = {
                "iter": float(it),
                "lr": float(lr),
                "train_loss": float(loss.detach().cpu().item()),
                "grad_norm": float(grad_norm.detach().cpu().item()),
                "elapsed_sec": float(time.perf_counter() - t0),
            }
            if val_dataset is not None:
                val_metrics = evaluate_loss(
                    model=model,
                    dataset=val_dataset,
                    batch_size=config.batch_size,
                    context_length=config.context_length,
                    num_batches=config.eval_batches,
                    device=config.device,
                )
                record["val_loss"] = val_metrics["loss"]
                record["val_perplexity"] = val_metrics["perplexity"]
            history.append(record)

    if checkpoint_path is not None:
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            iteration=config.max_iters,
            out=checkpoint_path,
        )

    prompt = torch.tensor(train_dataset[:8], dtype=torch.long, device=config.device)
    sampled = generate(
        model=model,
        prompt_ids=prompt,
        max_new_tokens=16,
        temperature=0.8,
        top_p=0.9,
    )

    return {
        "config": asdict(config),
        "history": history,
        "sampled_token_ids": sampled.detach().cpu().tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a compact CS336 Transformer LM.")
    parser.add_argument("--train-data", type=Path, default=None, help="Path to uint16/npy token dataset.")
    parser.add_argument("--val-data", type=Path, default=None, help="Optional validation token dataset.")
    parser.add_argument("--checkpoint-out", type=Path, default=None, help="Optional path to save checkpoint.")
    parser.add_argument("--log-out", type=Path, default=None, help="Optional JSON output path for training history.")
    parser.add_argument("--max-iters", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--context-length", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    config = TrainConfig(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        batch_size=args.batch_size,
        max_iters=args.max_iters,
        warmup_iters=max(1, args.max_iters // 5),
        cosine_cycle_iters=args.max_iters,
        eval_interval=max(1, args.max_iters // 4),
        device=args.device,
    )

    if args.train_data is not None:
        train_ds = load_token_dataset(args.train_data)
    else:
        rng = np.random.default_rng(config.seed)
        base_pattern = np.arange(0, 64, dtype=np.uint16)
        train_ds = np.tile(base_pattern, 64) + rng.integers(0, 2, size=64 * 64, dtype=np.uint16)

    val_ds = load_token_dataset(args.val_data) if args.val_data is not None else train_ds

    summary = run_training_loop(
        config=config,
        train_dataset=train_ds,
        val_dataset=val_ds,
        checkpoint_path=args.checkpoint_out,
    )
    if args.log_out is not None:
        args.log_out.parent.mkdir(parents=True, exist_ok=True)
        args.log_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    else:
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
