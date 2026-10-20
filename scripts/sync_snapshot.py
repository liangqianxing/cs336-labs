#!/usr/bin/env python3
"""Copy a reviewed CS336 snapshot; never mutate the source checkout."""

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

UPSTREAM_URL = "https://github.com/stanford-cs336/assignment1-basics"
UPSTREAM_COMMIT = "a158843b20107949f1a8d7df1b05cd33b9166712"
MAX_FILE_BYTES = 8 * 1024 * 1024

# Deliberately enumerate files. New files need human review before publication.
ALLOWLIST = (
    ".gitignore",
    "AGENTS.md",
    "CHANGELOG.md",
    "CLAUDE.md",
    "LICENSE",
    "README.md",
    "cs336_assignment1_basics.pdf",
    "cs336_basics/__init__.py",
    "cs336_basics/pretokenization_example.py",
    "cs336_basics/activations.py",
    "cs336_basics/attention.py",
    "cs336_basics/embedding.py",
    "cs336_basics/linear.py",
    "cs336_basics/rmsnorm.py",
    "cs336_basics/swiglu.py",
    "make_submission.sh",
    "pyproject.toml",
    "tests/__init__.py",
    "tests/_snapshots/test_4d_scaled_dot_product_attention.npz",
    "tests/_snapshots/test_adamw.npz",
    "tests/_snapshots/test_embedding.npz",
    "tests/_snapshots/test_linear.npz",
    "tests/_snapshots/test_multihead_self_attention.npz",
    "tests/_snapshots/test_multihead_self_attention_with_rope.npz",
    "tests/_snapshots/test_positionwise_feedforward.npz",
    "tests/_snapshots/test_rmsnorm.npz",
    "tests/_snapshots/test_rope.npz",
    "tests/_snapshots/test_scaled_dot_product_attention.npz",
    "tests/_snapshots/test_swiglu.npz",
    "tests/_snapshots/test_train_bpe_special_tokens.pkl",
    "tests/_snapshots/test_transformer_block.npz",
    "tests/_snapshots/test_transformer_lm.npz",
    "tests/_snapshots/test_transformer_lm_truncated_input.npz",
    "tests/adapters.py",
    "tests/common.py",
    "tests/conftest.py",
    "tests/fixtures/address.txt",
    "tests/fixtures/corpus.en",
    "tests/fixtures/german.txt",
    "tests/fixtures/gpt2_merges.txt",
    "tests/fixtures/gpt2_vocab.json",
    "tests/fixtures/special_token_double_newlines_non_whitespace.txt",
    "tests/fixtures/special_token_trailing_newlines.txt",
    "tests/fixtures/tinystories_sample.txt",
    "tests/fixtures/tinystories_sample_5M.txt",
    "tests/fixtures/train-bpe-reference-merges.txt",
    "tests/fixtures/train-bpe-reference-vocab.json",
    "tests/fixtures/ts_tests/model.pt",
    "tests/fixtures/ts_tests/model_config.json",
    "tests/test_data.py",
    "tests/test_model.py",
    "tests/test_nn_utils.py",
    "tests/test_optimizer.py",
    "tests/test_serialization.py",
    "tests/test_tokenizer.py",
    "tests/test_train_bpe.py",
    "uv.lock",
)

SECRET_PATTERNS = (
    re.compile(rb"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----"),
    re.compile(rb"\bgh[pousr]_[A-Za-z0-9]{36,}\b"),
    re.compile(rb"\bgithub_pat_[A-Za-z0-9_]{60,}\b"),
    re.compile(rb"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(rb"\bsk-(?:proj-)?[A-Za-z0-9_-]{32,}\b"),
)


def git_output(source, *args):
    return subprocess.run(
        ["git", "-C", str(source), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout


def reject_symlinks(root, relative):
    current = root
    for part in Path(relative).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"Symbolic link rejected: {relative}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    source = args.source.expanduser().resolve(strict=True)
    repository = Path(__file__).resolve().parent.parent
    destination = repository / "assignment1-basics"
    if source == repository or source in repository.parents or repository in source.parents:
        raise ValueError("Source and publication repository must be separate directories")

    source_head = git_output(source, "rev-parse", "HEAD").decode().strip()
    git_output(source, "cat-file", "-e", f"{UPSTREAM_COMMIT}^{{commit}}")
    payload = {}
    entries = {}

    # Validate the entire snapshot before creating or overwriting any file.
    for relative in ALLOWLIST:
        reject_symlinks(source, relative)
        reject_symlinks(repository, "assignment1-basics/" + relative)
        path = source / relative
        if not path.is_file():
            raise ValueError(f"Required file missing: {relative}")
        if path.stat().st_size > MAX_FILE_BYTES:
            raise ValueError(f"File size exceeds limit: {relative}")
        data = path.read_bytes()
        if len(data) > MAX_FILE_BYTES:
            raise ValueError(f"File size exceeds limit: {relative}")
        if any(pattern.search(data) for pattern in SECRET_PATTERNS):
            raise ValueError(f"Potential credential rejected: {relative}")
        if path.suffix in {".pdf", ".npz", ".pkl", ".pt"}:
            original = git_output(source, "show", f"{UPSTREAM_COMMIT}:{relative}")
            if data != original:
                raise ValueError(f"Official binary fixture changed: {relative}")
        payload[relative] = data
        entries[relative] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}

    reject_symlinks(repository, "docs/snapshot.json")
    if not args.dry_run:
        for relative, data in payload.items():
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        manifest = {
            "upstream_url": UPSTREAM_URL,
            "upstream_commit": UPSTREAM_COMMIT,
            "source_head": source_head,
            "captured_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "files": entries,
        }
        manifest_path = repository / "docs/snapshot.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    action = "Validated" if args.dry_run else "Copied"
    print(f"{action} {len(payload)} allowlisted files ({sum(len(data) for data in payload.values()):,} bytes).")
    print("Source checkout was read only. Review the diff and run the completed-component checks before publishing.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        # Avoid echoing source paths, credential contents or subprocess output.
        if isinstance(error, ValueError):
            raise SystemExit(str(error)) from None
        raise SystemExit("Snapshot validation failed; verify source files, permissions and upstream Git commit.") from None
