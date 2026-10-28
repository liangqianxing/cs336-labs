#!/usr/bin/env bash
# Run the complete official CS336 Assignment 1 test suite across all components.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root/assignment1-basics"
tests=(
  tests/test_data.py
  tests/test_model.py
  tests/test_nn_utils.py
  tests/test_optimizer.py
  tests/test_serialization.py
  tests/test_tokenizer.py
  tests/test_train_bpe.py
)
if [[ -n "${CS336_PYTHON:-}" ]]; then
  exec "$CS336_PYTHON" -m pytest -q "${tests[@]}"
fi
exec uv run --locked pytest -q "${tests[@]}"
