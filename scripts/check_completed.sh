#!/usr/bin/env bash
# Keep this list explicit: the remaining upstream tests are not implemented yet.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root/assignment1-basics"
tests=(
  tests/test_model.py::test_linear
  tests/test_model.py::test_embedding
  tests/test_model.py::test_rmsnorm
  tests/test_model.py::test_silu_matches_pytorch
  tests/test_model.py::test_swiglu
  tests/test_nn_utils.py::test_softmax_matches_pytorch
  tests/test_model.py::test_scaled_dot_product_attention
  tests/test_model.py::test_4d_scaled_dot_product_attention
  tests/test_model.py::test_multihead_self_attention
)
if [[ -n "${CS336_PYTHON:-}" ]]; then
  exec "$CS336_PYTHON" -m pytest -q "${tests[@]}"
fi
exec uv run --locked pytest -q "${tests[@]}"
