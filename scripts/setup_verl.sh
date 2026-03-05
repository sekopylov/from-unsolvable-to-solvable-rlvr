#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/seankopylov/projects/solve-hard-problems"
VERL_DIR="$ROOT/verl-main"
VERL_REPO="https://github.com/volcengine/verl.git"
VERL_COMMIT="d9d94b4da93fbacc06bb546629171c67c0a674aa"

if [ ! -d "$VERL_DIR/.git" ]; then
  git clone "$VERL_REPO" "$VERL_DIR"
fi

cd "$VERL_DIR"
git fetch origin
git checkout "$VERL_COMMIT"

echo "verl-main is ready at commit: $VERL_COMMIT"
