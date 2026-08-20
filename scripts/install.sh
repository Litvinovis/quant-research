#!/usr/bin/env bash
# Sets up .venv with all dependencies, including tinkoff-investments —
# which needs its own --no-deps install (see requirements.txt for why).
# Used identically by local dev setup and CI (test.yml/deploy.yml).
set -euo pipefail
cd "$(dirname "$0")/.."

python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements.txt
.venv/bin/pip install -q --no-deps \
  "git+https://github.com/RussianInvestments/invest-python.git@fcf0af1e27f053cd4bc75eb20863ece23a202d47"
