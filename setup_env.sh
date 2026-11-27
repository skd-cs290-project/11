#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python_bin=${CS290_CREATE_PYTHON:-python3.10}
if [[ ! -x .venv/bin/python ]]; then
    "$python_bin" -m venv .venv
fi
py=.venv/bin/python
"$py" -m pip install --upgrade 'pip<26'
"$py" -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
"$py" -m pip install -r requirements.txt
"$py" -m pip check
