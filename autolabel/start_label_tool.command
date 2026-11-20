#!/bin/sh
set -eu

app_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
python="$app_dir/.venv/bin/python"
if [ ! -x "$python" ]; then
    printf '%s\n' 'Set up the macOS virtual environment using README.md first.' >&2
    exit 1
fi

exec "$python" "$app_dir/label_tool.py" "$@"
