#!/bin/bash
set -euo pipefail
project_dir="$(cd "$(dirname "$0")" && pwd)"
python_path="$project_dir/.depth-venv/bin/python"
if [ ! -x "$python_path" ]; then
  echo 'Run bash setup.sh first.'
  exit 1
fi
sudo env PYTHONPATH="$HOME/Documents/RealSenseFace/librealsense/build-python/Release" "$python_path" "$project_dir/face_depth.py"
