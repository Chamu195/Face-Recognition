#!/bin/bash
set -euo pipefail
project_dir="$(cd "$(dirname "$0")" && pwd)"
cd "$project_dir"
python_path=/opt/homebrew/bin/python3.11
if [ ! -x "$python_path" ]; then
  echo 'Python 3.11 missing. Run: brew install python@3.11'
  exit 1
fi
"$python_path" -m venv .depth-venv
.depth-venv/bin/python -m pip install --upgrade pip
.depth-venv/bin/python -m pip install -r requirements.txt
if [ ! -s face_landmarker.task ]; then
  curl -fL --retry 2 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task' -o face_landmarker.task.download
  mv face_landmarker.task.download face_landmarker.task
fi
export PYTHONPATH="$HOME/Documents/RealSenseFace/librealsense/build-python/Release"
.depth-venv/bin/python -c 'import cv2, mediapipe, pyrealsense2; print("Imports OK. Setup complete.")'
echo 'Run: bash run.sh'
