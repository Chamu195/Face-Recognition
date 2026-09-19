REALSENSE FACE DEPTH - Cookie's D435i / Apple Silicon Mac

SETUP
1. Unzip this folder and move RealSenseFaceDepth into:
   Desktop/Face Recognition/
2. Stop face_app.py and any other camera apps first.
3. In Terminal run:
   cd "$HOME/Desktop/Face Recognition/RealSenseFaceDepth"
   bash setup.sh
   bash run.sh

Setup creates a separate .depth-venv with Python 3.11. It downloads Google's
face-landmark model once (internet required). Your working recognition
script and its environment remain available. The RealSense binding is reused
from Documents/RealSenseFace/librealsense/build-python/Release.
The model is not bundled. Download source:
https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

USE
Face the D435i, roughly 0.5-1 metre away, with light on your face.
Click the window so it receives keys:
Q / Escape: quit
M: show or hide landmark dots
S: save this frame's feature measurements (CSV) and visible surface (PLY)

WHAT YOU SEE
Left: colour image with model-predicted feature locations.
Right: aligned physical depth at pixels inside detected face regions.
Five numbered points: nose tip, forehead, cheek A, cheek B, chin.
Depth units are centimetres along the camera's Z axis, not radial range.
The feature locations are approximate model landmarks, not anatomical ground truth.
Purple dots show predicted locations only, not validated 3D measurements.
MediaPipe's predicted z coordinate is never used for physical distances.

The sensor depth is aligned to colour before sampling. A 5x5 local patch
provides a robust median. A missing central reading, insufficient valid
samples, out-of-range depth, or excessive local variation returns N/A.
No hole-filling or temporal carryover is used to manufacture measurements.
Values displayed to 0.1 cm do not imply that level of measurement accuracy.

Nose vs cheek-average Z = average cheek depth minus nose depth, in mm.
Positive means the nose is nearer the camera than the sampled cheeks.
This is a camera-axis comparison, NOT true pose-normalised nose protrusion.
It is hidden if cheek depths differ by more than 25 mm. This simple check
is not a guarantee that the face is frontal; pitch and anatomy still affect it.

Up to two faces are processed per frame. Slots 1/2 can swap as faces move;
they are not names or persistent IDs. Measurements are recomputed per frame.
The depth heat map automatically rescales, so colours are not comparable
between frames; use the numeric depths and on-screen scale.

SAVING
S creates scans/YYYYMMDD_HHMMSS beside this script. There is no automatic
recording. CSV includes unavailable rows with empty depth/XYZ. PLY uses
metres and the aligned colour-camera coordinate frame: X right, Y down,
Z forward. It represents visible face-region points, not a complete head
mesh or a multi-view reconstruction. Background leakage at edges is possible.
Treat saved scans as personal facial data; only save with participants' agreement.
Because camera access uses sudo, saved scan files may be owned by root.

SCOPE
This is the live measurement stage. It does not recognise names, train a
3D identity model, or determine whether a face is live. Your original LBPH
script handles colour-based names separately. Run only one camera app at a time.
A D435i reading plus a face landmark is not a secure anti-spoofing system.

TROUBLESHOOTING
- Model missing: rerun bash setup.sh (do not paste Python code into Terminal).
- Import error: use bash run.sh, not the old .venv or a bare python command.
- USB access error: close other camera apps, reconnect, run bash run.sh.
- N/A: face forward, move a little farther from the camera, improve lighting.
  Glasses, hair, occlusion, reflectivity and camera geometry can cause holes.
- No face: use a frontal face and avoid a bright window behind you.
- Installation/runtime error: copy the last 20 lines for troubleshooting.

VALIDATION
Syntax and synthetic depth-quality checks were run during preparation.
The Mac GUI, model execution and live D435i measurements could not be tested
here; they require your connected Mac and camera.

REFERENCES
MediaPipe Face Landmarker Python:
https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python
RealSense colour/depth alignment example:
https://github.com/realsenseai/librealsense/blob/master/wrappers/python/examples/align-depth2color.py
MediaPipe 0.10.21 macOS ARM64 wheel:
https://pypi.org/project/mediapipe/0.10.21/
