import argparse
import re
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs


parser = argparse.ArgumentParser()
parser.add_argument("--enroll", help="Name of the person to enrol")
args = parser.parse_args()

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "face_samples"
DATA.mkdir(exist_ok=True)

if args.enroll and not re.fullmatch(r"[A-Za-z0-9_-]+", args.enroll):
    raise SystemExit("Use letters, numbers, underscores or hyphens for the name.")

detector = cv2.CascadeClassifier(
    str(Path(__file__).resolve().parent / "haarcascade_frontalface_default.xml")
)
if detector.empty():
    raise SystemExit("Could not load the face detector.")

recognizer = cv2.face.LBPHFaceRecognizer_create()
names = []
images = []
labels = []

# Load previously enrolled face samples.
if not args.enroll:
    for folder in sorted(DATA.iterdir()):
        if not folder.is_dir():
            continue

        label = len(names)
        names.append(folder.name)

        for file in sorted(folder.glob("*.png")):
            sample = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
            if sample is not None:
                images.append(cv2.resize(sample, (200, 200)))
                labels.append(label)

    if not images:
        raise SystemExit("No faces enrolled. Run with --enroll Cookie first.")

    recognizer.train(images, np.array(labels, dtype=np.int32))


MATCH_THRESHOLD = 55.0

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

started = False
capturing = False
saved = 0
last_saved = 0.0
session = time.time_ns()
window = "RealSense Face Recognition"

try:
    pipeline.start(config)
    started = True
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    print("Click the camera window to use its keyboard controls.")
    print("Q: quit | E: start collecting enrollment samples")

    while True:
        frames = pipeline.wait_for_frames(10000)
        color = frames.get_color_frame()
        if not color:
            continue

        frame = np.asanyarray(color.get_data()).copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(90, 90),
        )

        for x, y, w, h in faces:
            sample = gray[y:y + h, x:x + w]
            sample = cv2.resize(sample, (200, 200))
            sample = cv2.equalizeHist(sample)

            text = "Face"
            box_color = (0, 200, 255)

            if args.enroll:
                # Save only when exactly one face is visible.
                if capturing and len(faces) == 1:
                    now = time.monotonic()

                    if now - last_saved >= 0.35 and saved < 40:
                        folder = DATA / args.enroll
                        folder.mkdir(exist_ok=True)

                        file = folder / f"{session}_{saved:03d}.png"
                        if not cv2.imwrite(str(file), sample):
                            raise RuntimeError(f"Could not save {file}")

                        saved += 1
                        last_saved = now

                text = f"{args.enroll}: {saved}/40"
            else:
                label, distance = recognizer.predict(sample)

                if distance < MATCH_THRESHOLD:
                    text = names[label]
                    box_color = (0, 220, 0)
                else:
                    text = "Unknown"
                    box_color = (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
            cv2.putText(
                frame, text, (x, max(25, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2
            )

        if args.enroll:
            if not capturing:
                status = "Only your face in view. Press E to start."
            elif len(faces) != 1:
                status = "Show exactly ONE face to collect samples."
            else:
                status = f"Collecting {saved}/40 - move your head slightly"
        else:
            status = "Recognition running | Q to quit"

        cv2.putText(
            frame, status, (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )

        cv2.imshow(window, frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("e") and args.enroll:
            capturing = True

        if key == ord("q") or cv2.getWindowProperty(
            window, cv2.WND_PROP_VISIBLE
        ) < 1:
            break

        if args.enroll and saved >= 40:
            print(f"Saved 40 face samples for {args.enroll}.")
            break

except KeyboardInterrupt:
    pass
finally:
    if started:
        pipeline.stop()
    cv2.destroyAllWindows()
