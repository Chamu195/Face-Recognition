#!/usr/bin/env python3
"""D435i colour landmarks + aligned sensor depth. No identity/liveness decisions."""
from pathlib import Path
import csv
import time
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent
FEATURES = [('Nose tip', 1), ('Forehead', 10), ('Cheek A', 50), ('Cheek B', 280), ('Chin', 152)]
MIN_M, MAX_M = 0.25, 2.0


def sample_depth(depth_m, x, y, radius=2):
    """Median of a 5x5 patch; reject holes, sparse patches and large spreads."""
    h, w = depth_m.shape
    if not (radius <= x < w-radius and radius <= y < h-radius):
        return None
    patch = depth_m[y-radius:y+radius+1, x-radius:x+radius+1]
    center = float(depth_m[y, x])
    if not np.isfinite(center) or not MIN_M <= center <= MAX_M:
        return None
    valid = patch[np.isfinite(patch) & (patch >= MIN_M) & (patch <= MAX_M)]
    if len(valid) < 0.7 * patch.size:
        return None
    median = float(np.median(valid))
    spread = float(np.percentile(valid, 90) - np.percentile(valid, 10))
    if spread > 0.025 or abs(center-median) > 0.015:
        return None
    return median


def nose_offset(values):
    """Camera-axis cheek-average minus nose; not a pose-normalised anatomy measure."""
    nose, a, b = (values.get(k) for k in ('Nose tip', 'Cheek A', 'Cheek B'))
    if nose is None or a is None or b is None or abs(a-b) > 0.025:
        return None
    return ((a+b)/2 - nose)*1000


def save_scan(records, depth, color, intrinsics, rs, cv2):
    """Explicit S key only: export this frame's landmarks and visible surface."""
    destination = ROOT / 'scans' / time.strftime('%Y%m%d_%H%M%S')
    destination.mkdir(parents=True, exist_ok=False)
    with (destination / 'features.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame_face_slot', 'feature', 'pixel_x', 'pixel_y', 'z_m', 'x_m', 'y_m', 'status'])
        for slot, points, values, hull in records:
            for name, (x, y) in points.items():
                z = values[name]
                xyz = rs.rs2_deproject_pixel_to_point(intrinsics, [float(x), float(y)], z) if z is not None else None
                writer.writerow([slot, name, x, y, z if z is not None else '',
                                 xyz[0] if xyz else '', xyz[1] if xyz else '',
                                 'valid' if xyz else 'unavailable'])
    cloud = []
    for slot, points, values, hull in records:
        mask = np.zeros(depth.shape, dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull, 255)
        valid_depths = [z for z in values.values() if z is not None]
        if not valid_depths:
            continue
        reference = float(np.median(valid_depths))
        ys, xs = np.where((mask > 0) & (depth >= MIN_M) & (depth <= MAX_M) & (np.abs(depth-reference) < 0.12))
        for y, x in zip(ys[::4], xs[::4]):
            xyz = rs.rs2_deproject_pixel_to_point(intrinsics, [float(x), float(y)], float(depth[y, x]))
            b, g, r = color[y, x]
            cloud.append((*xyz, int(r), int(g), int(b)))
    with (destination / 'face_surface.ply').open('w') as f:
        f.write('ply\nformat ascii 1.0\ncomment Visible face-region sensor points; metres; not a full head scan\n')
        f.write(f'element vertex {len(cloud)}\nproperty float x\nproperty float y\nproperty float z\n')
        f.write('property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
        for point in cloud:
            f.write(' '.join(map(str, point)) + '\n')
    print(f'Saved {destination}', flush=True)
    return destination.name


def main():
    import cv2
    import mediapipe as mp
    import pyrealsense2 as rs
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    model = ROOT / 'face_landmarker.task'
    if not model.is_file():
        raise RuntimeError('Missing face_landmarker.task. Run the download command in README.txt.')
    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=2,
        min_face_detection_confidence=0.6,
        min_face_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    align = rs.align(rs.stream.color)
    running = False
    window = 'Face Depth | Q quit | S save scan | M toggle dots'
    show_dots = True
    message, message_until = '', 0.0
    last_ts = -1
    start = time.monotonic()

    def label(img, text, x, y, scale=0.5, color=(220, 225, 235)):
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

    try:
        with vision.FaceLandmarker.create_from_options(options) as landmarker:
            profile = pipeline.start(config)
            running = True
            scale = profile.get_device().first_depth_sensor().get_depth_scale()
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window, 1280, 760)
            print('Point D435i at a well-lit face, roughly 0.5-1 metre away. Click window for keys.', flush=True)
            print('S saves current face-region data; Q quits. Face slots are NOT identities.', flush=True)
            while True:
                frames = align.process(pipeline.wait_for_frames(10000))
                depth_frame, color_frame = frames.get_depth_frame(), frames.get_color_frame()
                if not depth_frame or not color_frame:
                    continue
                original = np.asanyarray(color_frame.get_data()).copy()
                depth = np.asanyarray(depth_frame.get_data()).astype(np.float32)*scale
                intr = depth_frame.profile.as_video_stream_profile().intrinsics
                rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
                timestamp = max(last_ts+1, int((time.monotonic()-start)*1000))
                last_ts = timestamp
                result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp)
                overlay = original.copy()
                masks = np.zeros(depth.shape, dtype=np.uint8)
                records = []
                for slot, landmarks in enumerate(result.face_landmarks, 1):
                    pixels = np.array([(round(p.x*640), round(p.y*480)) for p in landmarks[:468]], dtype=np.int32)
                    hull = cv2.convexHull(pixels)
                    cv2.fillConvexPoly(masks, hull, 255)
                    cv2.polylines(overlay, [hull], True, (210, 130, 255), 1)
                    if show_dots:
                        for x, y in pixels:
                            if 0 <= x < 640 and 0 <= y < 480:
                                cv2.circle(overlay, (int(x), int(y)), 1, (190, 135, 220), -1)
                    points, values = {}, {}
                    for number, (name, idx) in enumerate(FEATURES, 1):
                        x, y = map(int, pixels[idx])
                        z = sample_depth(depth, x, y)
                        points[name], values[name] = (x, y), z
                        if 0 <= x < 640 and 0 <= y < 480:
                            color = (80, 240, 130) if z is not None else (0, 150, 255)
                            cv2.circle(overlay, (x, y), 4, color, -1)
                            label(overlay, str(number), x+6, y, color=color)
                    x, y, w, h = cv2.boundingRect(hull)
                    label(overlay, f'Face slot {slot}', max(0, min(x, 500)), max(25, min(y-8, 465)), color=(255, 255, 255))
                    records.append((slot, points, values, hull))
                valid = (masks > 0) & (depth >= MIN_M) & (depth <= MAX_M)
                heat = np.zeros_like(original)
                legend = 'No valid face depth'
                if np.any(valid):
                    low, high = map(float, np.percentile(depth[valid], [5, 95]))
                    high = max(high, low+0.04)
                    normalized = np.clip((depth-low)/(high-low), 0, 1)
                    heat = cv2.applyColorMap((normalized*255).astype(np.uint8), cv2.COLORMAP_TURBO)
                    heat[~valid] = 0
                    legend = f'Blue nearer ~{low*100:.1f}cm | red farther ~{high*100:.1f}cm (auto range)'
                for slot, points, values, hull in records:
                    for number, (name, point) in enumerate(points.items(), 1):
                        x, y = point
                        if 0 <= x < 640 and 0 <= y < 480:
                            cv2.circle(heat, (x, y), 4, (255, 255, 255), 1)
                            label(heat, str(number), x+6, y)
                panel = np.full((760, 1280, 3), (25, 20, 20), dtype=np.uint8)
                panel[40:520, :640] = overlay
                panel[40:520, 640:] = heat
                label(panel, 'COLOUR + FEATURE LOCATIONS', 16, 27, 0.65)
                label(panel, 'MEASURED DEPTH - FACE REGION', 656, 27, 0.65)
                label(panel, legend, 656, 544, 0.47)
                label(panel, 'Depth = camera-axis Z; N/A = missing or unstable sample.', 16, 544, 0.47)
                if not records:
                    label(panel, 'No face detected. Face the D435i in good light.', 16, 585, 0.7)
                for slot, points, values, hull in records:
                    left = (slot-1)*640 + 16
                    label(panel, f'FACE SLOT {slot} - frame-local, not an identity', left, 568, 0.5)
                    for i, (name, idx) in enumerate(FEATURES):
                        z = values[name]
                        text = f'{z*100:.1f} cm' if z is not None else 'N/A'
                        label(panel, f'{i+1}. {name}: {text}', left, 591+i*20, 0.5)
                    offset = nose_offset(values)
                    info = f'Nose vs cheek-average Z: {offset:+.1f} mm' if offset is not None else 'Nose offset: N/A - face forward / check depth'
                    label(panel, info, left, 697, 0.48)
                label(panel, 'Q quit   |   S save CSV + PLY   |   M dots   |   No liveness or identity decision', 16, 729, 0.52)
                footer = message if time.monotonic() < message_until else 'Depth readings are experimental. Pose, glasses, hair and lighting can affect them.'
                label(panel, footer, 16, 751, 0.46)
                cv2.imshow(window, panel)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), ord('Q'), 27) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key in (ord('m'), ord('M')):
                    show_dots = not show_dots
                if key in (ord('s'), ord('S')):
                    if records and np.any(valid):
                        try:
                            folder = save_scan(records, depth, original, intr, rs, cv2)
                            message = f'Saved scans/{folder} - contains facial data.'
                        except (OSError, RuntimeError) as error:
                            message = f'Could not save: {error}'
                            print(message, flush=True)
                    else:
                        message = 'Nothing saved: need a face with valid depth.'
                    message_until = time.monotonic()+5
    finally:
        if running:
            pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception as error:
        print(f'Face depth error: {error}', file=sys.stderr)
        print('Close other camera apps. Use the README launch command with sudo and the depth environment.', file=sys.stderr)
        sys.exit(1)
