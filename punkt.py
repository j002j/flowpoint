#dunkle Punkte als Mittepunkt der getrackten Personen mit bewegungslinie/schweif

import cv2
from ultralytics import YOLO
import numpy as np
import random

fade_strength = 0.97
max_track_length = 40
smoothing_factor = 0.4  # etwas stärker geglättet

brightness_offset = 80

thickness_glow = 12
thickness_mid = 6
thickness_core = 4  # dicker Kern

alpha_glow = 0.2
alpha_mid = 0.5

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Kamera konnte nicht geöffnet werden")
    exit()

trail_layer = None
tracks = {}
id_colors = {}


def get_color_for_id(track_id: int):
    if track_id not in id_colors:
        rng = random.Random(track_id)
        color = (
            rng.randint(100, 255),
            rng.randint(100, 255),
            rng.randint(100, 255)
        )
        id_colors[track_id] = color
    return id_colors[track_id]


def smooth_point(new_point, last_point, factor=0.4):
    if last_point is None:
        return new_point
    x = int(last_point[0] * (1 - factor) + new_point[0] * factor)
    y = int(last_point[1] * (1 - factor) + new_point[1] * factor)
    return (x, y)


window_name = "WKD Summercamp"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
screen_res = (1920, 1080)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    if trail_layer is None:
        trail_layer = np.ones_like(frame, dtype=np.uint8) * 255

    results = model.track(frame, persist=True, verbose=False)
    result = results[0]

    # Trail verblassen lassen
    trail_layer = cv2.addWeighted(
        trail_layer, fade_strength, np.ones_like(trail_layer) * 255, 1 - fade_strength, 0
    )

    if hasattr(result, "boxes") and result.boxes.id is not None:
        for box, cls, track_id in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.id):
            if int(cls) == 0:
                x1, y1, x2, y2 = map(int, box)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                tid = int(track_id)
                if tid not in tracks:
                    tracks[tid] = []

                last_point = tracks[tid][-1] if tracks[tid] else None
                smooth_pt = smooth_point((cx, cy), last_point, factor=smoothing_factor)
                tracks[tid].append(smooth_pt)

                if len(tracks[tid]) > max_track_length:
                    tracks[tid].pop(0)

                # Basisfarbe & Varianten
                base_color = np.array(get_color_for_id(tid), dtype=np.int32)
                bright = np.clip(base_color + brightness_offset, 0, 255)
                dark = np.clip(base_color - brightness_offset, 0, 255)

                num_points = len(tracks[tid])
                if num_points >= 2:
                    # Spur hell
                    for i in range(num_points - 1):
                        pt1 = tracks[tid][i]
                        pt2 = tracks[tid][i + 1]
                        color = bright.astype(np.uint8).tolist()

                        overlay = trail_layer.copy()
                        cv2.line(overlay, pt1, pt2, color, thickness=thickness_glow, lineType=cv2.LINE_AA)
                        trail_layer = cv2.addWeighted(overlay, alpha_glow, trail_layer, 1 - alpha_glow, 0)

                        overlay = trail_layer.copy()
                        cv2.line(overlay, pt1, pt2, color, thickness=thickness_mid, lineType=cv2.LINE_AA)
                        trail_layer = cv2.addWeighted(overlay, alpha_mid, trail_layer, 1 - alpha_mid, 0)

                        cv2.line(trail_layer, pt1, pt2, color, thickness=thickness_core, lineType=cv2.LINE_AA)

                    # Kopf dunkel und glatt, Radius proportional zur Kernlinie
                    head = tracks[tid][-1]
                    cv2.circle(trail_layer, head, radius=thickness_core, color=dark.astype(np.uint8).tolist(), thickness=-1, lineType=cv2.LINE_AA)

    frame_resized = cv2.resize(trail_layer, screen_res, interpolation=cv2.INTER_LINEAR)
    cv2.imshow(window_name, frame_resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
