# SCHWARZER HINTERGRUND: Schleier mit Bewegungslinie am Kopf nur!
import cv2
from ultralytics import YOLO
import numpy as np
import random

# =========================
# ░ Parameter / Look-Setup ░
# =========================

fade_strength = 0.97
max_track_length = 40
smoothing_factor = 0.4

brightness_offset = 80

# Neon/Glow-Linie
thickness_glow = 12
thickness_mid = 6
thickness_core = 4

# Vernetzung der Köpfe
connection_distance = 80   # maximale Entfernung in Pixel, um Linie zu zeichnen
connection_thickness = 1   # dünne Linie zwischen Köpfen

# zusätzliche Parameter für Kopfgröße
min_radius = 6        # kleinster Kopf-Radius (bei Bewegung)
max_radius = 30       # größter Kopf-Radius (bei Stillstand)
growth_speed = 0.2    # wie schnell der Radius wächst/schrumpft
movement_sensitivity = 2.0  # wie empfindlich auf Bewegung reagiert wird

# =================
# ░ YOLO vorbereit. ░
# =================

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Kamera konnte nicht geöffnet werden")
    exit()

trail_layer = None
tracks = {}
id_colors = {}

# Speicher für Radius pro ID
head_radii = {}
movement_avg = {}

# =================
# ░ Hilfsfunktionen ░
# =================

def get_color_for_id(track_id: int):
    """Jeder ID eine feste, leuchtende Farbe zuweisen"""
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
    """Exponential Glättung der Punktbewegung"""
    if last_point is None:
        return new_point
    x = int(last_point[0] * (1 - factor) + new_point[0] * factor)
    y = int(last_point[1] * (1 - factor) + new_point[1] * factor)
    return (x, y)

# ==========================
# ░ Fenster-/Anzeige-Setup ░
# ==========================

window_name = "WKD Summercamp"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
screen_res = (1920, 1080)

# ============
# ░ Main-Loop ░
# ============
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    if trail_layer is None:
        # Starten mit einem dunklen, aber nicht komplett schwarzen Hintergrund
        trail_layer = np.zeros_like(frame, dtype=np.uint8)

    # 1. Trail-Layer verblassen lassen
    trail_layer = cv2.addWeighted(
        trail_layer, fade_strength, np.zeros_like(trail_layer), 1 - fade_strength, 0
    )

    results = model.track(frame, persist=True, verbose=False)
    result = results[0]

    heads = []

    if hasattr(result, "boxes") and result.boxes.id is not None:
        for box, cls, track_id in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.id):
            if int(cls) == 0:   # nur Personen
                x1, y1, x2, y2 = map(int, box)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                tid = int(track_id)
                if tid not in tracks:
                    tracks[tid] = []

                # Punkt glätten
                last_point = tracks[tid][-1] if tracks[tid] else None
                smooth_pt = smooth_point((cx, cy), last_point, factor=smoothing_factor)
                tracks[tid].append(smooth_pt)
                if len(tracks[tid]) > max_track_length:
                    tracks[tid].pop(0)

                base_color = np.array(get_color_for_id(tid), dtype=np.int32)
                bright = np.clip(base_color + brightness_offset, 0, 255)
                dark = np.clip(base_color - brightness_offset, 0, 255)

                # ===================================
                # Bewegung berechnen + Radius anpassen
                # ===================================
                if last_point is not None:
                    movement = np.linalg.norm(np.array(smooth_pt) - np.array(last_point))
                else:
                    movement = 0

                if tid not in movement_avg:
                    movement_avg[tid] = movement
                else:
                    movement_avg[tid] = movement_avg[tid] * 0.8 + movement * 0.2

                if tid not in head_radii:
                    head_radii[tid] = min_radius

                if movement_avg[tid] < movement_sensitivity:
                    head_radii[tid] = min(max_radius, head_radii[tid] + growth_speed)
                else:
                    head_radii[tid] = max(min_radius, head_radii[tid] - growth_speed)

                # Kopf zeichnen mit dynamischem Radius
                head = tracks[tid][-1]
                heads.append(head)
                cv2.circle(
                    trail_layer,
                    head,
                    radius=int(head_radii[tid]),
                    color=dark.astype(np.uint8).tolist(),
                    thickness=-1,
                    lineType=cv2.LINE_AA
                )

                # Linie der Bewegung zeichnen
                num_points = len(tracks[tid])
                if num_points >= 2:
                    for i in range(num_points - 1):
                        pt1 = tracks[tid][i]
                        pt2 = tracks[tid][i + 1]
                        color = bright.astype(np.uint8).tolist()
                        cv2.line(trail_layer, pt1, pt2, color, thickness=thickness_glow, lineType=cv2.LINE_AA)
                        cv2.line(trail_layer, pt1, pt2, color, thickness=thickness_mid, lineType=cv2.LINE_AA)
                        cv2.line(trail_layer, pt1, pt2, color, thickness=thickness_core, lineType=cv2.LINE_AA)
    
    # 2. Verbindungs-Layer in jedem Frame neu erstellen
    connections_layer = np.zeros_like(frame, dtype=np.uint8)

    # ======= Dynamische Kopf-Vernetzung =======
    for i in range(len(heads)):
        for j in range(i + 1, len(heads)):
            pt1 = heads[i]
            pt2 = heads[j]
            dist = np.linalg.norm(np.array(pt1) - np.array(pt2))
            if dist <= connection_distance:
                cv2.line(connections_layer, pt1, pt2, (255, 255, 255), connection_thickness, lineType=cv2.LINE_AA)
    
    # 3. Alle Ebenen kombinieren
    temp_combined_layer = cv2.addWeighted(trail_layer, 1.0, connections_layer, 1.0, 0)
    
    # Finales Bild erstellen (über den schwarzen Hintergrund)
    final_image = cv2.addWeighted(np.zeros_like(frame), 1.0, temp_combined_layer, 1.0, 0)
    
    # Ausgabe skalieren
    frame_resized = cv2.resize(final_image, screen_res, interpolation=cv2.INTER_LINEAR)
    cv2.imshow(window_name, frame_resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()