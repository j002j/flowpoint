#punkte mit netz-Linien ohne Trail 
#punktNetzOhneTrail.py

import cv2
from ultralytics import YOLO
import numpy as np
import random

# =========================
# ░ Parameter / Look-Setup ░
# =========================

point_radius = 10              # Größe des Punktes für jede Person
brightness_offset = 80         # Farbe der Punkte

# Kopf-Vernetzung
connection_distance = 120      # maximale Entfernung für Linien
connection_thickness = 2       # Dicke der Linien zwischen Punkten

# =================
# ░ YOLO vorbereiten ░
# =================

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(1)  # Kamera

if not cap.isOpened():
    print("Kamera konnte nicht geöffnet werden")
    exit()

id_colors = {}  # Feste Farbe für jede Person-ID

# =================
# ░ Hilfsfunktionen ░
# =================

def get_color_for_id(track_id: int):
    """Jeder ID eine feste Farbe zuweisen."""
    if track_id not in id_colors:
        rng = random.Random(track_id)
        color = (
            rng.randint(100, 255),
            rng.randint(100, 255),
            rng.randint(100, 255)
        )
        id_colors[track_id] = color
    return id_colors[track_id]

# ==========================
# ░ Fenster-/Anzeige-Setup ░
# ==========================

window_name = "Personenpunkte"
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

    # Für jedes Frame erstellen wir einen neuen weißen Hintergrund
    display_frame = np.ones_like(frame, dtype=np.uint8) * 255

    # Personen erkennen
    results = model.track(frame, persist=True, verbose=False)
    result = results[0]

    heads = []  # Liste für aktuelle Punkte

    if hasattr(result, "boxes") and result.boxes.id is not None:
        for box, cls, track_id in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.id):
            if int(cls) == 0:  # nur Personen
                x1, y1, x2, y2 = map(int, box)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                # Farbe für diese Person
                base_color = np.array(get_color_for_id(int(track_id)), dtype=np.int32)
                color = np.clip(base_color - brightness_offset, 0, 255).astype(np.uint8).tolist()

                # Punkt zeichnen
                cv2.circle(display_frame, (cx, cy), radius=point_radius, color=color, thickness=-1, lineType=cv2.LINE_AA)
                heads.append((cx, cy))

    # Linien nur zwischen nahe beieinander liegenden Punkten
    for i in range(len(heads)):
        for j in range(i + 1, len(heads)):
            pt1 = heads[i]
            pt2 = heads[j]
            dist = np.linalg.norm(np.array(pt1) - np.array(pt2))
            if dist <= connection_distance:
                cv2.line(display_frame, pt1, pt2, (0, 0, 0), connection_thickness, lineType=cv2.LINE_AA)

    # Anzeige skalieren
    frame_resized = cv2.resize(display_frame, screen_res, interpolation=cv2.INTER_LINEAR)
    cv2.imshow(window_name, frame_resized)

    # Mit "q" beenden
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
