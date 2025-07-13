import cv2
import face_recognition
import numpy as np
import json
import os
import tkinter as tk
from tkinter.simpledialog import askstring

DATA_FILE = "face_data.json"
FONT = cv2.FONT_HERSHEY_SIMPLEX
ENCODING_THRESHOLD = 0.45
PROCESS_EVERY_N_FRAMES = 5

if os.path.exists(DATA_FILE):
    with open(DATA_FILE, "r") as f:
        face_names = json.load(f)
else:
    face_names = {}

known_encodings = []
known_ids = []
next_id = len(face_names)
clicked_pos = None
frame_count = 0
last_faces = [] 

tk_root = tk.Tk()
tk_root.withdraw()

def save_face_data():
    with open(DATA_FILE, "w") as f:
        json.dump(face_names, f, indent=2)

def get_matching_id(encoding):
    for idx, known_enc in enumerate(known_encodings):
        dist = np.linalg.norm(known_enc - encoding)
        if dist < ENCODING_THRESHOLD:
            return known_ids[idx]
    return None

def mouse_callback(event, x, y, flags, param):
    global clicked_pos
    if event == cv2.EVENT_LBUTTONDBLCLK:
        clicked_pos = (x, y)

cap = cv2.VideoCapture(0)
cv2.namedWindow("FaceTagger")
cv2.setMouseCallback("FaceTagger", mouse_callback)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    if frame_count % PROCESS_EVERY_N_FRAMES == 0:
        last_faces = []

        small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small)
        face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

        for i, (top, right, bottom, left) in enumerate(face_locations):
            face_enc = face_encodings[i]
            face_id = get_matching_id(face_enc)

            if not face_id:
                face_id = f"face_{next_id}"
                known_encodings.append(face_enc)
                known_ids.append(face_id)
                next_id += 1

            name = face_names.get(face_id, face_id)

            left *= 4
            top *= 4
            right *= 4
            bottom *= 4

            last_faces.append((left, top, right, bottom, face_id, name))

    for (left, top, right, bottom, face_id, name) in last_faces:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), FONT, 0.7, (255, 255, 255), 2)

        if clicked_pos and left <= clicked_pos[0] <= right and top <= clicked_pos[1] <= bottom:
            new_name = askstring("Rename Face", f"Enter name for {face_id}:", initialvalue=name)
            if new_name:
                face_names[face_id] = new_name.strip()
                save_face_data()
                print(f"Renamed {face_id} = {new_name.strip()}")
            clicked_pos = None

    cv2.imshow("FaceTagger", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
