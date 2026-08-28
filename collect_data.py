import csv
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

# Automatically gets the directory where this Python file is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Creates the correct path dynamically on any computer
dosya_yolu = os.path.join(BASE_DIR, "dataset.csv")
task_yolu = os.path.join(BASE_DIR, "hand_landmarker.task")

# Define the model file and settings
base_options = python.BaseOptions(model_asset_path=task_yolu)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1,
)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
etiket = 'Igni'  # The name of the sign you want to save
print(f"Collecting data for '{etiket}'. Press 's' to save, 'q' to quit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    result = detector.detect(mp_image)

    # Using a single waitKey to keep the key in memory
    key = cv2.waitKey(1) & 0xFF

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            for lm in hand_landmarks:
                cx, cy = int(lm.x * width), int(lm.y * height)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

            if key == ord('s'):
                row = []
                base_x = hand_landmarks[0].x
                base_y = hand_landmarks[0].y
                base_z = hand_landmarks[0].z

                for lm in hand_landmarks:
                    row.append(lm.x - base_x)
                    row.append(lm.y - base_y)
                    row.append(lm.z - base_z)

                row.append(etiket)

                with open(dosya_yolu, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(row)
                print(f'SUCCESS: Sample saved ({etiket})')

    cv2.imshow('Witcher Data Collection', frame)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()