import os
import pickle
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# You can update the username according to your device.
klasor_yolu = r"C:\Users\BURAK\Desktop\witsign"
model_yolu = os.path.join(klasor_yolu, "witcher_model.p")

# Load the trained model
try:
  with open(model_yolu, "rb") as f:
    model = pickle.load(f)
  print(f"Model yüklendi: {model_yolu}")
except FileNotFoundError:
  print(
      f"HATA: '{model_yolu}' bulunamadı! Önce train_model.py dosyasını"
      " çalıştırmalısınız."
  )
  exit()

# MediaPipe current hand detection settings
base_options = python.BaseOptions(model_asset_path=r"C:\Users\BURAK\Desktop\witsign\hand_landmarker.task")
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1,
)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
print("Witcher Büyü Sistemi Aktif! Çıkmak için 'q' tuşuna basın.")

while cap.isOpened():
  success, frame = cap.read()
  if not success:
    break

  frame = cv2.flip(frame, 1)
  height, width, _ = frame.shape

  rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

  result = detector.detect(mp_image)

  if result.hand_landmarks:
    for hand_landmarks in result.hand_landmarks:
      for lm in hand_landmarks:
        cx, cy = int(lm.x * width), int(lm.y * height)
        cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

      row = []
      base_x = hand_landmarks[0].x
      base_y = hand_landmarks[0].y
      base_z = hand_landmarks[0].z

      for lm in hand_landmarks:
        row.append(lm.x - base_x)
        row.append(lm.y - base_y)
        row.append(lm.z - base_z)

      # Predict using the model
      X_test = np.array([row])
      tahmin = model.predict(X_test)[0]

      # Display the sign name on the screen
      cv2.putText(
          frame,
          f"Sign: {tahmin}",
          (40, 50),
          cv2.FONT_HERSHEY_SIMPLEX,
          1,
          (0, 0, 255),
          2,
          cv2.LINE_AA,
      )

  cv2.imshow("Witcher Sign Detection", frame)

  if cv2.waitKey(1) & 0xFF == ord("q"):
    break

cap.release()
cv2.destroyAllWindows()