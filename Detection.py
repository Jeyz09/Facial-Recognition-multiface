import cv2
import numpy as np
import os

# ====== Paths ======
dataset_path = r"C:\Users\User\Pictures\FaceDataset"
model_path = os.path.join(dataset_path, "face_trainer.yml")

# ====== Load trained model ======
if not os.path.exists(model_path):
    print("❌ Model not found! Train first before running this.")
    exit()

model = cv2.face.LBPHFaceRecognizer_create()
model.read(model_path)

# ====== Build label-name mapping ======
people = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
label_dict = {i: name for i, name in enumerate(people)}

# ====== Load face detector ======
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Webcam not found.")
    exit()

print("🧠 Multi-face recognition running... Press ENTER to stop.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to capture frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (200, 200))

        result = model.predict(face_resized)
        confidence = int(100 * (1 - result[1] / 300))

        if confidence > 82:
            name = label_dict.get(result[0], "Unknown")
            color = (0, 255, 0)
            text = f"{name} ({confidence}%)"
        else:
            name = "Unknown"
            color = (0, 0, 255)
            text = "Unknown"

        # Draw bounding box and name for each detected face
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("Multi-Face Recognition", frame)

    if cv2.waitKey(1) == 13:  # ENTER key
        break

cap.release()
cv2.destroyAllWindows()
print("🟢 Recognition stopped.")
