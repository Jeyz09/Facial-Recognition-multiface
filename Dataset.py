import cv2
import os

# ====== Setup ======
# List of names you want to collect
persons = ["Bijey", "Jude", "nigga"]

# Base folder to save datasets (outside OneDrive to avoid red X)
dataset_path = r"C:\Users\User\Pictures\FaceDataSet"
os.makedirs(dataset_path, exist_ok=True)

# Load Haar Cascade for face detection
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def face_extractor(img):
    """Detect and return the cropped face."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return None
    for (x, y, w, h) in faces:
        return img[y:y+h, x:x+w]

# ====== Start collecting for each person ======
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ ERROR: Cannot access camera.")
    exit()

for person_name in persons:
    print(f"\n📸 Collecting samples for {person_name} — Press ENTER to stop early.")
    person_folder = os.path.join(dataset_path, person_name)
    os.makedirs(person_folder, exist_ok=True)

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to capture frame.")
            break

        face = face_extractor(frame)
        if face is not None:
            count += 1
            face = cv2.resize(face, (200, 200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            file_name_path = os.path.join(person_folder, f"{count}.jpg")
            cv2.imwrite(file_name_path, face)

            cv2.putText(frame, f"{person_name}: {count}/100", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No face detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Face Collector", frame)

        # ENTER = stop early, or stop after 100 samples
        if cv2.waitKey(1) == 13 or count >= 100:
            break

    print(f"✅ Finished collecting for {person_name}. Total samples: {count}")

cap.release()
cv2.destroyAllWindows()
print("\n🎯 Dataset collection complete for all persons!")
