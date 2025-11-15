import cv2
import numpy as np
import os

# ====== Paths ======
dataset_path = r"C:\Users\User\Pictures\FaceDataSet"  # same path as in collector
people = os.listdir(dataset_path)  # e.g. ['Bijey', 'Jude','nigga']

# ====== Prepare data ======
Training_Data, Labels = [], []
label_dict = {}  # for mapping numeric label → name
current_label = 0

print("🔍 Loading training data...")

for person in people:
    person_folder = os.path.join(dataset_path, person)
    if not os.path.isdir(person_folder):
        continue

    label_dict[current_label] = person

    for image_name in os.listdir(person_folder):
        image_path = os.path.join(person_folder, image_name)
        if not image_name.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue  # skip non-image files
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        Training_Data.append(np.asarray(img, dtype=np.uint8))
        Labels.append(current_label)

    print(f"✅ Loaded {len(os.listdir(person_folder))} images for {person}")
    current_label += 1

# ====== Train Model ======
if len(Labels) == 0:
    print("❌ No data found! Run the face collector first.")
    exit()

Labels = np.asarray(Labels, dtype=np.int32)
model = cv2.face.LBPHFaceRecognizer_create()
model.train(np.asarray(Training_Data), Labels)

# ====== Save Model ======
model_path = os.path.join(dataset_path, "face_trainer.yml")
model.save(model_path)

print("\n🎯 Dataset training complete!")
print(f"💾 Model saved to: {model_path}")
print("🧠 Label mapping:")
for label, name in label_dict.items():
    print(f"   {label} → {name}")
