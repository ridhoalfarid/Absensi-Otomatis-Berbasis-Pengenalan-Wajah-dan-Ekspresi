import cv2
import pickle
import numpy as np
from skimage import feature

# Kelas untuk Ekstraksi Fitur LBP
class LocalBinaryPatterns:
    def __init__(self, num_points, radius):
        self.num_points = num_points
        self.radius = radius

    def describe(self, image, eps=1e-7):
        lbp = feature.local_binary_pattern(image, self.num_points, self.radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),
                                 bins=np.arange(0, self.num_points + 3),
                                 range=(0, self.num_points + 2))
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)
        return hist

# Model dan Mapping
print("Memuat model...")
# Model Individu
with open("models/model_individu.pkl", "rb") as f:
    model_identitas = pickle.load(f)
with open("models/mapping_nama.pkl", "rb") as f:
    mapping_nama = pickle.load(f)

# Model Ekspresi
with open("models/model_ekspresi.pkl", "rb") as f:
    model_ekspresi = pickle.load(f)
with open("models/mapping_ekspresi.pkl", "rb") as f:
    mapping_ekspresi = pickle.load(f)
    
lbp_desc = LocalBinaryPatterns(24, 8)
face_cascade = cv2.CascadeClassifier('D:/SEMESTER 6/AI for DS/face_recognition/haarcascade_frontalface_default.xml')

# Buka kamera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Tidak bisa membuka kamera.")
    exit()
    
print("Aplikasi real-time berjalan... Tekan 'q' untuk keluar.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Potong dan proses wajah
        face_gray = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face_gray, (224, 224))
        
        # Ekstrak fitur LBP
        hist = lbp_desc.describe(face_resized)
        
        # --- Prediksi Identitas ---
        id_pred = model_identitas.predict([hist])[0]
        nama_pred = mapping_nama.get(id_pred, "Tidak Dikenal")

        # --- Prediksi Ekspresi ---
        id_ekspresi_pred = model_ekspresi.predict([hist])[0]
        ekspresi_pred = mapping_ekspresi.get(id_ekspresi_pred, "Error")
        
        # Tampilkan hasil
        label_text = f"{nama_pred} | Ekspresi: {ekspresi_pred}"
        
        # Gambar kotak dan label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow('Aplikasi Pengenalan Wajah dan Ekspresi', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()