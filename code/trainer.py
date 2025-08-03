import cv2
import os
import numpy as np
from skimage import feature
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

print("Memulai proses training model...")
# Path Setup
base_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(base_dir, "dataset")
models_dir = os.path.join(base_dir, "models")
eval_dir = os.path.join(base_dir, "evaluation_results")
os.makedirs(models_dir, exist_ok=True)
os.makedirs(eval_dir, exist_ok=True)

# Ekstraksi Fitur LBP
class LocalBinaryPatterns:
    def __init__(self, num_points, radius):
        self.num_points = num_points
        self.radius = radius

    def describe(self, image, eps=1e-7):
        # Hitung representasi LBP dari gambar
        lbp = feature.local_binary_pattern(image, self.num_points, self.radius, method="uniform")
        
        # Buat histogram dari pola LBP
        (hist, _) = np.histogram(lbp.ravel(),
                                 bins=np.arange(0, self.num_points + 3),
                                 range=(0, self.num_points + 2))
        
        # Normalisasi histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)
        
        return hist

# Muat Dataset dan Ekstrak Fitur
print("\nMemuat dataset dan mengekstrak fitur LBP...")

lbp_desc = LocalBinaryPatterns(24, 8)

data_fitur = []
label_individu = []
label_ekspresi = []
mapping_nama = {}
mapping_ekspresi = {"Netral": 0, "Senang": 1, "Sedih": 2}

for subjek_folder in sorted(os.listdir(dataset_dir)):
    if os.path.isdir(os.path.join(dataset_dir, subjek_folder)):
        try:
            nama, id_subjek_str = subjek_folder.rsplit('_', 1)
            id_subjek = int(id_subjek_str)
            mapping_nama[id_subjek] = nama
            
            subjek_path = os.path.join(dataset_dir, subjek_folder)
            for ekspresi_folder in sorted(os.listdir(subjek_path)):
                if ekspresi_folder in mapping_ekspresi:
                    ekspresi_path = os.path.join(subjek_path, ekspresi_folder)
                    id_ekspresi = mapping_ekspresi[ekspresi_folder]

                    for filename in os.listdir(ekspresi_path):
                        img_path = os.path.join(ekspresi_path, filename)
                        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        
                        if image is not None:
                            hist = lbp_desc.describe(image)
                            data_fitur.append(hist)
                            label_individu.append(id_subjek)
                            label_ekspresi.append(id_ekspresi)
        except ValueError:
            print(f"Melewatkan folder dengan format nama salah: {subjek_folder}")
            continue

print(f"Total {len(data_fitur)} gambar diproses.")
print(f"Mapping Nama: {mapping_nama}")

# Model Individu (SVM)
print("\n--- Melatih Model Individu (SVM) ---")
X_train, X_test, y_train, y_test = train_test_split(data_fitur, label_individu, test_size=0.2, random_state=42, stratify=label_individu)
print(f"Data Individu: {len(X_train)} training, {len(X_test)} testing.")

model_individu = SVC()
model_individu.fit(X_train, y_train)

# Evaluasi
y_pred = model_individu.predict(X_test)
#print(f"Akurasi Model Individu (SVM): {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nLaporan Klasifikasi Model Individu:")
target_names_individu = [mapping_nama[i] for i in model_individu.classes_]
print(classification_report(y_test, y_pred, target_names=target_names_individu, zero_division=0))
# Confusion Matrix - Individu
cm_individu = confusion_matrix(y_test, y_pred, labels=model_individu.classes_)
plt.figure(figsize=(10, 7))
sns.heatmap(cm_individu, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names_individu,
            yticklabels=target_names_individu)
plt.title('Confusion Matrix - Individu (SVM)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(eval_dir, 'confusion_matrix_individu.png'))
plt.close()

# Simpan model
with open(os.path.join(models_dir, "model_individu.pkl"), "wb") as f:
    pickle.dump(model_individu, f)
with open(os.path.join(models_dir, "mapping_nama.pkl"), "wb") as f:
    pickle.dump(mapping_nama, f)
print("Model Individu berhasil disimpan.")

# Model Ekspresi (KNN)
print("\n--- Melatih Model Ekspresi (KNN) ---")
X_train_e, X_test_e, y_train_e, y_test_e = train_test_split(data_fitur, label_ekspresi, test_size=0.2, random_state=42, stratify=label_ekspresi)
print(f"Data Ekspresi: {len(X_train_e)} training, {len(X_test_e)} testing.")

model_ekspresi = KNeighborsClassifier(weights='distance')
model_ekspresi.fit(X_train_e, y_train_e)

# Evaluasi
y_pred_e = model_ekspresi.predict(X_test_e)
#print(f"Akurasi Model Ekspresi (k-NN): {accuracy_score(y_test_e, y_pred_e) * 100:.2f}%")
print("\nLaporan Klasifikasi Model Ekspresi:")
mapping_ekspresi_inv = {v: k for k, v in mapping_ekspresi.items()}
target_names_ekspresi = [mapping_ekspresi_inv[i] for i in model_ekspresi.classes_]
print(classification_report(y_test_e, y_pred_e, target_names=target_names_ekspresi, zero_division=0))
# Confusion Matrix - Ekspresi
cm_ekspresi = confusion_matrix(y_test_e, y_pred_e, labels=model_ekspresi.classes_)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_ekspresi, annot=True, fmt='d', cmap='Greens',
            xticklabels=target_names_ekspresi,
            yticklabels=target_names_ekspresi)
plt.title('Confusion Matrix - Ekspresi (KNN)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(eval_dir, 'confusion_matrix_ekspresi.png'))
plt.close()

# Simpan model
with open(os.path.join(models_dir, "model_ekspresi.pkl"), "wb") as f:
    pickle.dump(model_ekspresi, f)
mapping_ekspresi_inv = {v: k for k, v in mapping_ekspresi.items()}
with open(os.path.join(models_dir, "mapping_ekspresi.pkl"), "wb") as f:
    pickle.dump(mapping_ekspresi_inv, f)
print("Model Ekspresi berhasil disimpan.")

print("\nTraining selesai. Semua model telah disimpan di folder 'models'.")