import cv2
import os
import time

# Konfigurasi Dinamis
print("--- Setup Pengambilan Data Wajah ---")
nama_subjek = input("Masukkan Nama: ")
id_subjek_str = input(f"Masukkan ID unik untuk {nama_subjek}: ")
ekspresi = input("Masukkan Ekspresi yang akan diambil (Netral/Senang/Sedih): ")
jumlah_sampel = 100

# Validasi input
if not nama_subjek or not id_subjek_str.isdigit() or ekspresi not in ['Netral', 'Senang', 'Sedih']:
    print("\nError: Input tidak valid. Pastikan ID adalah angka dan ekspresi adalah salah satu dari pilihan.")
    exit()

id_subjek = int(id_subjek_str)

# Path Setup
base_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(base_dir, "dataset")
subjek_dir = os.path.join(dataset_dir, f"{nama_subjek}_{id_subjek}")
ekspresi_dir = os.path.join(subjek_dir, ekspresi)

os.makedirs(ekspresi_dir, exist_ok=True)

# Inisialisasi Haar Cascade untuk deteksi wajah
face_cascade = cv2.CascadeClassifier('D:/SEMESTER 6/AI for DS/face_recognition/haarcascade_frontalface_default.xml')

# Buka kamera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Tidak bisa membuka kamera.")
    exit()

print(f"\n[INFO] Mempersiapkan pengambilan gambar untuk '{nama_subjek}' dengan ekspresi '{ekspresi}'.")
print("[INFO] Lihat ke kamera dan tekan 's' untuk mulai mengambil gambar...")
print("[INFO] Tekan 'q' kapan saja untuk keluar.")

hitung_sampel = 0
siap_ambil = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Tidak bisa menerima frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Gambar kotak di sekitar wajah
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Tampilkan instruksi
        if not siap_ambil:
             cv2.putText(frame, "Tekan 's' untuk mulai", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
             cv2.putText(frame, f"Mengambil: {hitung_sampel}/{jumlah_sampel}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if siap_ambil and hitung_sampel < jumlah_sampel:
            # Simpan gambar wajah yang telah dipotong dan di-grayscale
            face_img = gray[y:y+h, x:x+w]
            timestamp = int(time.time() * 1000)
            file_path = os.path.join(ekspresi_dir, f"{timestamp}.jpg")
            
            # Ubah ukuran sebelum menyimpan untuk konsistensi
            resized_face = cv2.resize(face_img, (224, 224))
            cv2.imwrite(file_path, resized_face)

            hitung_sampel += 1
            print(f"Gambar {hitung_sampel} disimpan di {file_path}")
            time.sleep(0.2)

    cv2.imshow('Pengumpul Data Wajah', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        siap_ambil = True
        print("[INFO] Mulai mengambil gambar...")
    elif key == ord('q') or hitung_sampel >= jumlah_sampel:
        break

print(f"\n[INFO] Pengambilan {hitung_sampel} gambar selesai.")
cap.release()
cv2.destroyAllWindows()