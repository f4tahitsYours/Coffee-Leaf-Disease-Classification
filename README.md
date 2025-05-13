# Deteksi Penyakit Daun Kopi Menggunakan CNN

Proyek ini bertujuan untuk membangun model klasifikasi citra berbasis **Convolutional Neural Network (CNN)** untuk mendeteksi berbagai jenis penyakit pada daun kopi dari gambar. Dataset citra daun diklasifikasikan ke dalam beberapa kelas penyakit untuk membantu proses diagnosis otomatis.

## 📂 Struktur Dataset
Dataset berisi gambar daun kopi yang telah dikategorikan ke dalam beberapa folder berdasarkan jenis penyakit. Dataset dibagi ke dalam tiga bagian:
- `train/` - data untuk pelatihan
- `validation/` - data untuk validasi selama pelatihan
- `test/` - data untuk evaluasi akhir model

## 🛠️ Alur Proyek
1. **Mount Google Drive** dan ekstraksi dataset `.zip` ke Google Colab.
2. **Preprocessing Dataset**:
   - Split dataset ke folder `train`, `validation`, dan `test`.
   - Visualisasi distribusi jumlah gambar per kelas.
3. **Augmentasi Gambar** menggunakan `ImageDataGenerator` untuk memperkuat generalisasi model.
4. **Model CNN**:
   - Dibuat dengan arsitektur sederhana (Conv2D + MaxPooling + Dense).
   - Menggunakan `Adam` optimizer dan `categorical_crossentropy` loss.
5. **Pelatihan Model**:
   - Training hingga 30 epoch dengan `EarlyStopping` dan `ModelCheckpoint`.
6. **Evaluasi**:
   - Akurasi model diukur menggunakan data test.
   - Visualisasi metrik seperti akurasi, loss, confusion matrix, dan classification report.

## 🧪 Hasil Sementara
- Akurasi training dan validasi divisualisasikan untuk memantau overfitting.
- Evaluasi akhir menghasilkan `Test Accuracy` dan laporan klasifikasi lengkap.
- Confusion matrix divisualisasikan menggunakan seaborn.

## 🧰 Teknologi & Library
- Python 3
- TensorFlow / Keras
- Pandas, NumPy, Matplotlib, Seaborn
- Google Colab (Cloud GPU support)

## 📁 Folder Penting
- `dataset.zip` → dataset citra daun kopi
- `penyakit_kopi/train/`, `validation/`, `test/` → struktur dataset hasil split
- `best_model.keras` → model terbaik hasil training

## ⚠️ Catatan
- Dataset diasumsikan telah diunggah ke Google Drive (`/MyDrive/dataset.zip`).
- File `dataset.zip` berisi struktur folder sesuai dengan nama kelas.

---

