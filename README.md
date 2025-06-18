# CitWorks – Deteksi Kesegaran Daging Sapi dengan CNN

Aplikasi ini dibuat oleh kelompok CitWorks untuk tugas akhir, bertujuan mendeteksi tingkat kesegaran daging sapi berdasarkan citra.

📌 Kategori klasifikasi:
- Segar
- Setengah Segar
- Busuk

🧠 Teknologi & Metode:
- Bahasa: Python
- Library: TensorFlow, Keras, OpenCV, PyQt5
- Arsitektur CNN:
  - Conv2D (32 filter) + MaxPooling2D
  - Conv2D (64 filter) + MaxPooling2D
  - Flatten
  - Dense(128) + Dropout(0.5)
  - Output: Dense(3) dengan Softmax
- Input gambar: 150x150 piksel

🔬 Selain klasifikasi CNN, fitur HSV dan tekstur ditampilkan:
- Rata-rata nilai Hue, Saturation, Value
- Fitur tekstur: kontras, entropi, varian, energi, dll

📂 Struktur Folder:
- percobaan.py → program utama GUI Deteksi Daging
- CitWorks.ui → desain GUI Qt Designer
- ModelCNN.py → Pelatihan Dataset
- Model CNN terlatih & Dataset →
- Aplikasi_CitWorks_Seluruh_Operasi → RTM 2

👨‍💻 Kelompok: CitWorks  
Anggota:
- 152023001 - Sintia Wati
- 152023005 - Nabilla Hasya Permana
- 152023200 - Fida Nujiya

📌 Catatan:
- Dataset tidak disertakan di repository ini sesuai ketentuan tugas.
- ModelCNN terlatih tidak disertakan karena melebihi batas maksimal GitHub yaitu 100 MB.

🎯 Tujuan akhir:  
Meningkatkan akurasi pendeteksian kondisi daging sapi secara otomatis dengan bantuan CNN dan fitur citra, serta mempermudah pengguna dalam menentukan kualitas daging berdasarkan foto.