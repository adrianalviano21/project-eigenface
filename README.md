# 🧠 Face Recognition with Eigenface

Sistem pengenalan wajah berbasis metode **Eigenface**, dibangun menggunakan Python dan Streamlit. Program ini mengenali wajah berdasarkan kemiripan fitur dengan gambar-gambar yang ada di dataset, menggunakan analisis vektor eigen dan pembobotan ruang wajah.

---

## 📋 Detail Content

1. [Basic Information](#1-basic-information)  
2. [Display Program](#2-display-program)  
3. [How to Run](#3-how-to-run)  
4. [Project Structure](#4-project-structure)

---

### 1. Basic Information

- **Project Name**: Face Recognition with Eigenface  
- **Metode**: Eigenface (PCA khusus wajah)  
- **Bahasa Pemrograman**: Python 3  
- **Framework Antarmuka**: Streamlit  
- **Library Utama**:
  - `numpy`
  - `opencv-python`
  - `streamlit`
  - `Pillow`
- **Input Format**: `.jpg`, `.jpeg`, `.png`
- **Ukuran Citra**: Semua wajah dikonversi ke grayscale 128×128

---

### 2. Display Program
![WhatsApp Image 2025-06-08 at 19 35 45_9db5dd6d](https://github.com/user-attachments/assets/b67a9dec-bf8e-4cbf-b266-f07ee8b84bdd)


Contoh placeholder:

```text
+--------------------+         +------------------------+
|   Upload Gambar    | ---->   |  Proses Eigenface      |
+--------------------+         +------------------------+
                                       |
                                       v
                          +------------------------+
                          |  Gambar Cocok Ditampilkan |
                          +------------------------+

```
### 3. How to Run 
✅ **Instalasi Library**
```text
pip install -r requirements.txt
```

🚀 **Jalankan Antarmuka Streamlit di Terminal VScode atau CMD**
```text
streamlit run interface.py
```

---
### 4. Project Structure
```text
.
├── README.md 
├── requirements.txt 
├── dataset/ 
└── src/ 
    ├── eigenface2.py 
    ├── interface.py
    └── main.py 
```

---

### Creators 
| Nama                         | NIM     |
|-----------------------------|---------|
| Adrian Alviano Susatyo      | L0124001|
| Daniel Ferdian Napitupulu   | L0124008|
| Diva Valencia Christianarta | L0124011|
| Fauzil Azhim                | L0124015|

