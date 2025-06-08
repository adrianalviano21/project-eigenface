import numpy as np
import os
import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def load_dataset(dataset_dir: str):
    """
    Membaca dataset wajah dari folder dan mengembalikan:
    - data: matriks vektor wajah (N, D)
    - mean_vec: rata-rata wajah (1, D)
    - selisih_vec: matriks selisih (N, D)
    - cov: matriks kovarian kecil (N, N)
    - eigvec: eigenvector asli (N, D)
    - weight_data: bobot dari semua wajah (N, N)
    """
    data = vectortoMatrix(dataset_dir)
    mean_vec = mean(data)
    selisih_vec = selisih(mean_vec, data)
    cov = covariance(selisih_vec)
    eigvec = eig(cov, selisih_vec)
    weight_data = weight_dataset(selisih_vec, eigvec)
    return data, mean_vec, selisih_vec, cov, eigvec, weight_data

def load_image(file):
    """
    Membaca file upload gambar (dari Streamlit) dan mengembalikan
    citra grayscale OpenCV.
    """
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return img

def predict_face(img_path, dataset_dir, mean_vec, eigvec, weight_data, threshold=80.0):
    """
    Wrapper untuk memanggil recogniseUnknownFace secara langsung.
    """
    return recogniseUnknownFace(
        dataset_dir=dataset_dir,
        test_path=img_path,
        datasetMean=mean_vec,
        eigVec=eigvec,
        weightDataset=weight_data,
        threshold=threshold,
    )

def imagetoVector(img_path: str) -> np.ndarray:
    """
    Membaca gambar, melakukan deteksi wajah, crop area wajah,
    mengubah ke grayscale, resize ke (128×128), lalu flatten menjadi vektor (1×D).
    """
    image = cv2.imread(img_path)
    if image is None:
        raise ValueError(f"Gambar tidak bisa dibaca: {img_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        raise ValueError(f"Tidak ada wajah terdeteksi di: {img_path}")

    # Ambil wajah pertama yang terdeteksi
    x, y, w, h = faces[0]
    face = gray[y : y + h, x : x + w]
    face_resized = cv2.resize(face, (128, 128), interpolation=cv2.INTER_AREA)

    return face_resized.flatten()


def vectortoMatrix(folder_path: str) -> np.ndarray:
    """
    Mengonversi semua gambar di folder_path menjadi matriks N×D,
    di mana N = jumlah gambar yang valid, D = 128×128 (→ 16384).
    File selain .jpg/.jpeg/.png akan diabaikan.
    """
    files = [
        f
        for f in os.listdir(folder_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    vectors = []
    for fname in files:
        full_path = os.path.join(folder_path, fname)
        try:
            vec = imagetoVector(full_path)  # (16384,)
            vectors.append(vec)
        except ValueError as e:
            # Jika file rusak atau tidak ada wajah, skip tetapi cetak pesan
            print(f"[WARN] {e}")

    if len(vectors) == 0:
        raise RuntimeError(f"Tidak ada gambar wajah valid di folder: {folder_path}")

    # Bentuk array → (N, 16384)
    return np.array(vectors)


def mean(matImgVec: np.ndarray) -> np.ndarray:
    """
    Menghitung rata‐rata (mean) setiap kolom vektor wajah.
    Keluaran → (1, D).
    """
    return np.mean(matImgVec, axis=0, keepdims=True)  # → (1, 16384)


def selisih(mean_vec: np.ndarray, matImgVec: np.ndarray) -> np.ndarray:
    """
    Mengurangi setiap baris pada matImgVec dengan mean_vec (broadcast).
    Keluaran → (N, D).
    """
    return matImgVec - mean_vec


def covariance(matSelisih: np.ndarray) -> np.ndarray:
    """
    Menghitung matriks kovarian kecil (N×N):
        C = (1/N) * (matSelisih @ matSelisihᵀ)
    """
    N = matSelisih.shape[0]
    cov = (matSelisih @ matSelisih.T) / float(N)
    return cov  # → (N, N)


def QR(M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Decomposisi QR dengan Householder untuk matriks persegi (N×N).
    Mengembalikan (Q, R) sehingga M = Q @ R.
    """
    (cntRows, _) = M.shape
    Q = np.eye(cntRows)
    R = M.copy().astype(float)

    for j in range(cntRows - 1):
        x = R[j:, j].copy()
        x[0] += np.copysign(np.linalg.norm(x), x[0])
        v = x / np.linalg.norm(x)
        H = np.eye(cntRows)
        H[j:, j:] -= 2.0 * np.outer(v, v)
        Q = Q @ H
        R = H @ R

    return Q, np.triu(R)


def eigQR(M: np.ndarray, iterations: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """
    Menghitung eigenvalues dan eigenvectors (small) dari matriks M (NxN) 
    menggunakan iterasi QR shift.
    Return:
      - eigVals (array shape (N,), berisi eigenvalue)
      - eigVecsSmall (array shape (N, N), kolom ke‐i adalah eigenvector ke‐i)
    """
    N = M.shape[0]
    eigVecs = np.eye(N)
    A = M.astype(float).copy()

    for _ in range(iterations):
        s = A[-1, -1] * np.eye(N)
        Q, R = QR(A - s)
        A = R @ Q + s
        eigVecs = eigVecs @ Q

    eigVals = np.diag(A)
    return eigVals, eigVecs


def eig(matCov: np.ndarray, matSelisih: np.ndarray) -> np.ndarray:
    """
    Menghitung eigenvectors di ruang asli (D‐dimensi) menggunakan trick:
      1) Hitung eigenvalues & eigenvectors (small) dari matCov (N×N).
      2) Proyeksikan ke ruang D: eigVecLarge = matSelisihᵀ @ eigVecSmall (→ D×N).
      3) Normalisasi setiap kolom → eigenvectors besar (D).
    Output:
      - Array shape (N, D) di mana baris ke‐i adalah eigenvector ke‐i di ruang asli.
    """
    eigVals, eigVecSmall = eigQR(matCov)
    eigVecSmall = eigVecSmall.T  # Sekarang shape → (N, N)

    # Proyeksi ke ruang D: matSelisihᵀ (D×N) @ eigVecSmall (N×N) → (D×N)
    eigVecLarge = matSelisih.T @ eigVecSmall  # → (D, N)

    # Normalisasi setiap kolom
    eigVecLarge = eigVecLarge / np.linalg.norm(eigVecLarge, axis=0)

    # Transpos → (N, D), di mana baris i adalah eigenvector ke‐i
    return eigVecLarge.T


def weight_dataset(matSelisih: np.ndarray, eigVec: np.ndarray) -> np.ndarray:
    """
    Menghitung bobot (weights) untuk setiap training image:
      - matSelisih: (N, D)
      - eigVec: (N, D)
    WeightDataset: setiap baris i = (matSelisih[i] @ eigVecᵀ) → vektor (N,)
    Keluaran: array shape (N, N)
    """
    # (N, D) @ (D, N) → (N, N)
    return matSelisih @ eigVec.T


def recogniseUnknownFace(
    dataset_dir: str,
    test_path: str,
    datasetMean: np.ndarray,
    eigVec: np.ndarray,
    weightDataset: np.ndarray,
    threshold: float,
) -> tuple[str | None, float]:
    """
    Cari gambar di dataset yang paling mirip dengan test face:
      1) Preprocessing test image → imagetoVector → vec (D,)
      2) Hitung selisih: vec_selisih = vec - datasetMean.flatten()  (D,)
      3) Hitung weightTest: (D,) @ eigVecᵀ (D, N) = (N,)
      4) Hitung jarak Euclidean between weightTest dan tiap baris weightDataset (shape (N, N))
         → distances (N,)
      5) Cari idx_min, min_dist, max_dist
      6) Hitung confidence = ((max_dist – min_dist) / max_dist)×100%
      7) Jika confidence < threshold, return (None, 0.0), else return (matched_path, confidence)
    """
    try:
        test_vec = imagetoVector(test_path)  # (D,)
    except Exception as e:
        print(f"[ERROR recogniseUnknownFace] {e}")
        return None, 0.0

    vec_selisih = test_vec - datasetMean.flatten()  # → (D,)

    # Hitung weightTest: (D,) @ (D, N) = (N,)
    weight_test = vec_selisih @ eigVec.T  # → (N,)

    # Hitung jarak Euclidean ke tiap baris training (N, N) vs (N,) → (N,)
    distances = np.linalg.norm(weightDataset - weight_test, axis=1)

    if distances.size == 0:
        return None, 0.0

    idx_min = np.argmin(distances)
    min_dist = distances[idx_min]
    max_dist = np.max(distances)

    # Hitung confidence
    if max_dist == 0:
        confidence = 100.0
    else:
        confidence = (max_dist - min_dist) / max_dist * 100.0

    # Jika confidence di bawah threshold, anggap tidak ada match
    if confidence < threshold:
        return None, 0.0

    # Ambil nama file training pada index idx_min
    files = [
        f
        for f in os.listdir(dataset_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    if idx_min >= len(files):
        return None, 0.0

    matched_filename = files[idx_min]
    matched_path = f"{dataset_dir}/{matched_filename}".replace("\\", "/")

    return matched_path, confidence

