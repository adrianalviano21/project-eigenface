import eigenface2

def run(
    dataset_dir: str,
    sample_path: str,
    threshold: float = 50.0
) -> tuple[str | None, float]:
    """
    Pipeline utama untuk face recognition dengan Eigenface:
      1) Vektorkan semua gambar di dataset_dir
      2) Hitung mean & selisih
      3) Hitung matriks covariance
      4) Hitung eigenvectors (ruang asli)
      5) Hitung weightDataset
      6) Panggil recogniseUnknownFace untuk test image dengan threshold
    """
    # 1) Convert folder dataset → matrix (N, D)
    datasetMat = eigenface2.vectortoMatrix(dataset_dir)  # (N, 16384)

    # 2) Hitung mean (1, D)
    datasetMean = eigenface2.mean(datasetMat)  # (1, 16384)

    # 3) Normalisasi training data: (N, D)
    normDataset = eigenface2.selisih(datasetMean, datasetMat)  # (N, 16384)

    # 4) Covariance kecil: (N, N)
    covDataset = eigenface2.covariance(normDataset)

    # 5) Hitung eigenvectors di ruang asli: output shape (N, D)
    eigVec = eigenface2.eig(covDataset, normDataset)  # (N, 16384)

    # 6) Hitung weights untuk setiap training: weightDataset → (N, N)
    weightDataset = eigenface2.weight_dataset(normDataset, eigVec)  # (N, N)

    # 7) Panggil recogniseUnknownFace dengan nilai threshold yang diteruskan
    result_path, match_percentage = eigenface2.recogniseUnknownFace(
        dataset_dir,
        sample_path,
        datasetMean,
        eigVec,
        weightDataset,
        threshold,
    )

    return result_path, match_percentage
