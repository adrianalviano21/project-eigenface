import streamlit as st
import cv2
import numpy as np
import time
from PIL import Image
import os
from eigenface2 import (
    vectortoMatrix,
    mean,
    selisih,
    covariance,
    eig,
    weight_dataset,
    imagetoVector,
    recogniseUnknownFace,
)

# === Custom Light Theme Styling ===
st.markdown("""
    <style>
    body {
        background-color: #f9f9f9;
    }
    .main {
        background-color: #ffffff;
        color: #333333;
        font-family: 'Segoe UI', sans-serif;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #2c3e50;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        height: 3em;
        width: 100%;
        border: none;
    }
    .stButton>button:hover {
        background-color: #2980b9;
    }
    .stSlider > div {
        color: #2c3e50;
    }
    .css-1aumxhk {
        background-color: #ffffff;
    }
    </style>
""", unsafe_allow_html=True)

# === Page Title ===
st.title("🧠 Face Recognition App")
st.markdown("<hr>", unsafe_allow_html=True)

# === Layout Columns ===
col1, col2, col3 = st.columns([1.3, 1.5, 1.5])

# === LEFT COLUMN ===
with col1:
    st.subheader("📂 Dataset Folder Path")
    dataset_path = st.text_input("Masukkan path folder dataset:", key="dataset_path")

    st.subheader("🖼️ Upload Gambar Test")
    test_image = st.file_uploader("Masukkan gambar wajah (jpg/jpeg/png)", type=["jpg", "jpeg", "png"], key="test_image")

    st.subheader("🎚️ Threshold Kemiripan")
    threshold = st.slider("Threshold", 0.0, 100.0, 70.0, step=1.0)

    st.subheader("🚀 Eksekusi")
    run = st.button("Run")

    st.subheader("📊 Hasil")
    result_placeholder = st.empty()

    st.subheader("⏱️ Waktu Eksekusi")
    time_placeholder = st.empty()

# === RIGHT COLUMNS ===
with col2:
    st.markdown("**Test Image**")
    test_img_placeholder = st.empty()

with col3:
    st.markdown("**Matched Image**")
    matched_img_placeholder = st.empty()

# === Main Processing ===
if run and dataset_path and test_image:
    try:
        start_time = time.time()

        # Load dataset
        matImgVec = vectortoMatrix(dataset_path)

        # Proses eigenface
        meanVec = mean(matImgVec)
        matSelisih = selisih(meanVec, matImgVec)
        matCov = covariance(matSelisih)
        eigVec = eig(matCov, matSelisih)
        weightDataset = weight_dataset(matSelisih, eigVec)

        # Proses gambar test
        test_np = np.asarray(bytearray(test_image.read()), dtype=np.uint8)
        test_cv = cv2.imdecode(test_np, cv2.IMREAD_COLOR)
        temp_test_path = "temp_test.jpg"
        cv2.imwrite(temp_test_path, test_cv)
        test_img_placeholder.image(temp_test_path, caption="Uploaded Test Image", channels="BGR")

        # Pengenalan wajah
        matched_path, confidence = recogniseUnknownFace(
            dataset_dir=dataset_path,
            test_path=temp_test_path,
            datasetMean=meanVec,
            eigVec=eigVec,
            weightDataset=weightDataset,
            threshold=threshold,
        )

        # Tampilkan hasil
        if matched_path:
            matched_img = Image.open(matched_path)
            matched_img_placeholder.image(matched_img, caption=f"Matched (Confidence: {confidence:.2f}%)")
            result_placeholder.success(f"Matched with confidence {confidence:.2f}%")
        else:
            result_placeholder.warning("No match found.")

        # Waktu eksekusi
        elapsed = time.time() - start_time
        time_placeholder.markdown(f"<span style='color:#2c3e50'>00:{elapsed:.2f}</span>", unsafe_allow_html=True)

    except Exception as e:
        result_placeholder.error(f"Error: {str(e)}")
