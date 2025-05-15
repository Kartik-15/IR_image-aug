
import streamlit as st
import cv2
import numpy as np
import io
import zipfile
import os
from PIL import Image

st.set_page_config(page_title="Image Augmentation Tool", layout="wide")
st.title("📸 Image Augmentation Tool")
st.caption("Generate synthetic image data with different tints and brightness levels.")

uploaded_files = st.file_uploader("Upload images (jpg/jpeg/png)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

def apply_augmentations(image):
    augmented_images = []
    brightness_factors = [0.6, 0.8, 1.2, 1.4]
    tint_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

    for factor in brightness_factors:
        bright_img = cv2.convertScaleAbs(image, alpha=factor, beta=0)
        augmented_images.append((bright_img, f"brightness_{factor}"))

    for tint in tint_colors:
        tint_img = cv2.addWeighted(image, 0.7, np.full_like(image, tint), 0.3, 0)
        augmented_images.append((tint_img, f"tint_{tint[0]}_{tint[1]}_{tint[2]}"))

    return augmented_images

def image_to_bytes(img_array):
    success, buffer = cv2.imencode('.jpg', img_array)
    if success:
        return buffer.tobytes()
    return None

if uploaded_files:
    st.success("Images uploaded! Generating augmented versions...")

    all_augmented_images_bytes = []
    all_augmented_filenames = []

    for file in uploaded_files:
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        augmented_images = apply_augmentations(image)

        for idx, (aug_img, label) in enumerate(augmented_images):
            img_bytes = image_to_bytes(aug_img)
            if img_bytes:
                all_augmented_images_bytes.append(img_bytes)
                base_filename = os.path.splitext(file.name)[0]
                all_augmented_filenames.append(f"{base_filename}_{label}.jpg")

    st.success("✅ Augmentation complete!")

    if all_augmented_images_bytes:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zip_file:
            for filename, filedata in zip(all_augmented_filenames, all_augmented_images_bytes):
                zip_file.writestr(filename, filedata)

        zip_buffer.seek(0)

        st.download_button(
            label="📦 Download All Augmented Images (ZIP)",
            data=zip_buffer,
            file_name="augmented_images.zip",
            mime="application/zip"
        )
