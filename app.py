import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image
import tempfile
import zipfile
from io import BytesIO

# === Augmentation Functions ===

def apply_overlay(base_img, overlay_img, alpha=0.3):
    overlay_resized = cv2.resize(overlay_img, (base_img.shape[1], base_img.shape[0]))

    # Ensure both images have 3 channels (RGB)
    if base_img.shape[2] != overlay_resized.shape[2]:
        if overlay_resized.shape[2] == 4:
            overlay_resized = cv2.cvtColor(overlay_resized, cv2.COLOR_BGRA2BGR)
        elif overlay_resized.shape[2] == 1:
            overlay_resized = cv2.cvtColor(overlay_resized, cv2.COLOR_GRAY2BGR)

        if base_img.shape[2] == 4:
            base_img = cv2.cvtColor(base_img, cv2.COLOR_BGRA2BGR)
        elif base_img.shape[2] == 1:
            base_img = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)

    return cv2.addWeighted(base_img, 1, overlay_resized, alpha, 0)

def apply_shadow(base_img):
    shadow = np.zeros_like(base_img)
    h, w = shadow.shape[:2]
    cv2.rectangle(shadow, (int(w * 0.3), 0), (w, h), (50, 50, 50), -1)
    return cv2.addWeighted(base_img, 1, shadow, 0.4, 0)

def apply_glass_reflection(base_img, reflection_img):
    return apply_overlay(base_img, reflection_img, alpha=0.4)

def apply_gaussian_blur(base_img):
    return cv2.GaussianBlur(base_img, (15, 15), 0)

def apply_random_occlusion(base_img):
    output = base_img.copy()
    h, w = output.shape[:2]
    for _ in range(3):
        x1, y1 = np.random.randint(0, w//2), np.random.randint(0, h//2)
        x2, y2 = x1 + np.random.randint(20, w//3), y1 + np.random.randint(20, h//3)
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 0, 0), -1)
    return output

def apply_perspective_transform(base_img):
    rows, cols = base_img.shape[:2]
    pts1 = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1], [cols - 1, rows - 1]])
    pts2 = np.float32([[10, 10], [cols - 30, 20], [30, rows - 20], [cols - 10, rows - 30]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(base_img, M, (cols, rows))

# === UI Setup ===

st.title("🧪 Custom Image Augmentation Tool")

input_folder = st.sidebar.file_uploader("Upload images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
overlay_file = st.sidebar.file_uploader("Optional: Upload glass reflection image (PNG/JPG)", type=["png", "jpg", "jpeg"])

augmentations = st.sidebar.multiselect("Select Augmentations", ["Shadow", "Reflection", "Blur", "Occlusion", "Perspective"])

if st.sidebar.button("Process Images") and input_folder:
    with st.spinner("Processing..."):

        temp_dir = tempfile.mkdtemp()
        result_dir = os.path.join(temp_dir, "results")
        os.makedirs(result_dir, exist_ok=True)

        overlay_img = None
        if overlay_file:
            overlay_img = Image.open(overlay_file).convert("RGB")
            overlay_img = np.array(overlay_img)

        for uploaded_file in input_folder:
            image = Image.open(uploaded_file).convert("RGB")
            img_np = np.array(image)

            for aug in augmentations:
                if aug == "Shadow":
                    aug_img = apply_shadow(img_np)
                elif aug == "Reflection":
                    if overlay_img is not None:
                        aug_img = apply_glass_reflection(img_np, overlay_img)
                    else:
                        st.warning("Reflection selected but no overlay image uploaded.")
                        continue
                elif aug == "Blur":
                    aug_img = apply_gaussian_blur(img_np)
                elif aug == "Occlusion":
                    aug_img = apply_random_occlusion(img_np)
                elif aug == "Perspective":
                    aug_img = apply_perspective_transform(img_np)
                else:
                    continue

                save_path = os.path.join(
                    result_dir,
                    f"{os.path.splitext(uploaded_file.name)[0]}_{aug.lower()}.jpg"
                )
                cv2.imwrite(save_path, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))

        # Create zip
        zip_buf = BytesIO()
        with zipfile.ZipFile(zip_buf, "w") as zipf:
            for root, _, files in os.walk(result_dir):
                for file in files:
                    zipf.write(os.path.join(root, file), arcname=file)

        st.success("Images processed successfully!")
        st.download_button("Download ZIP", data=zip_buf.getvalue(), file_name="augmented_images.zip", mime="application/zip")
