import streamlit as st
import cv2
import numpy as np
import os
import zipfile
import shutil
import tempfile
from io import BytesIO

# === Utility functions ===

def apply_tint(image, color, alpha=0.25):
    tint_layer = np.full_like(image, color, dtype=np.uint8)
    return cv2.addWeighted(image, 1 - alpha, tint_layer, alpha, 0)

def apply_shadow(image):
    shadow = np.zeros_like(image)
    h, w = image.shape[:2]
    top_x, top_y = np.random.randint(0, w // 2), np.random.randint(0, h // 2)
    bottom_x, bottom_y = np.random.randint(w // 2, w), np.random.randint(h // 2, h)
    cv2.rectangle(shadow, (top_x, top_y), (bottom_x, bottom_y), (0, 0, 0), -1)
    alpha = np.random.uniform(0.3, 0.7)
    return cv2.addWeighted(image, 1, shadow, alpha, 0)

def apply_reflection(image):
    flipped = cv2.flip(image, 0)
    mask = np.linspace(1, 0, image.shape[0])[:, None]
    reflection = (flipped * mask).astype(np.uint8)
    return np.vstack((image, reflection))

def apply_gaussian_blur(image):
    return cv2.GaussianBlur(image, (7, 7), 0)

def apply_random_occlusion(image):
    occluded = image.copy()
    h, w = occluded.shape[:2]
    for _ in range(np.random.randint(1, 4)):
        x1, y1 = np.random.randint(0, w - 30), np.random.randint(0, h - 30)
        x2, y2 = x1 + np.random.randint(20, 100), y1 + np.random.randint(20, 100)
        cv2.rectangle(occluded, (x1, y1), (x2, y2), (0, 0, 0), -1)
    return occluded

def apply_perspective_transform(image):
    h, w = image.shape[:2]
    pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    shift = np.random.randint(-30, 30, size=(4, 2)).astype(np.float32)
    pts2 = pts1 + shift
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(image, matrix, (w, h))

def augment_image(image, base_name, output_dir, tints, brightness_factors):
    for brightness_label, brightness in brightness_factors.items():
        bright_img = np.clip(image * brightness, 0, 255).astype(np.uint8)
        for tint_label, tint in tints.items():
            tinted = apply_tint(bright_img, tint)
            variants = {
                "shadow": apply_shadow(tinted),
                "reflection": apply_reflection(tinted),
                "blur": apply_gaussian_blur(tinted),
                "occlusion": apply_random_occlusion(tinted),
                "perspective": apply_perspective_transform(tinted),
            }
            for aug_label, aug_image in variants.items():
                filename = f"{base_name}_{brightness_label}_{tint_label}_{aug_label}.jpg"
                save_path = os.path.join(output_dir, filename)
                cv2.imwrite(save_path, cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))

# === Streamlit UI ===

st.title("🔁 Image Augmentation Tool")

uploaded_files = st.file_uploader("Upload images or a ZIP file", accept_multiple_files=True)

if uploaded_files:
    with tempfile.TemporaryDirectory() as input_dir, tempfile.TemporaryDirectory() as output_dir:
        # Handle ZIP files or images
        for file in uploaded_files:
            if file.name.endswith(".zip"):
                with zipfile.ZipFile(file, 'r') as zip_ref:
                    zip_ref.extractall(input_dir)
            else:
                with open(os.path.join(input_dir, file.name), "wb") as f:
                    f.write(file.read())

        tints = {
            "warm": (0, 30, 80),
            "cool": (80, 30, 0)
        }
        brightness_factors = {
            "normal": 1.2,
            "bright": 1.4
        }

        st.info("Processing images...")
        for filename in os.listdir(input_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                path = os.path.join(input_dir, filename)
                img = cv2.imread(path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    base_name = os.path.splitext(filename)[0]
                    augment_image(img, base_name, output_dir, tints, brightness_factors)

        # Create ZIP of output
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zipf:
            for root, _, files in os.walk(output_dir):
                for file in files:
                    full_path = os.path.join(root, file)
                    arcname = os.path.relpath(full_path, output_dir)
                    zipf.write(full_path, arcname)
        zip_buffer.seek(0)

        st.success("✅ Augmentation Complete!")
        st.download_button("Download Augmented Images (.zip)", zip_buffer, file_name="augmented_images.zip")
