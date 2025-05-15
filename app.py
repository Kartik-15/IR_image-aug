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

def apply_glass_reflection(image):
    overlay = np.zeros_like(image, dtype=np.uint8)
    h, w, _ = image.shape

    for i in range(0, w, w // 20):
        intensity = np.random.randint(50, 100)
        thickness = np.random.randint(1, 3)
        cv2.line(overlay, (i, 0), (i - h // 2, h), (intensity, intensity, intensity), thickness)

    alpha = 0.15
    return cv2.addWeighted(image, 1.0, overlay, alpha, 0)

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

def augment_image(image, base_name, output_dir, tints, brightness_factors, selected_augmentations):
    for brightness_label, brightness in brightness_factors.items():
        bright_img = np.clip(image * brightness, 0, 255).astype(np.uint8)
        for tint_label, tint in tints.items():
            tinted = apply_tint(bright_img, tint)
            if "Shadow" in selected_augmentations:
                save_augmented(tinted, apply_shadow, base_name, brightness_label, tint_label, "shadow", output_dir)
            if "Reflection" in selected_augmentations:
                save_augmented(tinted, apply_glass_reflection, base_name, brightness_label, tint_label, "reflection", output_dir)
            if "Blur" in selected_augmentations:
                save_augmented(tinted, apply_gaussian_blur, base_name, brightness_label, tint_label, "blur", output_dir)
            if "Occlusion" in selected_augmentations:
                save_augmented(tinted, apply_random_occlusion, base_name, brightness_label, tint_label, "occlusion", output_dir)
            if "Perspective" in selected_augmentations:
                save_augmented(tinted, apply_perspective_transform, base_name, brightness_label, tint_label, "perspective", output_dir)

def save_augmented(image, func, base_name, brightness_label, tint_label, aug_label, output_dir):
    try:
        aug_img = func(image)
        filename = f"{base_name}_{brightness_label}_{tint_label}_{aug_label}.jpg"
        save_path = os.path.join(output_dir, filename)
        cv2.imwrite(save_path, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
    except Exception as e:
        print(f"Skipping {aug_label} for {base_name} due to error: {e}")

# === Streamlit UI ===

st.title("🧪 Custom Image Augmentation Tool")

uploaded_files = st.file_uploader("Upload images or a ZIP file", accept_multiple_files=True)

st.sidebar.header("Augmentation Settings")

selected_augmentations = st.sidebar.multiselect(
    "Choose Augmentations to Apply:",
    ["Shadow", "Reflection", "Blur", "Occlusion", "Perspective"],
    default=["Shadow", "Reflection"]
)

brightness_option = st.sidebar.multiselect("Brightness Level", ["normal", "bright"])
brightness_factors = {
    "dark": 0.8,
    "normal": 1.2,
    "bright": 1.4,
    default=["normal"]
}

tint_option = st.sidebar.multiselect("Tint", ["warm", "cool"])
tints = {
    "warm": (0, 30, 80),
    "cool": (80, 30, 0),
    'cool_white': (220, 255, 255),     # cool white
    'warm_white': (255, 240, 200),     # warm aisle
    'fluorescent_green': (220, 255, 220),
    'bluish_white': (200, 220, 255),
    'soft_pink': (255, 220, 230),
    'daylight': (255, 255, 240),
    default=["warm_white"]
}

if uploaded_files:
    with tempfile.TemporaryDirectory() as input_dir, tempfile.TemporaryDirectory() as output_dir:
        for file in uploaded_files:
            if file.name.endswith(".zip"):
                with zipfile.ZipFile(file, 'r') as zip_ref:
                    zip_ref.extractall(input_dir)
            else:
                with open(os.path.join(input_dir, file.name), "wb") as f:
                    f.write(file.read())

        st.info("Processing images with selected augmentations...")

        for filename in os.listdir(input_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                path = os.path.join(input_dir, filename)
                img = cv2.imread(path)
                if img is not None:
                    try:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        base_name = os.path.splitext(filename)[0]
                        augment_image(
                            img,
                            base_name,
                            output_dir,
                            {tint_option: tints[tint_option]},
                            {brightness_option: brightness_factors[brightness_option]},
                            selected_augmentations
                        )
                    except Exception as e:
                        st.warning(f"Skipping {filename} due to error: {e}")

        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zipf:
            for root, _, files in os.walk(output_dir):
                for file in files:
                    full_path = os.path.join(root, file)
                    arcname = os.path.relpath(full_path, output_dir)
                    zipf.write(full_path, arcname)
        zip_buffer.seek(0)

        st.success("🎉 Augmentation complete!")
        st.download_button("Download Augmented Images (.zip)", zip_buffer, file_name="augmented_images.zip")
