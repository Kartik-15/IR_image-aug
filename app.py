import streamlit as st
import cv2
import numpy as np
import os
import zipfile
import shutil
import tempfile
from io import BytesIO

# === Utility Functions ===

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
    return cv2.addWeighted(image, 1.0, overlay, 0.15, 0)

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

def save_augmented(image, func, base_name, suffix, output_dir):
    try:
        aug_img = func(image)
        filename = f"{base_name}_{suffix}.jpg"
        save_path = os.path.join(output_dir, filename)
        cv2.imwrite(save_path, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
    except Exception as e:
        print(f"Skipping {suffix} for {base_name} due to error: {e}")

# === Streamlit UI ===

st.title("🧪 Custom Image Augmentation Tool")

uploaded_files = st.file_uploader("Upload images or a ZIP file", accept_multiple_files=True)

st.sidebar.header("Augmentation Settings")

augmentations = st.sidebar.multiselect(
    "Choose Augmentations",
    ["Shadow", "Reflection", "Blur", "Occlusion", "Perspective"]
)

brightness_options = st.sidebar.multiselect(
    "Brightness Levels",
    ["dark", "normal", "bright"],
    default=[]
)

tint_options = st.sidebar.multiselect(
    "Tints",
    [
        "warm", "cool", "cool_white", "warm_white",
        "fluorescent_green", "bluish_white", "soft_pink", "daylight"
    ],
    default=[]
)

brightness_factors = {
    "dark": 0.8,
    "normal": 1.2,
    "bright": 1.4
}

tint_colors = {
    "warm": (0, 30, 80),
    "cool": (80, 30, 0),
    "cool_white": (220, 255, 255),
    "warm_white": (255, 240, 200),
    "fluorescent_green": (220, 255, 220),
    "bluish_white": (200, 220, 255),
    "soft_pink": (255, 220, 230),
    "daylight": (255, 255, 240)
}

if uploaded_files:
    if not (augmentations or brightness_options or tint_options):
        st.error("⚠️ Please select at least one option from Brightness, Tint, or Augmentations.")
    else:
        with tempfile.TemporaryDirectory() as input_dir, tempfile.TemporaryDirectory() as output_dir:
            for file in uploaded_files:
                if file.name.endswith(".zip"):
                    with zipfile.ZipFile(file, 'r') as zip_ref:
                        zip_ref.extractall(input_dir)
                else:
                    with open(os.path.join(input_dir, file.name), "wb") as f:
                        f.write(file.read())

            st.info("🔄 Processing images...")

            for filename in os.listdir(input_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    path = os.path.join(input_dir, filename)
                    img = cv2.imread(path)
                    if img is None:
                        continue

                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    base_name = os.path.splitext(filename)[0]

                    # Handle all combinations
                    bright_levels = brightness_options if brightness_options else ["original"]
                    tint_levels = tint_options if tint_options else ["original"]

                    for b in bright_levels:
                        bright_img = img if b == "original" else np.clip(img * brightness_factors[b], 0, 255).astype(np.uint8)
                        for t in tint_levels:
                            final_img = bright_img if t == "original" else apply_tint(bright_img, tint_colors[t])

                            suffix_base = ""
                            if b != "original":
                                suffix_base += b
                            if t != "original":
                                suffix_base += f"_{t}"

                            if augmentations:
                                for aug in augmentations:
                                    suffix = f"{suffix_base}_{aug}" if suffix_base else aug
                                    func = {
                                        "Shadow": apply_shadow,
                                        "Reflection": apply_glass_reflection,
                                        "Blur": apply_gaussian_blur,
                                        "Occlusion": apply_random_occlusion,
                                        "Perspective": apply_perspective_transform
                                    }.get(aug)
                                    save_augmented(final_img, func, base_name, suffix, output_dir)
                            else:
                                suffix = suffix_base if suffix_base else "original"
                                save_path = os.path.join(output_dir, f"{base_name}_{suffix}.jpg")
                                cv2.imwrite(save_path, cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR))

            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, "w") as zipf:
                for root, _, files in os.walk(output_dir):
                    for file in files:
                        full_path = os.path.join(root, file)
                        arcname = os.path.relpath(full_path, output_dir)
                        zipf.write(full_path, arcname)
            zip_buffer.seek(0)

            st.success("✅ Augmentation complete!")
            st.download_button("Download Augmented Images (.zip)", zip_buffer, file_name="augmented_images.zip")
