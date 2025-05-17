import streamlit as st
import cv2
import numpy as np
import os
import zipfile
import shutil
import tempfile
from io import BytesIO
import glob

# === Helper Functions ===

def apply_overlay(base_img, overlay_img, alpha=0.3):
    overlay_resized = cv2.resize(overlay_img, (base_img.shape[1], base_img.shape[0]))
    return cv2.addWeighted(base_img, 1, overlay_resized, alpha, 0)

def apply_tint(image, color):
    tint_layer = np.full_like(image, color, dtype=np.uint8)
    return cv2.addWeighted(image, 0.7, tint_layer, 0.3, 0)

# Dummy augmentation functions
def apply_shadow(image): return image
def apply_glass_reflection(image): return image
def apply_gaussian_blur(image): return cv2.GaussianBlur(image, (15, 15), 0)
def apply_random_occlusion(image): return image
def apply_perspective_transform(image): return image

def save_augmented(image, func, base_name, suffix, output_dir):
    aug_img = func(image)
    filename = f"{base_name}_{suffix}.jpg"
    save_path = os.path.join(output_dir, filename)
    cv2.imwrite(save_path, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))

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

overlay_files = st.sidebar.file_uploader("Upload Overlay Images", accept_multiple_files=True, type=["png", "jpg", "jpeg"])

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

# Initialize overlay_images list
overlay_images = []

# Built-in Overlays Section
OVERLAY_DIR = "overlays"  # folder in your repo
overlay_paths = sorted(glob.glob(os.path.join(OVERLAY_DIR, "*.*")))

if overlay_paths:
    st.sidebar.subheader("📸 Built-in Glass Overlays")
    overlay_labels = [os.path.splitext(os.path.basename(p))[0] for p in overlay_paths]

    selections = []
    for label, path in zip(overlay_labels, overlay_paths):
        col1, col2 = st.sidebar.columns([1,4])
        with col1:
            st.image(path, use_container_width=True)
        with col2:
            if st.checkbox(label, key=f"ov_{label}"):
                selections.append((label, path))

    for label, path in selections:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            overlay_images.append((label, img))

# ========== PROCESSING ==========

if uploaded_files:
    if not (augmentations or brightness_options or tint_options or overlay_files or overlay_images):
        st.error("⚠️ Please select at least one option from Brightness, Tint, Augmentations or Overlay.")
    else:
        with tempfile.TemporaryDirectory() as input_dir, tempfile.TemporaryDirectory() as output_dir:
            for file in uploaded_files:
                if file.name.endswith(".zip"):
                    with zipfile.ZipFile(file, 'r') as zip_ref:
                        zip_ref.extractall(input_dir)
                else:
                    with open(os.path.join(input_dir, file.name), "wb") as f:
                        f.write(file.read())

            # Add user-uploaded overlays
            for overlay_file in overlay_files:
                file_bytes = np.asarray(bytearray(overlay_file.read()), dtype=np.uint8)
                overlay_img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
                if overlay_img is not None:
                    overlay_images.append((os.path.splitext(overlay_file.name)[0], overlay_img))

            st.info("🔄 Processing images...")

            for filename in os.listdir(input_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    path = os.path.join(input_dir, filename)
                    img = cv2.imread(path)
                    if img is None:
                        continue

                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    base_name = os.path.splitext(filename)[0]

                    bright_levels = brightness_options if brightness_options else ["original"]
                    tint_levels = tint_options if tint_options else ["original"]
                    overlay_levels = overlay_images if overlay_images else [("original", None)]

                    for b in bright_levels:
                        bright_img = img if b == "original" else np.clip(img * brightness_factors[b], 0, 255).astype(np.uint8)
                        for t in tint_levels:
                            tinted_img = bright_img if t == "original" else apply_tint(bright_img, tint_colors[t])
                            for overlay_name, overlay_img in overlay_levels:
                                final_img = (
                                    tinted_img
                                    if overlay_img is None
                                    else apply_overlay(tinted_img, overlay_img)
                                )

                                suffix_base = ""
                                if b != "original": suffix_base += b
                                if t != "original": suffix_base += f"_{t}"
                                if overlay_img is not None: suffix_base += f"_overlay_{overlay_name}"

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

            # Create zip
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
