import streamlit as st
from PIL import Image, ImageEnhance
import os
import random
import zipfile
import shutil
from io import BytesIO

# Define directories
SAMPLE_DIR = "Sample"
OVERLAY_DIR = "overlays"
OUTPUT_DIR = "output"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Function to apply overlays with opacity
def apply_overlay(base_img, overlay_img, opacity):
    overlay = overlay_img.convert("RGBA")
    base = base_img.convert("RGBA")

    # Resize overlay to match base image
    overlay = overlay.resize(base.size)

    # Adjust overlay opacity
    alpha = overlay.split()[3]
    alpha = alpha.point(lambda p: int(p * opacity))
    overlay.putalpha(alpha)

    return Image.alpha_composite(base, overlay)

# Function to apply brightness
def apply_brightness(img, brightness_factor):
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(brightness_factor)

# Function to apply tint
def apply_tint(img, tint_color, opacity):
    tint_layer = Image.new("RGBA", img.size, tint_color + (0,))
    alpha = int(255 * opacity)
    tint_layer.putalpha(alpha)
    return Image.alpha_composite(img.convert("RGBA"), tint_layer)

# Function to apply selected transformations
def augment_image(img, selected_overlays, overlay_opacity, brightness, tint_color, tint_opacity):
    augmented = img.convert("RGBA")
    for overlay_path in selected_overlays:
        overlay = Image.open(overlay_path).convert("RGBA")
        augmented = apply_overlay(augmented, overlay, overlay_opacity)

    if brightness != 1.0:
        augmented = apply_brightness(augmented, brightness)

    if tint_opacity > 0.0:
        augmented = apply_tint(augmented, tint_color, tint_opacity)

    return augmented.convert("RGB")

# Function to get overlay file paths
def get_overlay_files():
    files = []
    for file in os.listdir(OVERLAY_DIR):
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            files.append(os.path.join(OVERLAY_DIR, file))
    return files

# Function to get sample image file paths
def get_sample_images():
    files = []
    for file in os.listdir(SAMPLE_DIR):
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            files.append(os.path.join(SAMPLE_DIR, file))
    return files

# Sidebar UI settings
st.sidebar.title("Augmentation Settings")

# Upload section (merged)
uploaded_files = st.sidebar.file_uploader("Upload images (individual or ZIP)", type=["png", "jpg", "jpeg", "zip"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        if uploaded_file.name.endswith(".zip"):
            with zipfile.ZipFile(uploaded_file, "r") as zip_ref:
                zip_ref.extractall(SAMPLE_DIR)
        else:
            img = Image.open(uploaded_file)
            img.save(os.path.join(SAMPLE_DIR, uploaded_file.name))

# Overlay selection
st.sidebar.subheader("Select Overlays")
overlay_files = get_overlay_files()
selected_overlay_files = st.sidebar.multiselect("Choose overlays to apply", overlay_files, default=overlay_files)

# Opacity
overlay_opacity = st.sidebar.slider("Overlay Opacity", 0.0, 1.0, 0.5, 0.05)

# Brightness
brightness = st.sidebar.slider("Brightness", 0.0, 2.0, 1.0, 0.1)

# Tint options
tint_color = st.sidebar.color_picker("Tint Color", "#ffffff")
tint_color_rgb = tuple(int(tint_color[i:i+2], 16) for i in (1, 3, 5))
tint_opacity = st.sidebar.slider("Tint Opacity", 0.0, 1.0, 0.0, 0.05)

# Display thumbnails in column
st.title("Custom Image Augmentation Tool")
sample_images = get_sample_images()

if sample_images:
    selected_sample = st.session_state.get("selected_sample", sample_images[0])

    col1, col2 = st.columns([1, 3])
    with col1:
        st.subheader("Sample Images")
        for img_path in sample_images:
            img = Image.open(img_path)
            if st.button("", key=img_path):
                st.session_state.selected_sample = img_path
            st.image(img.resize((150, 150)), use_container_width=False, caption=os.path.basename(img_path),
                     channels="RGB", output_format="JPEG")

    with col2:
        selected_sample = st.session_state.get("selected_sample", sample_images[0])
        try:
            img = Image.open(selected_sample)
            augmented_img = augment_image(img, selected_overlay_files, overlay_opacity, brightness, tint_color_rgb, tint_opacity)
            st.image(augmented_img, caption="Augmented Image Preview", use_container_width=True)

            # Save button
            save_path = os.path.join(OUTPUT_DIR, f"augmented_{os.path.basename(selected_sample)}")
            augmented_img.save(save_path)
            with open(save_path, "rb") as f:
                img_bytes = f.read()
            st.download_button(label="Download Augmented Image", data=img_bytes, file_name=os.path.basename(save_path), mime="image/jpeg")
        except Exception as e:
            st.error(f"Error augmenting image: {e}")
else:
    st.info("Please upload sample images to begin.")
