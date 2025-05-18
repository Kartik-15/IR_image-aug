import streamlit as st
from PIL import Image, ImageEnhance
import os
import numpy as np
from io import BytesIO
import zipfile
from datetime import datetime

# ────────────────────────────── CONFIGURATION ──────────────────────────────

st.set_page_config(layout="wide")
st.title("🧪 Custom Image Augmentation Tool")

# Define required folders
SAMPLE_FOLDER = "Sample"
INPUT_FOLDER = "Input"
OVERLAY_FOLDER = "overlays"
OUTPUT_FOLDER = "Output"

# Create required folders if they don't exist
for folder in [SAMPLE_FOLDER, INPUT_FOLDER, OVERLAY_FOLDER, OUTPUT_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# ────────────────────────────── LOAD FILES ──────────────────────────────

sample_images = [f for f in os.listdir(SAMPLE_FOLDER) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
input_images = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
overlay_files = [f for f in os.listdir(OVERLAY_FOLDER) if f.lower().endswith('.png')]

overlay_imgs = {}
for f in overlay_files:
    try:
        overlay_imgs[f] = Image.open(os.path.join(OVERLAY_FOLDER, f)).convert("RGBA")
    except:
        pass

# ────────────────────────────── SIDEBAR: SETTINGS ──────────────────────────────

st.sidebar.header("🎛️ Augmentation Settings")

# Brightness
brightness_toggle = st.sidebar.checkbox("Brightness Adjustment", value=True)
brightness_opts = {}
if brightness_toggle:
    brightness_opts['min'] = st.sidebar.slider("Brightness Min", 0.5, 1.0, 0.8, 0.05)
    brightness_opts['max'] = st.sidebar.slider("Brightness Max", 1.0, 1.5, 1.2, 0.05)

# Tint
tint_toggle = st.sidebar.checkbox("Apply Tint", value=True)
tint_opts = {}
if tint_toggle:
    tint_opts['R'] = st.sidebar.slider("Red Channel", 0, 100, 0)
    tint_opts['G'] = st.sidebar.slider("Green Channel", 0, 100, 0)
    tint_opts['B'] = st.sidebar.slider("Blue Channel", 0, 100, 0)

# Overlay
overlay_toggle = st.sidebar.checkbox("Glass Overlay", value=True)
overlay_opts = {}
if overlay_toggle and overlay_imgs:
    selected_overlay = st.sidebar.selectbox("Select Overlay", list(overlay_imgs.keys()))
    overlay_opts['image'] = overlay_imgs[selected_overlay]
    overlay_opts['opacity'] = st.sidebar.slider("Overlay Opacity", 0.0, 1.0, 0.5, 0.05)

# ────────────────────────────── MAIN: FILE UPLOAD & PREVIEW ──────────────────────────────

st.subheader("📁 Upload Images")

uploaded_files = st.file_uploader("Upload individual image files", accept_multiple_files=True, type=["jpg", "jpeg", "png"])
if uploaded_files:
    for uploaded_file in uploaded_files:
        with open(os.path.join(INPUT_FOLDER, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.read())
    st.success("Uploaded individual images successfully!")

uploaded_zip = st.file_uploader("Or upload a .zip of images", type=["zip"])
if uploaded_zip:
    with zipfile.ZipFile(uploaded_zip, "r") as zip_ref:
        zip_ref.extractall(INPUT_FOLDER)
    st.success("Uploaded and extracted zip file successfully!")

# ────────────────────────────── MAIN: SAMPLE PREVIEW ──────────────────────────────

st.subheader("🔍 Preview on Sample Image")

if not sample_images:
    st.warning("No sample images found in the 'Sample' folder.")
else:
    selected_sample = st.selectbox("Choose a Sample Image", sample_images)
    sample_path = os.path.join(SAMPLE_FOLDER, selected_sample)
    sample_image = Image.open(sample_path).convert("RGB")

    preview_image = None
    try:
        preview_image = sample_image.copy()
        preview_image = apply_augmentations(
            preview_image,
            brightness_opts if brightness_toggle else None,
            tint_opts if tint_toggle else None,
            overlay_opts if overlay_toggle else None
        )
    except Exception as e:
        st.error(f"Error applying augmentations: {e}")

    col1, col2 = st.columns(2)
    with col1:
        st.image(sample_image, caption="Original Sample", use_container_width=True)
    with col2:
        if preview_image:
            st.image(preview_image, caption="Augmented Preview", use_container_width=True)

# ────────────────────────────── MAIN: OVERLAY IMAGE UPLOAD ──────────────────────────────

st.subheader("🌫️ Upload New Overlay Image")
overlay_upload = st.file_uploader("Upload a transparent PNG to 'overlays' folder", type=["png"], key="overlay_upload")
if overlay_upload:
    overlay_path = os.path.join(OVERLAY_FOLDER, overlay_upload.name)
    with open(overlay_path, "wb") as f:
        f.write(overlay_upload.read())
    st.success(f"Uploaded {overlay_upload.name} to overlays folder. Please refresh the page to use it.")

# ────────────────────────────── MAIN: PROCESS INPUT IMAGES ──────────────────────────────

st.subheader("📸 Process Input Images")

if not input_images:
    st.warning("No input images found in the 'Input' folder.")
else:
    if st.button("Run Augmentations"):
        with st.spinner("Processing images..."):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_output_dir = os.path.join(OUTPUT_FOLDER, f"aug_{timestamp}")
            os.makedirs(session_output_dir, exist_ok=True)

            for file in input_images:
                try:
                    path = os.path.join(INPUT_FOLDER, file)
                    img = Image.open(path).convert("RGB")
                    aug_img = apply_augmentations(
                        img,
                        brightness_opts if brightness_toggle else None,
                        tint_opts if tint_toggle else None,
                        overlay_opts if overlay_toggle else None
                    )
                    aug_img.save(os.path.join(session_output_dir, file))
                except Exception as e:
                    st.error(f"Error processing {file}: {e}")

            # Create zip
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, "w") as zipf:
                for f in os.listdir(session_output_dir):
                    zipf.write(os.path.join(session_output_dir, f), arcname=f)
            zip_buffer.seek(0)

            st.success("✅ Processing complete!")
            st.download_button("📦 Download Augmented Images", data=zip_buffer, file_name="augmented_images.zip", mime="application/zip")

# ────────────────────────────── FUNCTION: AUGMENTATION ──────────────────────────────

def apply_augmentations(img, brightness_opts=None, tint_opts=None, overlay_opts=None):
    img = img.convert("RGB")

    # Brightness
    if brightness_opts:
        enhancer = ImageEnhance.Brightness(img)
        factor = np.random.uniform(brightness_opts['min'], brightness_opts['max'])
        img = enhancer.enhance(factor)

    # Tint
    if tint_opts:
        r, g, b = tint_opts['R'], tint_opts['G'], tint_opts['B']
        tint_layer = Image.new("RGB", img.size, (r, g, b))
        img = Image.blend(img, tint_layer, alpha=0.2)

    # Overlay
    if overlay_opts:
        base = img.convert("RGBA")
        overlay = overlay_opts['image'].resize(img.size).convert("RGBA")
        opacity = overlay_opts['opacity']
        blended_overlay = Image.blend(Image.new("RGBA", img.size, (0, 0, 0, 0)), overlay, opacity)
        img = Image.alpha_composite(base, blended_overlay).convert("RGB")

    return img
