import streamlit as st
import os
import zipfile
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import random
import shutil

# ───────────────────────────── SETUP ─────────────────────────────
SAMPLE_FOLDER = "Sample"
OVERLAY_FOLDER = "overlays"
OUTPUT_FOLDER = "output"
INPUT_FOLDER = "Input"

for folder in [SAMPLE_FOLDER, OVERLAY_FOLDER, OUTPUT_FOLDER, INPUT_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# ────────────────────────── HELPER FUNCTIONS ──────────────────────────
def apply_augmentation(image, overlay_images, blur=False, brightness=None, tint=None):
    if blur:
        image = image.filter(ImageFilter.GaussianBlur(radius=2))

    if brightness:
        enhancer = ImageEnhance.Brightness(image)
        factor = {"Dark": 0.6, "Normal": 1.0, "Bright": 1.4}.get(brightness, 1.0)
        image = enhancer.enhance(factor)

    if tint:
        tint_colors = {
            "Cool": (173, 216, 230),
            "Warm": (255, 228, 196),
            "Daylight": (255, 255, 240),
        }
        if tint in tint_colors:
            overlay = Image.new("RGB", image.size, tint_colors[tint])
            image = Image.blend(image, overlay, alpha=0.3)

    if overlay_images:
        selected_overlay_path = random.choice(overlay_images)
        overlay = Image.open(selected_overlay_path).resize(image.size).convert("RGBA")
        image = image.convert("RGBA")
        image = Image.alpha_composite(image, overlay).convert("RGB")

    return image

def save_augmented_images(image_path, overlay_images, augment_options, brightness_levels, tint_options):
    original = Image.open(image_path).convert("RGB")
    filename = os.path.splitext(os.path.basename(image_path))[0]
    count = 0

    for blur in augment_options:
        for bright in brightness_levels:
            for tint in tint_options:
                augmented = apply_augmentation(original.copy(), overlay_images, blur, bright, tint)
                output_path = os.path.join(OUTPUT_FOLDER, f"{filename}_b{blur}_l{bright}_t{tint}.jpg")
                augmented.save(output_path)
                count += 1
    return count

def list_images(folder):
    return [f for f in os.listdir(folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))]

# ────────────────────────────── SIDEBAR ──────────────────────────────
st.sidebar.header("Settings")

# Augmentation settings
augment_options = st.sidebar.multiselect("Add augmentation", ["Blur"], default=[])
apply_blur = "Blur" in augment_options

brightness_levels = st.sidebar.multiselect("Add Brightness", ["Dark", "Normal", "Bright"], default=["Normal"])
tint_options = st.sidebar.multiselect("Add Tints", ["Cool", "Warm", "Daylight"], default=[])

# Overlay uploader
st.sidebar.markdown("---")
st.sidebar.markdown("### Upload Overlay Images")
overlay_files = st.sidebar.file_uploader("Upload overlay images (png, jpg)", accept_multiple_files=True)
if overlay_files:
    for file in overlay_files:
        with open(os.path.join(OVERLAY_FOLDER, file.name), "wb") as f:
            f.write(file.read())

# ────────────────────────────── MAIN PAGE ──────────────────────────────
st.title("Custom Image Augmentation Tool")

# Upload section
st.markdown("### Upload Images")
uploaded = st.file_uploader("Upload image or zip", type=["jpg", "jpeg", "png", "zip"], accept_multiple_files=True)
if uploaded:
    for item in uploaded:
        if item.name.endswith(".zip"):
            with zipfile.ZipFile(item, 'r') as zip_ref:
                zip_ref.extractall(INPUT_FOLDER)
        else:
            with open(os.path.join(INPUT_FOLDER, item.name), "wb") as f:
                f.write(item.read())

# Display sample images for preview
sample_images = list_images(SAMPLE_FOLDER)
overlay_images = [os.path.join(OVERLAY_FOLDER, f) for f in list_images(OVERLAY_FOLDER)]
input_images = [os.path.join(INPUT_FOLDER, f) for f in list_images(INPUT_FOLDER)]

if input_images:
    selected_img_name = st.selectbox("Choose an image to preview from uploaded", [os.path.basename(path) for path in input_images])
    selected_img_path = os.path.join(INPUT_FOLDER, selected_img_name)

    st.markdown("---")
    st.markdown("### Sample Images")

    col1, col2 = st.columns([1, 3])
    with col1:
        for img_file in sample_images:
            img_path = os.path.join(SAMPLE_FOLDER, img_file)
            thumbnail = Image.open(img_path).resize((100, 100))
            if st.button(img_file, key=img_file):
                selected_img_path = img_path
            st.image(thumbnail, caption=img_file, use_column_width=True)

    with col2:
        st.markdown("### Augmented Preview")
        preview_img = apply_augmentation(Image.open(selected_img_path), overlay_images, apply_blur, brightness_levels[0] if brightness_levels else None, tint_options[0] if tint_options else None)
        st.image(preview_img, caption="Augmented Image", use_column_width=True)

    # Process and save
    if st.button("Run Augmentation"):
        total = 0
        for img_path in input_images:
            total += save_augmented_images(img_path, overlay_images, [apply_blur], brightness_levels, tint_options)
        st.success(f"Saved {total} augmented images in '{OUTPUT_FOLDER}' folder.")
else:
    st.info("Please upload images to begin.")
