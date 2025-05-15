
import streamlit as st
import cv2
import numpy as np
import io
import zipfile
import os
from PIL import Image

st.set_page_config(page_title="Image Augmentation Tool", layout="wide")
st.title("📸 Image Augmentation Tool")
st.caption("Generate synthetic image data with specific tints and brightness levels.")

uploaded_files = st.file_uploader("Upload images (jpg/jpeg/png)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# Define tints (BGR format)
tints = {
    "warm": (0, 30, 80),
    "cool": (80, 30, 0),
}

# Define brightness levels
brightness_factors = {
    "normal": 1.2,
    "bright": 1.4
}

alpha = 0.25  # Tint blending strength

def apply_transformations(image, base_name):
    augmented_images = []

    for brightness_label, brightness in brightness_factors.items():
        # Apply brightness
        bright_image = np.clip(image * brightness, 0, 255).astype(np.uint8)

        for tint_label, tint_color in tints.items():
            # Apply tint using blending
            tint_layer = np.full_like(bright_image, tint_color, dtype=np.uint8)
            final_image = cv2.addWeighted(bright_image, 1 - alpha, tint_layer, alpha, 0)

            output_filename = f"{base_name}_{brightness_label}_{tint_label}.jpg"
            _, buffer = cv2.imencode('.jpg', cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))
            augmented_images.append((output_filename, buffer.tobytes()))

    return augmented_images

if uploaded_files:
    st.success("Images uploaded! Generating augmented versions...")

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zip_file:
        for file in uploaded_files:
            file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            base_name = os.path.splitext(file.name)[0]
            augmented = apply_transformations(image, base_name)
            for filename, file_bytes in augmented:
                zip_file.writestr(filename, file_bytes)

    zip_buffer.seek(0)
    st.success("✅ Augmentation complete!")

    st.download_button(
        label="📦 Download All Augmented Images (ZIP)",
        data=zip_buffer,
        file_name="augmented_images.zip",
        mime="application/zip"
    )
