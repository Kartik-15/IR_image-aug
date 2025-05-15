
import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image
import tempfile

st.set_page_config(page_title="Image Augmentation Tool", layout="wide")
st.title("📸 Image Augmentation Tool")
st.markdown("Generate synthetic image data with different tints and brightness levels.")

# File uploader
uploaded_files = st.file_uploader("Upload images (jpg/jpeg/png)", accept_multiple_files=True, type=["jpg", "jpeg", "png"])

# Brightness options
brightness_factors = {
    "Normal": 1.2,
    "Bright": 1.4
}

# Tint options (in RGB)
tints = {
    "Warm": (255, 180, 100),
    "Cool": (100, 180, 255)
}

alpha = 0.25  # Tint blending strength

if uploaded_files:
    output_dir = tempfile.mkdtemp()
    st.success("Images uploaded! Generating augmented versions...")

    for uploaded_file in uploaded_files:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        base_name = os.path.splitext(uploaded_file.name)[0]

        for brightness_label, brightness in brightness_factors.items():
            bright_image = np.clip(image * brightness, 0, 255).astype(np.uint8)

            for tint_label, tint_color in tints.items():
                tint_layer = np.full_like(bright_image, tint_color, dtype=np.uint8)
                final_image = cv2.addWeighted(bright_image, 1 - alpha, tint_layer, alpha, 0)

                output_filename = f"{base_name}_{brightness_label}_{tint_label}.jpg"
                output_path = os.path.join(output_dir, output_filename)
                cv2.imwrite(output_path, cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))

    st.success("✅ Augmentation complete!")

    # Download all files
    with open(output_path, "rb") as f:
        btn = st.download_button(
            label="Download Last Image",
            data=f,
            file_name=output_filename,
            mime="image/jpeg"
        )

    st.info("For full batch download, use the ZIP version from your local directory.")
else:
    st.warning("Please upload at least one image to begin.")
