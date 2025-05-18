import streamlit as st
from PIL import Image, ImageEnhance, ImageOps
import os
import io

# Set page config
st.set_page_config(page_title="Image Augmentation Tool", layout="wide")

# Directory containing overlay images
overlay_dir = "overlays"
sample_image_dir = "sample_images"

# Load overlays
def load_overlay_images():
    overlays = {}
    for file in os.listdir(overlay_dir):
        if file.endswith(".png"):
            name = os.path.splitext(file)[0]
            overlays[name] = Image.open(os.path.join(overlay_dir, file)).convert("RGBA")
    return overlays

# Apply overlay
def apply_overlay(base_image, overlay_image, opacity):
    overlay = overlay_image.resize(base_image.size)
    overlay = Image.blend(Image.new("RGBA", base_image.size, (0,0,0,0)), overlay, opacity)
    return Image.alpha_composite(base_image.convert("RGBA"), overlay)

# Apply brightness and tint
def apply_augmentations(image, brightness=1.0, tint_color=None, tint_intensity=0.0):
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(brightness)
    if tint_color and tint_intensity > 0:
        tint_layer = Image.new("RGBA", image.size, tint_color)
        image = Image.blend(image, tint_layer, tint_intensity)
    return image

# Load sample images
def load_sample_images():
    samples = {}
    for file in os.listdir(sample_image_dir):
        if file.lower().endswith(('png', 'jpg', 'jpeg')):
            name = os.path.splitext(file)[0]
            samples[name] = os.path.join(sample_image_dir, file)
    return samples

# UI Elements
st.sidebar.header("Settings")

# Load overlays and sample images
overlay_options = load_overlay_images()
sample_images = load_sample_images()

# Controls
selected_overlay = st.sidebar.selectbox("Select Overlay", list(overlay_options.keys()))
overlay_opacity = st.sidebar.slider("Overlay Opacity", 0.0, 1.0, 0.5, 0.05)

brightness = st.sidebar.slider("Brightness", 0.5, 2.0, 1.0, 0.1)
tint_color = st.sidebar.color_picker("Tint Color", value="#000000")
tint_intensity = st.sidebar.slider("Tint Intensity", 0.0, 1.0, 0.0, 0.05)

# Layout using columns
sample_col, preview_col, upload_col = st.columns([1, 2, 1])

with sample_col:
    st.subheader("Sample Images")
    selected_sample = st.radio("", list(sample_images.keys()), index=0)
    sample_path = sample_images[selected_sample]
    sample_image = Image.open(sample_path).convert("RGBA")
    st.image(sample_image.resize((150, 150)), caption=selected_sample)

with preview_col:
    st.subheader("Preview")
    preview_image = apply_augmentations(sample_image.copy(), brightness, tint_color, tint_intensity)
    preview_image = apply_overlay(preview_image, overlay_options[selected_overlay], overlay_opacity)
    st.image(preview_image.resize((300, 300)), caption="Preview Image")

with preview_col:
    st.markdown("### Transformation Controls")
    st.markdown(f"**Brightness:** {brightness}")
    st.markdown(f"**Tint Color:** {tint_color}")
    st.markdown(f"**Tint Intensity:** {tint_intensity}")
    st.markdown(f"**Overlay:** {selected_overlay} @ {overlay_opacity*100:.0f}%")

with upload_col:
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        user_image = Image.open(uploaded_file).convert("RGBA")
        processed_user_image = apply_augmentations(user_image.copy(), brightness, tint_color, tint_intensity)
        processed_user_image = apply_overlay(processed_user_image, overlay_options[selected_overlay], overlay_opacity)

        st.image(user_image.resize((200, 200)), caption="Original Upload")
        st.image(processed_user_image.resize((200, 200)), caption="Augmented Upload")

        # Download link
        buf = io.BytesIO()
        processed_user_image.save(buf, format="PNG")
        byte_im = buf.getvalue()
        st.download_button("Download Augmented Image", data=byte_im, file_name="augmented_image.png", mime="image/png")
