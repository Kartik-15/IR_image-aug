import streamlit as st
import os
from PIL import Image
from utils.transformations import apply_transformations, list_images_in_folder
from utils.overlay_loader import load_preloaded_overlays

st.set_page_config(layout="wide", page_title="Image Augmentation Tool")
st.title("🖼️ Synthetic Image Generator for Cooler Conditions")

# Sidebar Settings
with st.sidebar:
    st.header("⚙️ Settings")

    shadow_strength = st.slider("Shadow Strength", 0.0, 1.0, 0.5, 0.05)
    reflection_strength = st.slider("Reflection Strength", 0.0, 1.0, 0.5, 0.05)
    overlay_opacity = st.slider("Overlay Opacity", 0.0, 1.0, 0.5, 0.05)
    tint_strength = st.slider("Tint Strength", 0.0, 1.0, 0.3, 0.05)

    brightness_factor = st.slider("Brightness Factor", 0.5, 1.5, 1.0, 0.05)
    contrast_factor = st.slider("Contrast Factor", 0.5, 1.5, 1.0, 0.05)

    show_shadow = st.checkbox("Add Shadow", value=True)
    show_reflection = st.checkbox("Add Reflection", value=True)
    show_overlay = st.checkbox("Add Overlay", value=True)
    show_tint = st.checkbox("Add Tint", value=True)
    show_brightness = st.checkbox("Adjust Brightness", value=True)
    show_contrast = st.checkbox("Adjust Contrast", value=True)

    overlays_folder = "overlays"
    preloaded_overlays = load_preloaded_overlays(overlays_folder)
    uploaded_overlays = st.file_uploader("Upload Overlay Images", type=["png"], accept_multiple_files=True)

    overlay_images = uploaded_overlays if uploaded_overlays else preloaded_overlays

    uploaded_image = st.file_uploader("Upload Image to Preview Transformations", type=["png", "jpg", "jpeg"], key="preview_image")

# Sample Image Selector and Preview
sample_images_folder = "sample_images"
sample_images = list_images_in_folder(sample_images_folder)

if sample_images:
    st.subheader("🎯 Select a Sample Image")
    sample_cols = st.columns(len(sample_images))
    selected_sample = None
    for i, (sample_img, col) in enumerate(zip(sample_images, sample_cols)):
        with col:
            if st.button("Use This", key=f"sample_btn_{i}"):
                selected_sample = sample_img
            st.image(sample_img, width=100)
else:
    st.warning("No sample images found in folder.")

# Image Preview Section
st.markdown("---")
preview_container = st.container()

with preview_container:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.write("**Upload/Selected Image**")
        if uploaded_image:
            image = Image.open(uploaded_image).convert("RGBA")
            st.image(image, caption="Uploaded Image", width=250)
        elif selected_sample:
            image = Image.open(selected_sample).convert("RGBA")
            st.image(image, caption="Sample Image", width=250)
        else:
            image = None
            st.info("Upload or select an image to preview transformations.")

    with col2:
        st.write("**🔍 Preview with Transformations**")
        if image:
            preview_img = apply_transformations(
                image=image.copy(),
                overlays=overlay_images,
                shadow_strength=shadow_strength if show_shadow else 0.0,
                reflection_strength=reflection_strength if show_reflection else 0.0,
                overlay_opacity=overlay_opacity if show_overlay else 0.0,
                tint_strength=tint_strength if show_tint else 0.0,
                brightness_factor=brightness_factor if show_brightness else 1.0,
                contrast_factor=contrast_factor if show_contrast else 1.0,
                only_one_overlay=True  # for preview performance
            )
            st.image(preview_img, caption="Transformed Preview", width=400)

# Output Image Count Estimation
selected_transforms = [show_shadow, show_reflection, show_overlay, show_tint, show_brightness, show_contrast]
num_combinations = 1
for transform in selected_transforms:
    if transform:
        num_combinations *= 2  # each toggle doubles combinations

st.markdown(f"### 📦 Estimated Output Image Variants: `{num_combinations}`")

st.markdown("---")

# Preserve your existing processing and download code hereafter without touching it.
# Example placeholder:
# if st.button("Process Images"):
#     ... (your core logic untouched)
