import streamlit as st
import cv2, numpy as np, os, zipfile, tempfile, glob
from io import BytesIO

# === New Overlay Function ===
def apply_overlay(base_img, overlay_img, alpha=0.3):
    overlay_resized = cv2.resize(overlay_img, (base_img.shape[1], base_img.shape[0]))
    return cv2.addWeighted(base_img, 1, overlay_resized, alpha, 0)

# ---------------------------------------
# Streamlit UI
# ---------------------------------------
st.title("🧪 Custom Image Augmentation Tool")

uploaded_files = st.file_uploader("Upload images or a ZIP file", accept_multiple_files=True)
st.sidebar.header("Augmentation Settings")

augmentations = st.sidebar.multiselect("Choose Augmentations",
    ["Shadow", "Reflection", "Blur", "Occlusion", "Perspective"])

brightness_options = st.sidebar.multiselect("Brightness Levels",
    ["dark", "normal", "bright"], default=[])

tint_options = st.sidebar.multiselect("Tints",
    ["warm","cool","cool_white","warm_white","fluorescent_green",
     "bluish_white","soft_pink","daylight"], default=[])

overlay_files = st.sidebar.file_uploader("Upload Overlay Images",
    accept_multiple_files=True, type=["png","jpg","jpeg"])

# ------------------------------------------------------------------
# CONSTANTS / LOOK-UP TABLES
# ------------------------------------------------------------------
brightness_factors = {"dark":0.8,"normal":1.2,"bright":1.4}
tint_colors = {"warm":(0,30,80),"cool":(80,30,0),"cool_white":(220,255,255),
               "warm_white":(255,240,200),"fluorescent_green":(220,255,220),
               "bluish_white":(200,220,255),"soft_pink":(255,220,230),
               "daylight":(255,255,240)}

# ---------------------------------------
# SECTION: built-in overlay gallery
# ---------------------------------------
OVERLAY_DIR = "overlays"
overlay_paths  = sorted(glob.glob(os.path.join(OVERLAY_DIR, "*.*")))
overlay_labels = [os.path.splitext(os.path.basename(p))[0] for p in overlay_paths]

overlay_images = []   # ### FIX ### initialise before use

if overlay_paths:
    st.sidebar.subheader("📸 Built-in Glass Overlays")

    selections = []
    for label, path in zip(overlay_labels, overlay_paths):
        col1, col2 = st.sidebar.columns([1,4])
        with col1:
            st.image(path, use_container_width=True)   # updated param
        with col2:
            if st.checkbox(label, key=f"ov_{label}"):
                selections.append((label, path))

    # Load selected overlay files
    for label, path in selections:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            overlay_images.append((label, img))

# ------------------------------------------------------------------
# MAIN PROCESSING
# ------------------------------------------------------------------
if uploaded_files:
    if not (augmentations or brightness_options or tint_options or overlay_files or overlay_images):
        st.error("⚠️ Please select at least one option from Brightness, Tint, Augmentations or Overlay.")
    else:
        with tempfile.TemporaryDirectory() as input_dir, tempfile.TemporaryDirectory() as output_dir:

            # save uploads / unzip
            for file in uploaded_files:
                if file.name.endswith(".zip"):
                    with zipfile.ZipFile(file, 'r') as z: z.extractall(input_dir)
                else:
                    with open(os.path.join(input_dir, file.name), "wb") as f:
                        f.write(file.read())

            # load overlays uploaded through widget
            if overlay_files:                               # ### FIX ###
                for ofile in overlay_files:
                    data = np.frombuffer(ofile.read(), np.uint8)
                    oimg = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
                    if oimg is not None:
                        overlay_images.append((os.path.splitext(ofile.name)[0], oimg))

            st.info("🔄 Processing images...")

            # ( ... rest of your loops stay unchanged ...)
            # -----------------------------------------------------------------
            #  final ZIP packaging & download button (unchanged)
            # -----------------------------------------------------------------
