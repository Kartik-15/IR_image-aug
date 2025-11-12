# app.py (replace your existing file with this)
import streamlit as st, cv2, numpy as np, os, glob, zipfile, tempfile, gc
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

# ──────────────────────────────────
# Aug helpers (unchanged logic, minor safety tweaks)
# ──────────────────────────────────
def apply_tint(img, color, a=0.25):
    layer = np.full_like(img, color, np.uint8)
    return cv2.addWeighted(img, 1-a, layer, a, 0)

def apply_shadow(img, strength=0.45):
    h, w = img.shape[:2]
    mask = np.zeros_like(img)
    cv2.rectangle(mask, (int(w*0.3),0), (w,int(h*0.7)), (0,0,0), -1)
    return cv2.addWeighted(img, 1, mask, strength, 0)

def apply_glass_reflection(img, intensity=0.12):
    over = np.zeros_like(img)
    h,w = img.shape[:2]
    step = max(1, w//20)
    for x in range(0,w,step):
        # keep same visual behaviour
        cv2.line(over,(x,0),(x-h//2,h),(255,255,255),1)
    over = cv2.GaussianBlur(over,(0,0),5)
    return cv2.addWeighted(img, 1, over, intensity, 0)

def apply_gaussian_blur(img, k=7):
    k = int(k)
    k = k if k % 2 == 1 else k+1
    return cv2.GaussianBlur(img, (k,k), 0)

def apply_random_occlusion(img):
    return img  # stub kept intentionally (no change)

def apply_perspective_transform(i):
    return i    # stub kept intentionally (no change)

def apply_overlay(base, overlay, alpha=0.3):
    # resize overlay to base, handle alpha channel if present
    ov = cv2.resize(overlay, (base.shape[1], base.shape[0]))
    if ov.ndim == 3 and ov.shape[2] == 4:
        ov = cv2.cvtColor(ov, cv2.COLOR_BGRA2BGR)
    if base.ndim == 3 and base.shape[2] == 4:
        base = cv2.cvtColor(base, cv2.COLOR_BGRA2BGR)
    return cv2.addWeighted(base, 1, ov, alpha, 0)

def save_aug(img, func, name, suf, out_dir):
    out = func(img)
    # ensure correct color conversion (our internal images are RGB)
    path = os.path.join(out_dir, f"{name}_{suf}.jpg")
    cv2.imwrite(path, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    return path

# ──────────────────────────────────
# Sidebar options (UI preserved)
# ──────────────────────────────────
st.title("🧪 Custom Image Augmentation Tool")

up_files = st.file_uploader("Upload images / zip", accept_multiple_files=True)
ov_uploads = st.sidebar.file_uploader("Upload Overlay(s)", type=["png","jpg"], accept_multiple_files=True)

st.sidebar.header("Settings")
augmentations = st.sidebar.multiselect("Augmentations",
    ["Shadow","Reflection","Blur","Occlusion","Perspective"])
brightness_opts = st.sidebar.multiselect("Brightness", ["dark","normal","bright"])
tint_opts = st.sidebar.multiselect("Tints",
    ["warm","cool","cool_white","warm_white","fluorescent_green",
     "bluish_white","soft_pink","daylight"])

brightness_vals = {"dark":0.8, "normal":1.2, "bright":1.4}
tint_vals = {
    "warm":(0,30,80),"cool":(80,30,0),"cool_white":(220,255,255),
    "warm_white":(255,240,200),"fluorescent_green":(220,255,220),
    "bluish_white":(200,220,255),"soft_pink":(255,220,230),"daylight":(255,255,240)
}

# Add non-intrusive performance controls in sidebar (safe defaults based on your environment)
st.sidebar.markdown("---")
st.sidebar.header("Performance / Stability")
batch_size = st.sidebar.number_input("Batch size (images at once)", min_value=1, max_value=64, value=8, help="Process N input images per batch to limit memory use.")
use_parallel = st.sidebar.checkbox("Use parallel processing", value=True, help="Process multiple input files in parallel (bounded).")
max_workers = st.sidebar.slider("Max workers (parallel)", 1, 8, 4, help="Number of parallel workers for processing input files.")
preview_max_size = st.sidebar.selectbox("Preview downscale (px)", [512, 720, 1024], index=1)

# ──────────────────────────────────
# Overlay images in sidebar (preserve behavior)
# ──────────────────────────────────
overlay_imgs = []
OV_DIR = "overlays"
for p in glob.glob(f"{OV_DIR}/*.*"):
    lbl = os.path.splitext(os.path.basename(p))[0]
    col1, col2 = st.sidebar.columns([1, 4])
    with col1:
        st.image(p, use_container_width=True)
    with col2:
        if st.checkbox(lbl, key=f"ov_{lbl}"):
            img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            overlay_imgs.append((lbl, img))

# include any uploaded overlays
for f in ov_uploads or []:
    data = np.frombuffer(f.read(), np.uint8)
    if data.size == 0:
        continue
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    overlay_imgs.append((os.path.splitext(f.name)[0], img))

# ──────────────────────────────────
# Live Preview Section (preserve UI appearance & controls)
# ──────────────────────────────────
st.markdown("---")
st.header("🔍 Live Preview (Sample Image)")

sample_files = glob.glob(os.path.join("Sample", "*.jpg"))
if sample_files:
    sample_labels = [os.path.basename(f) for f in sample_files]
    selected_label = st.sidebar.selectbox("Select a Sample Image", sample_labels)
    selected_sample = os.path.join("Sample", selected_label)
    preview_img = cv2.cvtColor(cv2.imread(selected_sample), cv2.COLOR_BGR2RGB)
    st.sidebar.image(preview_img, caption="Preview", use_container_width=True)

    img_prev = preview_img.copy()

    img_col, spacer_col, slider_col = st.columns([2, 0.1, 1])
    with slider_col:
        st.markdown("#### 🔧 Preview Controls")
        shadow_strength = st.slider("Shadow Strength", 0.0, 1.0, 0.45, step=0.05)
        reflection_intensity = st.slider("Reflection Intensity", 0.0, 1.0, 0.12, step=0.02)
        blur_strength = st.slider("Blur Kernel (odd)", 1, 15, 7, step=2)

        # Tint controls (checkbox + opacity sliders) - same as before, keys are important
        enabled_tints_local = []
        for tint in tint_opts:
            enable = st.checkbox(f"Enable {tint}", value=True, key=f"tint_en_{tint}")
            if enable:
                opacity = st.slider(f"{tint} Opacity", 0.0, 1.0, 0.25, step=0.05, key=f"tint_op_{tint}")
                enabled_tints_local.append((tint, opacity))

        # Overlay controls (checkbox + opacity)
        enabled_overlays_local = []
        for name, img in overlay_imgs:
            enable = st.checkbox(f"Enable Overlay: {name}", value=True, key=f"ov_en_{name}")
            if enable:
                ov_alpha = st.slider(f"{name} Opacity", 0.0, 1.0, 0.3, step=0.05, key=f"ov_op_{name}")
                enabled_overlays_local.append((name, img, ov_alpha))

    # apply preview choices to the preview image
    for tint, alpha in enabled_tints_local:
        img_prev = apply_tint(img_prev, tint_vals[tint], alpha)
    for _, ov_img, ov_alpha in enabled_overlays_local:
        img_prev = apply_overlay(img_prev, ov_img, ov_alpha)
    if "Shadow" in augmentations:
        img_prev = apply_shadow(img_prev, shadow_strength)
    if "Reflection" in augmentations:
        img_prev = apply_glass_reflection(img_prev, reflection_intensity)
    if "Blur" in augmentations:
        img_prev = apply_gaussian_blur(img_prev, blur_strength)

    with img_col:
        # downscale just for display to reduce memory & rendering load
        display_img = img_prev.copy()
        h,w = display_img.shape[:2]
        max_dim = preview_max_size
        if max(h,w) > max_dim:
            scale = max_dim / max(h,w)
            display_img = cv2.resize(display_img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
        st.image(display_img, caption="Live Preview", use_container_width=True)
else:
    st.warning("No sample image found in the 'Sample' folder. Please add at least one .jpg file.")

# ──────────────────────────────────
# Helper: read checkbox/slider state values (we created keys above in preview UI)
# This allows the main PROCESS logic to reuse the same user choices without recreating UI elements.
# ──────────────────────────────────
def get_enabled_tints():
    out = []
    for tint in tint_opts:
        if st.session_state.get(f"tint_en_{tint}", False):
            alpha = st.session_state.get(f"tint_op_{tint}", 0.25)
            out.append((tint, alpha))
    return out

def get_enabled_overlays():
    out = []
    for name, img in overlay_imgs:
        if st.session_state.get(f"ov_en_{name}", False):
            alpha = st.session_state.get(f"ov_op_{name}", 0.3)
            out.append((name, img, alpha))
    return out

# ──────────────────────────────────
# Core image processing for a single file (writes results immediately to disk)
# This keeps memory bounded: we don't collect all outputs in a list.
# ──────────────────────────────────
def process_single_file(filepath, brightness_opts_local, enabled_tints, enabled_overlays, augmentations_local,
                        shadow_strength, reflection_intensity, blur_strength, out_dir):
    """Process one input image file and write generated images to out_dir. Returns number of images written."""
    written = 0
    # safe read
    img_bgr = cv2.imread(filepath)
    if img_bgr is None:
        return 0
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    base = os.path.splitext(os.path.basename(filepath))[0]

    # ensure there's at least the original option to iterate over
    brightness_iter = brightness_opts_local or ["original"]
    tints_iter = enabled_tints or [("original", None)]
    overlays_iter = enabled_overlays or [("orig", None, None)]

    for b in brightness_iter:
        if b == "original":
            img_b = img
        else:
            # safely apply brightness scaling: multiply then clip
            img_b = np.clip(img * brightness_vals.get(b, 1.0), 0, 255).astype(np.uint8)

        for t, t_alpha in tints_iter:
            if t == "original":
                img_bt = img_b
            else:
                img_bt = apply_tint(img_b, tint_vals[t], t_alpha)

            for ov_name, ov_img, ov_alpha in overlays_iter:
                if ov_img is None:
                    img_bto = img_bt
                else:
                    img_bto = apply_overlay(img_bt, ov_img, ov_alpha)

                suffix = "_".join([s for s in [b, t] if s != "original"])
                if ov_img is not None:
                    suffix += f"_ov_{ov_name}"
                suffix = suffix or "original"

                # if augmentations selected -> apply each and save
                if augmentations_local:
                    for aug in augmentations_local:
                        func = {
                            "Shadow": lambda x: apply_shadow(x, shadow_strength),
                            "Reflection": lambda x: apply_glass_reflection(x, reflection_intensity),
                            "Blur": lambda x: apply_gaussian_blur(x, blur_strength),
                            "Occlusion": apply_random_occlusion,
                            "Perspective": apply_perspective_transform
                        }[aug]
                        try:
                            path = save_aug(img_bto, func, base, f"{suffix}_{aug.lower()}", out_dir)
                            written += 1
                        except Exception as e:
                            # skip failing image but log to streamlit for debugging
                            st.error(f"Failed to write augmented image for {base} ({aug}): {e}")
                else:
                    # save base combination
                    out_path = os.path.join(out_dir, f"{base}_{suffix}.jpg")
                    try:
                        cv2.imwrite(out_path, cv2.cvtColor(img_bto, cv2.COLOR_RGB2BGR))
                        written += 1
                    except Exception as e:
                        st.error(f"Failed to write image {out_path}: {e}")

    # attempt to free local RAM
    gc.collect()
    return written

# ──────────────────────────────────
# PROCESS
# ──────────────────────────────────
if up_files and (augmentations or brightness_opts or tint_opts or overlay_imgs):
    if st.button("✅ Process Images"):
        # read the user's preview-based choices from session_state so behaviour matches preview exactly
        enabled_tints = get_enabled_tints()
        enabled_overlays = get_enabled_overlays()

        # validate inputs
        if not enabled_tints and tint_opts:
            st.warning("You selected tint options but no tint checkboxes are enabled in the preview. Enable them to apply tints.")
        if not enabled_overlays and overlay_imgs:
            st.info("No overlays enabled — continuing without overlays.")

        # prepare input files in a temp dir (extract zips safely)
        with tempfile.TemporaryDirectory() as inp_dir, tempfile.TemporaryDirectory() as out_dir:
            # write uploaded files / extract zips into inp_dir
            for f in up_files:
                if f.name.endswith(".zip"):
                    # write zip bytes to temp file then extract
                    tmpzip = os.path.join(inp_dir, f.name)
                    with open(tmpzip, "wb") as zf:
                        zf.write(f.read())
                    try:
                        with zipfile.ZipFile(tmpzip, "r") as z:
                            z.extractall(inp_dir)
                    except zipfile.BadZipFile:
                        st.error(f"Uploaded zip file {f.name} appears corrupted — skipping.")
                else:
                    path = os.path.join(inp_dir, f.name)
                    with open(path, "wb") as out_f:
                        out_f.write(f.read())

            # list only image files in inp_dir
            all_input_files = [os.path.join(inp_dir, p) for p in os.listdir(inp_dir)
                               if p.lower().endswith((".jpg",".jpeg",".png"))]

            total_inputs = len(all_input_files)
            if total_inputs == 0:
                st.warning("No image files found in the uploaded input(s).")
            else:
                st.info("Processing...")
                progress_bar = st.progress(0)
                status_text = st.empty()
                total_written = 0
                # chunk input files by batch size to limit memory/parallelism
                batches = [all_input_files[i:i+batch_size] for i in range(0, total_inputs, batch_size)]
                batch_index = 0

                # prepare function partial for worker threads
                worker_fn = partial(process_single_file,
                                    brightness_opts_local=brightness_opts,
                                    enabled_tints=enabled_tints,
                                    enabled_overlays=enabled_overlays,
                                    augmentations_local=augmentations,
                                    shadow_strength=st.session_state.get("Shadow Strength", 0.45) if "Shadow Strength" in st.session_state else st.sidebar.session_state.get("Shadow Strength", 0.45) if False else locals().get('shadow_strength', 0.45),
                                    reflection_intensity=st.session_state.get("Reflection Intensity", 0.12) if "Reflection Intensity" in st.session_state else locals().get('reflection_intensity', 0.12),
                                    blur_strength=st.session_state.get("Blur Kernel (odd)", 7) if "Blur Kernel (odd)" in st.session_state else locals().get('blur_strength', 7),
                                    out_dir=out_dir)

                # NOTE: earlier UI stores these variables in local scope; however to ensure we always have the values:
                # fall back to the local variable values if session_state keys are not present.
                # update worker_fn with actual values retrieved from locals if possible
                # but safer: read from st.session_state keys we defined earlier (their keys are unique)
                # We'll try to read the slider values by the labels used above:
                # Fallback approach: attempt to read by keys created above (they were created without explicit keys),
                # so use the local variables defined in preview. They exist if preview UI was rendered; fallback to defaults if not.

                # Process batches
                with ThreadPoolExecutor(max_workers=max_workers if use_parallel else 1) as executor:
                    # iterate per batch
                    for batch in batches:
                        batch_index += 1
                        status_text.info(f"Processing batch {batch_index}/{len(batches)} — {len(batch)} files in this batch.")
                        # submit batch jobs to executor
                        futures = {executor.submit(process_single_file, fpath,
                                                   brightness_opts, enabled_tints, enabled_overlays,
                                                   augmentations,
                                                   # use the local UI values if they're in scope; else defaults
                                                   shadow_strength if 'shadow_strength' in locals() else 0.45,
                                                   reflection_intensity if 'reflection_intensity' in locals() else 0.12,
                                                   blur_strength if 'blur_strength' in locals() else 7,
                                                   out_dir): fpath for fpath in batch}

                        # as each completes, update counters & progress
                        for fut in as_completed(futures):
                            inp = futures[fut]
                            try:
                                written = fut.result()
                                total_written += written
                            except Exception as e:
                                st.error(f"Error processing {inp}: {e}")
                            # update progress relative to total inputs and (approximated) generated images
                            # since total generated images is unknown upfront, we update based on processed inputs
                            completed_inputs = sum(1 for b in all_input_files if os.path.exists(b) and os.path.basename(b) not in [])
                            # progress fraction by batches:
                            progress_fraction = min(1.0, (batch_index - 1) / len(batches) + 0.01)
                            progress_bar.progress(min(1.0, (batch_index / len(batches))))
                            status_text.text(f"Finished {total_written} output images so far...")
                            # force garbage collection periodically
                            gc.collect()

                        # explicit cleanup between batches
                        gc.collect()

                # after processing all batches
                st.success("✅ Done")
                st.markdown(f"**Total images created: {total_written}**")

                # Prepare download: if few files show inline, else zip
                output_files = [os.path.join(out_dir, p) for p in os.listdir(out_dir)]
                if len(output_files) == 0:
                    st.warning("No output images were created.")
                elif len(output_files) < 6:
                    cols = st.columns(min(4, len(output_files)))
                    for c, path in zip(cols, output_files):
                        fname = os.path.basename(path)
                        with open(path, "rb") as f:
                            img_bytes = f.read()
                        with c:
                            st.image(img_bytes, caption=fname, use_column_width=False)
                            st.download_button("Download", img_bytes, file_name=fname)
                else:
                    buf = BytesIO()
                    with zipfile.ZipFile(buf, "w") as z:
                        for fpath in output_files:
                            z.write(fpath, os.path.basename(fpath))
                    buf.seek(0)
                    st.download_button("Download ZIP", buf.getvalue(), "augmented_images.zip", "application/zip")

                # final cleanup
                gc.collect()
else:
    if up_files:
        st.warning("Select at least one transformation or overlay.")
