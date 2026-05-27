import streamlit as st
import cv2, numpy as np, os, glob, zipfile, tempfile, gc
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Constants ─────────────────────────────────────────────────────────────────
BRIGHTNESS_VALS = {"dark": 0.8, "normal": 1.2, "bright": 1.4}
TINT_VALS = {
    "warm": (0, 30, 80), "cool": (80, 30, 0), "cool_white": (220, 255, 255),
    "warm_white": (255, 240, 200), "fluorescent_green": (220, 255, 220),
    "bluish_white": (200, 220, 255), "soft_pink": (255, 220, 230), "daylight": (255, 255, 240),
}

# ── Augmentation helpers ──────────────────────────────────────────────────────
def apply_tint(img, color, a=0.25):
    layer = np.full_like(img, color, np.uint8)
    return cv2.addWeighted(img, 1 - a, layer, a, 0)

def apply_shadow(img, strength=0.45):
    h, w = img.shape[:2]
    mask = np.zeros_like(img)
    cv2.rectangle(mask, (int(w * 0.3), 0), (w, int(h * 0.7)), (0, 0, 0), -1)
    return cv2.addWeighted(img, 1, mask, strength, 0)

def apply_glass_reflection(img, intensity=0.12):
    over = np.zeros_like(img)
    h, w = img.shape[:2]
    step = max(1, w // 20)
    for x in range(0, w, step):
        cv2.line(over, (x, 0), (x - h // 2, h), (255, 255, 255), 1)
    over = cv2.GaussianBlur(over, (0, 0), 5)
    return cv2.addWeighted(img, 1, over, intensity, 0)

def apply_gaussian_blur(img, k=7):
    k = int(k)
    k = k if k % 2 == 1 else k + 1
    return cv2.GaussianBlur(img, (k, k), 0)

def apply_random_occlusion(img):
    return img

def apply_perspective_transform(img):
    return img

def apply_overlay(base, overlay, alpha=0.3):
    ov = cv2.resize(overlay, (base.shape[1], base.shape[0]))
    if ov.ndim == 3 and ov.shape[2] == 4:
        ov = cv2.cvtColor(ov, cv2.COLOR_BGRA2BGR)
    if base.ndim == 3 and base.shape[2] == 4:
        base = cv2.cvtColor(base, cv2.COLOR_BGRA2BGR)
    return cv2.addWeighted(base, 1, ov, alpha, 0)

# ── Worker: no st.* calls — safe to run in threads ────────────────────────────
def process_single_file(filepath, brightness_opts_local, enabled_tints, enabled_overlays,
                        augmentations_local, shadow_strength, reflection_intensity,
                        blur_strength, out_dir):
    """Returns (written_count, [error_strings]). Never calls st.*."""
    errors, written = [], 0

    img_bgr = cv2.imread(filepath)
    if img_bgr is None:
        return 0, [f"Could not read: {os.path.basename(filepath)}"]
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    base = os.path.splitext(os.path.basename(filepath))[0]

    brightness_iter = brightness_opts_local or ["original"]
    tints_iter      = enabled_tints or [("original", None)]
    overlays_iter   = enabled_overlays or [("orig", None, None)]

    aug_map = {
        "Shadow":      lambda x: apply_shadow(x, shadow_strength),
        "Reflection":  lambda x: apply_glass_reflection(x, reflection_intensity),
        "Blur":        lambda x: apply_gaussian_blur(x, blur_strength),
        "Occlusion":   apply_random_occlusion,
        "Perspective": apply_perspective_transform,
    }

    for b in brightness_iter:
        img_b = img if b == "original" else np.clip(img * BRIGHTNESS_VALS.get(b, 1.0), 0, 255).astype(np.uint8)
        for t, t_alpha in tints_iter:
            img_bt = img_b if t == "original" else apply_tint(img_b, TINT_VALS[t], t_alpha)
            for ov_name, ov_img, ov_alpha in overlays_iter:
                img_bto = img_bt if ov_img is None else apply_overlay(img_bt, ov_img, ov_alpha)

                parts = [s for s in [b, t] if s != "original"]
                if ov_img is not None:
                    parts.append(f"ov_{ov_name}")
                suffix = "_".join(parts) or "original"

                if augmentations_local:
                    for aug in augmentations_local:
                        func = aug_map.get(aug)
                        if not func:
                            continue
                        try:
                            out = func(img_bto)
                            cv2.imwrite(os.path.join(out_dir, f"{base}_{suffix}_{aug.lower()}.jpg"),
                                        cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
                            written += 1
                        except Exception as e:
                            errors.append(f"{base} ({aug}): {e}")
                else:
                    try:
                        cv2.imwrite(os.path.join(out_dir, f"{base}_{suffix}.jpg"),
                                    cv2.cvtColor(img_bto, cv2.COLOR_RGB2BGR))
                        written += 1
                    except Exception as e:
                        errors.append(f"{base}_{suffix}: {e}")

    gc.collect()
    return written, errors

# ── Cached ZIP reader: reads once per unique path, avoids re-reading every rerun
@st.cache_data(show_spinner=False)
def _read_zip(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🧪 Custom Image Augmentation Tool")

up_files  = st.file_uploader("Upload images / zip", accept_multiple_files=True)
ov_uploads = st.sidebar.file_uploader("Upload Overlay(s)", type=["png", "jpg"], accept_multiple_files=True)

st.sidebar.header("Settings")
augmentations = st.sidebar.multiselect("Augmentations",
    ["Shadow", "Reflection", "Blur", "Occlusion", "Perspective"])
brightness_opts = st.sidebar.multiselect("Brightness", ["dark", "normal", "bright"])
tint_opts = st.sidebar.multiselect("Tints",
    ["warm", "cool", "cool_white", "warm_white", "fluorescent_green",
     "bluish_white", "soft_pink", "daylight"])

st.sidebar.markdown("---")
st.sidebar.header("Performance / Stability")
batch_size      = st.sidebar.number_input("Batch size", min_value=1, max_value=64, value=8,
    help="Process N input images per batch.")
use_parallel    = st.sidebar.checkbox("Parallel processing", value=True)
max_workers     = st.sidebar.slider("Max workers", 1, 8, 4)
preview_max_size = st.sidebar.selectbox("Preview downscale (px)", [512, 720, 1024], index=1)

# ── Overlay images ────────────────────────────────────────────────────────────
overlay_imgs = []
OV_DIR = "overlays"
for p in glob.glob(f"{OV_DIR}/*.*"):
    lbl = os.path.splitext(os.path.basename(p))[0]
    c1, c2 = st.sidebar.columns([1, 4])
    with c1:
        st.image(p, width="stretch")
    with c2:
        if st.checkbox(lbl, key=f"ov_{lbl}"):
            overlay_imgs.append((lbl, cv2.imread(p, cv2.IMREAD_UNCHANGED)))

for f in ov_uploads or []:
    f.seek(0)
    data = np.frombuffer(f.read(), np.uint8)
    if data.size == 0:
        continue
    overlay_imgs.append((os.path.splitext(f.name)[0], cv2.imdecode(data, cv2.IMREAD_UNCHANGED)))

# ── Live Preview ──────────────────────────────────────────────────────────────
st.markdown("---")
st.header("🔍 Live Preview (Sample Image)")

sample_files = glob.glob(os.path.join("Sample", "*.jpg"))
if sample_files:
    sample_labels = [os.path.basename(f) for f in sample_files]
    selected_label   = st.sidebar.selectbox("Select a Sample Image", sample_labels)
    preview_img      = cv2.cvtColor(cv2.imread(os.path.join("Sample", selected_label)), cv2.COLOR_BGR2RGB)
    st.sidebar.image(preview_img, caption="Preview", width="stretch")

    img_prev = preview_img.copy()
    img_col, _, slider_col = st.columns([2, 0.1, 1])

    with slider_col:
        st.markdown("#### 🔧 Preview Controls")
        shadow_strength    = st.slider("Shadow Strength",    0.0, 1.0, 0.45, step=0.05, key="shadow_strength")
        reflection_intensity = st.slider("Reflection Intensity", 0.0, 1.0, 0.12, step=0.02, key="reflection_intensity")
        blur_strength      = st.slider("Blur Kernel (odd)", 1,   15,  7,    step=2,    key="blur_strength")

        enabled_tints_local = []
        for tint in tint_opts:
            if st.checkbox(f"Enable {tint}", value=True, key=f"tint_en_{tint}"):
                opacity = st.slider(f"{tint} Opacity", 0.0, 1.0, 0.25, step=0.05, key=f"tint_op_{tint}")
                enabled_tints_local.append((tint, opacity))

        enabled_overlays_local = []
        for name, ov_img in overlay_imgs:
            if st.checkbox(f"Enable Overlay: {name}", value=True, key=f"ov_en_{name}"):
                ov_alpha = st.slider(f"{name} Opacity", 0.0, 1.0, 0.3, step=0.05, key=f"ov_op_{name}")
                enabled_overlays_local.append((name, ov_img, ov_alpha))

    for tint, alpha in enabled_tints_local:
        img_prev = apply_tint(img_prev, TINT_VALS[tint], alpha)
    for _, ov_img, ov_alpha in enabled_overlays_local:
        img_prev = apply_overlay(img_prev, ov_img, ov_alpha)
    if "Shadow"     in augmentations: img_prev = apply_shadow(img_prev, shadow_strength)
    if "Reflection" in augmentations: img_prev = apply_glass_reflection(img_prev, reflection_intensity)
    if "Blur"       in augmentations: img_prev = apply_gaussian_blur(img_prev, blur_strength)

    with img_col:
        d = img_prev.copy()
        h, w = d.shape[:2]
        if max(h, w) > preview_max_size:
            s = preview_max_size / max(h, w)
            d = cv2.resize(d, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        st.image(d, caption="Live Preview", width="stretch")
else:
    st.warning("No sample image found in the 'Sample' folder. Please add at least one .jpg file.")

# ── Helpers ───────────────────────────────────────────────────────────────────
def get_enabled_tints():
    return [(t, st.session_state.get(f"tint_op_{t}", 0.25))
            for t in tint_opts if st.session_state.get(f"tint_en_{t}", False)]

def get_enabled_overlays():
    return [(n, img, st.session_state.get(f"ov_op_{n}", 0.3))
            for n, img in overlay_imgs if st.session_state.get(f"ov_en_{n}", False)]

# ── Process section ───────────────────────────────────────────────────────────
st.markdown("---")
if up_files and (augmentations or brightness_opts or tint_opts or overlay_imgs):

    # Estimate output count so users can spot large batches before starting
    _et = get_enabled_tints()
    _eo = get_enabled_overlays()
    est = len(up_files) * max(1, len(brightness_opts)) * max(1, len(_et)) \
        * max(1, len(_eo)) * max(1, len(augmentations))
    if est > 300:
        st.warning(f"⚠️ Estimated **{est}** output images. Consider reducing filter combinations "
                   f"to stay under Streamlit Cloud's memory limit.")
    else:
        st.info(f"Estimated output images: **{est}**")

    if st.button("✅ Process Images"):
        enabled_tints   = get_enabled_tints()
        enabled_overlays = get_enabled_overlays()

        proc_shadow     = st.session_state.get("shadow_strength", 0.45)
        proc_reflection = st.session_state.get("reflection_intensity", 0.12)
        proc_blur       = int(st.session_state.get("blur_strength", 7))

        with tempfile.TemporaryDirectory() as inp_dir, tempfile.TemporaryDirectory() as out_dir:
            for f in up_files:
                f.seek(0)  # reset in case file was read in a previous run
                if f.name.lower().endswith(".zip"):
                    tmpzip = os.path.join(inp_dir, f.name)
                    with open(tmpzip, "wb") as zf:
                        zf.write(f.read())
                    try:
                        with zipfile.ZipFile(tmpzip, "r") as z:
                            z.extractall(inp_dir)
                    except zipfile.BadZipFile:
                        st.error(f"Corrupted zip: {f.name} — skipping.")
                else:
                    with open(os.path.join(inp_dir, f.name), "wb") as out_f:
                        out_f.write(f.read())

            all_input_files = [
                os.path.join(inp_dir, p) for p in os.listdir(inp_dir)
                if p.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
            total_inputs = len(all_input_files)

            if total_inputs == 0:
                st.warning("No image files found in the uploaded input(s).")
            else:
                progress_bar = st.progress(0)
                status_text  = st.empty()
                total_written, completed_inputs, all_errors = 0, 0, []
                batches = [all_input_files[i:i + batch_size]
                           for i in range(0, total_inputs, batch_size)]

                with ThreadPoolExecutor(max_workers=max_workers if use_parallel else 1) as executor:
                    for batch in batches:
                        futures = {
                            executor.submit(
                                process_single_file, fp,
                                brightness_opts, enabled_tints, enabled_overlays,
                                augmentations, proc_shadow, proc_reflection, proc_blur,
                                out_dir
                            ): fp for fp in batch
                        }
                        for fut in as_completed(futures):
                            try:
                                written, errors = fut.result()
                                total_written += written
                                all_errors.extend(errors)
                            except Exception as e:
                                all_errors.append(f"{os.path.basename(futures[fut])}: {e}")
                            completed_inputs += 1
                            progress_bar.progress(completed_inputs / total_inputs)
                            status_text.text(f"Processed {completed_inputs}/{total_inputs} — "
                                             f"{total_written} outputs so far...")
                        gc.collect()

                progress_bar.progress(1.0)
                status_text.empty()
                for err in all_errors:
                    st.error(err)

                output_files = sorted([
                    os.path.join(out_dir, p) for p in os.listdir(out_dir)
                    if p.lower().endswith((".jpg", ".jpeg", ".png"))
                ])

                if not output_files:
                    st.warning("No output images were created.")
                else:
                    # Write ZIP directly to a named temp file on disk — this avoids
                    # loading the entire ZIP into RAM twice (BytesIO + .getvalue() copy).
                    # Only the path is stored in session_state; bytes are loaded from
                    # disk once per download via @st.cache_data.
                    old_path = st.session_state.pop("zip_path", None)
                    if old_path and os.path.exists(old_path):
                        try:
                            os.unlink(old_path)
                            _read_zip.clear()
                        except OSError:
                            pass

                    zip_fd, zip_path = tempfile.mkstemp(suffix=".zip", prefix="aug_")
                    os.close(zip_fd)
                    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as z:
                        for fp in output_files:
                            z.write(fp, os.path.basename(fp))

                    st.session_state["zip_path"]    = zip_path
                    st.session_state["output_count"] = total_written
                    gc.collect()

    # Download section — renders in the same script run as processing completes,
    # so no st.rerun() is needed. Also re-renders on every subsequent rerun as
    # long as the ZIP file still exists on disk.
    zip_path = st.session_state.get("zip_path")
    if zip_path and os.path.exists(zip_path):
        st.success(f"✅ {st.session_state.get('output_count', '?')} images generated — ready to download.")
        col_dl, col_clr = st.columns([3, 1])
        with col_dl:
            # _read_zip is cached by path, so the disk read only happens once
            # even across multiple reruns caused by widget interactions.
            st.download_button(
                "⬇️ Download ZIP",
                data=_read_zip(zip_path),
                file_name="augmented_images.zip",
                mime="application/zip",
                key="dl_zip",
            )
        with col_clr:
            if st.button("🗑️ Clear", key="clear_results"):
                old = st.session_state.pop("zip_path", None)
                if old and os.path.exists(old):
                    try:
                        os.unlink(old)
                        _read_zip.clear()
                    except OSError:
                        pass
                st.session_state.pop("output_count", None)
                st.rerun()

elif up_files:
    st.warning("Select at least one transformation or overlay.")
