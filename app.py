import streamlit as st, cv2, numpy as np, os, glob, zipfile, tempfile
from io import BytesIO

# ──────────────────────────────────
# Aug helpers
# ──────────────────────────────────
def apply_tint(img, color, a=0.25):
    layer = np.full_like(img, color, np.uint8)
    return cv2.addWeighted(img, 1-a, layer, a, 0)

def apply_shadow(img, strength=0.45):
    h, w = img.shape[:2]
    mask = np.zeros_like(img)
    cv2.rectangle(mask, (int(w*0.3),0), (w,int(h*0.7)), (0,0,0), -1)
    return cv2.addWeighted(img,1, mask,strength,0)

def apply_glass_reflection(img, strength=0.12):
    over = np.zeros_like(img)
    h,w = img.shape[:2]
    for x in range(0,w,w//20):
        cv2.line(over,(x,0),(x-h//2,h),(255,255,255),1)
    over = cv2.GaussianBlur(over,(0,0),5)
    return cv2.addWeighted(img,1,over,strength,0)

def apply_gaussian_blur(img): return cv2.GaussianBlur(img,(7,7),0)
def apply_random_occlusion(img): return img

def apply_perspective_transform(img): return img

def apply_overlay(base, overlay, alpha=0.3):
    ov = cv2.resize(overlay,(base.shape[1],base.shape[0]))
    if ov.shape[2]==4: ov = cv2.cvtColor(ov, cv2.COLOR_BGRA2BGR)
    if base.shape[2]==4: base = cv2.cvtColor(base, cv2.COLOR_BGRA2BGR)
    return cv2.addWeighted(base,1, ov,alpha,0)

def save_aug(img, func, name, suf, out_dir):
    out = func(img)
    path = os.path.join(out_dir,f"{name}_{suf}.jpg")
    cv2.imwrite(path, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    return path

# ──────────────────────────────────
# UI Setup
# ──────────────────────────────────
st.set_page_config(layout="wide")
st.title("🧪 Custom Image Augmentation Tool")

left_col, mid_col, right_col = st.columns([1, 1.5, 1.5])

# Sidebar settings
with st.sidebar:
    st.header("🔧 Settings")
    augmentations = st.multiselect("Augmentations", ["Shadow","Reflection","Blur","Occlusion","Perspective"])
    brightness_opts = st.multiselect("Brightness", ["dark","normal","bright"])
    tint_opts = st.multiselect("Tints", [
        "warm","cool","cool_white","warm_white","fluorescent_green",
        "bluish_white","soft_pink","daylight"])

    brightness_vals = {"dark":0.8, "normal":1.2, "bright":1.4}
    tint_vals = {
        "warm":(0,30,80),"cool":(80,30,0),"cool_white":(220,255,255),
        "warm_white":(255,240,200),"fluorescent_green":(220,255,220),
        "bluish_white":(200,220,255),"soft_pink":(255,220,230),"daylight":(255,255,240)
    }

# Upload section
with right_col:
    up_files = st.file_uploader("Upload images / zip", accept_multiple_files=True)
    ov_uploads = st.file_uploader("Upload Overlay(s)", type=["png","jpg"], accept_multiple_files=True)

# Sample image selection
with mid_col:
    st.subheader("🖼 Sample Image Preview")
    sample_imgs = sorted(glob.glob("Sample/*.*"))
    sample_col_imgs = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in sample_imgs]
    sel_sample = None
    sample_idx = 0

    cols = st.columns(len(sample_imgs))
    for i, (col, img, path) in enumerate(zip(cols, sample_col_imgs, sample_imgs)):
        if col.button(os.path.basename(path), key=f"sample_btn_{i}"):
            sample_idx = i
    sel_sample = sample_col_imgs[sample_idx]
    st.image(sel_sample, caption=f"Selected Sample: {os.path.basename(sample_imgs[sample_idx])}", width=300)

# Live Preview
with right_col:
    st.subheader("🔍 Live Preview")
    preview_img = sel_sample.copy()
    shadow_opacity, refl_opacity = 0.45, 0.12
    overlay_opacities = {}
    tint_strengths = {}

    if "Shadow" in augmentations:
        shadow_opacity = st.slider("Shadow Strength", 0.0, 1.0, 0.45, 0.05)
        preview_img = apply_shadow(preview_img, strength=shadow_opacity)

    if "Reflection" in augmentations:
        refl_opacity = st.slider("Reflection Strength", 0.0, 1.0, 0.12, 0.01)
        preview_img = apply_glass_reflection(preview_img, strength=refl_opacity)

    # Overlay loading
    overlay_imgs=[]
    OV_DIR="overlays"
    for p in glob.glob(f"{OV_DIR}/*.*"):
        lbl=os.path.splitext(os.path.basename(p))[0]
        if st.checkbox(lbl,key=f"ov_{lbl}"):
            img=cv2.imread(p,cv2.IMREAD_UNCHANGED); overlay_imgs.append((lbl,img))

    for f in ov_uploads:
        data=np.frombuffer(f.read(),np.uint8)
        img=cv2.imdecode(data,cv2.IMREAD_UNCHANGED)
        overlay_imgs.append((os.path.splitext(f.name)[0],img))

    for name, ov_img in overlay_imgs:
        overlay_opacities[name] = st.slider(f"Overlay '{name}' Opacity", 0.0, 1.0, 0.3, 0.05, key=name)
        preview_img = apply_overlay(preview_img, ov_img, alpha=overlay_opacities[name])

    for tint in tint_opts:
        tint_strengths[tint] = st.slider(f"Tint '{tint}' Strength", 0.0, 1.0, 0.25, 0.05, key=f"tint_{tint}")
        preview_img = apply_tint(preview_img, tint_vals[tint], a=tint_strengths[tint])

    for b in brightness_opts:
        preview_img = np.clip(preview_img * brightness_vals[b], 0, 255).astype(np.uint8)

    st.image(preview_img, caption="Combined Preview", width=500)

# ──────────────────────────────────
# PROCESS
# ──────────────────────────────────
if up_files and (augmentations or brightness_opts or tint_opts or overlay_imgs):
    if st.button("✅ Process Images"):
        with tempfile.TemporaryDirectory() as inp_dir, tempfile.TemporaryDirectory() as out_dir:
            for f in up_files:
                path=os.path.join(inp_dir,f.name)
                if f.name.endswith(".zip"):
                    with zipfile.ZipFile(f,"r") as z:z.extractall(inp_dir)
                else:
                    open(path,"wb").write(f.read())

            st.info("Processing...")
            output_files = []
            for fname in os.listdir(inp_dir):
                if not fname.lower().endswith((".jpg",".jpeg",".png")): continue
                img=cv2.cvtColor(cv2.imread(os.path.join(inp_dir,fname)), cv2.COLOR_BGR2RGB)
                base=os.path.splitext(fname)[0]

                for b in (brightness_opts or ["original"]):
                    img_b = img if b=="original" else np.clip(img*brightness_vals[b],0,255).astype(np.uint8)
                    for t in (tint_opts or ["original"]):
                        a_val = tint_strengths[t] if t in tint_strengths else 0.25
                        img_bt = img_b if t=="original" else apply_tint(img_b, tint_vals[t], a=a_val)
                        for ov_name,ov_img in (overlay_imgs or [("orig",None)]):
                            ov_val = overlay_opacities.get(ov_name, 0.3)
                            img_bto = img_bt if ov_img is None else apply_overlay(img_bt,ov_img, alpha=ov_val)

                            suffix="_".join([s for s in [b,t] if s!="original"])
                            if ov_img is not None: suffix += f"_ov_{ov_name}"
                            suffix = suffix or "original"

                            if augmentations:
                                for aug in augmentations:
                                    func={
                                        "Shadow":lambda x: apply_shadow(x, strength=shadow_opacity),
                                        "Reflection":lambda x: apply_glass_reflection(x, strength=refl_opacity),
                                        "Blur":apply_gaussian_blur,
                                        "Occlusion":apply_random_occlusion,
                                        "Perspective":apply_perspective_transform
                                    }[aug]
                                    path = save_aug(img_bto, func, base, suffix+"_"+aug.lower(), out_dir)
                                    output_files.append(path)
                            else:
                                path = os.path.join(out_dir,f"{base}_{suffix}.jpg")
                                cv2.imwrite(path, cv2.cvtColor(img_bto, cv2.COLOR_RGB2BGR))
                                output_files.append(path)

            zip_path = os.path.join(tempfile.gettempdir(), "augmented_images.zip")
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for file in output_files:
                    zipf.write(file, os.path.basename(file))

            st.success(f"Done! Total output images: {len(output_files)}")
            with open(zip_path, "rb") as f:
                st.download_button("📦 Download ZIP", f.read(), file_name="augmented_images.zip")
