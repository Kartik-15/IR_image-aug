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
    return cv2.addWeighted(img, 1, mask, strength, 0)

def apply_glass_reflection(img, intensity=0.12):
    over = np.zeros_like(img)
    h,w = img.shape[:2]
    for x in range(0,w,w//20):
        cv2.line(over,(x,0),(x-h//2,h),(255,255,255),1)
    over = cv2.GaussianBlur(over,(0,0),5)
    return cv2.addWeighted(img, 1, over, intensity, 0)

def apply_gaussian_blur(img, k=7):
    k = k if k % 2 == 1 else k+1
    return cv2.GaussianBlur(img, (k,k), 0)

def apply_random_occlusion(img):   return img  # stub
def apply_perspective_transform(i):return i    # stub

def apply_overlay(base, overlay, alpha=0.3):
    ov = cv2.resize(overlay,(base.shape[1],base.shape[0]))
    if ov.shape[2]==4: ov = cv2.cvtColor(ov,cv2.COLOR_BGRA2BGR)
    if base.shape[2]==4: base = cv2.cvtColor(base,cv2.COLOR_BGRA2BGR)
    return cv2.addWeighted(base,1, ov,alpha,0)

def save_aug(img, func, name, suf, out_dir):
    out = func(img)
    path = os.path.join(out_dir,f"{name}_{suf}.jpg")
    cv2.imwrite(path, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    return path

# ──────────────────────────────────
# Sidebar options
# ──────────────────────────────────
st.title("🧪 Custom Image Augmentation Tool")

up_files   = st.file_uploader("Upload images / zip", accept_multiple_files=True)
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

# ──────────────────────────────────
# Sample Image Selection + Previews
# ──────────────────────────────────
st.sidebar.subheader("🔍 Select Sample Image")
sample_files = glob.glob(os.path.join("Sample", "*.jpg"))
sample_dict = {os.path.basename(f): f for f in sample_files}
if sample_files:
    sample_name = st.sidebar.selectbox("Choose image", list(sample_dict.keys()))
    preview_img = cv2.cvtColor(cv2.imread(sample_dict[sample_name]), cv2.COLOR_BGR2RGB)
    st.sidebar.image(preview_img, caption="Sample Preview", use_container_width=True)
else:
    preview_img = None
    st.sidebar.warning("No sample image found in the 'Sample' folder.")

# ──────────────────────────────────
# Live Preview Section
# ──────────────────────────────────
st.markdown("---")
st.header("🔍 Live Preview (Sample Image)")

if preview_img is not None:
    st.sidebar.subheader("Preview Settings")
    tint_preview = st.sidebar.slider("Tint Opacity", 0.0, 1.0, 0.25, step=0.05)
    shadow_strength = st.sidebar.slider("Shadow Strength", 0.0, 1.0, 0.45, step=0.05)
    reflection_intensity = st.sidebar.slider("Reflection Intensity", 0.0, 1.0, 0.12, step=0.02)
    blur_strength = st.sidebar.slider("Blur Kernel (odd)", 1, 15, 7, step=2)

    # Collect overlays from sidebar checkboxes
    overlay_imgs=[]
    OV_DIR="overlays"
    for p in glob.glob(f"{OV_DIR}/*.*"):
        lbl=os.path.splitext(os.path.basename(p))[0]
        col1,col2=st.sidebar.columns([1,4])
        with col1: st.image(p,use_container_width=True)
        with col2:
            if st.checkbox(lbl,key=f"ov_{lbl}"):
                img=cv2.imread(p,cv2.IMREAD_UNCHANGED); overlay_imgs.append((lbl,img))

    for f in ov_uploads:
        data=np.frombuffer(f.read(),np.uint8)
        img=cv2.imdecode(data,cv2.IMREAD_UNCHANGED)
        overlay_imgs.append((os.path.splitext(f.name)[0],img))

    # Build preview image
    img_prev = preview_img.copy()
    if tint_opts:
        img_prev = apply_tint(img_prev, tint_vals[tint_opts[0]], tint_preview)
    if "Shadow" in augmentations:
        img_prev = apply_shadow(img_prev, shadow_strength)
    if "Reflection" in augmentations:
        img_prev = apply_glass_reflection(img_prev, reflection_intensity)
    if "Blur" in augmentations:
        img_prev = apply_gaussian_blur(img_prev, blur_strength)
    if overlay_imgs:
        img_prev = apply_overlay(img_prev, overlay_imgs[0][1])

    st.image(img_prev, caption="Live Preview", width=400)
else:
    st.warning("No preview image available.")

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
                        img_bt = img_b if t=="original" else apply_tint(img_b, tint_vals[t])
                        for ov_name,ov_img in (overlay_imgs or [("orig",None)]):
                            img_bto = img_bt if ov_img is None else apply_overlay(img_bt,ov_img)

                            suffix="_".join([s for s in [b,t] if s!="original"])
                            if ov_img is not None: suffix += f"_ov_{ov_name}"
                            suffix = suffix or "original"

                            if augmentations:
                                for aug in augmentations:
                                    func={
                                        "Shadow":apply_shadow,
                                        "Reflection":apply_glass_reflection,
                                        "Blur":apply_gaussian_blur,
                                        "Occlusion":apply_random_occlusion,
                                        "Perspective":apply_perspective_transform
                                    }[aug]
                                    path = save_aug(img_bto,func,base,f"{suffix}_{aug.lower()}",out_dir)
                                    output_files.append(path)
                            else:
                                path = os.path.join(out_dir,f"{base}_{suffix}.jpg")
                                cv2.imwrite(path,cv2.cvtColor(img_bto,cv2.COLOR_RGB2BGR))
                                output_files.append(path)

            st.success("✅ Done")

            if len(output_files) < 5:
                for path in output_files:
                    fname = os.path.basename(path)
                    with open(path, "rb") as f:
                        img_bytes = f.read()
                        st.image(img_bytes, caption=fname, use_container_width=True)
                        st.download_button("Download "+fname, img_bytes, file_name=fname)
            else:
                buf=BytesIO()
                with zipfile.ZipFile(buf,"w") as z:
                    for f in output_files:
                        z.write(f, os.path.basename(f))
                buf.seek(0)
                st.download_button("Download ZIP",buf.getvalue(),"augmented_images.zip","application/zip")
else:
    if up_files:
        st.warning("Select at least one transformation or overlay.")
