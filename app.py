"""
app.py — Perbandingan Inferensi: SAHI+YOLOv8s-seg vs Baseline YOLOv8s-seg
Deteksi Cacat Biji Kopi + SCA Grading
Jalankan: streamlit run app.py
"""

import os
import gc
import math
import tempfile
from collections import Counter
from pathlib import Path

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
import numpy as np
import streamlit as st
try:
    import torch
    import torchvision
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ── Coba import ultralytics & sahi ──────────────────────────────────────────
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="Coffee Defect Comparator",
    page_icon="☕",
    layout="wide",
)

# ── Force Light Mode ─────────────────────────────────────────
st.markdown("""
<style>
  /* Paksa background putih dan teks gelap di semua elemen utama */
  html, body, [data-testid="stAppViewContainer"],
  [data-testid="stMain"], [data-testid="block-container"],
  section[data-testid="stSidebar"] {
      background-color: #ffffff !important;
      color: #1a1a1a !important;
  }
  /* Sidebar */
  section[data-testid="stSidebar"] > div {
      background-color: #f5f5f5 !important;
  }
  /* Semua teks */
  p, span, label, div, h1, h2, h3, h4, h5, h6,
  .stMarkdown, .stText {
      color: #1a1a1a !important;
  }
  /* Input, selectbox, slider */
  .stTextInput input, .stSelectbox select {
      background-color: #ffffff !important;
      color: #1a1a1a !important;
  }
  /* Metric */
  [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
      color: #1a1a1a !important;
  }
  /* Expander */
  .streamlit-expanderHeader {
      color: #1a1a1a !important;
      background-color: #f0f0f0 !important;
  }
  /* Dataframe */
  .dataframe { color: #1a1a1a !important; }
  /* Divider */
  hr { border-color: #cccccc !important; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# KONSTANTA & ATURAN SCA
# ============================================================
SCA_RULES = {
    # Cacat Primer (Kategori 1)
    "full black"    : {"kategori": 1, "divisor": 1},
    "full sour"     : {"kategori": 1, "divisor": 1},
    "fungus"        : {"kategori": 1, "divisor": 1},
    "foreign matter": {"kategori": 1, "divisor": 1},
    "cherry pod"    : {"kategori": 1, "divisor": 1},
    "severe insect" : {"kategori": 1, "divisor": 5},  # FIXED: div=5
    # Cacat Sekunder (Kategori 2)
    "partial black" : {"kategori": 2, "divisor": 3},
    "partial sour"  : {"kategori": 2, "divisor": 3},
    "parchment"     : {"kategori": 2, "divisor": 5},
    "broken"        : {"kategori": 2, "divisor": 5},
    "withered"      : {"kategori": 2, "divisor": 5},
    "immature"      : {"kategori": 2, "divisor": 5},
    "hull"          : {"kategori": 2, "divisor": 5},
    "shell"         : {"kategori": 2, "divisor": 5},
    "floater"       : {"kategori": 2, "divisor": 5},
    "slight insect" : {"kategori": 2, "divisor": 10},
}

# Palet warna per class_id (BGR → dikonversi ke RGB untuk tampilan)
PALETTE = [
    (255, 56,  56),   # 0  merah
    (255, 157,  51),  # 1  oranye
    (255, 112, 255),  # 2  pink
    (0,   194, 255),  # 3  cyan
    (0,   255, 124),  # 4  hijau
    (183, 128, 255),  # 5  ungu
    (255, 215,   0),  # 6  emas
    ( 55, 126, 184),  # 7  biru
    (228,  26,  28),  # 8  merah tua
    ( 77, 175,  74),  # 9  hijau tua
    (152,  78, 163),  # 10 ungu tua
    (255, 127,   0),  # 11 oranye tua
]

def get_color(cls_id: int) -> tuple:
    """Return (R, G, B) untuk class id."""
    return PALETTE[cls_id % len(PALETTE)]


# ============================================================
# FUNGSI INFERENSI BASELINE
# ============================================================
@st.cache_resource
def load_model(model_path: str):
    """Load YOLO model (cached agar tidak reload setiap run)."""
    if not YOLO_AVAILABLE:
        return None
    return YOLO(model_path)


def run_baseline_inference(image_path: str, model, conf: float = 0.25, iou: float = 0.45, device: str = "cpu"):
    """Jalankan inferensi YOLOv8s-seg baseline biasa."""
    img_bgr = cv2.imread(image_path)
    with torch.no_grad():
        results = model(image_path, conf=conf, iou=iou, device=device, verbose=False)
    res = results[0]

    boxes, scores, cls_ids, masks = [], [], [], []
    if res.boxes is not None:
        for i, box in enumerate(res.boxes):
            boxes.append(list(map(float, box.xyxy[0])))
            scores.append(float(box.conf.item()))
            cls_ids.append(int(box.cls.item()))
            if res.masks is not None and i < len(res.masks.xy):
                masks.append(res.masks.xy[i].copy())
            else:
                masks.append(None)

    return {
        "boxes": boxes,
        "scores": scores,
        "cls_ids": cls_ids,
        "masks": masks,
        "img": img_bgr,
        "class_names": res.names,
    }


# ============================================================
# FUNGSI INFERENSI SAHI (MANUAL SLICING)
# ============================================================
def slice_image_for_infer(img: np.ndarray, slice_h: int, slice_w: int, overlap: float):
    H, W = img.shape[:2]
    step_x = max(1, int(slice_w * (1 - overlap)))
    step_y = max(1, int(slice_h * (1 - overlap)))
    slices = []
    for y0 in range(0, H, step_y):
        for x0 in range(0, W, step_x):
            x1 = min(x0 + slice_w, W)
            y1 = min(y0 + slice_h, H)
            patch = img[y0:y1, x0:x1]
            slices.append((patch, x0, y0))
    return slices


def run_sahi_inference(image_path: str, model,
                       slice_h: int = 640, slice_w: int = 640,
                       overlap: float = 0.2, conf: float = 0.25,
                       nms_iou: float = 0.5, device: str = "cpu"):
    """Jalankan inferensi SAHI-style sliced prediction + NMS."""
    img_orig = cv2.imread(image_path)
    H_orig, W_orig = img_orig.shape[:2]

    all_boxes, all_scores, all_cls_ids, all_masks = [], [], [], []
    last_names = {}

    slices = slice_image_for_infer(img_orig, slice_h, slice_w, overlap)

    for patch, x_off, y_off in slices:
        with torch.no_grad():
            results = model(patch, conf=conf, device=device, verbose=False)
        res = results[0]
        last_names = res.names

        if res.boxes is None or len(res.boxes) == 0:
            continue

        for i, box in enumerate(res.boxes):
            bx1, by1, bx2, by2 = map(float, box.xyxy[0])
            bx1 += x_off; bx2 += x_off
            by1 += y_off; by2 += y_off
            bx1 = max(0, min(bx1, W_orig)); bx2 = max(0, min(bx2, W_orig))
            by1 = max(0, min(by1, H_orig)); by2 = max(0, min(by2, H_orig))

            all_boxes.append([bx1, by1, bx2, by2])
            all_scores.append(float(box.conf.item()))
            all_cls_ids.append(int(box.cls.item()))

            if res.masks is not None and i < len(res.masks.xy):
                mask_pts = res.masks.xy[i].copy()
                mask_pts[:, 0] += x_off
                mask_pts[:, 1] += y_off
                all_masks.append(mask_pts)
            else:
                all_masks.append(None)

    if not all_boxes:
        return {"boxes": [], "scores": [], "cls_ids": [], "masks": [],
                "img": img_orig, "class_names": last_names}

    # NMS manual
    boxes_t  = torch.tensor(all_boxes,  dtype=torch.float32)
    scores_t = torch.tensor(all_scores, dtype=torch.float32)
    keep_idx = torchvision.ops.nms(boxes_t, scores_t, nms_iou).tolist()

    return {
        "boxes"      : [all_boxes[k]   for k in keep_idx],
        "scores"     : [all_scores[k]  for k in keep_idx],
        "cls_ids"    : [all_cls_ids[k] for k in keep_idx],
        "masks"      : [all_masks[k]   for k in keep_idx],
        "img"        : img_orig,
        "class_names": last_names,
    }


# ============================================================
# VISUALISASI
# ============================================================
def draw_detections(result: dict, title: str = "") -> np.ndarray:
    """Gambar BBox + Instance Mask di atas gambar asli. Return RGB numpy array."""
    img = result["img"].copy()
    overlay = img.copy()
    names = result["class_names"]

    for box, score, cls_id, mask_pts in zip(
            result["boxes"], result["scores"],
            result["cls_ids"], result["masks"]):

        r, g, b = get_color(cls_id)
        color_bgr = (b, g, r)

        # Isi mask
        if mask_pts is not None and len(mask_pts) > 2:
            pts = mask_pts.astype(np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [pts], color_bgr)

        # BBox
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), color_bgr, 2)

        # Label
        label = f"{names.get(cls_id, str(cls_id))} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(img, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color_bgr, -1)
        cv2.putText(img, label, (x1 + 2, max(th, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # Blend
    vis = cv2.addWeighted(overlay, 0.40, img, 0.60, 0)

    # Watermark judul
    if title:
        cv2.putText(vis, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 0, 0), 1, cv2.LINE_AA)

    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)


# ============================================================
# SCA GRADING
# ============================================================

def compute_sca_grade(result: dict):
    """Hitung defect counts, SCA TDC, dan grade dari hasil inferensi."""
    names = result["class_names"]
    cls_ids = result["cls_ids"]
    detected = [names.get(c, str(c)).lower().strip() for c in cls_ids]
    counts = Counter(detected)

    cat1_tdc = 0
    cat2_tdc = 0
    breakdown = []

    for defect, cnt in sorted(counts.items()):
        key = defect.lower().strip()
        if key in SCA_RULES:
            rule = SCA_RULES[key]
            tdc = math.floor(cnt / rule["divisor"])
            tipe = "Primer" if rule["kategori"] == 1 else "Sekunder"
            breakdown.append({
                "Kelas": defect,
                "Tipe Cacat": tipe,
                "Jumlah Biji": cnt,
                "Kategori": rule["kategori"],
                "Ekuivalens": rule["divisor"],
                "TDC": tdc,
            })
            if rule["kategori"] == 1:
                cat1_tdc += tdc
            else:
                cat2_tdc += tdc
        else:
            breakdown.append({
                "Kelas": defect,
                "Tipe Cacat": "?",
                "Jumlah Biji": cnt,
                "Kategori": "?",
                "Divisor": "-",
                "TDC": "-",
            })

    total_tdc = cat1_tdc + cat2_tdc

    is_specialty = (cat1_tdc == 0 and cat2_tdc <= 5)
    grade_label = "✅ SPECIALTY GRADE" if is_specialty else "❌ BELOW SPECIALTY"
    grade_color = "green" if is_specialty else "red"

    return {
        "counts"       : counts,
        "cat1_tdc"     : cat1_tdc,
        "cat2_tdc"     : cat2_tdc,
        "total_tdc"    : total_tdc,
        "is_specialty" : is_specialty,
        "grade_label"  : grade_label,
        "grade_color"  : grade_color,
        "breakdown"    : breakdown,
        "total_defects": len(cls_ids),
    }


def render_grade_card(grade_info: dict, model_name: str):
    """Render kartu SCA grading di Streamlit."""
    if grade_info['is_specialty']:
        bg_color     = "#d4edda"
        border_color = "#28a745"
        title_color  = "#155724"
    else:
        bg_color     = "#f8d7da"
        border_color = "#dc3545"
        title_color  = "#7b1a22"

    st.markdown(f"""
    <div style="
        background: {bg_color};
        border: 2px solid {border_color};
        border-radius: 10px; padding: 16px; margin-top: 8px;">
      <h4 style="margin:0; color: {title_color};">
        {grade_info['grade_label']}
      </h4>
      <p style="margin: 4px 0; font-size: 0.9em; color: #1a1a1a;">
        <b>Total Deteksi:</b> {grade_info['total_defects']} biji<br>
        <b>TDC Primer (Kat-1):</b> {grade_info['cat1_tdc']} &nbsp;|&nbsp;
        <b>TDC Sekunder (Kat-2):</b> {grade_info['cat2_tdc']} &nbsp;|&nbsp;
        <b>Total TDC:</b> {grade_info['total_tdc']}
      </p>
    </div>
    """, unsafe_allow_html=True)

    if grade_info["breakdown"]:
        import pandas as pd
        df = pd.DataFrame(grade_info["breakdown"])

        def highlight_tipe(row):
            if row["Tipe Cacat"] == "Primer":
                return ["background-color: #f8d7da; color: #7b1a22; font-weight: 600;"] * len(row)
            elif row["Tipe Cacat"] == "Sekunder":
                return ["background-color: #fff3cd; color: #6b5400; font-weight: 600;"] * len(row)
            else:
                return ["color: #1a1a1a;"] * len(row)

        styled_df = df.style.apply(highlight_tipe, axis=1)
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        st.caption("🔴 Merah muda = Cacat Primer (Kat-1) &nbsp;&nbsp; 🟡 Kuning = Cacat Sekunder (Kat-2)")


# ============================================================
# LEGEND WARNA
# ============================================================
def render_legend(class_names: dict):
    """Tampilkan legenda warna kelas."""
    if not class_names:
        return
    cols = st.columns(min(len(class_names), 4))
    for i, (cls_id, name) in enumerate(sorted(class_names.items())):
        r, g, b = get_color(cls_id)
        hex_color = f"#{r:02x}{g:02x}{b:02x}"
        with cols[i % len(cols)]:
            st.markdown(
                f'<span style="background:{hex_color};'
                f'border-radius:4px;padding:2px 10px;'
                f'color:white;font-size:0.8em;">{name}</span>',
                unsafe_allow_html=True
            )


# ============================================================
# MAIN APP
# ============================================================
def main():
    # ── Header ──────────────────────────────────────────────
    st.markdown("""
    <h1 style="text-align:center;">☕ Coffee Defect — Model Comparator</h1>
    <p style="text-align:center; color:#666;">
        SAHI + YOLOv8s-seg &nbsp;vs&nbsp; Baseline YOLOv8s-seg · SCA Grading
    </p>
    <hr>
    """, unsafe_allow_html=True)

    # ── Cek dependency wajib ────────────────────────────────
    missing = []
    if not CV2_AVAILABLE:
        missing.append("`opencv-python-headless`")
    if not TORCH_AVAILABLE:
        missing.append("`torch` dan `torchvision`")
    if not YOLO_AVAILABLE:
        missing.append("`ultralytics`")
    if missing:
        st.error(
            "❌ Library berikut belum terinstall: " + ", ".join(missing) + "\n\n"
            "Pastikan `requirements.txt` sudah ada di root repo dan berisi:\n"
            "```\nopencv-python-headless\nultralytics\ntorch\ntorchvision\n```"
        )
        st.stop()

    # ── Sidebar: konfigurasi ─────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Konfigurasi")

        st.subheader("📂 Path Model")
        baseline_path = st.text_input(
            "Baseline best.pt",
            value="coffee_baseline_v2/yolov8s_seg_baseline/weights/best.pt",
            help="Path ke model baseline YOLOv8s-seg"
        )
        sahi_path = st.text_input(
            "SAHI best.pt",
            value="sahi_checkpoint_backup/best.pt",
            help="Path ke model yang dilatih dengan SAFT v2"
        )

        st.divider()
        st.subheader("🔧 Parameter Inferensi")

        device_opt = st.selectbox("Device", ["cpu", "cuda:0"], index=0)
        conf_thr   = st.slider("Confidence threshold", 0.1, 0.9, 0.25, 0.05)
        iou_thr    = st.slider("NMS IoU threshold", 0.1, 0.9, 0.45, 0.05)

        st.divider()
        st.subheader("🔲 SAHI Slicing")
        slice_size = st.slider("Slice size (px)", 320, 1280, 640, 64)
        overlap    = st.slider("Overlap ratio", 0.0, 0.7, 0.2, 0.05)

        st.divider()
        run_btn = st.button("🚀 Jalankan Inferensi", type="primary", use_container_width=True)

    # ── Upload gambar ────────────────────────────────────────
    st.subheader("📷 Upload Gambar Biji Kopi")
    uploaded = st.file_uploader(
        "Pilih gambar (JPG / PNG)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    if not uploaded:
        st.info("⬆️ Upload minimal 1 gambar untuk memulai perbandingan.")
        st.stop()

    # ── Validasi model path ──────────────────────────────────
    baseline_ok = os.path.isfile(baseline_path)
    sahi_ok     = os.path.isfile(sahi_path)

    col_a, col_b = st.columns(2)
    with col_a:
        if baseline_ok:
            st.success(f"✅ Baseline model ditemukan")
        else:
            st.error(f"❌ Baseline tidak ditemukan: `{baseline_path}`")
    with col_b:
        if sahi_ok:
            st.success(f"✅ SAHI model ditemukan")
        else:
            st.error(f"❌ SAHI model tidak ditemukan: `{sahi_path}`")

    if not (baseline_ok and sahi_ok):
        st.warning("Pastikan kedua path model benar sebelum menjalankan inferensi.")
        st.stop()

    # ── Load model ────────────────────────────────────────────
    if run_btn:
        with st.spinner("📦 Memuat model..."):
            baseline_model = load_model(baseline_path)
            sahi_model     = load_model(sahi_path)

        # ── Loop per gambar ──────────────────────────────────
        for file_idx, uploaded_file in enumerate(uploaded):
            st.markdown(f"---\n### 🖼️ Gambar {file_idx+1}: `{uploaded_file.name}`")

            # Simpan ke file temp
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            # ── Inferensi ────────────────────────────────────
            with st.spinner("🔍 Menjalankan Baseline inference..."):
                baseline_result = run_baseline_inference(
                    tmp_path, baseline_model,
                    conf=conf_thr, iou=iou_thr,
                    device=device_opt
                )

            with st.spinner("🔍 Menjalankan SAHI inference..."):
                sahi_result = run_sahi_inference(
                    tmp_path, sahi_model,
                    slice_h=slice_size, slice_w=slice_size,
                    overlap=overlap, conf=conf_thr,
                    nms_iou=iou_thr, device=device_opt
                )

            # ── Visualisasi gambar (head-to-head) ────────────
            baseline_vis = draw_detections(baseline_result, "Baseline YOLOv8s-seg")
            sahi_vis     = draw_detections(sahi_result,     "SAHI + YOLOv8s-seg")

            col1, col2 = st.columns(2)
            with col1:
                st.image(baseline_vis, caption="🔵 Baseline YOLOv8s-seg",
                         use_container_width=True)
                n_base = len(baseline_result["boxes"])
                st.metric("Total deteksi", f"{n_base} cacat")
            with col2:
                st.image(sahi_vis, caption="🟢 SAHI + YOLOv8s-seg",
                         use_container_width=True)
                n_sahi = len(sahi_result["boxes"])
                st.metric("Total deteksi", f"{n_sahi} cacat",
                          delta=f"{n_sahi - n_base:+d} vs Baseline")

            # ── Legenda warna ────────────────────────────────
            all_names = {**baseline_result["class_names"], **sahi_result["class_names"]}
            if all_names:
                with st.expander("🎨 Legenda Kelas", expanded=False):
                    render_legend(all_names)

            # ── SCA Grading ──────────────────────────────────
            st.markdown("#### 📋 SCA Grading")
            baseline_grade = compute_sca_grade(baseline_result)
            sahi_grade     = compute_sca_grade(sahi_result)

            gcol1, gcol2 = st.columns(2)
            with gcol1:
                st.markdown("**🔵 Baseline**")
                render_grade_card(baseline_grade, "Baseline")
            with gcol2:
                st.markdown("**🟢 SAHI**")
                render_grade_card(sahi_grade, "SAHI")

            # ── Ringkasan Perbandingan Grade Head-to-Head ────
            st.markdown("#### ⚖️ Ringkasan Perbandingan Grade")
            import pandas as pd
            summary_rows = [
                {
                    "Model": "Baseline",
                    "TDC Primer": baseline_grade["cat1_tdc"],
                    "TDC Sekunder": baseline_grade["cat2_tdc"],
                    "Total TDC": baseline_grade["total_tdc"],
                    "Status Specialty": "✅ Ya" if baseline_grade["is_specialty"] else "❌ Tidak",
                },
                {
                    "Model": "SAHI",
                    "TDC Primer": sahi_grade["cat1_tdc"],
                    "TDC Sekunder": sahi_grade["cat2_tdc"],
                    "Total TDC": sahi_grade["total_tdc"],
                    "Status Specialty": "✅ Ya" if sahi_grade["is_specialty"] else "❌ Tidak",
                },
            ]
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

            # ── Ringkasan per-kelas ──────────────────────────
            with st.expander("📊 Perbandingan Jumlah per Kelas"):
                import pandas as pd
                all_classes = sorted(set(
                    list(baseline_grade["counts"].keys()) +
                    list(sahi_grade["counts"].keys())
                ))
                rows = []
                for cls in all_classes:
                    rows.append({
                        "Kelas": cls,
                        "Baseline": baseline_grade["counts"].get(cls, 0),
                        "SAHI": sahi_grade["counts"].get(cls, 0),
                        "Δ (SAHI−Base)": (
                            sahi_grade["counts"].get(cls, 0) -
                            baseline_grade["counts"].get(cls, 0)
                        ),
                    })
                if rows:
                    df_cmp = pd.DataFrame(rows)
                    st.dataframe(df_cmp, use_container_width=True, hide_index=True)

            # Cleanup
            os.unlink(tmp_path)
            del baseline_result, sahi_result, baseline_vis, sahi_vis
            gc.collect()

        # ── Selesai ──────────────────────────────────────────
        st.success("✅ Inferensi selesai untuk semua gambar!")

    else:
        # Preview gambar yang diupload sebelum run
        st.subheader("Preview Gambar")
        preview_cols = st.columns(min(len(uploaded), 4))
        for i, f in enumerate(uploaded):
            with preview_cols[i % 4]:
                st.image(f, caption=f.name, use_container_width=True)


if __name__ == "__main__":
    main()
