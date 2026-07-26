from __future__ import annotations

import os
import html
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from model_utils import (
    DEFAULT_BUNDLE_PATH,
    DEFAULT_SEGMENTED_ROOTS,
    find_sam3_segmented_image_simple,
    format_text_report,
    load_caption_system,
    run_caption_pipeline,
    save_uploaded_image,
)

APP_DIR = Path(__file__).resolve().parent
SAMPLE_DIR = APP_DIR / "sample_images"


st.set_page_config(
    page_title="Pakaian Adat Captioning",
    page_icon="ID",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def cached_caption_system(bundle_path: str, device: str):
    return load_caption_system(bundle_path=bundle_path, device=device)


def list_sample_images():
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    if not SAMPLE_DIR.exists():
        return []
    return sorted([p for p in SAMPLE_DIR.iterdir() if p.suffix.lower() in exts])


def render_image_pair(original_path: str, segmented_path: str | None):
    if segmented_path:
        left, right = st.columns(2)
        with left:
            st.image(Image.open(original_path).convert("RGB"), caption="Original Image", use_container_width=True)
        with right:
            st.image(Image.open(segmented_path).convert("RGB"), caption="SAM3 Segmented", use_container_width=True)
        st.caption(f"SAM3 segmented path: {segmented_path}")
    else:
        st.image(Image.open(original_path).convert("RGB"), caption="Original Image", width=420)
        st.info("SAM3 segmented image tidak ditemukan untuk nama gambar ini.")


def render_wrapped_table(df: pd.DataFrame, columns: list[str]):
    table_df = df[columns].copy()

    def format_value(value):
        if isinstance(value, float):
            return f"{value:.4f}"
        return html.escape(str(value))

    headers = "".join(f"<th>{html.escape(col)}</th>" for col in table_df.columns)
    rows = []
    for _, row in table_df.iterrows():
        cells = "".join(f"<td>{format_value(row[col])}</td>" for col in table_df.columns)
        rows.append(f"<tr>{cells}</tr>")

    st.markdown(
        """
        <style>
        .wrapped-table {
            border-collapse: collapse;
            width: 100%;
            table-layout: fixed;
            font-size: 0.92rem;
        }
        .wrapped-table th,
        .wrapped-table td {
            border: 1px solid rgba(49, 51, 63, 0.18);
            padding: 0.65rem 0.75rem;
            text-align: left;
            vertical-align: top;
            white-space: normal;
            overflow-wrap: anywhere;
            word-break: normal;
        }
        .wrapped-table th {
            background: rgba(49, 51, 63, 0.06);
            font-weight: 700;
        }
        .wrapped-table th:nth-child(1),
        .wrapped-table td:nth-child(1) {
            width: 4.5rem;
            text-align: center;
        }
        .wrapped-table th:nth-child(2),
        .wrapped-table td:nth-child(2) {
            width: 58%;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<table class='wrapped-table'><thead><tr>{headers}</tr></thead><tbody>{''.join(rows)}</tbody></table>",
        unsafe_allow_html=True,
    )


def render_results(result: dict):
    st.subheader("Generated Caption")
    st.write(result["main_caption"])

    st.subheader("Enhanced Caption")
    st.write(result["enhanced_caption"])

    st.subheader("Zero-Shot Retrieval")
    zero_df = pd.DataFrame(result["zero_shot_results"])
    render_wrapped_table(zero_df, ["rank", "caption", "raw_cosine_sim", "normalized"])

    st.subheader("Late Fusion")
    fusion_df = pd.DataFrame(result["fusion_results"])
    st.caption("Formula: Final Score = (CLIP x 0.4) + (BLEU4 x 0.3) + (METEOR x 0.3)")
    render_wrapped_table(
        fusion_df,
        ["rank", "caption", "clip_score", "bleu4_score", "meteor_score", "final_score"],
    )

    st.subheader("Best Caption Selected")
    st.success(result["best_caption"]["caption"])

    st.subheader("Final Evaluation Metrics")
    m1, m2, m3 = st.columns(3)
    m1.metric("clip score", f"{result['final_metrics']['clip_score']:.4f}")
    m2.metric("bleu4-score", f"{result['final_metrics']['bleu4_score']:.4f}")
    m3.metric("meteor", f"{result['final_metrics']['meteor']:.4f}")

    with st.expander("Console-style report"):
        st.code(format_text_report(result), language="text")


st.title("Pakaian Adat Captioning")
st.caption("Dataset-backed captioning, zero-shot retrieval, late fusion, and SAM3 segmented display.")

with st.sidebar:
    st.header("Runtime")
    bundle_path = st.text_input("Model bundle", value=str(DEFAULT_BUNDLE_PATH))
    device = st.selectbox("Device", options=["cpu"], index=0)
    show_segmented = st.toggle("Tampilkan SAM3 segmented", value=True)

    extra_segmented_root = st.text_input(
        "Folder segmentasi tambahan",
        value=os.environ.get("SEGMENTED_ROOT", ""),
        placeholder="opsional",
    )

    st.divider()
    st.write("Sample image")
    samples = list_sample_images()
    sample_labels = ["Tidak pakai sample"] + [p.name for p in samples]
    sample_choice = st.selectbox("Pilih sample", options=sample_labels)

uploaded_file = st.file_uploader("Upload gambar pakaian adat", type=["jpg", "jpeg", "png", "webp"])
run_button = st.button("Run Captioning", type="primary", use_container_width=True)

selected_image_path = None
if uploaded_file is not None:
    selected_image_path = save_uploaded_image(uploaded_file)
elif sample_choice != "Tidak pakai sample" and samples:
    selected_image_path = str(samples[sample_labels.index(sample_choice) - 1])

if selected_image_path:
    st.subheader("Input Image")
    segmented_roots = list(DEFAULT_SEGMENTED_ROOTS)
    if extra_segmented_root.strip():
        segmented_roots.insert(0, Path(extra_segmented_root.strip()))
    segmented_path = find_sam3_segmented_image_simple(selected_image_path, segmented_roots) if show_segmented else None
    render_image_pair(selected_image_path, segmented_path)
else:
    st.info("Upload gambar atau pilih sample, lalu tekan Run Captioning.")

if run_button:
    if not selected_image_path:
        st.warning("Pilih gambar terlebih dahulu.")
        st.stop()

    if not Path(bundle_path).exists():
        st.error("Model bundle belum ditemukan. Jalankan scripts/build_model_bundle.py terlebih dahulu.")
        st.code("python scripts/build_model_bundle.py", language="bash")
        st.stop()

    with st.spinner("Loading model dan menjalankan captioning..."):
        system = cached_caption_system(bundle_path, device)
        result = run_caption_pipeline(system, selected_image_path)
        if show_segmented:
            result["segmented_image_path"] = segmented_path

    render_results(result)
