from html import escape
from io import BytesIO
from pathlib import Path
import tempfile

from PIL import Image
import streamlit as st

from plant_classifier.model_store import (
    bundle_summary,
    list_model_bundles,
    load_model_bundle,
    predict_image,
    preferred_model_path,
)


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"


st.set_page_config(
    page_title="LeafLens — Plant Disease Classifier",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.markdown(
    """
    <style>
      :root {
        --leaf-ink: #14251d;
        --leaf-muted: #6d7d74;
        --leaf-line: #dce5df;
        --leaf-canvas: #f5f7f3;
        --leaf-forest: #123d2b;
        --leaf-green: #1d6847;
        --leaf-lime: #c9e66d;
        --leaf-soft: #eef4ef;
      }

      .stApp {
        background:
          radial-gradient(circle at 8% 1%, rgba(201, 230, 109, 0.12), transparent 23rem),
          radial-gradient(circle at 95% 35%, rgba(29, 104, 71, 0.07), transparent 27rem),
          var(--leaf-canvas);
        color: var(--leaf-ink);
      }

      [data-testid="stHeader"] {
        background: transparent;
      }

      [data-testid="stMainBlockContainer"] {
        max-width: 1240px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
      }

      #MainMenu,
      footer {
        visibility: hidden;
      }

      h1, h2, h3 {
        color: var(--leaf-ink);
        letter-spacing: -0.035em;
      }

      .leaf-header {
        min-height: 66px;
        margin-bottom: 3.8rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        border-bottom: 1px solid rgba(200, 213, 205, 0.8);
      }

      .leaf-brand {
        display: flex;
        align-items: center;
        gap: 11px;
        color: var(--leaf-ink);
        font-size: 18px;
        font-weight: 760;
      }

      .leaf-brand-mark {
        width: 33px;
        height: 33px;
        display: grid;
        place-items: center;
        border-radius: 10px;
        background: var(--leaf-forest);
        color: var(--leaf-lime);
        font-size: 18px;
      }

      .leaf-system {
        display: flex;
        align-items: center;
        gap: 9px;
        color: var(--leaf-muted);
        font-size: 12px;
        font-weight: 650;
      }

      .leaf-system-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #48a878;
        box-shadow: 0 0 0 4px rgba(72, 168, 120, 0.13);
      }

      .leaf-hero {
        margin-bottom: 2.5rem;
        display: grid;
        grid-template-columns: minmax(0, 1.55fr) minmax(280px, .65fr);
        gap: 4rem;
        align-items: end;
      }

      .leaf-eyebrow {
        margin: 0 0 .65rem;
        color: var(--leaf-green);
        font-size: 11px;
        font-weight: 800;
        letter-spacing: .13em;
        text-transform: uppercase;
      }

      .leaf-hero h1 {
        max-width: 760px;
        margin: 0;
        font-size: clamp(42px, 5.6vw, 74px);
        font-weight: 690;
        line-height: 1;
        letter-spacing: -.055em;
      }

      .leaf-hero-copy {
        margin: 0;
        color: #34483e;
        font-size: 16px;
        line-height: 1.7;
      }

      .leaf-steps {
        margin-bottom: 1rem;
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        overflow: hidden;
        border: 1px solid var(--leaf-line);
        border-radius: 13px;
        background: rgba(255,255,255,.6);
      }

      .leaf-step {
        padding: 1rem 1.25rem;
        color: var(--leaf-muted);
        font-size: 11px;
        font-weight: 700;
      }

      .leaf-step + .leaf-step {
        border-left: 1px solid var(--leaf-line);
      }

      .leaf-step strong {
        margin-right: .6rem;
        color: #95a39b;
        font-size: 9px;
        letter-spacing: .1em;
      }

      .leaf-step.active {
        color: var(--leaf-forest);
        background: var(--leaf-soft);
        box-shadow: inset 0 -2px var(--leaf-green);
      }

      [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: var(--leaf-line);
        border-radius: 15px;
        background: rgba(255,255,255,.92);
        box-shadow: 0 18px 50px rgba(20, 47, 34, .07);
      }

      [data-testid="stFileUploaderDropzone"] {
        min-height: 240px;
        padding: 2rem;
        border: 1.5px dashed #a9bdb0;
        border-radius: 12px;
        background:
          linear-gradient(rgba(255,255,255,.72), rgba(255,255,255,.72)),
          repeating-linear-gradient(45deg, #edf3ee 0, #edf3ee 1px, transparent 1px, transparent 14px);
      }

      [data-testid="stFileUploaderDropzoneInstructions"] span {
        color: var(--leaf-ink);
        font-weight: 700;
      }

      .stButton > button,
      [data-testid="stFileUploaderDropzone"] button {
        border-color: var(--leaf-forest);
        border-radius: 9px;
        background: var(--leaf-forest);
        color: white;
        font-weight: 700;
      }

      .stButton > button:hover,
      [data-testid="stFileUploaderDropzone"] button:hover {
        border-color: #0b2f20;
        background: #0b2f20;
        color: white;
      }

      .stSelectbox label,
      .stFileUploader label,
      [data-testid="stCameraInput"] label {
        color: var(--leaf-green);
        font-size: 11px;
        font-weight: 800;
        letter-spacing: .1em;
        text-transform: uppercase;
      }

      .leaf-section-title {
        margin: 0 0 1rem;
      }

      .leaf-section-title span {
        display: block;
        margin-bottom: .35rem;
        color: var(--leaf-green);
        font-size: 10px;
        font-weight: 800;
        letter-spacing: .12em;
        text-transform: uppercase;
      }

      .leaf-section-title h2 {
        margin: 0;
        font-size: 22px;
      }

      .leaf-model-note {
        margin-top: .75rem;
        color: var(--leaf-muted);
        font-size: 11px;
      }

      .leaf-empty {
        min-height: 430px;
        display: grid;
        place-content: center;
        justify-items: center;
        padding: 2rem;
        text-align: center;
      }

      .leaf-empty-icon {
        width: 94px;
        height: 94px;
        margin-bottom: 1.25rem;
        display: grid;
        place-items: center;
        border: 1px solid #d8e4dc;
        border-radius: 50%;
        background: linear-gradient(145deg, #f8fbf8, #eaf2ec);
        color: var(--leaf-green);
        font-size: 37px;
      }

      .leaf-empty h3 {
        margin: 0 0 .5rem;
        font-size: 20px;
      }

      .leaf-empty p {
        max-width: 300px;
        margin: 0;
        color: var(--leaf-muted);
        font-size: 13px;
        line-height: 1.6;
      }

      .leaf-status {
        display: inline-flex;
        margin-bottom: 1rem;
        padding: .35rem .6rem;
        border-radius: 999px;
        font-size: 9px;
        font-weight: 800;
        letter-spacing: .08em;
        text-transform: uppercase;
      }

      .leaf-status.healthy {
        background: #e2f2e8;
        color: #206643;
      }

      .leaf-status.disease {
        background: #f8eae7;
        color: #a54135;
      }

      .leaf-diagnosis-plant {
        margin-bottom: .5rem;
        color: var(--leaf-green);
        font-size: 11px;
        font-weight: 800;
        letter-spacing: .11em;
        text-transform: uppercase;
      }

      .leaf-diagnosis-title {
        margin: 0 0 .7rem;
        font-size: clamp(32px, 4vw, 44px);
        line-height: 1;
      }

      .leaf-summary {
        margin-bottom: 1.5rem;
        color: var(--leaf-muted);
        font-size: 13px;
        line-height: 1.6;
      }

      .leaf-confidence {
        padding-top: 1rem;
        border-top: 1px solid var(--leaf-line);
        color: #34483e;
        font-size: 12px;
        font-weight: 700;
      }

      .leaf-match {
        padding: .5rem 0 .25rem;
        display: flex;
        justify-content: space-between;
        gap: 1rem;
        color: #34483e;
        font-size: 11px;
      }

      .leaf-recommendation {
        margin-top: 1rem;
        padding: .9rem;
        border: 1px solid #d9e5dc;
        border-radius: 10px;
        background: var(--leaf-soft);
        color: var(--leaf-muted);
        font-size: 11px;
        line-height: 1.55;
      }

      .leaf-recommendation strong {
        display: block;
        margin-bottom: .2rem;
        color: var(--leaf-ink);
      }

      .leaf-crops {
        margin-top: 1rem;
        padding: 1rem .2rem;
        display: flex;
        align-items: center;
        gap: .6rem;
        flex-wrap: wrap;
        color: var(--leaf-muted);
        font-size: 10px;
      }

      .leaf-crops span {
        padding: .35rem .65rem;
        border: 1px solid #c8d5cd;
        border-radius: 999px;
        background: rgba(255,255,255,.55);
        color: #34483e;
        font-weight: 700;
      }

      .leaf-crops em {
        margin-left: auto;
        font-style: normal;
      }

      @media (max-width: 760px) {
        .leaf-header {
          margin-bottom: 2.5rem;
        }

        .leaf-system {
          font-size: 0;
        }

        .leaf-hero {
          grid-template-columns: 1fr;
          gap: 1.25rem;
        }

        .leaf-hero h1 {
          font-size: clamp(42px, 12vw, 58px);
        }

        .leaf-step {
          padding: .8rem .4rem;
          text-align: center;
          font-size: 9px;
        }

        .leaf-step strong {
          display: none;
        }

        .leaf-crops em {
          width: 100%;
          margin-left: 0;
        }
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_model_catalog():
    model_paths = list_model_bundles(MODEL_DIR)
    summaries = {path.name: bundle_summary(path) for path in model_paths}
    preferred = preferred_model_path(model_paths)
    return summaries, preferred.name if preferred else None


@st.cache_resource(show_spinner=False)
def get_model(model_file):
    return load_model_bundle(MODEL_DIR / model_file)


@st.cache_data(show_spinner=False)
def analyze_image(image_bytes, model_file):
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temporary:
        temporary.write(image_bytes)
        temporary_path = Path(temporary.name)
    try:
        return predict_image(get_model(model_file), temporary_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def friendly_variant(value):
    variants = {
        "camera": "Camera optimized",
        "camera_fast": "Camera optimized",
        "camera_robust": "Camera optimized",
        "fast": "Fast analysis",
        "full": "Detailed analysis",
        "regular": "Detailed analysis",
    }
    return variants.get(str(value).lower(), str(value).title())


def split_diagnosis(label):
    plant, separator, condition = str(label).partition(" - ")
    return plant, condition if separator else plant


st.markdown(
    """
    <div class="leaf-header">
      <div class="leaf-brand">
        <span class="leaf-brand-mark">◒</span>
        <span>LeafLens</span>
      </div>
      <div class="leaf-system">
        <span class="leaf-system-dot"></span>
        Analysis system ready
      </div>
    </div>
    <section class="leaf-hero">
      <div>
        <p class="leaf-eyebrow">Computer vision for plant care</p>
        <h1>Understand what your leaf is telling you.</h1>
      </div>
      <p class="leaf-hero-copy">
        Upload a clear leaf photo for an instant screening across peach,
        bell pepper, and strawberry plants.
      </p>
    </section>
    <div class="leaf-steps">
      <div class="leaf-step active"><strong>01</strong>Add a leaf photo</div>
      <div class="leaf-step"><strong>02</strong>AI image analysis</div>
      <div class="leaf-step"><strong>03</strong>Review the result</div>
    </div>
    """,
    unsafe_allow_html=True,
)


catalog, default_model = get_model_catalog()

if not catalog:
    st.error("No trained model bundles were found in the models folder.")
    st.stop()

model_files = list(catalog)
default_index = model_files.index(default_model) if default_model in model_files else 0

input_column, result_column = st.columns([1.55, 0.85], gap="medium")

with input_column:
    with st.container(border=True):
        st.markdown(
            """
            <div class="leaf-section-title">
              <span>Leaf image</span>
              <h2>Upload or take a photo</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )

        selected_model = st.selectbox(
            "Analysis model",
            options=model_files,
            index=default_index,
            format_func=lambda model_file: (
                f"{catalog[model_file]['name']} · "
                f"{friendly_variant(catalog[model_file]['variant'])}"
            ),
        )

        selected_summary = catalog[selected_model]
        f1_score = selected_summary["metrics"].get("f1_weighted")
        score_text = f" · validation F1 {f1_score:.0%}" if isinstance(f1_score, (int, float)) else ""
        st.markdown(
            (
                f"<p class='leaf-model-note'>{friendly_variant(selected_summary['variant'])}"
                f" · {selected_summary['feature_count']} features{score_text}</p>"
            ),
            unsafe_allow_html=True,
        )

        upload_tab, camera_tab = st.tabs(["Upload photo", "Use camera"])
        with upload_tab:
            uploaded_file = st.file_uploader(
                "Leaf photo",
                type=["jpg", "jpeg", "png", "webp"],
                help="Use one clear, well-lit leaf that fills most of the frame.",
            )
        with camera_tab:
            camera_file = st.camera_input(
                "Take a leaf photo",
                help="Position one leaf in the center of the frame.",
            )

        source_file = uploaded_file or camera_file
        image_bytes = source_file.getvalue() if source_file else None

        if image_bytes:
            try:
                preview = Image.open(BytesIO(image_bytes)).convert("RGB")
                st.image(preview, caption="Selected leaf image", use_container_width=True)
            except Exception:
                st.error("The selected image could not be read. Please try another file.")
                image_bytes = None

        st.caption(
            "For a reliable screening, use one leaf, fill most of the frame, "
            "and avoid harsh shadows. Your image is processed only for this analysis."
        )

with result_column:
    with st.container(border=True):
        st.markdown(
            """
            <div class="leaf-section-title">
              <span>Screening result</span>
              <h2>Leaf assessment</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if not image_bytes:
            st.markdown(
                """
                <div class="leaf-empty">
                  <div class="leaf-empty-icon">⌁</div>
                  <h3>Ready when you are</h3>
                  <p>
                    Add a leaf photo to see the predicted condition,
                    confidence score, and closest matches.
                  </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            with st.spinner("Examining leaf patterns…"):
                try:
                    prediction = analyze_image(image_bytes, selected_model)
                except Exception as error:
                    st.error(f"The image could not be analyzed: {error}")
                    st.stop()

            plant, condition = split_diagnosis(prediction["label"])
            is_healthy = "healthy" in prediction["label"].lower()
            status_class = "healthy" if is_healthy else "disease"
            status_text = "Healthy pattern" if is_healthy else "Disease pattern"
            summary_text = (
                "The visible patterns most closely match a healthy leaf in the trained dataset."
                if is_healthy
                else (
                    "The visible patterns most closely match "
                    f"{condition.lower()} in the trained dataset."
                )
            )

            st.markdown(
                f"""
                <span class="leaf-status {status_class}">{status_text}</span>
                <div class="leaf-diagnosis-plant">{escape(plant)}</div>
                <h3 class="leaf-diagnosis-title">{escape(condition)}</h3>
                <p class="leaf-summary">{escape(summary_text)}</p>
                <div class="leaf-confidence">Model confidence</div>
                """,
                unsafe_allow_html=True,
            )

            confidence = float(prediction.get("confidence") or 0)
            confidence_column, score_column = st.columns([3, 1])
            with confidence_column:
                st.progress(confidence)
            with score_column:
                st.metric("Confidence", f"{confidence:.0%}", label_visibility="collapsed")

            st.markdown(
                "<p class='leaf-eyebrow' style='margin-top:1.4rem'>Top matches</p>",
                unsafe_allow_html=True,
            )
            for item in prediction.get("top_predictions", []):
                item_score = float(item["confidence"])
                st.markdown(
                    (
                        "<div class='leaf-match'>"
                        f"<strong>{escape(item['label'])}</strong>"
                        f"<span>{item_score:.0%}</span>"
                        "</div>"
                    ),
                    unsafe_allow_html=True,
                )
                st.progress(item_score)

            recommendation_title = "Keep monitoring" if is_healthy else "Inspect and isolate"
            recommendation_text = (
                "Continue routine care and screen again if new spots, discoloration, or wilting appear."
                if is_healthy
                else (
                    "Separate affected foliage where practical and confirm the result "
                    "with a local plant-care expert before treatment."
                )
            )
            st.markdown(
                (
                    "<div class='leaf-recommendation'>"
                    f"<strong>{recommendation_title}</strong>"
                    f"{recommendation_text}"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )

st.markdown(
    """
    <div class="leaf-crops">
      Currently trained to screen
      <span>Peach</span>
      <span>Bell pepper</span>
      <span>Strawberry</span>
      <em>Screening support only — not a substitute for professional plant pathology advice.</em>
    </div>
    """,
    unsafe_allow_html=True,
)
