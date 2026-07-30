# LeafLens — Plant Disease Classifier

LeafLens is a local computer-vision web application that screens leaf photos for common conditions in peach, bell pepper, and strawberry plants. It combines handcrafted image features with ensemble machine-learning models and presents the result in a clear, responsive diagnostic interface.

## Highlights

- Drag-and-drop, file picker, clipboard paste, and live camera input
- Automatic image preparation and local Flask inference
- Three model variants for camera, fast, and detailed analysis
- Ranked predictions with confidence scores
- Responsive, accessible interface with clear loading and error states
- Local processing with no external API or cloud upload
- Flask and Streamlit front ends backed by the same inference pipeline

## Supported classes

- Peach — Healthy / Bacterial spot
- Bell pepper — Healthy / Bacterial spot
- Strawberry — Healthy / Leaf scorch

## Project structure

```text
app.py                 Flask application and JSON endpoints
streamlit_app.py       Streamlit Community Cloud entry point
plant_classifier/      Image feature extraction and model inference
models/                Serialized trained model bundles
templates/             Application HTML
static/                Interface styles and browser interactions
scripts/               Model training utilities
tests/                 Feature, model, and API tests
*.ipynb                Experimentation and model-development notebooks
```

## Run locally

Create a Python 3.12 virtual environment and install the dependencies. The saved
models use pinned scikit-learn and XGBoost versions for reproducible inference:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python app.py
```

Then open [http://127.0.0.1:5000](http://127.0.0.1:5000).

To run the Streamlit edition:

```powershell
.\.venv\Scripts\streamlit run streamlit_app.py
```

## Run the test suite

```powershell
.\.venv\Scripts\python -m unittest discover -s tests -v
```

## How inference works

1. The browser captures or uploads an image.
2. The image is cropped to the analysis frame and sent to the local Flask server.
3. OpenCV extracts color, texture, leaf-mask, and spot features for the selected model variant.
4. The trained classifier returns a ranked prediction and confidence score.
5. The interface presents the screening result with cautious next-step guidance.

## Scope

LeafLens is an educational screening tool trained on six classes from three crops. Its output should not replace diagnosis by a plant pathologist or local agricultural professional.
