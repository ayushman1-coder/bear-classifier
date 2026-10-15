from pathlib import Path

import gradio as gr
from fastai.vision.all import load_learner

# Put export.pkl in the same Hugging Face Space repository as this app.py.
APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "export.pkl"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Model file 'export.pkl' was not found. "
        "Run learn.export() in your training notebook and upload export.pkl "
        "to the same Hugging Face Space as app.py."
    )

# Load on CPU so the Space does not require a GPU.
learn = load_learner(MODEL_PATH, cpu=True)


def classify_image(img):
    """Return probabilities using the class order saved in the trained model."""
    if img is None:
        return {}

    _pred, _idx, probs = learn.predict(img)
    return {
        str(learn.dls.vocab[i]): float(probs[i])
        for i in range(len(probs))
    }


# Include example images only if you upload the local bear/ dataset folder.
# Example folder structure: bear/<class-name>/<image-file>
bear_dir = APP_DIR / "bear"
examples = []
if bear_dir.exists():
    for class_dir in sorted(p for p in bear_dir.iterdir() if p.is_dir()):
        for image_path in sorted(class_dir.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in {
                ".jpg", ".jpeg", ".png", ".webp", ".bmp"
            }:
                examples.append([str(image_path)])
            if len(examples) >= 6:
                break
        if len(examples) >= 6:
            break

demo = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil", label="Upload a bear image"),
    outputs=gr.Label(num_top_classes=3, label="Prediction probabilities"),
    examples=examples or None,
    title="Bear Image Classifier",
    description=(
        "Upload an image to classify it using a fast.ai ResNet-18 model. "
        "The class names and their order come from the exported model."
    ),
)

if __name__ == "__main__":
    demo.launch()
