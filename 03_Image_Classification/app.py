from pathlib import Path

import gradio as gr
import torch
from PIL import Image, ImageOps
from torchvision import models, transforms


ROOT = Path(__file__).parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_OPTIONS = {
    "Fine-tuned + augmentation": "Oxford_pets_finetune_augment.pth",
    "Fine-tuned, no augmentation": "Oxford_pets_finetune_noaugment.pth",
    "From scratch + augmentation": "Oxford_pets_scratch_augment.pth",
    "From scratch, no augmentation": "Oxford_pets_scratch_noaugment.pth",
}

BREED_NAMES = [
    "Abyssinian",
    "Bengal",
    "Birman",
    "Bombay",
    "British Shorthair",
    "Egyptian Mau",
    "Maine Coon",
    "Persian",
    "Ragdoll",
    "Russian Blue",
    "Siamese",
    "Sphynx",
    "American Bulldog",
    "American Pit Bull Terrier",
    "Basset Hound",
    "Beagle",
    "Boxer",
    "Chihuahua",
    "English Cocker Spaniel",
    "English Setter",
    "German Shorthaired",
    "Great Pyrenees",
    "Havanese",
    "Japanese Chin",
    "Keeshond",
    "Leonberger",
    "Miniature Pinscher",
    "Newfoundland",
    "Pomeranian",
    "Pug",
    "Saint Bernard",
    "Samoyed",
    "Scottish Terrier",
    "Shiba Inu",
    "Staffordshire Bull Terrier",
    "Wheaten Terrier",
    "Yorkshire Terrier",
]


def _build_model(checkpoint):
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(checkpoint["class_names"]))
    model.load_state_dict(checkpoint["model_state"])
    return model.to(DEVICE).eval()


@torch.inference_mode()
def predict(image, model_name):
    if image is None:
        raise gr.Error("Upload a cat or dog image first.")

    path = ROOT / MODEL_OPTIONS[model_name]
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = _build_model(checkpoint)
    transform = transforms.Compose(
        [
            transforms.Resize((checkpoint["img_size"], checkpoint["img_size"])),
            transforms.ToTensor(),
            transforms.Normalize(checkpoint["norm_mean"], checkpoint["norm_std"]),
        ]
    )

    image = ImageOps.exif_transpose(image).convert("RGB")
    scores = torch.softmax(model(transform(image).unsqueeze(0).to(DEVICE)), dim=1)[0]
    top_scores, top_indices = scores.topk(5)
    return {
        checkpoint["class_names"][index.item()].replace("_", " ").title(): float(score)
        for score, index in zip(top_scores, top_indices)
    }


def model_details(model_name):
    checkpoint = torch.load(ROOT / MODEL_OPTIONS[model_name], map_location="cpu", weights_only=True)
    accuracy = checkpoint.get("val_accuracy")
    accuracy_text = f"{accuracy:.1%} validation accuracy" if accuracy is not None else "validation accuracy unavailable"
    return f"**{len(checkpoint['class_names'])} breeds**  ·  {accuracy_text}  ·  {checkpoint['arch']}  ·  epoch {checkpoint.get('epoch', '?')}"


with gr.Blocks(
    theme=gr.themes.Base(
        primary_hue="orange",
        secondary_hue="slate",
        neutral_hue="stone",
        font=["DM Sans", "ui-sans-serif", "sans-serif"],
    ),
    css="""
    .gradio-container { max-width: 1120px !important; margin: auto; }
    .hero { padding: 18px 0 8px; }
    .hero h1 { font-size: 2.6rem; letter-spacing: -0.04em; margin-bottom: 0.25rem; }
    .hero p { color: #64748b; font-size: 1.05rem; }
    .model-panel { border: 1px solid #e2e8f0; border-radius: 10px; padding: 18px; background: #fffaf4; }
    .result-panel { border: 1px solid #e2e8f0; border-radius: 10px; padding: 18px; }
    footer { display: none !important; }
    """,
) as demo:
    gr.Markdown(
        """
        <div class="hero">
        <h1>Oxford Pets · breed lens</h1>
        <p>Choose a trained ResNet18 checkpoint, then test it on a cat or dog image.</p>
        </div>
        """
    )

    with gr.Row(equal_height=False):
        with gr.Column(scale=1, elem_classes="model-panel"):
            model_selector = gr.Dropdown(
                choices=list(MODEL_OPTIONS),
                value="Fine-tuned + augmentation",
                label="Model checkpoint",
            )
            details = gr.Markdown(model_details("Fine-tuned + augmentation"))
            image = gr.Image(type="pil", label="Pet image", height=360)
            classify = gr.Button("Classify breed", variant="primary")

        with gr.Column(scale=1, elem_classes="result-panel"):
            gr.Markdown("### Predictions\nThe five highest-scoring breeds will appear here.")
            result = gr.Label(num_top_classes=5, label="Breed probabilities")

    with gr.Accordion("37 supported breeds", open=False):
        gr.Markdown("  ·  ".join(BREED_NAMES))

    model_selector.change(model_details, inputs=model_selector, outputs=details)
    classify.click(predict, inputs=[image, model_selector], outputs=result)
    image.change(predict, inputs=[image, model_selector], outputs=result)


if __name__ == "__main__":
    demo.launch()