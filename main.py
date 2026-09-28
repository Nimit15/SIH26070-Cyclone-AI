
from pathlib import Path
from io import BytesIO
import base64

import numpy as np
import torch
import torch.nn as nn
import timm

from PIL import Image
from torchvision import transforms
from nicegui import ui


CATEGORIES = ["CS", "D", "DD", "SCS", "SuCS", "VSCS"]
SUCS_THRESHOLD = 117.5

PROJECT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT / "checkpoints" / "cyclone_best.pt"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


class CycloneModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=False,
            num_classes=0,
        )

        features = self.backbone.num_features

        self.classifier_head = nn.Linear(
            features,
            len(CATEGORIES),
        )

        self.regression_head = nn.Linear(
            features,
            1,
        )

    def forward(self, x):

        features = self.backbone(x)

        logits = self.classifier_head(features)

        wind = self.regression_head(
            features
        ).squeeze(1)

        return logits, wind


if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model checkpoint not found: {MODEL_PATH}"
    )


model = CycloneModel().to(DEVICE)

state = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
)

model.load_state_dict(
    state,
    strict=True,
)

model.eval()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def image_to_data_url(image):

    buffer = BytesIO()

    image.save(
        buffer,
        format="PNG",
    )

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode()

    return (
        "data:image/png;base64,"
        + encoded
    )


def generate_gradcam(
    image_tensor,
    class_index,
):

    activations = []
    gradients = []

    target_layer = model.backbone.conv_head

    def forward_hook(_, __, output):
        activations.append(output)

    def backward_hook(_, __, grad_output):
        gradients.append(
            grad_output[0]
        )

    forward_handle = (
        target_layer.register_forward_hook(
            forward_hook
        )
    )

    backward_handle = (
        target_layer.register_full_backward_hook(
            backward_hook
        )
    )

    try:

        model.zero_grad()

        logits, _ = model(
            image_tensor
        )

        logits[
            0,
            class_index
        ].backward()

        activation = activations[0].detach()
        gradient = gradients[0].detach()

    finally:

        forward_handle.remove()
        backward_handle.remove()

    weights = gradient.mean(
        dim=(2, 3),
        keepdim=True,
    )

    cam = (
        weights * activation
    ).sum(
        dim=1,
        keepdim=True,
    )

    cam = torch.relu(cam)

    cam = torch.nn.functional.interpolate(
        cam,
        size=(224, 224),
        mode="bilinear",
        align_corners=False,
    )

    cam = (
        cam
        .squeeze()
        .cpu()
        .numpy()
    )

    cam -= cam.min()

    if cam.max() > 0:
        cam /= cam.max()

    return cam


def create_gradcam_image(
    image,
    cam,
):

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    image = image.resize(
        (224, 224)
    )

    base = (
        np.asarray(
            image,
            dtype=np.float32,
        )
        / 255.0
    )

    heatmap = (
        plt.get_cmap("jet")(cam)[..., :3]
    )

    blended = (
        0.55 * base
        + 0.45 * heatmap
    )

    blended = np.clip(
        blended * 255,
        0,
        255,
    ).astype(np.uint8)

    return Image.fromarray(
        blended
    )


async def analyze_upload(
    event,
    original_view,
    gradcam_view,
    category_label,
    wind_label,
    confidence_label,
    classifier_label,
    decision_label,
):

    try:

        data = await event.file.read()

        image = Image.open(
            BytesIO(data)
        ).convert("RGB")

        tensor = (
            transform(image)
            .unsqueeze(0)
            .to(DEVICE)
        )

        with torch.no_grad():

            logits, wind = model(
                tensor
            )

            probabilities = torch.softmax(
                logits,
                dim=1,
            )[0]

            class_index = torch.argmax(
                probabilities
            ).item()

            classifier_category = (
                CATEGORIES[class_index]
            )

            confidence = probabilities[
                class_index
            ].item()

            predicted_wind = wind.item()

        if predicted_wind >= SUCS_THRESHOLD:

            final_category = "SuCS"

            reason = (
                f"Wind estimate "
                f"{predicted_wind:.1f} kt "
                f"exceeds the "
                f"{SUCS_THRESHOLD:.1f} kt "
                f"SuCS threshold."
            )

        else:

            final_category = (
                classifier_category
            )

            reason = (
                "Final category follows "
                "the visual classifier."
            )

        cam = generate_gradcam(
            tensor,
            class_index,
        )

        gradcam = create_gradcam_image(
            image,
            cam,
        )

        original_view.set_source(
            image_to_data_url(
                image.resize(
                    (224, 224)
                )
            )
        )

        gradcam_view.set_source(
            image_to_data_url(
                gradcam
            )
        )

        category_label.set_text(
            final_category
        )

        wind_label.set_text(
            f"{predicted_wind:.1f} kt"
        )

        confidence_label.set_text(
            f"{confidence:.1%}"
        )

        classifier_label.set_text(
            classifier_category
        )

        decision_label.set_text(
            reason
        )

    except Exception as exc:

        ui.notify(
            f"Analysis failed: {exc}",
            type="negative",
        )


ui.add_css("""
body {
    background: #07111f;
}

.app {
    max-width: 1400px;
    margin: 0 auto;
}

.hero {
    background: #0d1b2d;
    border: 1px solid #203752;
    border-radius: 18px;
}

.panel {
    background: #0d1929;
    border: 1px solid #1d334d;
    border-radius: 16px;
}

.metric {
    background: #101f32;
    border: 1px solid #213c59;
    border-radius: 14px;
}

.muted {
    color: #90a6bd;
}

.metric-title {
    color: #90a6bd;
    font-size: 13px;
}

.metric-value {
    font-size: 28px;
    font-weight: 700;
}
""", shared=True)


@ui.page("/")
def main_page():

    with ui.column().classes(
        "app w-full p-6 gap-5"
    ):

        with ui.card().classes(
            "hero w-full p-6"
        ):

            ui.label(
                "CYCLONE INTELLIGENCE SYSTEM"
            ).classes(
                "text-3xl font-bold text-white"
            )

            ui.label(
                "Explainable cyclone intensity "
                "analysis from satellite imagery"
            ).classes(
                "muted text-base mt-1"
            )

            ui.label(
                f"EfficientNet-B0 | "
                f"Device: {DEVICE.type.upper()} | "
                f"Grad-CAM enabled"
            ).classes(
                "muted text-sm mt-3"
            )

        with ui.row().classes(
            "w-full gap-5 items-stretch"
        ):

            with ui.card().classes(
                "panel flex-1 p-5"
            ):

                ui.label(
                    "Satellite Image"
                ).classes(
                    "text-xl font-semibold text-white"
                )

                ui.label(
                    "Upload a cyclone satellite image."
                ).classes(
                    "muted mb-4"
                )

                original_view = ui.image().classes(
                    "w-full rounded-xl mt-5"
                )

                ui.upload(
                    on_upload=lambda event:
                        analyze_upload(
                            event,
                            original_view,
                            gradcam_view,
                            category_label,
                            wind_label,
                            confidence_label,
                            classifier_label,
                            decision_label,
                        ),
                    auto_upload=True,
                ).props(
                    "accept=.jpg,.jpeg,.png"
                ).classes(
                    "w-full"
                )

            with ui.card().classes(
                "panel flex-1 p-5"
            ):

                ui.label(
                    "Analysis"
                ).classes(
                    "text-xl font-semibold text-white"
                )

                with ui.row().classes(
                    "w-full gap-3 mt-5"
                ):

                    with ui.card().classes(
                        "metric flex-1 p-4"
                    ):

                        ui.label(
                            "Final Category"
                        ).classes(
                            "metric-title"
                        )

                        category_label = (
                            ui.label("--")
                            .classes(
                                "metric-value text-white"
                            )
                        )

                    with ui.card().classes(
                        "metric flex-1 p-4"
                    ):

                        ui.label(
                            "Estimated Wind"
                        ).classes(
                            "metric-title"
                        )

                        wind_label = (
                            ui.label("--")
                            .classes(
                                "metric-value text-white"
                            )
                        )

                    with ui.card().classes(
                        "metric flex-1 p-4"
                    ):

                        ui.label(
                            "Confidence"
                        ).classes(
                            "metric-title"
                        )

                        confidence_label = (
                            ui.label("--")
                            .classes(
                                "metric-value text-white"
                            )
                        )

                ui.separator().classes(
                    "my-5"
                )

                ui.label(
                    "Visual classifier"
                ).classes(
                    "metric-title"
                )

                classifier_label = (
                    ui.label("--")
                    .classes(
                        "text-lg font-semibold text-white"
                    )
                )

                ui.label(
                    "Decision"
                ).classes(
                    "metric-title mt-4"
                )

                decision_label = (
                    ui.label(
                        "Upload a satellite image "
                        "to begin."
                    )
                    .classes(
                        "text-sm text-white"
                    )
                )

        with ui.card().classes(
            "panel w-full p-5"
        ):

            ui.label(
                "Explainability"
            ).classes(
                "text-xl font-semibold text-white"
            )

            ui.label(
                "Grad-CAM highlights the regions "
                "used by the visual classifier."
            ).classes(
                "muted mb-4"
            )

            gradcam_view = ui.image().classes(
                "w-full max-w-4xl mx-auto rounded-xl"
            )


ui.run(
    host="0.0.0.0",
    port=8000,
    title="Cyclone Intelligence System",
)
