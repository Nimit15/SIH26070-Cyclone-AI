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
CATEGORY_NAMES = {
    "D": "Depression",
    "DD": "Deep Depression",
    "CS": "Cyclonic Storm",
    "SCS": "Severe Cyclonic Storm",
    "VSCS": "Very Severe Cyclonic Storm",
    "SuCS": "Super Cyclonic Storm",
}

CATEGORY_ADVICE = {
    "D": "Mild disturbance. Stay aware of official weather updates.",
    "DD": "Monitor official updates and keep basic emergency supplies ready.",
    "CS": "A cyclonic storm is indicated. Check local warnings and prepare.",
    "SCS": "A severe storm is indicated. Follow local authority guidance.",
    "VSCS": "A very severe storm is indicated. Pay close attention to official alerts.",
    "SuCS": "A super cyclonic storm is indicated. Follow official evacuation guidance.",
}

LANGUAGES = {
    "en": "English",
    "hi": "हिन्दी",
    "bn": "বাংলা",
    "te": "తెలుగు",
    "ta": "தமிழ்",
    "or": "ଓଡ଼ିଆ",
    "gu": "ગુજરાતી",
}

SAFETY_MESSAGES = {
    "en": {
        "D": "Mild disturbance. Stay aware of official updates.",
        "DD": "Monitor official updates and keep basic emergency supplies ready.",
        "CS": "A cyclonic storm is indicated. Check local warnings and prepare.",
        "SCS": "A severe storm is indicated. Follow local authority guidance.",
        "VSCS": "A very severe storm is indicated. Pay close attention to official alerts.",
        "SuCS": "A super cyclonic storm is indicated. Follow official evacuation guidance.",
    },
    "hi": {
        "D": "हल्का विक्षोभ है। आधिकारिक मौसम अपडेट देखते रहें।",
        "DD": "आधिकारिक अपडेट देखते रहें और जरूरी सामान तैयार रखें।",
        "CS": "चक्रवाती तूफान का संकेत है। स्थानीय चेतावनियां देखें और तैयारी करें।",
        "SCS": "गंभीर तूफान का संकेत है। स्थानीय अधिकारियों के निर्देश मानें।",
        "VSCS": "बहुत गंभीर तूफान का संकेत है। आधिकारिक चेतावनियों पर ध्यान दें।",
        "SuCS": "सुपर साइक्लोनिक स्टॉर्म का संकेत है। आधिकारिक निकासी निर्देशों का पालन करें।",
    },
    "bn": {
        "D": "মৃদু ঝড়ের ইঙ্গিত। সরকারি আবহাওয়া আপডেট দেখুন।",
        "DD": "সরকারি আপডেট দেখুন এবং জরুরি সামগ্রী প্রস্তুত রাখুন।",
        "CS": "ঘূর্ণিঝড়ের ইঙ্গিত। স্থানীয় সতর্কতা দেখুন এবং প্রস্তুতি নিন।",
        "SCS": "তীব্র ঝড়ের ইঙ্গিত। স্থানীয় কর্তৃপক্ষের নির্দেশ মানুন।",
        "VSCS": "অত্যন্ত তীব্র ঝড়ের ইঙ্গিত। সরকারি সতর্কতার দিকে নজর রাখুন।",
        "SuCS": "সুপার ঘূর্ণিঝড়ের ইঙ্গিত। সরকারি সরিয়ে নেওয়ার নির্দেশ অনুসরণ করুন।",
    },
    "te": {
        "D": "తేలికపాటి వాతావరణ వ్యవస్థ. అధికారిక అప్‌డేట్‌లను గమనించండి.",
        "DD": "అధికారిక అప్‌డేట్‌లను గమనించి అవసరమైన సామగ్రిని సిద్ధంగా ఉంచండి.",
        "CS": "తుఫాను సూచన ఉంది. స్థానిక హెచ్చరికలను గమనించి సిద్ధం అవ్వండి.",
        "SCS": "తీవ్రమైన తుఫాను సూచన ఉంది. స్థానిక అధికారుల సూచనలు పాటించండి.",
        "VSCS": "చాలా తీవ్రమైన తుఫాను సూచన ఉంది. అధికారిక హెచ్చరికలను గమనించండి.",
        "SuCS": "సూపర్ సైక్లోనిక్ స్టార్మ్ సూచన ఉంది. అధికారిక తరలింపు సూచనలు పాటించండి.",
    },
    "ta": {
        "D": "லேசான வானிலை அமைப்பு. அதிகாரப்பூர்வ புதுப்பிப்புகளை கவனிக்கவும்.",
        "DD": "அதிகாரப்பூர்வ தகவல்களை கவனித்து தேவையான பொருட்களை தயார் வைக்கவும்.",
        "CS": "புயல் இருப்பதற்கான அறிகுறி. உள்ளூர் எச்சரிக்கைகளை பார்த்து தயாராகுங்கள்.",
        "SCS": "கடுமையான புயல் இருப்பதற்கான அறிகுறி. அதிகாரிகளின் அறிவுறுத்தல்களை பின்பற்றுங்கள்.",
        "VSCS": "மிகவும் கடுமையான புயல் இருப்பதற்கான அறிகுறி. அதிகாரப்பூர்வ எச்சரிக்கைகளை கவனிக்கவும்.",
        "SuCS": "சூப்பர் சைக்ளோனிக் புயல் இருப்பதற்கான அறிகுறி. அதிகாரப்பூர்வ வெளியேற்ற அறிவுறுத்தல்களை பின்பற்றுங்கள்.",
    },
    "or": {
        "D": "ହାଲୁକା ବିକ୍ଷୋଭ। ସରକାରୀ ପାଣିପାଗ ସୂଚନା ଦେଖନ୍ତୁ।",
        "DD": "ସରକାରୀ ସୂଚନା ଦେଖନ୍ତୁ ଏବଂ ଆବଶ୍ୟକ ସାମଗ୍ରୀ ପ୍ରସ୍ତୁତ ରଖନ୍ତୁ।",
        "CS": "ବାତ୍ୟାର ସଙ୍କେତ ରହିଛି। ସ୍ଥାନୀୟ ସତର୍କତା ଦେଖି ପ୍ରସ୍ତୁତ ହୁଅନ୍ତୁ।",
        "SCS": "ତୀବ୍ର ବାତ୍ୟାର ସଙ୍କେତ ରହିଛି। ସ୍ଥାନୀୟ କର୍ତ୍ତୃପକ୍ଷଙ୍କ ନିର୍ଦ୍ଦେଶ ମାନନ୍ତୁ।",
        "VSCS": "ଅତି ତୀବ୍ର ବାତ୍ୟାର ସଙ୍କେତ ରହିଛି। ସରକାରୀ ସତର୍କତା ଉପରେ ଧ୍ୟାନ ଦିଅନ୍ତୁ।",
        "SuCS": "ସୁପର ସାଇକ୍ଲୋନିକ୍ ଷ୍ଟର୍ମର ସଙ୍କେତ ରହିଛି। ସରକାରୀ ସ୍ଥାନାନ୍ତର ନିର୍ଦ୍ଦେଶ ମାନନ୍ତୁ।",
    },
    "gu": {
        "D": "હળવા વિક્ષેપનો સંકેત છે. સત્તાવાર હવામાન અપડેટ જુઓ.",
        "DD": "સત્તાવાર અપડેટ જુઓ અને જરૂરી સામગ્રી તૈયાર રાખો.",
        "CS": "ચક્રવાતી તોફાનનો સંકેત છે. સ્થાનિક ચેતવણીઓ જુઓ અને તૈયારી કરો.",
        "SCS": "ગંભીર તોફાનનો સંકેત છે. સ્થાનિક અધિકારીઓની સૂચનાઓનું પાલન કરો.",
        "VSCS": "ખૂબ ગંભીર તોફાનનો સંકેત છે. સત્તાવાર ચેતવણીઓ પર ધ્યાન આપો.",
        "SuCS": "સુપર સાયક્લોનિક સ્ટોર્મનો સંકેત છે. સત્તાવાર સ્થળાંતર સૂચનાઓનું પાલન કરો.",
    },
}

PROJECT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT / "checkpoints" / "cyclone_best.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CycloneModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=False,
            num_classes=0,
        )
        features = self.backbone.num_features
        self.classifier_head = nn.Linear(features, len(CATEGORIES))
        self.regression_head = nn.Linear(features, 1)

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier_head(features)
        wind = self.regression_head(features).squeeze(1)
        return logits, wind


if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model checkpoint not found: {MODEL_PATH}"
    )

model = CycloneModel().to(DEVICE)
state = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state, strict=True)
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
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode()
    return "data:image/png;base64," + encoded


def make_gradcam(image_tensor, class_index):
    activations = []
    gradients = []
    layer = model.backbone.conv_head

    def save_activation(_, __, output):
        activations.append(output)

    def save_gradient(_, __, grad_output):
        gradients.append(grad_output[0])

    forward_handle = layer.register_forward_hook(save_activation)
    backward_handle = layer.register_full_backward_hook(save_gradient)

    try:
        model.zero_grad()
        logits, _ = model(image_tensor)
        logits[0, class_index].backward()
        activation = activations[0].detach()
        gradient = gradients[0].detach()
    finally:
        forward_handle.remove()
        backward_handle.remove()

    weights = gradient.mean(dim=(2, 3), keepdim=True)
    cam = (weights * activation).sum(dim=1, keepdim=True)
    cam = torch.relu(cam)
    cam = torch.nn.functional.interpolate(
        cam,
        size=(224, 224),
        mode="bilinear",
        align_corners=False,
    )
    cam = cam.squeeze().cpu().numpy()
    cam -= cam.min()

    if cam.max() > 0:
        cam /= cam.max()

    return cam


def render_gradcam(image, cam):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base = np.asarray(
        image.resize((224, 224)),
        dtype=np.float32,
    ) / 255.0
    heatmap = plt.get_cmap("viridis")(cam)[..., :3]
    blended = np.clip(
        (0.55 * base + 0.45 * heatmap) * 255,
        0,
        255,
    ).astype(np.uint8)
    return Image.fromarray(blended)


def predict(image):
    tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits, wind = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        class_index = torch.argmax(probabilities).item()

    category = CATEGORIES[class_index]
    confidence = probabilities[class_index].item()
    wind_speed = wind.item()

    cam = make_gradcam(tensor, class_index)
    explanation = render_gradcam(image, cam)

    return {
        "category": category,
        "name": CATEGORY_NAMES[category],
        "confidence": confidence,
        "wind": wind_speed,
        "explanation": explanation,
    }


def update_language(category, labels):
    for code, label in labels.items():
        message = SAFETY_MESSAGES.get(code, SAFETY_MESSAGES["en"]).get(
            category,
            CATEGORY_ADVICE[category],
        )
        label.set_text(message)


def build_satellite_url(date_text):
    return (
        "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
        "?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap"
        "&LAYERS=VIIRS_SNPP_CorrectedReflectance_TrueColor"
        "&CRS=EPSG:4326"
        "&BBOX=5,68,25,95"
        "&WIDTH=900&HEIGHT=650"
        "&FORMAT=image/jpeg"
        f"&TIME={date_text}"
    )


ui.add_css("""
body { background: #07171b; }
.app { max-width: 1380px; margin: 0 auto; }
.hero { background: #0d2a30; border: 1px solid #20505a; border-radius: 20px; }
.panel { background: #0d2328; border: 1px solid #1d4148; border-radius: 18px; }
.metric { background: #102d32; border: 1px solid #28555d; border-radius: 14px; }
.muted { color: #8fb2b7; }
.metric-title { color: #8fb2b7; font-size: 13px; }
.metric-value { color: #edfafa; font-size: 25px; font-weight: 700; }
.notice { background: #102d32; border: 1px solid #28555d; border-radius: 12px; }
""", shared=True)


@ui.page("/")
def main_page():
    state = {"result": None}

    with ui.column().classes("app w-full p-6 gap-5"):
        with ui.card().classes("hero w-full p-6"):
            ui.label("Cyclone Intelligence System").classes(
                "text-3xl font-bold text-white"
            )
            ui.label(
                "Satellite-based cyclone intensity analysis with visual explanation."
            ).classes("muted text-base mt-1")
            ui.label(
                f"EfficientNet-B0  •  {DEVICE.type.upper()}  •  Grad-CAM"
            ).classes("muted text-sm mt-3")

        with ui.card().classes("notice w-full p-4"):
            ui.label(
                "Model estimates are decision-support outputs. "
                "Use official IMD, NDMA and local authority warnings for real-world action."
            ).classes("text-sm text-white")

        with ui.row().classes("w-full gap-5 items-stretch"):
            with ui.card().classes("panel flex-1 p-5"):
                ui.label("Satellite image").classes(
                    "text-xl font-semibold text-white"
                )
                ui.label(
                    "Upload a compatible cyclone satellite image."
                ).classes("muted mb-4")

                original_view = ui.image().classes(
                    "w-full rounded-xl mt-2"
                )

                async def handle_upload(event):
                    try:
                        data = await event.file.read()
                        image = Image.open(
                            BytesIO(data)
                        ).convert("RGB")

                        result = predict(image)
                        state["result"] = result

                        original_view.set_source(
                            image_to_data_url(
                                image.resize((224, 224))
                            )
                        )
                        explanation_view.set_source(
                            image_to_data_url(result["explanation"])
                        )
                        category_value.set_text(
                            f"{result['category']} — {result['name']}"
                        )
                        wind_value.set_text(
                            f"{result['wind']:.1f} kt"
                        )
                        confidence_value.set_text(
                            f"{result['confidence']:.1%}"
                        )
                        decision_value.set_text(
                            CATEGORY_ADVICE[result["category"]]
                        )
                        update_language(
                            result["category"],
                            language_labels,
                        )

                    except Exception:
                        ui.notify(
                            "The image could not be analysed. "
                            "Please check the file and try again.",
                            type="negative",
                        )

                ui.upload(
                    on_upload=handle_upload,
                    auto_upload=True,
                ).props(
                    "accept=.jpg,.jpeg,.png"
                ).classes("w-full")

            with ui.card().classes("panel flex-1 p-5"):
                ui.label("Analysis").classes(
                    "text-xl font-semibold text-white"
                )

                with ui.row().classes("w-full gap-3 mt-5"):
                    with ui.card().classes("metric flex-1 p-4"):
                        ui.label("Storm category").classes("metric-title")
                        category_value = ui.label("--").classes("metric-value")

                    with ui.card().classes("metric flex-1 p-4"):
                        ui.label("Wind estimate").classes("metric-title")
                        wind_value = ui.label("--").classes("metric-value")

                    with ui.card().classes("metric flex-1 p-4"):
                        ui.label("Classifier confidence").classes("metric-title")
                        confidence_value = ui.label("--").classes("metric-value")

                ui.label("Decision support").classes(
                    "metric-title mt-5"
                )
                decision_value = ui.label(
                    "Upload an image to begin."
                ).classes("text-sm text-white mt-1")

                ui.separator().classes("my-5")

                ui.label("Safety message").classes(
                    "metric-title"
                )
                language_labels = {}
                with ui.grid(columns=2).classes("w-full gap-2 mt-2"):
                    for code, name in LANGUAGES.items():
                        with ui.card().classes("metric p-3"):
                            ui.label(name).classes("metric-title")
                            language_labels[code] = ui.label(
                                "Waiting for a result."
                            ).classes("text-sm text-white")

        with ui.card().classes("panel w-full p-5"):
            ui.label("Visual explanation").classes(
                "text-xl font-semibold text-white"
            )
            ui.label(
                "Grad-CAM highlights image regions that contributed to the selected visual class."
            ).classes("muted mb-4")
            explanation_view = ui.image().classes(
                "w-full max-w-4xl mx-auto rounded-xl"
            )

        with ui.card().classes("panel w-full p-5"):
            ui.label("Recent satellite view").classes(
                "text-xl font-semibold text-white"
            )
            ui.label(
                "This view is for situational awareness only. "
                "It is not passed directly into the trained image model."
            ).classes("muted mb-3")

            satellite_view = ui.image().classes(
                "w-full max-w-5xl mx-auto rounded-xl"
            )
            date_input = ui.input(
                label="Date (YYYY-MM-DD)",
                value="",
            ).classes("w-full max-w-sm")

            def load_satellite():
                date_text = date_input.value.strip()
                if len(date_text) != 10:
                    ui.notify("Enter a date as YYYY-MM-DD.", type="warning")
                    return
                satellite_view.set_source(build_satellite_url(date_text))

            ui.button(
                "Load satellite frame",
                on_click=load_satellite,
            ).classes("mt-2")

        with ui.card().classes("panel w-full p-5"):
            ui.label("Emergency information").classes(
                "text-xl font-semibold text-white"
            )
            ui.label(
                "For an immediate emergency in India, call 112. "
                "For cyclone decisions, follow official local instructions."
            ).classes("text-sm text-white")
            ui.link(
                "NDMA",
                "https://ndma.gov.in/",
                new_tab=True,
            ).classes("text-teal-300 mt-2")
            ui.link(
                "India Meteorological Department",
                "https://mausam.imd.gov.in/",
                new_tab=True,
            ).classes("text-teal-300")
            ui.link(
                "SACHET alerts",
                "https://sachet.ndma.gov.in/",
                new_tab=True,
            ).classes("text-teal-300")

        with ui.card().classes("panel w-full p-5"):
            ui.label("About the model").classes(
                "text-xl font-semibold text-white"
            )
            ui.label(
                "The visual model classifies cyclone intensity and estimates maximum "
                "sustained wind from the satellite image. The project also includes a "
                "separate storm-level temporal forecasting branch for +6h and +12h evaluation."
            ).classes("text-sm text-white")
            ui.label(
                "The web interface keeps the visual inference path separate from the "
                "forecasting artifacts so the demo does not imply that a single uploaded "
                "image is a full temporal forecast."
            ).classes("text-sm muted mt-2")

if __name__ in {"__main__", "__mp_main__"}:
    ui.run(
        host="0.0.0.0",
        port=8000,
        title="Cyclone Intelligence System",
    )
