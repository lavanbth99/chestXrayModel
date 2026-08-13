import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm
from huggingface_hub import hf_hub_download

# ==============================
# CONFIG
# ==============================
HF_REPO_ID = "LavanBathija/chest-xray"
CHECKPOINT_FILENAME = "best_coocc_model.pth"

NUM_CLASSES = 14
THRESHOLD = 0.5

LABEL_COLUMNS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion",
    "Lung Opacity", "No Finding", "Pleural Effusion",
    "Pleural Other", "Pneumonia", "Pneumothorax", "Support Devices"
]

st.set_page_config(page_title="Chest X-ray Classifier (research demo)")


# ==============================
# MODEL  (unchanged from your script)
# ==============================
class XRayModel(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.backbone = timm.create_model(
            "convnext_tiny.fb_in22k_ft_in1k_384",
            pretrained=False,
            num_classes=0
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.backbone.num_features, num_classes)
        )

    def forward(self, x):
        feats = self.backbone(x)
        return self.classifier(feats)


@st.cache_resource
def load_model():
    """Downloads weights from the HF Hub once, then caches the model in memory.
    Without cache_resource this would re-run on every widget interaction."""
    ckpt_path = hf_hub_download(repo_id=HF_REPO_ID, filename=CHECKPOINT_FILENAME)
    model = XRayModel()
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


# EXACTLY the transforms from your inference script.
# No transforms.Normalize(), because your script doesn't have one either.
test_tfms = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
])


@torch.no_grad()
def predict(model, image: Image.Image):
    x = test_tfms(image.convert("RGB")).unsqueeze(0)
    probs = torch.sigmoid(model(x))[0].numpy()
    return {label: float(p) for label, p in zip(LABEL_COLUMNS, probs)}


# ==============================
# UI
# ==============================
st.title("Chest X-ray Multi-Label Classifier")

st.error(
     "This is a research model. It estimates the likelihood of 14 chest findings: "
    "atelectasis, cardiomegaly, consolidation, edema, enlarged cardiomediastinum, "
    "fracture, lung lesion, lung opacity, no finding, pleural effusion, pleural "
    "other, pneumonia, pneumothorax, and support devices.
)

uploaded = st.file_uploader("Upload a chest X-ray", type=["png", "jpg", "jpeg"])

if uploaded is not None:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded image", width=350)

    with st.spinner("Running the model (first run downloads weights, ~1 min)..."):
        model = load_model()
        scores = predict(model, image)

    st.subheader("Predicted probability per finding")
    st.caption(
        "These are independent scores, one per finding. Several can be high at "
        "once and they do not add up to 1."
    )

    for label in sorted(scores, key=scores.get, reverse=True):
        st.progress(scores[label], text=f"{label} — {scores[label]:.3f}")

    flagged = [l for l in LABEL_COLUMNS if scores[l] >= THRESHOLD]
    if flagged:
        st.info(f"Above the {THRESHOLD:.2f} threshold: " + ", ".join(flagged))
    else:
        st.info(
            f"Nothing above the {THRESHOLD:.2f} threshold. That is not the same "
            "as the X-ray being normal."
        )
else:
    st.write("Upload an image to run the model.")
