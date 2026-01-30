import streamlit as st
import torch
from torchvision import transforms
from transformers import ViTForImageClassification
from PIL import Image
import torch.nn.functional as F

# -----------------------
# App Title
# -----------------------
st.set_page_config(page_title="ViT Image Classifier", layout="centered")
st.title("🧠 Vision Transformer Image Classifier")
st.write("Upload an image and let the AI predict its class!")

# -----------------------
# Classes (CIFAR-10)
# -----------------------
classes = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# -----------------------
# Device
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# Load Model (once)
# -----------------------
@st.cache_resource
def load_model():
    model = ViTForImageClassification.from_pretrained("results/vit-cifar10")
    model.to(device)
    model.eval()
    return model

model = load_model()

# -----------------------
# Image Transform
# -----------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# -----------------------
# File Uploader
# -----------------------
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_column_width=True)

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor).logits
        probs = F.softmax(outputs, dim=1)
        confidence, predicted_index = torch.max(probs, dim=1)

    predicted_class = classes[predicted_index.item()]
    confidence = confidence.item() * 100

    st.success(f"🧠 Predicted Class: **{predicted_class}**")
    st.info(f"📊 Confidence: **{confidence:.2f}%**")