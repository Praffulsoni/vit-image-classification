import os
import sys
import torch
from torchvision import transforms
from transformers import ViTForImageClassification
from PIL import Image
import torch.nn.functional as F

print("PREDICT.PY STARTED 🔮")

# -----------------------
# Config
# -----------------------
MODEL_PATH = "results/vit-cifar10"
IMAGE_PATH = sys.argv[1]

classes = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------
# Load Model
# -----------------------
model = ViTForImageClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()

print("Model loaded successfully!")

# -----------------------
# Image Transform
# -----------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# -----------------------
# Prediction Function
# -----------------------
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor).logits
        probs = F.softmax(outputs, dim=1)

        confidence, predicted_index = torch.max(probs, dim=1)

    predicted_class = classes[predicted_index.item()]
    confidence = confidence.item() * 100

    print(f"\nImage: {image_path}")
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence:.2f}%")

# -----------------------
# Single or Batch Mode
# -----------------------
if os.path.isfile(IMAGE_PATH):
    predict_image(IMAGE_PATH)

elif os.path.isdir(IMAGE_PATH):
    print("Batch mode enabled 📂")

    for file in os.listdir(IMAGE_PATH):
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            predict_image(os.path.join(IMAGE_PATH, file))

else:
    print("Invalid path provided ❌")