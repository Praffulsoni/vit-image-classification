# 🧠 Vision Transformer (ViT) Image Classification on CIFAR-10

This project demonstrates an end-to-end deep learning pipeline using a **Vision Transformer (ViT)** model for image classification.
The model is trained on the **CIFAR-10 dataset** and deployed for inference using both:

* ✅ Command-line interface (CLI)
* ✅ Web application using **Streamlit**

Instead of retraining every time, the trained model is saved and reused for fast predictions on new images.


## 📌 Features

* 🚀 Train a Vision Transformer on CIFAR-10
* 💾 Save and reload trained model
* 🖼️ Predict class of any input image
* 📊 Display prediction confidence
* 📂 Batch prediction support (multiple images)
* 🌐 Streamlit web app for easy usage
* ⚡ GPU acceleration (CUDA supported)


## 🗂️ Project Structure

```
vit-image-classification/
│
├── train.py          # Trains the ViT model and saves it
├── predict.py        # Predicts class for image(s) using saved model
├── app.py            # Streamlit web application
├── requirements.txt # Required Python packages
├── README.md         # Project documentation
│
├── results/          # Saved model and checkpoints
├── predict_image/    # Images for prediction
├── data/             # CIFAR-10 dataset (auto-downloaded)
└── venv/             # Virtual environment (optional)
```

## 🧠 Model Used

* **Vision Transformer (ViT)**
* Pretrained model: `google/vit-base-patch16-224`
* Dataset: **CIFAR-10** (10 classes)

Classes:

airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck


## ⚙️ Installation

### 1️⃣ Create virtual environment (optional but recommended)

python -m venv venv
venv\Scripts\activate

### 2️⃣ Install dependencies

pip install -r requirements.txt


## 🏋️ Training the Model

Run:

python train.py

This will:

* Download CIFAR-10
* Train the ViT model
* Save the trained model to:

results/vit-cifar10/


## 🔮 Predicting on Images (CLI)

### Single image:

python predict.py predict_image/puppy.jpg

### Batch prediction (folder):

python predict.py predict_image/

Output example:

Image: puppy.jpg
Predicted class: dog
Confidence: 98.84%


## 🌐 Web App (Streamlit)

Run:

streamlit run app.py

Then open in browser:

http://localhost:8501

Features:

* Upload an image
* Displays:

  * Predicted class
  * Confidence score
  * Uploaded image preview


## 📊 Confidence Score

The confidence score represents how sure the model is about its prediction using Softmax probability.

Example:

Predicted class: dog  
Confidence: 98.84%

This means the model is 98.84% confident in its prediction.


## 🧪 Why This Project?

This project demonstrates:

* Practical use of Transformers in vision tasks
* Training vs inference separation
* Model persistence (save & reload)
* GPU acceleration
* Deployment with a simple UI

It simulates a real-world ML workflow:
**Train → Save → Load → Predict → Deploy**


## 🚀 Future Improvements

* Top-3 predictions
* Grad-CAM heatmap visualization
* Support custom datasets
* Online deployment
* Mobile-friendly UI


## 📜 Requirements

See `requirements.txt`

Main libraries:

* torch
* torchvision
* transformers
* streamlit
* pillow
* numpy


## 👨‍💻 Author

**Prafful Rajesh Soni**
B.Tech IT Student
Passionate about AI, ML, and Deep Learning 🚀


## ⭐ Acknowledgements

* HuggingFace Transformers
* PyTorch
* CIFAR-10 Dataset
* Streamlit