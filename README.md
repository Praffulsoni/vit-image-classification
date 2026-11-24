# 🧠 Vision Transformer (ViT) Image Classification on CIFAR-10

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/🤗%20Transformers-5.0-FFD21E)
![Streamlit](https://img.shields.io/badge/Streamlit-1.53-FF4B4B?logo=streamlit&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-optional-76B900?logo=nvidia&logoColor=white)
![Test Accuracy](https://img.shields.io/badge/Test%20Accuracy-98.45%25-brightgreen)

This project fine-tunes a pretrained **Vision Transformer (`google/vit-base-patch16-224`)** on the **CIFAR-10** dataset and serves it in two ways:

- 🖥️ a **command-line tool** for single-image or whole-folder predictions
- 🌐 a **Streamlit web app** where you upload an image and get the predicted class with a confidence score

You train once, the fine-tuned weights are saved to disk, and every later prediction just loads them. The workflow is **Train → Save → Load → Predict → Deploy**.

---

## 📑 Table of Contents

- [How It Works](#-how-it-works)
- [Features](#-features)
- [Project Structure](#️-project-structure)
- [Model and Dataset](#-model-and-dataset)
- [Installation](#️-installation)
- [Training](#️-training)
- [Prediction (CLI)](#-prediction-cli)
- [Web App (Streamlit)](#-web-app-streamlit)
- [Results](#-results)
- [Training Configuration](#️-training-configuration)
- [Troubleshooting](#-troubleshooting)
- [Future Improvements](#-future-improvements)
- [Author](#-author)
- [Acknowledgements](#-acknowledgements)

---

## 🔍 How It Works

```
                ┌──────────────────────────────┐
 CIFAR-10  ───► │ Resize 32×32 → 224×224       │
 (50k train)    │ ToTensor  (values in 0–1)    │
                └──────────────┬───────────────┘
                               ▼
                ┌──────────────────────────────┐
                │ ViT-Base/16 (ImageNet-21k    │
                │ pretrained), new 10-class    │
                │ head, fine-tuned with AdamW  │
                └──────────────┬───────────────┘
                               ▼
                  results/vit-cifar10/  (saved weights)
                               │
             ┌─────────────────┴─────────────────┐
             ▼                                   ▼
      predict.py (CLI)                    app.py (Streamlit)
   single image / folder             upload → class + confidence
```

1. **Patching.** ViT cuts each 224×224 image into 196 patches of 16×16 pixels. Each patch becomes a token, the same way words are tokens in a language model.
2. **Self-attention.** 12 Transformer encoder layers (12 attention heads, 768 hidden size) let every patch attend to every other patch, so the model can relate distant parts of the image.
3. **Classification.** The `[CLS]` token's final embedding goes through a new linear head with **10 outputs**, one per CIFAR-10 class. The original 1000-class ImageNet head is discarded (`ignore_mismatched_sizes=True`).
4. **Confidence.** At inference time the logits are passed through **softmax**, and the highest probability is reported as the confidence score.

---

## ✨ Features

| Feature | Details |
| :--- | :--- |
| 🚀 Transfer learning | Fine-tunes a pretrained ViT instead of training from scratch |
| 💾 Checkpointing | Saves model and optimizer state after every epoch and resumes automatically |
| 📊 Built-in evaluation | Reports test accuracy on the 10,000 CIFAR-10 test images after training |
| 🖼️ Single-image prediction | `python predict.py path/to/image.jpg` |
| 📂 Batch prediction | Pass a folder and every `.jpg` / `.jpeg` / `.png` in it is classified |
| 🌐 Web UI | Streamlit app with image upload, preview, predicted class and confidence |
| ⚡ GPU acceleration | Uses CUDA automatically when available, falls back to CPU |
| 🧠 Cached model loading | The Streamlit app loads the model once with `@st.cache_resource` |

---

## 🗂️ Project Structure

```
vit-image-classification/
├── train.py             # Fine-tunes ViT on CIFAR-10, evaluates it, saves the model
├── predict.py           # CLI inference on one image or a folder of images
├── app.py               # Streamlit web application
├── requirements.txt     # Python dependencies
├── README.md
│
├── predict_image/       # Sample images to try (airplane, bird, cat, car, horse)
├── data/                # CIFAR-10, auto-downloaded on first run   (git-ignored)
└── results/             # Created by train.py                       (git-ignored)
    ├── checkpoint.pth   #   resumable training checkpoint (~1 GB)
    └── vit-cifar10/     #   final model: config.json + model.safetensors (~340 MB)
```

> **Note:** `data/` and `results/` are listed in `.gitignore` because they are large. After cloning you need to run `train.py` once to produce `results/vit-cifar10/` before `predict.py` or `app.py` will work.

---

## 🧠 Model and Dataset

### Model: `google/vit-base-patch16-224`

| Property | Value |
| :--- | :--- |
| Architecture | ViT-Base, patch size 16 |
| Input resolution | 224 × 224 × 3 |
| Encoder layers | 12 |
| Attention heads | 12 |
| Hidden size / MLP size | 768 / 3072 |
| Parameters | ~86 M |
| Pretraining | ImageNet-21k, fine-tuned on ImageNet-1k |
| Output head | Replaced with a 10-class linear layer |

### Dataset: CIFAR-10

60,000 colour images (32×32) in 10 balanced classes: **50,000 for training** and **10,000 for testing**. `torchvision` downloads it into `./data` automatically.

| Index | Class | Index | Class |
| :---: | :--- | :---: | :--- |
| 0 | ✈️ airplane | 5 | 🐶 dog |
| 1 | 🚗 automobile | 6 | 🐸 frog |
| 2 | 🐦 bird | 7 | 🐴 horse |
| 3 | 🐱 cat | 8 | 🚢 ship |
| 4 | 🦌 deer | 9 | 🚚 truck |

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/Praffulsoni/vit-image-classification.git
cd vit-image-classification
```

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU users:** the default `pip install torch` may give you a CPU-only build. For CUDA, install PyTorch first using the command from [pytorch.org/get-started](https://pytorch.org/get-started/locally/), then run the line above.

Check that the GPU is visible:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 🏋️ Training

```bash
python train.py
```

What happens:

1. CIFAR-10 is downloaded to `./data` (first run only).
2. `google/vit-base-patch16-224` is downloaded from the Hugging Face Hub and given a new 10-class head.
3. The model is fine-tuned, printing the loss every 100 batches.
4. A checkpoint is saved to `results/checkpoint.pth` after each epoch.
5. The model is evaluated on the 10,000 test images and the **test accuracy** is printed.
6. The final model is saved to `results/vit-cifar10/`.

Sample console output:

```
Using device: cuda
Dataset loaded successfully!
Train samples: 50000
Test samples: 10000
ViT model loaded!

Epoch [1/1] 🚀
Batch 0, Loss: 2.3xxx
Batch 100, Loss: 0.xxxx
...
Epoch completed ✅ Avg Loss: 0.xxxx
Checkpoint saved 💾

Starting evaluation 🔍
Test Accuracy: xx.xx%
Model saved to results/vit-cifar10 🎉
```

> ⏱️ One epoch is 1,563 batches of 32 images at 224×224. A modern GPU finishes it in minutes; on CPU it can take hours.

---

## 🔮 Prediction (CLI)

### Single image

```bash
python predict.py predict_image/cat.jpg
```

### Whole folder (batch mode)

```bash
python predict.py predict_image/
```

### Example output

```
Using device: cuda
Model loaded successfully!

Image: predict_image/cat.jpg
Predicted class: cat
Confidence: 98.84%
```

Any image size works: it is converted to RGB and resized to 224×224 before inference. Keep in mind that the model only knows the 10 CIFAR-10 classes, so a photo of anything else will still be mapped to the closest of those 10.

---

## 🌐 Web App (Streamlit)

```bash
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

1. Upload a `.jpg`, `.jpeg` or `.png` image.
2. The app shows a preview of the image.
3. You get the **predicted class** and the **confidence score**.

The model is loaded once and cached, so predictions after the first one are fast.

### Understanding the confidence score

The model outputs 10 raw scores (logits). Softmax turns them into probabilities that add up to 100%, and the app reports the highest one:

$$P(\text{class}_i) = \frac{e^{z_i}}{\sum_{j=1}^{10} e^{z_j}}$$

`Confidence: 98.84%` means the model gave 98.84% of the probability to that class. High confidence is not proof of correctness, especially for images outside CIFAR-10's categories.

---

## 📈 Results

The saved model (`results/vit-cifar10`, fine-tuned for 1 epoch) was evaluated on all **10,000 CIFAR-10 test images** using the same transform as training:

**Test accuracy: 98.45%**

| Class | Accuracy | Class | Accuracy |
| :--- | :---: | :--- | :---: |
| ✈️ airplane | 98.70% | 🐶 dog | 97.20% |
| 🚗 automobile | 98.50% | 🐸 frog | 99.80% |
| 🐦 bird | 98.80% | 🐴 horse | 99.40% |
| 🐱 cat | 96.60% | 🚢 ship | 99.10% |
| 🦌 deer | 97.20% | 🚚 truck | 99.20% |

Cat (96.6%), deer and dog (97.2%) are the hardest classes. Frog is the easiest (99.8%).

---

## 🛠️ Training Configuration

All hyperparameters live at the top of `main()` in [`train.py`](train.py):

| Parameter | Value | Notes |
| :--- | :--- | :--- |
| `BATCH_SIZE` | 32 | Lower it if you run out of GPU memory |
| `EPOCHS` | 1 | Increase for higher accuracy |
| `LEARNING_RATE` | 2e-5 | Small learning rate, typical for fine-tuning |
| Optimizer | AdamW | |
| Loss | CrossEntropyLoss | |
| Input transform | `Resize(224, 224)` + `ToTensor()` | The same transform is used in `predict.py` and `app.py` |
| DataLoader | `num_workers=4`, `pin_memory=True` | |

---

## 🧯 Troubleshooting

| Problem | Fix |
| :--- | :--- |
| `OSError` / "can't find results/vit-cifar10" when predicting | The trained model isn't in the repo. Run `python train.py` first. |
| Raised `EPOCHS` but training ends immediately | `train.py` resumes from `results/checkpoint.pth`. If the checkpoint's epoch already reached `EPOCHS`, nothing is left to train. Raise `EPOCHS` above the saved epoch, or delete the checkpoint to start fresh. |
| `CUDA out of memory` | Lower `BATCH_SIZE` (for example to 16 or 8). |
| DataLoader errors on Windows | The training code already sits under `if __name__ == "__main__":`, which Windows needs. If problems continue, set `num_workers=0`. |
| `Using device: cpu` even though you have a GPU | Reinstall PyTorch with CUDA support (see [Installation](#️-installation)). |
| `IndexError` when running `predict.py` | Pass an image or folder path: `python predict.py <path>`. |

---

## 🚀 Future Improvements

- [ ] Show the top-3 predictions with a probability bar chart
- [ ] Grad-CAM / attention-map visualisation of what the model looks at
- [ ] Data augmentation (random crop, horizontal flip) and ImageNet-style normalisation
- [ ] Learning-rate scheduler and multi-epoch training with a validation split
- [ ] Mixed-precision (AMP) training for speed
- [ ] Confusion matrix and per-class accuracy report
- [ ] Support for custom datasets
- [ ] Online deployment (Hugging Face Spaces / Streamlit Community Cloud)

---

## 👨‍💻 Author

**Prafful Soni**
B.Tech IT student, interested in AI, Machine Learning and Deep Learning 🚀

GitHub: [@Praffulsoni](https://github.com/Praffulsoni)

---

## ⭐ Acknowledgements

- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [Google Research: Vision Transformer](https://github.com/google-research/vision_transformer), from *Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (ICLR 2021)*
- [PyTorch](https://pytorch.org/) and [torchvision](https://pytorch.org/vision/)
- [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) by Alex Krizhevsky
- [Streamlit](https://streamlit.io/)

If you found this project useful, consider giving it a ⭐!
