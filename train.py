# -----------------------
# Imports
# -----------------------
import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import ViTForImageClassification

# -----------------------
# Main function (IMPORTANT)
# -----------------------
def main():
    print("TRAIN.PY LOADED 🧠")
    print("TRAIN.PY STARTED 🚀🔥")

    # -----------------------
    # Basic Config
    # -----------------------
    NUM_CLASSES = 10
    BATCH_SIZE = 32
    EPOCHS = 1
    LEARNING_RATE = 2e-5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # -----------------------
    # Image Transforms
    # -----------------------
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # -----------------------
    # Dataset
    # -----------------------
    train_dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print("Dataset loaded successfully!")
    print("Train samples:", len(train_dataset))
    print("Test samples:", len(test_dataset))

    # -----------------------
    # Model
    # -----------------------
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True
    )

    model.to(device)
    print("ViT model loaded!")

    # -----------------------
    # Loss & Optimizer
    # -----------------------
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # -----------------------
    # Checkpoint
    # -----------------------
    os.makedirs("results", exist_ok=True)
    checkpoint_path = "results/checkpoint.pth"
    start_epoch = 0

    if os.path.exists(checkpoint_path):
        print("Loading checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resuming from epoch {start_epoch}")

    # -----------------------
    # Training
    # -----------------------
    model.train()

    for epoch in range(start_epoch, EPOCHS):
        print(f"\nEpoch [{epoch+1}/{EPOCHS}] 🚀")
        running_loss = 0.0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images).logits
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch completed ✅ Avg Loss: {avg_loss:.4f}")

        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict()
        }, checkpoint_path)

        print("Checkpoint saved 💾")

    # -----------------------
    # Evaluation
    # -----------------------
    print("\nStarting evaluation 🔍")
    model.eval()

    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images).logits
            preds = torch.argmax(outputs, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

    # -----------------------
    # Save final model
    # -----------------------
    model.save_pretrained("results/vit-cifar10")
    print("Model saved to results/vit-cifar10 🎉")


# -----------------------
# REQUIRED FOR WINDOWS
# -----------------------
if __name__ == "__main__":
    main()
