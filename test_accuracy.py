import os

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoModelForImageClassification


def load_model():
    """Load the Tengeop model from Hugging Face"""
    model = AutoModelForImageClassification.from_pretrained(
        "galeio-research/nereus-sar-1-tengeop", token=os.environ["HF_TOKEN"]
    ).to("cuda")
    model.eval()
    return model


def generate_test_data(num_samples=100):
    """Generate synthetic test data for evaluation"""
    # Generate random SAR images (single channel)
    images = torch.randn(num_samples, 1, 256, 256).to("cuda")

    # Generate random ground truth labels (10 classes)
    labels = torch.randint(0, 2, (num_samples, 10)).to("cuda")

    return images, labels


def evaluate_model(model, images, labels):
    """Evaluate model accuracy on test data"""
    with torch.no_grad():
        outputs = model(images)
        predictions = torch.sigmoid(outputs.logits) > 0.5

    # Convert to numpy for sklearn metrics
    predictions_np = predictions.cpu().numpy()
    labels_np = labels.cpu().numpy()

    # Calculate accuracy for each class
    accuracies = []
    for i in range(10):
        acc = accuracy_score(labels_np[:, i], predictions_np[:, i])
        accuracies.append(acc)

    # Print detailed classification report
    print("\nClassification Report:")
    print(classification_report(labels_np, predictions_np))

    # Print average accuracy
    print(f"\nAverage Accuracy: {np.mean(accuracies):.4f}")
    print("\nPer-class Accuracies:")
    for i, acc in enumerate(accuracies):
        print(f"Class {i}: {acc:.4f}")


def main():
    # Load model
    model = load_model()

    # Generate test data
    images, labels = generate_test_data()

    # Evaluate model
    evaluate_model(model, images, labels)


if __name__ == "__main__":
    main()
