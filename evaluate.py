import torch
import json
import mlflow
from model import ResNet50
from utils import setup_logger
from data import get_data_loaders
from setup_mlflow import setup_mlflow

def evaluate_model(config):
    logger = setup_logger()
    mlflow = setup_mlflow("cnn_evaluation_experiment")

    _, _, test_loader = get_data_loaders(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet50().to(device)
    model.load_state_dict(torch.load(config.model_path))
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    logger.info(f"Test Accuracy: {accuracy:.4f}")

    with mlflow.start_run() as run:
        mlflow.log_metric("test_accuracy", accuracy)

        with open("metrics.json", "w") as f:
            json.dump({"accuracy": accuracy}, f)
        mlflow.log_artifact("metrics.json")

if __name__ == "__main__":
    from config import Config
    config = Config()
    evaluate_model(config)
