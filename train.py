import os
import torch
import mlflow
from config import Config
from data import get_data_loaders
from model import ResNet50
from utils import setup_logger, plot_training_curves
from train_utils import train_model as train_fn
from setup_mlflow import setup_mlflow

def train_model(config):
    logger = setup_logger()
    mlflow = setup_mlflow("cnn_training_experiment")

    train_loader, val_loader, test_loader = get_data_loaders(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet50().to(device)

    with mlflow.start_run() as run:
        mlflow.log_param("num_epochs", config.num_epochs)
        mlflow.log_param("learning_rate", config.lr)
        mlflow.log_param("batch_size", config.batch_size)
        mlflow.log_param("train_val_split", config.train_val_split)
        mlflow.log_param("combination_method", config.combination_method)
        mlflow.log_param("selected_batches", config.selected_batches)
        
        train_losses, val_losses, val_accuracies = train_fn(model, train_loader, val_loader,
                                                            config.num_epochs, config.lr, device, logger)
        
        for epoch, (train_loss, val_loss, val_acc) in enumerate(zip(train_losses, val_losses, val_accuracies)):
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)

        model_path = config.model_path
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(model_path)

        plot_training_curves(train_losses, val_losses, torch.tensor(val_accuracies))

if __name__ == "__main__":
    config = Config()
    train_model(config)
