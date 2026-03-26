"""
Main module of the logo detector project.
This script can be run in two modes:
1. Training mode: `python main.py --mode train` - This will train the model on the dataset and save the trained weights.
2. Serving mode: `python main.py --mode serve` - This will start the FastAPI server to serve predictions using the trained model.
"""

import argparse
import os

import uvicorn

from app import app
from config import config
from dataset import get_dataloaders
from evaluator import evaluate
from model import get_model
from trainer import Trainer
from utils import download_kaggle_dataset

def main():
    """
    Parse CLI arguments and run training or API serving mode
    """
    
    parser = argparse.ArgumentParser(description="Logo Detector")
    parser.add_argument("--mode", choices=["train", "serve"], required=True, help="Mode to run the application in: 'train' or 'serve'")
    args = parser.parse_args()

    if args.mode == "train":

        if not os.path.isdir("data/flickr_logos_27_dataset"):
            print("Dataset not found locally. Downloading from Kaggle...")
            download_kaggle_dataset()

        print("Starting training...")
        model = get_model()
        train_loader, validation_loader = get_dataloaders()

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=validation_loader,
            lr=config["training"]["learning_rate"],
            epochs=config["training"]["num_epochs"],
            save_path=config["training"]["save_path"]
        )
        
        trainer.train()
        print("Training completed successfully.")
        
        evaluation_metrics = evaluate(model, validation_loader)
        print("Evaluation Metrics:")
        for metric, value in evaluation_metrics.items():
            print(f"  {metric.capitalize()}: {value}")

    elif args.mode == "serve":
        uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
