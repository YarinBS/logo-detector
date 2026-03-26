"""
Trainer module
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

class Trainer:
    def __init__(self, 
                 model: nn.Module, 
                 train_loader: DataLoader, 
                 val_loader: DataLoader, 
                 lr: float,
                 epochs: int,
                 save_path: str):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.save_path = save_path

        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(
            (p for p in self.model.parameters() if p.requires_grad),
            lr=lr
        )
        self.best_val_loss = float("inf")
    
    def train(self):
        """
        Runs the full training loop across epochs and saves the best checkpoint
        """

        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1}/{self.epochs}", end=" - ")
            train_loss = self.train_epoch()
            val_loss = self.validate_epoch()

            print(f"Train Loss: {round(train_loss, 4)}; Val Loss: {round(val_loss, 4)}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.save_path)
                print(f"New best model saved with val loss: {round(val_loss, 4)}")
        
        print("Training complete. Best validation loss: ", round(self.best_val_loss, 4))
        print(f"Best model saved at: {self.save_path}")

    def train_epoch(self):
        """
        Trains the model for one epoch and returns mean training loss
        """
        
        self.model.train()
        total_loss = 0
        for images, labels in tqdm(self.train_loader, desc="Training"):
            self.optimizer.zero_grad()

            outputs = self.model(images)
            loss = self.criterion(outputs.squeeze(), labels.float())
            
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    def validate_epoch(self):
        """
        Evaluates the model for one epoch on validation data and returns mean loss
        """

        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc="Validation"):
                outputs = self.model(images)
                loss = self.criterion(outputs.squeeze(), labels.float())
                total_loss += loss.item()
        return total_loss / len(self.val_loader)