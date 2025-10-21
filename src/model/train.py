"""
Training pipeline for lesion classification CNN.

Handles dataset loading, model training, validation, and checkpointing.
"""

import os
import argparse
import yaml
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets
from tqdm import tqdm

from src.model.cnn import create_model
from src.preprocessing.image_processing import ImagePreprocessor


class LesionDataset(Dataset):
    """
    Custom dataset for lesion images.

    Expects directory structure:
    data_dir/
        class1/
            image1.jpg
            image2.jpg
        class2/
            image1.jpg
    """

    def __init__(self, data_dir: str, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        self.classes = []

        self._load_samples()

    def _load_samples(self):
        """Load all image paths and labels."""
        # Get all class directories
        class_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        self.classes = sorted([d.name for d in class_dirs])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        # Load all samples
        for class_dir in class_dirs:
            class_idx = self.class_to_idx[class_dir.name]
            for img_path in class_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    self.samples.append((str(img_path), class_idx))

        print(f"Loaded {len(self.samples)} samples from {len(self.classes)} classes")
        print(f"Classes: {self.classes}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # Load and preprocess image
        if self.transform:
            image = self.transform.preprocess(img_path)
        else:
            from PIL import Image
            image = Image.open(img_path).convert('RGB')
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

        return image, label


class Trainer:
    """
    Handles model training and validation.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: str = 'cuda',
        save_dir: str = 'data/models'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc='Training')
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss / (pbar.n + 1),
                'acc': 100 * correct / total
            })

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100 * correct / total

        return epoch_loss, epoch_acc

    def validate(self) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc='Validation'):
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100 * correct / total

        return epoch_loss, epoch_acc

    def train(self, num_epochs: int, early_stopping_patience: int = 10):
        """
        Train the model for multiple epochs.

        Args:
            num_epochs: Number of epochs to train
            early_stopping_patience: Stop if no improvement for this many epochs
        """
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            # Train
            train_loss, train_acc = self.train_epoch()
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)

            # Validate
            val_loss, val_acc = self.validate()
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)

            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_checkpoint('best_model.pth')
                print("Saved best model!")
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping after {epoch + 1} epochs")
                break

            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pth')

        # Save final model
        self.save_checkpoint('final_model.pth')
        self.save_training_history()

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
        }
        torch.save(checkpoint, self.save_dir / filename)

    def save_training_history(self):
        """Save training history as YAML."""
        history = {
            'train_losses': [float(x) for x in self.train_losses],
            'val_losses': [float(x) for x in self.val_losses],
            'train_accuracies': [float(x) for x in self.train_accuracies],
            'val_accuracies': [float(x) for x in self.val_accuracies],
        }

        with open(self.save_dir / 'training_history.yaml', 'w') as f:
            yaml.dump(history, f)


def main():
    parser = argparse.ArgumentParser(description='Train lesion classification CNN')
    parser.add_argument('--data-dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--model-type', type=str, default='cnn', choices=['cnn', 'resnet'], help='Model architecture')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--val-split', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--num-classes', type=int, default=None, help='Number of classes (auto-detected if not specified)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    parser.add_argument('--save-dir', type=str, default='data/models', help='Directory to save models')

    args = parser.parse_args()

    print("=" * 60)
    print("Lesion Classification CNN Training")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Model type: {args.model_type}")
    print(f"Device: {args.device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print("=" * 60)

    # Create preprocessor with augmentation
    preprocessor = ImagePreprocessor(augment=True)

    # Load dataset
    dataset = LesionDataset(args.data_dir, transform=preprocessor)

    # Determine number of classes
    num_classes = args.num_classes if args.num_classes else len(dataset.classes)
    print(f"\nNumber of classes: {num_classes}")
    print(f"Class names: {dataset.classes}")

    # Split into train and validation
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"\nTrain samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Create model
    model = create_model(model_type=args.model_type, num_classes=num_classes)
    print(f"\nModel created with {sum(p.numel() for p in model.parameters())} parameters")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=args.device,
        save_dir=args.save_dir
    )

    # Train
    print("\nStarting training...")
    trainer.train(num_epochs=args.epochs)

    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Models saved to: {args.save_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
