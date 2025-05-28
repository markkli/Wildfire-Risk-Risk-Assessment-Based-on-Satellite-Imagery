"""
Image Classifier Training Module
Handles training of ResNet-based fire risk classifier using satellite imagery.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import numpy as np
import pickle
import argparse
import logging
from pathlib import Path
import json
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FireRiskDataset(Dataset):
    """Custom dataset for fire risk classification"""
    
    def __init__(self, data, transform=None, image_size=(224, 224)):
        self.data = data
        self.transform = transform
        self.image_size = image_size
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        example = self.data[idx]
        
        # Get image and resize for memory efficiency
        image = example['image']
        if self.image_size:
            image = image.resize(self.image_size, Image.BILINEAR)
        
        # Get label
        label = example['label']
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label

class FireRiskClassifier(nn.Module):
    """ResNet-based classifier for fire risk prediction"""
    
    def __init__(self, num_classes=4, pretrained=True):
        super(FireRiskClassifier, self).__init__()
        
        # Use ResNet18 as backbone
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # Replace final layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        return self.backbone(x)

class FireRiskTrainer:
    """Handles training of the fire risk classifier"""
    
    def __init__(self, model, device='auto', class_names=None):
        self.model = model
        self.device = self._setup_device(device)
        self.model.to(self.device)
        self.class_names = class_names or ["High", "Low", "Moderate", "Non-burnable"]
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rates': []
        }
        
    def _setup_device(self, device):
        """Setup training device"""
        if device == 'auto':
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)
        
        logger.info(f"Using device: {device}")
        return device
    
    def create_data_loaders(self, train_data, val_data, batch_size=32, num_workers=2):
        """Create training and validation data loaders"""
        
        # Define transforms
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Create datasets
        train_dataset = FireRiskDataset(train_data, transform=train_transform)
        val_dataset = FireRiskDataset(val_data, transform=val_transform)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader, criterion, optimizer, scheduler=None):
        """Train for one epoch"""
        self.model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='Training')
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100 * correct / total:.2f}%'
            })
        
        # Update learning rate
        if scheduler:
            scheduler.step()
            
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, val_loader, criterion):
        """Validate for one epoch"""
        self.model.eval()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Validation')
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                # Statistics
                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Update progress bar
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100 * correct / total:.2f}%'
                })
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self, train_data, val_data, num_epochs=10, batch_size=32, 
              learning_rate=0.001, weight_decay=1e-4, save_path='models/fire_risk_model.pth'):
        """Main training loop"""
        
        # Create data loaders
        train_loader, val_loader = self.create_data_loaders(
            train_data, val_data, batch_size=batch_size
        )
        
        # Setup loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        
        # Track best model
        best_val_loss = float('inf')
        best_model_state = None
        
        logger.info(f"Starting training for {num_epochs} epochs...")
        logger.info(f"Training batches: {len(train_loader)}, Validation batches: {len(val_loader)}")
        
        for epoch in range(num_epochs):
            logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
            logger.info("-" * 50)
            
            # Training phase
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer, scheduler)
            
            # Validation phase
            val_loss, val_acc = self.validate_epoch(val_loader, criterion)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(optimizer.param_groups[0]['lr'])
            
            # Print epoch results
            logger.info(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            logger.info(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict().copy()
                logger.info(f"New best model saved with validation loss: {best_val_loss:.4f}")
        
        # Load best model and save
        if best_model_state:
            self.model.load_state_dict(best_model_state)
            
        # Save model and training info
        self.save_model(save_path)
        
        logger.info(f"\nTraining completed! Best validation loss: {best_val_loss:.4f}")
        return self.history
    
    def save_model(self, save_path):
        """Save model and metadata"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model state dict
        torch.save(self.model.state_dict(), save_path)
        
        # Save training metadata
        metadata = {
            'model_architecture': 'ResNet18',
            'num_classes': len(self.class_names),
            'class_names': self.class_names,
            'final_train_acc': self.history['train_acc'][-1] if self.history['train_acc'] else 0,
            'final_val_acc': self.history['val_acc'][-1] if self.history['val_acc'] else 0,
            'best_val_loss': min(self.history['val_loss']) if self.history['val_loss'] else float('inf')
        }
        
        metadata_path = save_path.parent / f"{save_path.stem}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save training history
        history_path = save_path.parent / f"{save_path.stem}_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        logger.info(f"Model saved to {save_path}")
        logger.info(f"Metadata saved to {metadata_path}")
        logger.info(f"Training history saved to {history_path}")

def plot_training_history(history, save_path=None):
    """Plot training history"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss plot
    ax1.plot(history['train_loss'], label='Training Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(history['train_acc'], label='Training Accuracy')
    ax2.plot(history['val_acc'], label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    # Learning rate plot
    ax3.plot(history['learning_rates'])
    ax3.set_title('Learning Rate Schedule')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.set_yscale('log')
    ax3.grid(True)
    
    # Combined loss and accuracy
    ax4_twin = ax4.twinx()
    ax4.plot(history['train_loss'], 'b-', label='Train Loss')
    ax4.plot(history['val_loss'], 'b--', label='Val Loss')
    ax4_twin.plot(history['train_acc'], 'r-', label='Train Acc')
    ax4_twin.plot(history['val_acc'], 'r--', label='Val Acc')
    
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss', color='b')
    ax4_twin.set_ylabel('Accuracy', color='r')
    ax4.set_title('Loss and Accuracy Combined')
    ax4.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training plots saved to {save_path}")
    
    plt.show()

def load_processed_data(data_dir):
    """Load preprocessed data"""
    data_dir = Path(data_dir)
    
    train_data = None
    val_data = None
    
    # Load training data
    train_file = data_dir / "train_processed.pkl"
    if train_file.exists():
        with open(train_file, 'rb') as f:
            train_data = pickle.load(f)
        logger.info(f"Loaded training data: {len(train_data)} samples")
    
    # Load validation data (or use test data)
    for val_file_name in ["validation_processed.pkl", "test_processed.pkl"]:
        val_file = data_dir / val_file_name
        if val_file.exists():
            with open(val_file, 'rb') as f:
                val_data = pickle.load(f)
            logger.info(f"Loaded validation data from {val_file_name}: {len(val_data)} samples")
            break
    
    if train_data is None or val_data is None:
        raise FileNotFoundError("Could not find preprocessed training or validation data")
    
    return train_data, val_data

def main():
    parser = argparse.ArgumentParser(description='Train Fire Risk Image Classifier')
    parser.add_argument('--data-dir', default='data/processed/', help='Directory with preprocessed data')
    parser.add_argument('--output-dir', default='models/', help='Output directory for trained model')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--device', default='auto', help='Training device (auto, cpu, cuda)')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained ResNet weights')
    
    args = parser.parse_args()
    
    # Load data
    train_data, val_data = load_processed_data(args.data_dir)
    
    # Create model
    model = FireRiskClassifier(num_classes=4, pretrained=args.pretrained)
    
    # Create trainer
    trainer = FireRiskTrainer(model, device=args.device)
    
    # Train model
    history = trainer.train(
        train_data=train_data,
        val_data=val_data,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        save_path=f"{args.output_dir}/fire_risk_model.pth"
    )
    
    # Plot training history
    plot_training_history(history, save_path=f"{args.output_dir}/training_history.png")
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()
