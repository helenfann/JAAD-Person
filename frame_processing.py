import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

# Define base directory for your dataset
base_dir = 'images'  # Adjust this path to your actual data directory

# Define transforms for data augmentation (equivalent to TensorFlow's ImageDataGenerator)
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(15),           # rotation_range=15
    transforms.RandomHorizontalFlip(),       # horizontal_flip=True
    transforms.RandomAffine(                 # width_shift, height_shift, shear, zoom
        degrees=0, 
        translate=(0.1, 0.1),               # width_shift_range=0.1, height_shift_range=0.1
        scale=(0.9, 1.1),                   # zoom_range=0.1
        shear=0.1                           # shear_range=0.1
    ),
    transforms.ToTensor(),                   # rescale=1./255 (converts to [0,1] range)
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),                   # rescale=1./255
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create datasets (equivalent to flow_from_directory)
train_dataset = datasets.ImageFolder(
    root=f'{base_dir}/train',
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    root=f'{base_dir}/val',
    transform=val_transform
)

# Create data loaders (equivalent to generators)
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# Utility functions
def get_dataset_info():
    """Print information about the datasets"""
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Number of classes: {len(train_dataset.classes)}")
    print(f"Class names: {train_dataset.classes}")
    print(f"Class to index mapping: {train_dataset.class_to_idx}")

def test_data_loading():
    """Test the data loading functionality"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test one batch
    for batch_idx, (data, target) in enumerate(train_loader):
        print(f"Batch shape: {data.shape}")  # Should be [32, 3, 224, 224]
        print(f"Target shape: {target.shape}")  # Should be [32]
        print(f"Data range: [{data.min():.3f}, {data.max():.3f}]")
        print(f"Targets: {target}")
        break

# Example CNN model definition (PyTorch equivalent)
def create_cnn_model(num_classes=2):
    """
    Create a CNN model equivalent to your TensorFlow model:
    
    Original TensorFlow model:
    Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary output
    """
    import torch.nn as nn
    
    model = nn.Sequential(
        # First conv block
        nn.Conv2d(3, 32, kernel_size=3, padding=1),  # input: 3 channels, output: 32
        nn.ReLU(),
        nn.MaxPool2d(2, 2),  # 224x224 -> 112x112
        
        # Second conv block
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),  # 112x112 -> 56x56
        
        # Third conv block
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),  # 56x56 -> 28x28
        
        # Flatten and dense layers
        nn.Flatten(),  # 128 * 28 * 28 = 100352
        nn.Linear(128 * 28 * 28, 512),
        nn.ReLU(),
        nn.Linear(512, num_classes)  # For binary: use 2 classes, not 1
    )
    
    return model

# Uncomment to test the data loading
# if __name__ == "__main__":
#     get_dataset_info()
#     test_data_loading()