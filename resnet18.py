"""
Task 2: ResNet-18 Fine-Tuning on EMNIST
Efficient fine-tuning using linear probe and partial unfreeze strategies
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision import transforms, models
from torchvision.datasets import EMNIST
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import time
import random


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_trainable_params(model):
    """Count total trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_trainable_params_millions(model):
    """Count trainable parameters in millions"""
    return count_trainable_params(model) / 1e6


# ============================================================================
# MODEL FUNCTIONS
# ============================================================================

def get_resnet18_model(num_classes=62):
    """Load pretrained ResNet-18 and modify final layer"""
    model = models.resnet18(pretrained=True)

    # Replace final FC layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    return model


def freeze_backbone(model):
    """Freeze all layers except final FC layer"""
    for name, param in model.named_parameters():
        if 'fc' not in name:
            param.requires_grad = False
    print(f"Frozen backbone. Trainable params: {count_trainable_params_millions(model):.2f}M")


def unfreeze_last_block(model):
    """Unfreeze layer4 and FC layer"""
    for name, param in model.named_parameters():
        if 'layer4' in name or 'fc' in name:
            param.requires_grad = True
    print(f"Unfroze layer4. Trainable params: {count_trainable_params_millions(model):.2f}M")


def unfreeze_all(model):
    """Unfreeze all layers"""
    for param in model.parameters():
        param.requires_grad = True
    print(f"Unfroze all params. Trainable params: {count_trainable_params_millions(model):.2f}M")


# ============================================================================
# DATA LOADING
# ============================================================================

def get_data_transforms():
    """Get data transforms for EMNIST"""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        transforms.RandomRotation((-90, -90)),  # Rotate -90 degrees
        transforms.RandomHorizontalFlip(p=1.0),  # Flip horizontally
        transforms.Resize((224, 224)),  # Resize to 224 x 224
        transforms.Grayscale(num_output_channels=3),  # Convert to 3 channels
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    return transform


def load_datasets():
    """Load and split EMNIST datasets"""
    transform = get_data_transforms()

    # Load train dataset
    full_train_dataset = EMNIST(
        root='./data',
        split='byclass',
        train=True,
        download=True,
        transform=transform
    )

    # Split into train and validation (90/10)
    train_size = int(0.9 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    # Load test dataset
    test_dataset = EMNIST(
        root='./data',
        split='byclass',
        train=False,
        download=True,
        transform=transform
    )

    print(f"Image size: {full_train_dataset[0][0].shape}")
    print(f"Dataset length: {len(full_train_dataset)}")
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    return train_dataset, val_dataset, test_dataset


def get_data_loaders(train_dataset, val_dataset, test_dataset, batch_size=256):
    """Create data loaders"""
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_one_epoch(model, train_loader, criterion, optimizer, device, use_amp=False):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # Create GradScaler for AMP if enabled
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        if use_amp:
            # Use autocast for mixed precision
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            # Scale loss and backprop
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Normal training without AMP
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        return epoch_loss, epoch_acc


def train_regime(model, train_loader, val_loader, criterion, optimizer, scheduler,
                num_epochs, device, regime_name="", use_amp=False):
    """Train a complete regime"""
    print(f"\n{'='*70}")
    print(f"Training Regime: {regime_name}")
    if use_amp:
        print(f"Mixed Precision (AMP): Enabled")
    print(f"{'='*70}")

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    best_val_acc = 0.0
    start_time = time.time()

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, use_amp=use_amp
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if scheduler:
            scheduler.step()

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'{regime_name}_best_model.pth')

    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time/60:.2f} minutes")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")

    return history, training_time


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def evaluate_full_metrics(model, test_loader, device):
    """Evaluate model and compute all metrics"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        conf_matrix = confusion_matrix(all_labels, all_preds)

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': conf_matrix,
            'predictions': all_preds,
            'labels': all_labels
        }


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_training_curves(history1, history2, regime1_name, regime2_name):
    """Plot training and validation curves"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Loss plot
    axes[0].plot(history1['train_loss'], label=f'{regime1_name} Train')
    axes[0].plot(history1['val_loss'], label=f'{regime1_name} Val')
    axes[0].plot(range(len(history1['train_loss']),
                      len(history1['train_loss']) + len(history2['train_loss'])),
                history2['train_loss'], label=f'{regime2_name} Train')
    axes[0].plot(range(len(history1['val_loss']),
                      len(history1['val_loss']) + len(history2['val_loss'])),
                history2['val_loss'], label=f'{regime2_name} Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)

    # Accuracy plot
    axes[1].plot(history1['train_acc'], label=f'{regime1_name} Train')
    axes[1].plot(history1['val_acc'], label=f'{regime1_name} Val')
    axes[1].plot(range(len(history1['train_acc']),
                      len(history1['train_acc']) + len(history2['train_acc'])),
                history2['train_acc'], label=f'{regime2_name} Train')
    axes[1].plot(range(len(history1['val_acc']),
                      len(history1['val_acc']) + len(history2['val_acc'])),
                history2['val_acc'], label=f'{regime2_name} Val')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    print("Saved: training_curves.png")


def plot_confusion_matrix(conf_matrix, num_classes=62):
    """Plot confusion matrix"""
    plt.figure(figsize=(12, 10))
    plt.imshow(conf_matrix, cmap='Blues', interpolation='nearest')
    plt.colorbar()
    plt.title('Confusion Matrix (EMNIST ByClass - 62 classes)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("Saved: confusion_matrix.png")


def save_results_summary(test_metrics, trainable_params_regime1, trainable_params_regime2,
                        time_regime1, time_regime2, device, train_dataset):
    """Save results to text file"""
    total_time = time_regime1 + time_regime2

    with open('results_summary.txt', 'w') as f:
        f.write("TASK 2: ResNet-18 EMNIST Fine-Tuning Results\n")
        f.write("="*70 + "\n\n")

        f.write(f"Test Accuracy: {test_metrics['accuracy']*100:.2f}%\n")
        f.write(f"Test Precision (macro): {test_metrics['precision']:.4f}\n")
        f.write(f"Test Recall (macro): {test_metrics['recall']:.4f}\n")
        f.write(f"Test F1-Score (macro): {test_metrics['f1']:.4f}\n\n")

        f.write("Efficiency Accounting:\n")
        f.write(f"Regime 1 (Linear Probe): {trainable_params_regime1:.2f}M params, "
                f"{time_regime1/60:.2f} min\n")
        f.write(f"Regime 2 (Partial Unfreeze): {trainable_params_regime2:.2f}M params, "
                f"{time_regime2/60:.2f} min\n")
        f.write(f"Total time: {total_time/60:.2f} minutes\n")
        f.write(f"Device: {device}\n")

    print("\nSaved: results_summary.txt")


def print_efficiency_table(test_metrics, trainable_params_regime1, trainable_params_regime2,
                          time_regime1, time_regime2, device, train_dataset):
    """Print efficiency accounting table"""
    print(f"\n{'='*70}")
    print("EFFICIENCY ACCOUNTING")
    print(f"{'='*70}")

    total_train_samples = len(train_dataset)

    print(f"\n{'Regime':<20} {'Test Acc':<12} {'Params (M)':<12} {'Epochs×Samples':<15} "
          f"{'Hardware':<15} {'Time (min)':<12}")
    print("-" * 100)

    # Regime 1
    epochs_regime1 = 10
    print(f"{'Linear Probe':<20} {test_metrics['accuracy']*100:>10.2f}% "
          f"{trainable_params_regime1:>10.2f}M "
          f"{epochs_regime1}×{total_train_samples:<8} {str(device):<15} {time_regime1/60:>10.2f}")

    # Regime 2
    epochs_regime2 = 15
    total_time = time_regime1 + time_regime2
    print(f"{'Partial Unfreeze':<20} {test_metrics['accuracy']*100:>10.2f}% "
          f"{trainable_params_regime2:>10.2f}M "
          f"{epochs_regime2}×{total_train_samples:<8} {str(device):<15} {time_regime2/60:>10.2f}")

    print(f"\nTotal training time: {total_time/60:.2f} minutes")


# ============================================================================
# MAIN TRAINING SCRIPT
# ============================================================================

def main():
    """Main training function"""
    # Set random seed
    set_seed(42)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load datasets
    train_dataset, val_dataset, test_dataset = load_datasets()

    # Create data loaders
    train_loader, val_loader, test_loader = get_data_loaders(
        train_dataset, val_dataset, test_dataset, batch_size=256
    )

    # ========================================================================
    # REGIME 1: LINEAR PROBE
    # ========================================================================
    print(f"\n{'='*70}")
    print("REGIME 1: LINEAR PROBE")
    print(f"{'='*70}")

    model = get_resnet18_model().to(device)
    freeze_backbone(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    trainable_params_regime1 = count_trainable_params_millions(model)

    history_regime1, time_regime1 = train_regime(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        num_epochs=10, device=device, regime_name="linear_probe",
        use_amp=torch.cuda.is_available()
    )

    # ========================================================================
    # REGIME 2: PARTIAL UNFREEZE
    # ========================================================================
    print(f"\n{'='*70}")
    print("REGIME 2: PARTIAL UNFREEZE (LAYER4 + FC)")
    print(f"{'='*70}")

    # Load best model from Regime 1
    model.load_state_dict(torch.load('linear_probe_best_model.pth'))
    unfreeze_last_block(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

    trainable_params_regime2 = count_trainable_params_millions(model)

    history_regime2, time_regime2 = train_regime(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        num_epochs=15, device=device, regime_name="partial_unfreeze",
        use_amp=torch.cuda.is_available()
    )

    # ========================================================================
    # EVALUATE ON TEST SET
    # ========================================================================
    model.load_state_dict(torch.load('partial_unfreeze_best_model.pth'))

    print(f"\n{'='*70}")
    print("EVALUATING ON TEST SET")
    print(f"{'='*70}")

    test_metrics = evaluate_full_metrics(model, test_loader, device)
    print(f"Test Accuracy: {test_metrics['accuracy']*100:.2f}%")
    print(f"Test Precision (macro): {test_metrics['precision']:.4f}")
    print(f"Test Recall (macro): {test_metrics['recall']:.4f}")
    print(f"Test F1-Score (macro): {test_metrics['f1']:.4f}")

    # ========================================================================
    # GENERATE PLOTS AND REPORTS
    # ========================================================================
    plot_training_curves(history_regime1, history_regime2, "Linear Probe", "Partial Unfreeze")
    plot_confusion_matrix(test_metrics['confusion_matrix'])

    print_efficiency_table(test_metrics, trainable_params_regime1, trainable_params_regime2,
                          time_regime1, time_regime2, device, train_dataset)

    save_results_summary(test_metrics, trainable_params_regime1, trainable_params_regime2,
                        time_regime1, time_regime2, device, train_dataset)

    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
