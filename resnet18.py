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
import os


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

class EarlyStopping:
    """Early stopping handler to prevent overfitting"""
    def __init__(self, patience=5, min_delta=0.0, mode='max'):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for metrics like accuracy, 'min' for loss
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.early_stop = False

    def __call__(self, current_value):
        if self.best_value is None:
            self.best_value = current_value
            return False

        if self.mode == 'max':
            if current_value > self.best_value + self.min_delta:
                self.best_value = current_value
                self.counter = 0
            else:
                self.counter += 1
        else:  # mode == 'min'
            if current_value < self.best_value - self.min_delta:
                self.best_value = current_value
                self.counter = 0
            else:
                self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True
            return True
        return False


def get_warmup_scheduler(optimizer, warmup_epochs=3):
    """Create a warmup learning rate scheduler"""
    def warmup_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 1.0

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)


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
                num_epochs, device, regime_name="", use_amp=False, output_dir="./output",
                early_stopping_patience=None, warmup_epochs=0):
    """Train a complete regime with optional early stopping and warmup"""
    print(f"\n{'='*70}")
    print(f"Training Regime: {regime_name}")
    if use_amp:
        print(f"Mixed Precision (AMP): Enabled")
    if early_stopping_patience:
        print(f"Early Stopping: Enabled (patience={early_stopping_patience})")
    if warmup_epochs > 0:
        print(f"Learning Rate Warmup: {warmup_epochs} epochs")
    print(f"{'='*70}")

    # Create output directory for this regime
    os.makedirs(output_dir, exist_ok=True)

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    # Initialize early stopping if enabled
    early_stopper = None
    if early_stopping_patience:
        early_stopper = EarlyStopping(patience=early_stopping_patience, min_delta=0.001, mode='max')

    # Initialize warmup scheduler if enabled
    warmup_scheduler = None
    if warmup_epochs > 0:
        warmup_scheduler = get_warmup_scheduler(optimizer, warmup_epochs)

    best_val_acc = 0.0
    start_time = time.time()
    epochs_trained = 0

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, use_amp=use_amp
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Apply warmup scheduler for first few epochs
        if warmup_scheduler and epoch < warmup_epochs:
            warmup_scheduler.step()
        # Apply main scheduler after warmup
        elif scheduler and epoch >= warmup_epochs:
            scheduler.step()

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"LR: {current_lr:.6f} | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = os.path.join(output_dir, 'best_model.pth')
            torch.save(model.state_dict(), model_path)

        # Check early stopping
        if early_stopper and early_stopper(val_acc):
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            print(f"No improvement for {early_stopping_patience} epochs")
            epochs_trained = epoch + 1
            break

        epochs_trained = epoch + 1

    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time/60:.2f} minutes")
    print(f"Epochs trained: {epochs_trained}/{num_epochs}")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")

    return history, training_time, best_val_acc, epochs_trained


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

def plot_training_curves(history, regime_name, output_dir):
    """Plot training and validation curves for a single regime"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Loss plot
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'{regime_name} - Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)

    # Accuracy plot
    axes[1].plot(history['train_acc'], label='Train Accuracy', marker='o')
    axes[1].plot(history['val_acc'], label='Val Accuracy', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title(f'{regime_name} - Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_comparison_curves(history1, history2, regime1_name, regime2_name, output_dir):
    """Plot comparison of two regimes side by side"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Regime 1 Loss
    axes[0, 0].plot(history1['train_loss'], label='Train Loss', marker='o')
    axes[0, 0].plot(history1['val_loss'], label='Val Loss', marker='s')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title(f'{regime1_name} - Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Regime 1 Accuracy
    axes[0, 1].plot(history1['train_acc'], label='Train Acc', marker='o')
    axes[0, 1].plot(history1['val_acc'], label='Val Acc', marker='s')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title(f'{regime1_name} - Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Regime 2 Loss
    axes[1, 0].plot(history2['train_loss'], label='Train Loss', marker='o', color='orange')
    axes[1, 0].plot(history2['val_loss'], label='Val Loss', marker='s', color='red')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title(f'{regime2_name} - Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Regime 2 Accuracy
    axes[1, 1].plot(history2['train_acc'], label='Train Acc', marker='o', color='orange')
    axes[1, 1].plot(history2['val_acc'], label='Val Acc', marker='s', color='red')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy (%)')
    axes[1, 1].set_title(f'{regime2_name} - Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'regime_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_confusion_matrix(conf_matrix, regime_name, output_dir, num_classes=62):
    """Plot confusion matrix"""
    plt.figure(figsize=(12, 10))
    plt.imshow(conf_matrix, cmap='Blues', interpolation='nearest')
    plt.colorbar()
    plt.title(f'Confusion Matrix - {regime_name} (EMNIST ByClass - 62 classes)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def save_regime_summary(test_metrics, regime_name, trainable_params, training_time,
                       num_epochs, epochs_trained, device, train_dataset, output_dir):
    """Save results summary for a single regime"""
    save_path = os.path.join(output_dir, 'results_summary.txt')

    with open(save_path, 'w') as f:
        f.write(f"TASK 2: ResNet-18 EMNIST Fine-Tuning Results\n")
        f.write(f"Regime: {regime_name}\n")
        f.write("="*70 + "\n\n")

        f.write("Test Metrics:\n")
        f.write(f"  Accuracy: {test_metrics['accuracy']*100:.2f}%\n")
        f.write(f"  Precision (macro): {test_metrics['precision']:.4f}\n")
        f.write(f"  Recall (macro): {test_metrics['recall']:.4f}\n")
        f.write(f"  F1-Score (macro): {test_metrics['f1']:.4f}\n\n")

        f.write("Efficiency Accounting:\n")
        f.write(f"  Trainable Parameters: {trainable_params:.2f}M\n")
        f.write(f"  Max Epochs: {num_epochs}\n")
        f.write(f"  Epochs Trained: {epochs_trained}")
        if epochs_trained < num_epochs:
            f.write(f" (early stopped)\n")
        else:
            f.write("\n")
        f.write(f"  Samples per Epoch: {len(train_dataset)}\n")
        f.write(f"  Total Update Budget: {epochs_trained * len(train_dataset)}\n")
        f.write(f"  Training Time: {training_time/60:.2f} minutes\n")
        f.write(f"  Device: {device}\n")

    print(f"Saved: {save_path}")


def save_comparison_summary(results1, results2, regime1_name, regime2_name, output_dir):
    """Save comparison summary for both regimes"""
    save_path = os.path.join(output_dir, 'regime_comparison_summary.txt')

    with open(save_path, 'w') as f:
        f.write("TASK 2: ResNet-18 EMNIST Fine-Tuning - Regime Comparison\n")
        f.write("="*80 + "\n\n")

        # Table header
        f.write(f"{'Regime':<25} {'Test Acc':<12} {'Params (M)':<12} {'Epochs':<10} "
                f"{'Time (min)':<12}\n")
        f.write("-" * 80 + "\n")

        # Regime 1
        f.write(f"{regime1_name:<25} "
                f"{results1['test_acc']*100:>10.2f}% "
                f"{results1['trainable_params']:>10.2f}M "
                f"{results1['num_epochs']:>8} "
                f"{results1['training_time']/60:>10.2f}\n")

        # Regime 2
        f.write(f"{regime2_name:<25} "
                f"{results2['test_acc']*100:>10.2f}% "
                f"{results2['test_acc']*100:>10.2f}% "
                f"{results2['trainable_params']:>10.2f}M "
                f"{results2['num_epochs']:>8} "
                f"{results2['training_time']/60:>10.2f}\n")

        f.write("\n" + "="*80 + "\n\n")

        # Detailed comparison
        f.write("DETAILED COMPARISON:\n\n")

        f.write(f"Regime 1: {regime1_name}\n")
        f.write(f"  Test Accuracy: {results1['test_acc']*100:.2f}%\n")
        f.write(f"  Precision: {results1['precision']:.4f}\n")
        f.write(f"  Recall: {results1['recall']:.4f}\n")
        f.write(f"  F1-Score: {results1['f1']:.4f}\n")
        f.write(f"  Trainable Params: {results1['trainable_params']:.2f}M\n")
        f.write(f"  Training Time: {results1['training_time']/60:.2f} min\n\n")

        f.write(f"Regime 2: {regime2_name}\n")
        f.write(f"  Test Accuracy: {results2['test_acc']*100:.2f}%\n")
        f.write(f"  Precision: {results2['precision']:.4f}\n")
        f.write(f"  Recall: {results2['recall']:.4f}\n")
        f.write(f"  F1-Score: {results2['f1']:.4f}\n")
        f.write(f"  Trainable Params: {results2['trainable_params']:.2f}M\n")
        f.write(f"  Training Time: {results2['training_time']/60:.2f} min\n\n")

        # Trade-off analysis
        f.write("TRADE-OFF ANALYSIS:\n")
        acc_diff = (results2['test_acc'] - results1['test_acc']) * 100
        time_diff = (results2['training_time'] - results1['training_time']) / 60
        param_diff = results2['trainable_params'] - results1['trainable_params']

        f.write(f"  Accuracy improvement (Regime 2 - Regime 1): {acc_diff:+.2f}%\n")
        f.write(f"  Additional training time: {time_diff:+.2f} minutes\n")
        f.write(f"  Additional trainable params: {param_diff:+.2f}M\n")

    print(f"Saved: {save_path}")


# ============================================================================
# MAIN TRAINING SCRIPT
# ============================================================================

def main():
    """Main training function - trains two independent regimes for comparison"""
    # Set random seed
    set_seed(42)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load datasets
    train_dataset, val_dataset, test_dataset = load_datasets()

    # Create data loaders with smaller batch size for efficiency
    batch_size = 128  # Reduced from 256 for better memory efficiency
    train_loader, val_loader, test_loader = get_data_loaders(
        train_dataset, val_dataset, test_dataset, batch_size=batch_size
    )
    print(f"Batch size: {batch_size}")

    # Create main output directory
    main_output_dir = './output'
    os.makedirs(main_output_dir, exist_ok=True)

    # ========================================================================
    # REGIME 1: LINEAR PROBE (ONLY FC LAYER)
    # ========================================================================
    print(f"\n{'='*70}")
    print("REGIME 1: LINEAR PROBE (Freeze backbone, train only FC)")
    print(f"{'='*70}")

    regime1_output_dir = os.path.join(main_output_dir, 'regime1_linear_probe')
    os.makedirs(regime1_output_dir, exist_ok=True)

    # Reset seed for reproducibility
    set_seed(42)

    # Create fresh model
    model1 = get_resnet18_model().to(device)
    freeze_backbone(model1)

    criterion = nn.CrossEntropyLoss()
    optimizer1 = optim.SGD(model1.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler1 = optim.lr_scheduler.StepLR(optimizer1, step_size=5, gamma=0.1)

    trainable_params_regime1 = count_trainable_params_millions(model1)
    num_epochs_regime1 = 15  # Max epochs, but will use early stopping

    # Train with early stopping
    history_regime1, time_regime1, best_val_acc_regime1, epochs_trained_regime1 = train_regime(
        model1, train_loader, val_loader, criterion, optimizer1, scheduler1,
        num_epochs=num_epochs_regime1, device=device, regime_name="Regime 1: Linear Probe",
        use_amp=torch.cuda.is_available(), output_dir=regime1_output_dir,
        early_stopping_patience=5  # Stop if no improvement for 5 epochs
    )

    # Evaluate Regime 1 on test set
    model1.load_state_dict(torch.load(os.path.join(regime1_output_dir, 'best_model.pth')))
    test_metrics1 = evaluate_full_metrics(model1, test_loader, device)

    print(f"\nRegime 1 Test Results:")
    print(f"  Accuracy: {test_metrics1['accuracy']*100:.2f}%")
    print(f"  Precision: {test_metrics1['precision']:.4f}")
    print(f"  Recall: {test_metrics1['recall']:.4f}")
    print(f"  F1-Score: {test_metrics1['f1']:.4f}")

    # Save Regime 1 results
    plot_training_curves(history_regime1, "Regime 1: Linear Probe", regime1_output_dir)
    plot_confusion_matrix(test_metrics1['confusion_matrix'], "Regime 1: Linear Probe", regime1_output_dir)
    save_regime_summary(test_metrics1, "Regime 1: Linear Probe", trainable_params_regime1,
                       time_regime1, num_epochs_regime1, epochs_trained_regime1, device, train_dataset, regime1_output_dir)

    # ========================================================================
    # REGIME 2: PARTIAL UNFREEZE (LAYER4 + FC from scratch)
    # ========================================================================
    print(f"\n{'='*70}")
    print("REGIME 2: PARTIAL UNFREEZE (Unfreeze layer4 + FC from start)")
    print(f"{'='*70}")

    regime2_output_dir = os.path.join(main_output_dir, 'regime2_partial_unfreeze')
    os.makedirs(regime2_output_dir, exist_ok=True)

    # Reset seed for reproducibility
    set_seed(42)

    # Create fresh model (independent from Regime 1)
    model2 = get_resnet18_model().to(device)
    unfreeze_last_block(model2)  # Unfreeze layer4 + FC

    criterion = nn.CrossEntropyLoss()
    optimizer2 = optim.SGD(model2.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=20)

    trainable_params_regime2 = count_trainable_params_millions(model2)
    num_epochs_regime2 = 20  # Max epochs, but will use early stopping

    # Train with warmup and early stopping
    history_regime2, time_regime2, best_val_acc_regime2, epochs_trained_regime2 = train_regime(
        model2, train_loader, val_loader, criterion, optimizer2, scheduler2,
        num_epochs=num_epochs_regime2, device=device, regime_name="Regime 2: Partial Unfreeze",
        use_amp=torch.cuda.is_available(), output_dir=regime2_output_dir,
        early_stopping_patience=5,  # Stop if no improvement for 5 epochs
        warmup_epochs=3  # Warmup for first 3 epochs
    )

    # Evaluate Regime 2 on test set
    model2.load_state_dict(torch.load(os.path.join(regime2_output_dir, 'best_model.pth')))
    test_metrics2 = evaluate_full_metrics(model2, test_loader, device)

    print(f"\nRegime 2 Test Results:")
    print(f"  Accuracy: {test_metrics2['accuracy']*100:.2f}%")
    print(f"  Precision: {test_metrics2['precision']:.4f}")
    print(f"  Recall: {test_metrics2['recall']:.4f}")
    print(f"  F1-Score: {test_metrics2['f1']:.4f}")

    # Save Regime 2 results
    plot_training_curves(history_regime2, "Regime 2: Partial Unfreeze", regime2_output_dir)
    plot_confusion_matrix(test_metrics2['confusion_matrix'], "Regime 2: Partial Unfreeze", regime2_output_dir)
    save_regime_summary(test_metrics2, "Regime 2: Partial Unfreeze", trainable_params_regime2,
                       time_regime2, num_epochs_regime2, epochs_trained_regime2, device, train_dataset, regime2_output_dir)

    # ========================================================================
    # GENERATE COMPARISON REPORTS
    # ========================================================================
    print(f"\n{'='*70}")
    print("GENERATING COMPARISON REPORTS")
    print(f"{'='*70}")

    # Prepare results dictionaries
    results1 = {
        'test_acc': test_metrics1['accuracy'],
        'precision': test_metrics1['precision'],
        'recall': test_metrics1['recall'],
        'f1': test_metrics1['f1'],
        'trainable_params': trainable_params_regime1,
        'num_epochs': epochs_trained_regime1,  # Use actual epochs trained
        'training_time': time_regime1
    }

    results2 = {
        'test_acc': test_metrics2['accuracy'],
        'precision': test_metrics2['precision'],
        'recall': test_metrics2['recall'],
        'f1': test_metrics2['f1'],
        'trainable_params': trainable_params_regime2,
        'num_epochs': epochs_trained_regime2,  # Use actual epochs trained
        'training_time': time_regime2
    }

    # Save comparison visualizations and summary
    plot_comparison_curves(history_regime1, history_regime2,
                          "Regime 1: Linear Probe", "Regime 2: Partial Unfreeze",
                          main_output_dir)
    save_comparison_summary(results1, results2,
                           "Regime 1: Linear Probe", "Regime 2: Partial Unfreeze",
                           main_output_dir)

    # Print final comparison table
    print(f"\n{'='*80}")
    print("FINAL COMPARISON TABLE")
    print(f"{'='*80}")
    print(f"{'Regime':<30} {'Test Acc':<12} {'Params (M)':<12} {'Epochs':<10} {'Time (min)':<12}")
    print("-" * 80)
    print(f"{'Regime 1: Linear Probe':<30} {results1['test_acc']*100:>10.2f}% "
          f"{results1['trainable_params']:>10.2f}M {results1['num_epochs']:>8} "
          f"{results1['training_time']/60:>10.2f}")
    print(f"{'Regime 2: Partial Unfreeze':<30} {results2['test_acc']*100:>10.2f}% "
          f"{results2['trainable_params']:>10.2f}M {results2['num_epochs']:>8} "
          f"{results2['training_time']/60:>10.2f}")
    print(f"{'='*80}")

    acc_improvement = (results2['test_acc'] - results1['test_acc']) * 100
    print(f"\nAccuracy improvement (Regime 2 - Regime 1): {acc_improvement:+.2f}%")
    print(f"Additional training time: {(results2['training_time'] - results1['training_time'])/60:+.2f} min")
    print(f"Additional trainable params: {(results2['trainable_params'] - results1['trainable_params']):+.2f}M")

    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print(f"Results saved to: {main_output_dir}/")
    print("="*70)


if __name__ == "__main__":
    main()
