# Computer Vision Assignment 1

Deep learning assignment focused on CNN implementation from scratch and ResNet-18 fine-tuning.

## Project Structure

### Training Scripts
- `cnn_from_scratch_empty.py` - Custom CNN implementation from scratch using PyTorch primitives
- `resnet18.py` - ResNet-18 fine-tuning on EMNIST dataset with two training regimes

### SLURM Job Scripts
- `run_cnn.sbatch` - Submit CNN training job (4 hours, single GPU)
- `run_resnet.sbatch` - Submit ResNet training job (6 hours, single GPU)

### Datasets
- **CNN**: MNIST (handwritten digits, 10 classes)
- **ResNet**: EMNIST ByClass (62 classes - digits, uppercase, lowercase)

## CNN from Scratch

Implements forward/backward propagation for:
- Convolutional layers (Conv2D)
- Pooling layers (MaxPool2D)
- Fully connected layers (Linear)
- Activation functions (ReLU)
- Cross-entropy loss

**Architecture**: Conv(1→8) → ReLU → MaxPool → Conv(8→16) → ReLU → MaxPool → Flatten → Linear(784→10)

**Training**: 5 epochs, batch size 640, SGD optimizer

## ResNet-18 Fine-Tuning

Two training regimes compared:

### Regime 1: Linear Probe
- Freeze backbone, train only final FC layer
- 15 max epochs with early stopping
- Learning rate: 0.01
- Trainable params: ~0.03M

### Regime 2: Partial Unfreeze
- Unfreeze layer4 + FC from start
- 20 max epochs with early stopping + warmup
- Learning rate: 0.0005
- Trainable params: ~3.1M

Both use mixed precision training (AMP) and comprehensive evaluation metrics.

## SLURM Execution

Both scripts run inside a Singularity container with CUDA support:
- CUDA 12.6.2, cuDNN 9.5.0
- Ubuntu 24.04.1
- Environment setup via `setup_env.sh`
- Automated git pulls before execution
- Job output logs: `{cnn,resnet}_training_*.{out,err}`

## Requirements

- PyTorch with CUDA support
- torchvision
- scikit-learn
- matplotlib
- numpy

## Output

Training results saved to timestamped directories with:
- Training curves
- Confusion matrices
- Model checkpoints
- Performance metrics
- Comparison summaries (for ResNet)
