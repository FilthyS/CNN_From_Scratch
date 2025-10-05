#!/bin/bash
#SBATCH --job-name=resnet_training
#SBATCH --account=csci_ua_0480_042-2025fa
#SBATCH --partition=g2-standard-12
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --open-mode=append
#SBATCH --output=resnet_training_%j.out
#SBATCH --error=resnet_training_%j.err
#SBATCH --export=ALL
#SBATCH --requeue

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"

# Set up environment
cd /scratch/cx2275

# Run singularity
echo "Starting Singularity container..."
singularity exec --nv --overlay ./overlay-25GB-500K.ext3:rw cuda12.6.2-cudnn9.5.0-devel-ubuntu24.04.1.sif bash -c "
    # Setup conda environment
    source ./setup_env.sh

    # Move to project dir
    cd CNN_From_Scratch

    # Print environment info
    echo 'Python version:' \$(python --version)
    echo 'PyTorch version:' \$(python -c 'import torch; print(torch.__version__)')
    echo 'CUDA available:' \$(python -c 'import torch; print(torch.cuda.is_available())')
    echo 'GPU count:' \$(python -c 'import torch; print(torch.cuda.device_count())')
    if python -c 'import torch; exit(0 if torch.cuda.is_available() else 1)'; then
        echo 'GPU name:' \$(python -c 'import torch; print(torch.cuda.get_device_name(0))')
    fi
    echo '=========================='

    # Run ResNet-18 training
    echo 'Starting ResNet-18 fine-tuning on EMNIST...'
    python resnet18.py

    echo 'Training completed at: ' \$(date)
"

echo "Job completed at: $(date)"
