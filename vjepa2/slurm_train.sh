#!/bin/bash 
#SBATCH --account=sci-demelo 
#SBATCH --nodes=1 
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --time=6:00:00
#SBATCH --partition=gpu-batch
#SBATCH --exclude=gx01
#SBATCH --constraint="ARCH:X86&GPU_MEM:40GB" 
#SBATCH --mem=100G 

cd /sc/home/konrad.goldenbaum/thesis/vjepa2

# Find a free port for distributed training
find_free_port() {
    local start_port=29500
    local end_port=65535
    
    for ((port=start_port; port<=end_port; port++)); do
        # Check if port is not in use
        if ! ss -tuln | grep -q ":$port "; then
            echo $port
            return 0
        fi
    done
    
    echo "ERROR: Could not find free port" >&2
    exit 1
}

# Find and export the master port
export MASTER_PORT=$(find_free_port)
echo "Using MASTER_PORT=$MASTER_PORT"

# Run training - pass port explicitly to torchrun
uv run torchrun --nproc_per_node=2 --master_port=$MASTER_PORT -m app.train_mvfouldl
