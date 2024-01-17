#!/bin/bash
#SBATCH --job-name=pretrain 		       # Job name
#SBATCH --nodes=1                      # Number of nodes. We only have one node at the moment.
#SBATCH --ntasks=1                     # Number of CPU tasks. Typically, one task is started per node.
#SBATCH --cpus-per-task=2              # Specifies the number of CPUs (which might be interpreted as cores or threads) you wish to allocate to each of those tasks.
#SBATCH --gres=gpu:2                   # Request 2 GPUs. GRE stands for generic resources.
#SBATCH --time=01:00:00                # Time limit hrs:min:sec
#SBATCH --output=job_%j.out            # Standard output and error log. %j interpolates the job ID

# Activate your environment (if you're using conda or another virtual environment)
conda activate dreamdiffusion

# Use srun to run the job
srun python -m torch.distributed.run --nproc-per-node=2 code/stageA1_eeg_pretrain.py
