#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=28G
#SBATCH --time=100:00:00
export OMP_NUM_THREADS=1

source activate laies

python main.py --config=ISA_3m_ss --env-config=sc2 with env_args.map_name=3m env_args.reward_sparse=True t_max=2100000 seed=2024 label=0613 wandb=False

