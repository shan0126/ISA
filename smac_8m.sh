#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=112G
#SBATCH --time=100:00:00
export OMP_NUM_THREADS=1

source activate ISA

python main.py --config=ISA_8m_ss --env-config=sc2 with env_args.map_name=8m env_args.reward_sparse=True t_max=2100000 seed=2024 label=ISA wandb=False
python main.py --config=ISA_8m_ss --env-config=sc2 with env_args.map_name=8m env_args.reward_sparse=True t_max=2100000 seed=2025 label=ISA wandb=False
python main.py --config=ISA_8m_ss --env-config=sc2 with env_args.map_name=8m env_args.reward_sparse=True t_max=2100000 seed=2026 label=ISA wandb=False
python main.py --config=ISA_8m_ss --env-config=sc2 with env_args.map_name=8m env_args.reward_sparse=True t_max=2100000 seed=2027 label=ISA wandb=False
python main.py --config=ISA_8m_ss --env-config=sc2 with env_args.map_name=8m env_args.reward_sparse=True t_max=2100000 seed=2028 label=ISA wandb=False






