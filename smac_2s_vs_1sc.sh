#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=28G
#SBATCH --time=120:00:00
export OMP_NUM_THREADS=1

source activate ISA

python main.py --config=ISA_2s_vs_1sc_ss --env-config=sc2 with env_args.map_name=2s_vs_1sc env_args.reward_sparse=True t_max=2100000 seed=2024 label=ISA wandb=False
python main.py --config=ISA_2s_vs_1sc_ss --env-config=sc2 with env_args.map_name=2s_vs_1sc env_args.reward_sparse=True t_max=2100000 seed=2025 label=ISA wandb=False
python main.py --config=ISA_2s_vs_1sc_ss --env-config=sc2 with env_args.map_name=2s_vs_1sc env_args.reward_sparse=True t_max=2100000 seed=2026 label=ISA wandb=False
python main.py --config=ISA_2s_vs_1sc_ss --env-config=sc2 with env_args.map_name=2s_vs_1sc env_args.reward_sparse=True t_max=2100000 seed=2027 label=ISA wandb=False
python main.py --config=ISA_2s_vs_1sc_ss --env-config=sc2 with env_args.map_name=2s_vs_1sc env_args.reward_sparse=True t_max=2100000 seed=2028 label=ISA wandb=False

