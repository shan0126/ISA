# Influence Scope for Agents (ISA)

This is the code for "Credit Assignment and Focused Exploration for Sparse-reward Multi-agent Deep Reinforcement Learning", which has been accepted by Reinforcement Learning Conference 2026.

## Requirements
* Python 3.8
* OpenAI Gym
* PyTorch (CPU)
* SMAC

## Install instructions

    conda create -n ISA python=3.8 -y
    conda activate ISA
    pip install -r requirements.txt
    bash install_sc2.sh


## Run the code

The code can be run with the following .sh scripts for each tasks.

    sh smac_3m.sh
    sh smac_2s_vs_1sc.sh
    sh smac_8m.sh
    sh mpe_navigation.sh
    sh mpe_shooting.sh
    sh mpe_unlock.sh



    
