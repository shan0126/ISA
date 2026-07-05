from functools import partial
import sys
import os
from .multiagentenv import MultiAgentEnv

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}

# grf version
try:
    from .grf import GRFootballEnv
    REGISTRY["grf"] = partial(env_fn, env=GRFootballEnv)
except TypeError:
    print("grf is not installed")

# from .grf import GRFootballEnv

# REGISTRY["grf"] = partial(env_fn, env=GRFootballEnv)


# smac version
from .starcraft import StarCraft2Env
from .matrix_game import OneStepMatrixGame
from .mpe import MPEenv


REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
REGISTRY["one_step_matrix_game"] = partial(env_fn, env=OneStepMatrixGame)
REGISTRY["mpe"] = partial(env_fn, env=MPEenv)



if sys.platform == "linux":
    os.environ.setdefault("SC2PATH", "~/StarCraftII")
