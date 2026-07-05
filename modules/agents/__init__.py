REGISTRY = {}

from .n_rnn_agent import NRNNAgent
from .g_rnn_agent import GRNNAgent

REGISTRY["n_rnn"] = NRNNAgent
REGISTRY["group_rnn"] = GRNNAgent


