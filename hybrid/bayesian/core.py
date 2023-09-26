import os

import torch
from torch.nn import Tanh

import numpy as np

from collections import OrderedDict

from learner.network import SimpleNN
from heuristics.core import follow_the_center_line as heuristic


N_DECISIONS: int = 5

THIS_FOLDER: str = os.path.dirname(__file__)
PATH_TO_MODEL: str = os.path.join(THIS_FOLDER, "bayesian_a2c_weights.zip")


def rebuild_a2c_model():
    model: SimpleNN = SimpleNN(input_size=48, output_size=4,
                               trainable_parameter=(128, 128),
                               activation=Tanh)
    state_dict: dict = torch.load(PATH_TO_MODEL)
    policy_parameter: OrderedDict = state_dict['actor_net_state_dict']
    model.load_state_dict(policy_parameter)
    return model


MODEL: SimpleNN = rebuild_a2c_model()


def model_predict(observation: np.ndarray):
    observation = torch.from_numpy(observation).float()
    observation_fold = observation.repeat(N_DECISIONS, 1)

    with torch.no_grad():
        output = MODEL(observation_fold)
    mu, log_sigma = output.chunk(2, dim=1)

    action = mu.mean(dim=0)
    action_std = mu.std(dim=0).max()
    action_mse = ((mu - action) ** 2).mean()

    action = action.numpy()
    action_std = action_std.item()
    action_mse = action_mse.item()
    return action, action_std, action_mse


def model_and_heuristic_combined(observation: np.ndarray, theta: float = 0.01):
    action, max_action_std, action_mse = model_predict(observation)
    use_heuristic = action_mse >= theta

    if use_heuristic:
        action = heuristic(observation)
    return action
