import os

import numpy as np

import torch
from torch.nn import Tanh

from sklearn.mixture import GaussianMixture

from collections import OrderedDict

from learner.network import SimpleNN
from heuristics.core import follow_the_center_line as heuristic
from hybrid.gmm.analysis import load_model as load_gmm_model


N_COMPS: int = 1
THIS_FOLDER: str = os.path.dirname(__file__)
PATH_TO_A2C_MODEL: str = os.path.join(THIS_FOLDER, "models/a2c_model.zip")
PATH_TO_GMM_MODEL: str = os.path.join(THIS_FOLDER, "models/gmm_model.pkl")


def rebuild_a2c_model(path: str):
    model: SimpleNN = SimpleNN(input_size=48, output_size=4,
                               trainable_parameter=(128, 128),
                               activation=Tanh)
    state_dict: dict = torch.load(path)
    policy_parameter: OrderedDict = state_dict['actor_net_state_dict']
    model.load_state_dict(policy_parameter)

    return model


A2C_MODEL: SimpleNN = rebuild_a2c_model(PATH_TO_A2C_MODEL)
GMM_MODEL: GaussianMixture = load_gmm_model(PATH_TO_GMM_MODEL)


def a2c_predict(observation: np.ndarray):
    observation = torch.from_numpy(observation).float()
    with torch.no_grad():
        output = A2C_MODEL(observation)
    action, _ = output.chunk(2, dim=1)
    action = action.numpy()
    return action


def estimate_uncertainty(observation: np.ndarray):
    log_likelihood: float = np.max(GMM_MODEL.score_samples(observation))
    return log_likelihood


def model_and_heuristic_combined(observation: np.ndarray, theta: float = 40):
    uncertainty: float = estimate_uncertainty(observation)
    if uncertainty <= theta:
        action = heuristic(observation)
        return action
    else:
        action = a2c_predict(observation)
        return action.squeeze()
