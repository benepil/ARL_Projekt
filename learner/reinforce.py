import numpy as np

from copy import deepcopy
from datetime import datetime

from configparser import ConfigParser
from parser.tools import get_value

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dists
import torch.nn.utils as utils

from environment.wrapper import UnityToPythonWrapper
from learner.base import Learner, TrainingHistory
from learner.network import SimpleNN


class Reinforce(Learner):

    def __init__(self,
                 environment: UnityToPythonWrapper,
                 trainable_parameter: (int,) = (128, 128),
                 train_every_n_episode: int = 5,
                 gamma: float = 0.99,
                 learning_rate: float = 0.01,
                 max_grad_norm: float = 1.5,
                 seed: int = 42):

        super().__init__(environment)
        torch.manual_seed(int(seed))

        self.trainable_parameter: (int,) = (int(p) for p in trainable_parameter)
        self.gamma: float = float(gamma)
        self.train_every_n_episode: int = int(train_every_n_episode)
        self.learning_rate: float = float(learning_rate)
        self.seed: int = int(seed)
        self.max_grad_norm = float(max_grad_norm)

        self.observation_size: int = int(self.env.observation_space.shape[0])
        self.number_of_actions: int = int(self.env.action_space.shape[0])

        self.policy_net: SimpleNN = SimpleNN(input_size=self.observation_size,
                                             output_size=2 * self.number_of_actions,
                                             trainable_parameter=trainable_parameter,
                                             activation=nn.Tanh)

        self.optimizer: optim.Adam = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        self.model_checkpoint: dict = deepcopy(self.policy_net.state_dict())

    def training_mode(self):
        self.policy_net.train()

    def evaluation_mode(self):
        self.policy_net.eval()

    def policy(self, observation: np.ndarray):
        observation = torch.from_numpy(observation).float()
        mu, log_sigma = self.policy_net(observation).squeeze().chunk(2)
        sigma = torch.exp(log_sigma)
        cov_mat = torch.diag_embed(sigma)

        distribution = dists.MultivariateNormal(mu, cov_mat)
        action = distribution.sample()
        log_prob = distribution.log_prob(action)
        action = action.numpy()

        return action, log_prob

    def predict(self, observation: np.ndarray):
        with torch.no_grad():
            observation = torch.from_numpy(observation).float()
            action, log_std = self.policy_net(observation).squeeze().chunk(2)
            uncertainty = log_std.exp()
            action = action.numpy()
        return action, uncertainty, 0.0

    def learn(self, total_timesteps: int) -> TrainingHistory:
        global_step_count: int = 0
        current_episode: int = 0

        # checkpoint every 5% progress
        checkpoint_interval: int = max(int(0.1 * total_timesteps), 1)
        previous_checkpoint_episode: int = 0
        previous_score: float = 0.0
        checkpoint_flag: bool = False

        # global training data
        collect_episode_reward: list = []
        collect_episode_length: list = []

        # local training data
        collect_log_probs: list = []
        collect_rewards: list = []
        collect_dones: list = []

        learn: bool = global_step_count < total_timesteps

        # -------- training start -------- #

        while learn:

            state, info = self.env.reset()
            episode_reward = 0
            local_step_count = 0
            done = False

            # -------- episode start -------- #

            while not done:
                action, log_prob = self.policy(state)
                state, reward, done, info = self.env.step(action)

                collect_log_probs.append(log_prob)
                collect_rewards.append(reward)
                collect_dones.append(done)

                episode_reward += reward
                local_step_count += 1
                global_step_count += 1

                learn = global_step_count < total_timesteps
                checkpoint_flag |= global_step_count % checkpoint_interval == 0

            # -------- episode end -------- #

            collect_episode_reward.append(episode_reward)
            collect_episode_length.append(local_step_count)
            current_episode += 1

            # learn from rollout
            if self.train_every_n_episode > 0:
                if current_episode % self.train_every_n_episode == 0:
                    self.update_policy(collect_log_probs, collect_rewards, collect_dones)
                    collect_log_probs.clear()
                    collect_rewards.clear()
                    collect_dones.clear()

            # -------- episode end: checkpoint -------- #

            if checkpoint_flag:
                lookback = current_episode - previous_checkpoint_episode
                checkpoint_flag = False

                if lookback > 0:
                    score = np.mean(collect_episode_reward[-lookback:]).item()
                    previous_checkpoint_episode = current_episode
                    msg_type = "info "

                    if score > 1.05 * previous_score:
                        self.model_checkpoint = deepcopy(self.policy_net.state_dict())
                        previous_score = score
                        msg_type = "saved"

                    print(f'{datetime.now().replace(microsecond=0)}     {msg_type}     '
                          f'time steps: {global_step_count:>8}/{total_timesteps} '
                          f'episode count: {current_episode:>6} '
                          f'average score: {score:>8.2f} [last {lookback:>3} episodes]')

        # -------- training end -------- #

        # finally load best model
        self.policy_net.load_state_dict(self.model_checkpoint)

        training_history: pd.DataFrame = pd.DataFrame({
            "episodes": list(range(len(collect_episode_reward))),
            "episode_reward": collect_episode_reward,
            "episode_length": collect_episode_length
        })

        return training_history

    def update_policy(self, log_probs: [torch.Tensor], rewards: [float], dones: [bool], normalize: bool = True):
        G = 0
        returns = torch.empty(len(rewards), dtype=torch.float32)
        index = len(rewards) - 1
        for reward, done in zip(rewards[::-1], dones[::-1]):
            G = reward + self.gamma * G * (1 - int(done))
            returns[index] = G
            index -= 1

        if normalize:
            returns = (returns - returns.mean()) / (returns.std() + 0.00001)

        log_probs = torch.stack(log_probs)
        policy_loss = (-log_probs * returns).sum()

        self.optimizer.zero_grad()
        policy_loss.backward()
        utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
        self.optimizer.step()

    def load(self, path: str):
        self.policy_net.load_state_dict(torch.load(path))

    def save(self, path: str):
        torch.save(self.policy_net.state_dict(), path)


class BayesianReinforce(Reinforce):

    def __init__(self,
                 environment: UnityToPythonWrapper,
                 n_decisions: int = 5,
                 **kwargs
                 ):

        super().__init__(environment, **kwargs)
        self.n_decisions = int(n_decisions)

    def evaluation_mode(self):
        # allways keep dropout layers online
        pass

    def policy(self, observation: np.ndarray):
        observation = torch.from_numpy(observation).float()
        observation_fold = observation.repeat(self.n_decisions, 1)

        mu, log_sigma = self.policy_net(observation_fold).chunk(2, dim=1)
        sigma = torch.exp(log_sigma)
        cov_mat = torch.diag_embed(sigma)

        distribution = dists.MultivariateNormal(mu, cov_mat)
        actions = distribution.sample()
        log_probs = distribution.log_prob(actions)
        expected_log_prob = log_probs.mean()
        action = actions.mean(dim=0).numpy()

        return action, expected_log_prob

    def predict(self, observation: np.ndarray):
        with torch.no_grad():
            observation = torch.from_numpy(observation).float()
            observation_fold = observation.repeat(self.n_decisions, 1)
            mu, _ = self.policy_net(observation_fold).chunk(2, dim=1)

            action = mu.mean(dim=0)
            uncertainty = mu.std(dim=0)
            mse_uncertainty = ((mu - action)**2).mean(dim=0)

        return action, uncertainty, mse_uncertainty


def build_reinforce(environment: UnityToPythonWrapper, configuration: ConfigParser) -> Reinforce:
    train_every_n_episode = get_value(configuration, "Hyperparameter", "train_every_n_episode", int)
    learning_rate = get_value(configuration, "Hyperparameter", "learning_rate", float)
    gamma = get_value(configuration, "Hyperparameter", "gamma", float)
    max_grad_norm = get_value(configuration, "Hyperparameter", "max_grad_norm", float)
    seed = get_value(configuration, "Global", "seed", int)
    trainable_parameter = get_value(configuration, "Hyperparameter", "trainable_parameter", str)
    trainable_parameter: (int,) = (int(p) for p in trainable_parameter.split(","))

    return Reinforce(environment=environment,
                     trainable_parameter=trainable_parameter,
                     train_every_n_episode=train_every_n_episode,
                     learning_rate=learning_rate,
                     gamma=gamma,
                     max_grad_norm=max_grad_norm,
                     seed=seed)


def build_bayesian_reinforce(environment: UnityToPythonWrapper, configuration: ConfigParser) -> Reinforce:
    train_every_n_episode = get_value(configuration, "Hyperparameter", "train_every_n_episode", int)
    learning_rate = get_value(configuration, "Hyperparameter", "learning_rate", float)
    gamma = get_value(configuration, "Hyperparameter", "gamma", float)
    max_grad_norm = get_value(configuration, "Hyperparameter", "max_grad_norm", float)
    n_decisions = get_value(configuration, "Hyperparameter", "n_decisions", int)
    seed = get_value(configuration, "Global", "seed", int)
    trainable_parameter = get_value(configuration, "Hyperparameter", "trainable_parameter", str)
    trainable_parameter: (int,) = (int(p) for p in trainable_parameter.split(","))

    return BayesianReinforce(environment=environment,
                             trainable_parameter=trainable_parameter,
                             n_decisions=n_decisions,
                             train_every_n_episode=train_every_n_episode,
                             learning_rate=learning_rate,
                             gamma=gamma,
                             max_grad_norm=max_grad_norm,
                             seed=seed)
