import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as opt
import torch.distributions as dists
import torch.nn.utils as utils

from copy import deepcopy
from datetime import datetime
from collections import OrderedDict

from configparser import ConfigParser
from parser.tools import get_value

from environment.wrapper import UnityToPythonWrapper
from learner.base import Learner, TrainingHistory
from learner.network import SimpleNN


class ActorCritic(Learner):

    def __init__(self,
                 environment: UnityToPythonWrapper,
                 pi_parameters: (int, int) = (32, 16),
                 vf_parameters: (int, int) = (32, 16),
                 train_every_n_episode: int = 1,
                 lr_actor: float = 0.0005,
                 lr_critic: float = 0.0005,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 max_grad_norm: float = 1.0,
                 seed: int = 42):

        super().__init__(environment)
        torch.manual_seed(int(seed))

        self.pi_parameters: (int, ) = (int(p) for p in pi_parameters)
        self.vf_parameters: (int, ) = (int(p) for p in vf_parameters)
        self.train_every_n_episode: int = int(train_every_n_episode)
        self.lr_actor: float = float(lr_actor)
        self.lr_critic: float = float(lr_critic)
        self.gamma: float = float(gamma)
        self.gae_lambda: float = float(gae_lambda)
        self.max_grad_norm: float = max_grad_norm
        self.seed: int = int(seed)

        self.observation_size: int = int(self.env.observation_space.shape[0])
        self.number_of_actions: int = int(self.env.action_space.shape[0])

        self.actor_net: SimpleNN = SimpleNN(input_size=self.observation_size,
                                            output_size=2*self.number_of_actions,
                                            trainable_parameter=pi_parameters,
                                            activation=nn.Tanh)

        self.critic_net: SimpleNN = SimpleNN(input_size=self.observation_size,
                                             output_size=1,
                                             trainable_parameter=vf_parameters,
                                             activation=nn.Tanh)

        self.actor_optimizer: opt.Adam = opt.Adam(self.actor_net.parameters(), lr=lr_actor)
        self.critic_optimizer: opt.Adam = opt.Adam(self.critic_net.parameters(), lr=lr_critic)

        self.actor_checkpoint: OrderedDict  = deepcopy(self.actor_net.state_dict())
        self.critic_checkpoint: OrderedDict = deepcopy(self.critic_net.state_dict())

        self.training_mode()

    def evaluation_mode(self):
        self.actor_net.eval()
        self.critic_net.eval()

    def training_mode(self):
        self.actor_net.train()
        self.critic_net.train()

    def evaluate_states(self, observation: np.ndarray):
        observation = torch.from_numpy(observation).float()
        value_estimation = self.critic_net(observation).squeeze()
        return value_estimation

    def policy(self, observation: np.ndarray):
        observation = torch.from_numpy(observation).float()
        mu, log_sigma = self.actor_net(observation).squeeze().chunk(2)
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
            action, log_std = self.actor_net(observation).squeeze().chunk(2)
            uncertainty = log_std.exp()
            action = action.numpy()
            mse_uncertainty = 0.0
        return action, uncertainty, mse_uncertainty

    def learn(self, total_timesteps: int) -> TrainingHistory:
        global_step_count = 0
        current_episode = 0

        # checkpoint every 5% progress
        checkpoint_interval: int = max(int(0.05 * total_timesteps), 1)
        previous_checkpoint_episode: int = 0
        previous_score: float = 0
        checkpoint_flag: bool = False

        # global training data
        collect_episode_reward = []
        collect_episode_length = []

        # local training data
        collect_log_probs: list = []
        collect_rewards: list = []
        collect_values: list = []
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
                value = self.evaluate_states(state)
                state, reward, done, info = self.env.step(action)

                collect_log_probs.append(log_prob)
                collect_rewards.append(reward)
                collect_values.append(value)
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

            # -------- episode end: update model -------- #

            if self.train_every_n_episode > 0:
                if current_episode % self.train_every_n_episode == 0:
                    self.update_policy(collect_log_probs,
                                       collect_values,
                                       collect_rewards,
                                       collect_dones)
                    collect_log_probs.clear()
                    collect_rewards.clear()
                    collect_values.clear()
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
                        self.actor_checkpoint = deepcopy(self.actor_net.state_dict())
                        self.critic_checkpoint = deepcopy(self.critic_net.state_dict())
                        previous_score = score
                        msg_type = "saved"

                    print(f'{datetime.now().replace(microsecond=0)}     {msg_type}     '
                          f'time steps: {global_step_count:>8}/{total_timesteps} '
                          f'episode count: {current_episode:>6} '
                          f'average score: {score:>8.2f} [last {lookback:>3} episodes]')

        # -------- training end -------- #

        # load last checkpoint
        self.actor_net.load_state_dict(self.actor_checkpoint)
        self.critic_net.load_state_dict(self.critic_checkpoint)

        training_history: pd.DataFrame = pd.DataFrame({
            "episodes": list(range(len(collect_episode_reward))),
            "episode_reward": collect_episode_reward,
            "episode_length": collect_episode_length
        })

        return training_history

    def update_policy(self, log_probs: [torch.Tensor], values: [torch.Tensor], rewards: [float],
                      dones: [bool], normalize: bool = True):

        values: torch.Tensor = torch.stack(values).squeeze()
        values_detached: torch.Tensor = values.detach().clone()
        log_probs: torch.Tensor = torch.stack(log_probs)

        gae = 0
        n_rewards = len(rewards)
        advantage = torch.zeros(n_rewards, dtype=torch.float32)
        for k in reversed(range(n_rewards - 1)):
            delta = rewards[k] + self.gamma * values_detached[k + 1] * (1 - int(dones[k])) - values_detached[k]
            gae = delta + self.gamma * self.gae_lambda * (1 - int(dones[k])) * gae
            advantage[k] = gae
        advantage[n_rewards - 1] = rewards[n_rewards - 1]

        returns = advantage + values_detached

        if normalize:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 0.00001)

        critic_loss = ((returns - values)**2).mean()
        actor_loss = -(advantage * log_probs).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

    def load(self, path: str):
        checkpoint: dict = torch.load(path)
        self.actor_net.load_state_dict(checkpoint['actor_net_state_dict'])
        self.critic_net.load_state_dict(checkpoint['critic_net_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])

    def save(self, path: str):
        state: dict = {
            'actor_net_state_dict': self.actor_net.state_dict(),
            'critic_net_state_dict': self.critic_net.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict()}
        torch.save(state, path)


class BayesianActorCritic(ActorCritic):

    def __init__(self,
                 environment: UnityToPythonWrapper,
                 n_decisions: int = 5,
                 *args,
                 **kwargs
                 ):

        super().__init__(environment, *args, **kwargs)
        self.n_decisions = int(n_decisions)

    def evaluation_mode(self):
        # allways keep dropout layers online
        pass

    def policy(self, observation: np.ndarray):
        observation = torch.from_numpy(observation).float()
        observation_fold = observation.repeat(self.n_decisions, 1)

        output = self.actor_net(observation_fold)
        mu, log_sigma = output.chunk(2, dim=1)
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
            output = self.actor_net(observation_fold)
            mu, log_sigma = output.chunk(2, dim=1)

            action = mu.mean(dim=0)
            uncertainty = mu.std(dim=0)
            mse_uncertainty = ((mu - action)**2).mean(dim=0)

        return action, uncertainty, mse_uncertainty


def build_actor_critic(environment: UnityToPythonWrapper, configuration: ConfigParser) -> ActorCritic:
    train_every_n_episode = get_value(configuration, "Hyperparameter", "train_every_n_episode", int)
    lr_actor = get_value(configuration, "Hyperparameter", "lr_actor", float)
    lr_critic = get_value(configuration, "Hyperparameter", "lr_critic", float)
    gamma = get_value(configuration, "Hyperparameter", "gamma", float)
    gae_lambda = get_value(configuration, "Hyperparameter", "gae_lambda", float)
    max_grad_norm = get_value(configuration, "Hyperparameter", "max_grad_norm", float)
    seed = get_value(configuration, "Global", "seed", int)
    pi_parameters = get_value(configuration, "Hyperparameter", "pi_parameters", str)
    vf_parameters = get_value(configuration, "Hyperparameter", "vf_parameters", str)
    pi_parameters: (int,) = (int(p) for p in pi_parameters.split(","))
    vf_parameters: (int,) = (int(p) for p in vf_parameters.split(","))

    return ActorCritic(environment=environment,
                       pi_parameters=pi_parameters,
                       vf_parameters=vf_parameters,
                       train_every_n_episode=train_every_n_episode,
                       lr_actor=lr_actor,
                       lr_critic=lr_critic,
                       gamma=gamma,
                       gae_lambda=gae_lambda,
                       max_grad_norm=max_grad_norm,
                       seed=seed)


def build_bayesian_actor_critic(environment: UnityToPythonWrapper, configuration: ConfigParser) -> BayesianActorCritic:
    n_decisions = get_value(configuration, "Hyperparameter", "n_decisions", int)
    train_every_n_episode = get_value(configuration, "Hyperparameter", "train_every_n_episode", int)
    lr_actor = get_value(configuration, "Hyperparameter", "lr_actor", float)
    lr_critic = get_value(configuration, "Hyperparameter", "lr_critic", float)
    gamma = get_value(configuration, "Hyperparameter", "gamma", float)
    gae_lambda = get_value(configuration, "Hyperparameter", "gae_lambda", float)
    max_grad_norm = get_value(configuration, "Hyperparameter", "max_grad_norm", float)
    seed = get_value(configuration, "Global", "seed", int)
    pi_parameters = get_value(configuration, "Hyperparameter", "pi_parameters", str)
    vf_parameters = get_value(configuration, "Hyperparameter", "vf_parameters", str)
    pi_parameters: (int,) = (int(p) for p in pi_parameters.split(","))
    vf_parameters: (int,) = (int(p) for p in vf_parameters.split(","))

    return BayesianActorCritic(environment=environment,
                               pi_parameters=pi_parameters,
                               vf_parameters=vf_parameters,
                               n_decisions=n_decisions,
                               train_every_n_episode=train_every_n_episode,
                               lr_actor=lr_actor,
                               lr_critic=lr_critic,
                               gamma=gamma,
                               gae_lambda=gae_lambda,
                               max_grad_norm=max_grad_norm,
                               seed=seed)
