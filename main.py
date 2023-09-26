import os
import time
import shutil

import argparse
from argparse import Namespace, RawTextHelpFormatter

import numpy as np
import pandas as pd

import torch
import random

from filelock import FileLock
from datetime import datetime, timedelta

import matplotlib.pyplot as plt

from collections import deque
import multiprocessing as mp

from pathlib import Path
from typing import Optional

from configparser import ConfigParser
from parser.tools import get_value

from environment.wrapper import UnityToPythonWrapper
from environment.wrapper import build_environment

from learner.base import Learner, TrainingHistory
from learner.a2c import build_actor_critic, build_bayesian_actor_critic
from learner.reinforce import build_reinforce, build_bayesian_reinforce

from analysis.plots import plot_training_history


EVALUATION_SEED: int = 1337

# store all data here
RESULTS_DIRECTORY: Path = Path("./results")

# make the training multiprocessing save
IO_ACCESS_LOCK_FILE: Path = RESULTS_DIRECTORY / "access.lock"
IO_ACCESS_LOCK: FileLock = FileLock(IO_ACCESS_LOCK_FILE)

LIVE_PLOT_FRAME_SIZE: int = 200


def parse_cmd_args() -> Namespace:
    print("Parsing Command line arguments")

    parser = argparse.ArgumentParser(description="Uncertainty in Reinforcement Learning",
                                     formatter_class=RawTextHelpFormatter)

    parser.add_argument("mode",
                        help="select a mode: [train, eval]")

    parser.add_argument("path",
                        help="value depends on the selected mode:\n"
                             "\t --mode=train \t path/to/any/configuration/file.ini\n"
                             "\t --mode=eval  \t path/to/any/experiment/folder/\n")

    parser.add_argument("--level", type=int, default=0,
                        help="select a track [0, 1, 2, 3]")

    parser.add_argument("--render", action="store_true", default=False,
                        help="render the environment (default: False)")

    parser.add_argument("--dry-run", action="store_true", default=False,
                        help="data will not be saved (default: False)")

    args = parser.parse_args()

    msg: str = "Command line arguments: "
    msg += " , ".join((f"{arg} = {getattr(args, arg)}" for arg in vars(args)))
    print(msg)

    return args


def set_global_seed(seed: int):
    if seed > 0:
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
    else:
        raise ValueError(f"Negative seed {seed}!")


def make_folder() -> Path:
    with IO_ACCESS_LOCK:
        time_stamp: datetime = datetime.now().replace(microsecond=0)
        time_stamp_string: str = time_stamp.strftime("%Y-%m-%d_%H-%M_%S")

        new_folder_path: Path = RESULTS_DIRECTORY / time_stamp_string
        print(f"Creating experiment folder @ {new_folder_path}")
        os.mkdir(new_folder_path)
        time.sleep(1)

    return new_folder_path


def assert_experiment_folder(path_to_experiment_folder: Path):
    print(f"Asserting experiment folder @ {path_to_experiment_folder}")

    if not path_to_experiment_folder.exists():
        raise ValueError(f"Experiment folder @ {path_to_experiment_folder} does not exist!")

    if not path_to_experiment_folder.is_dir():
        raise ValueError(f"Path {path_to_experiment_folder} does not lead to a folder!")

    find_experiment_config(path_to_experiment_folder)

    path_to_done_flag_file: Path = path_to_experiment_folder / "done"
    if not path_to_done_flag_file.is_file():
        raise ValueError(f"Experiment folder @ {path_to_experiment_folder} was not marked"
                         "as done! The experiment might still be running.")

    path_to_model: Path = path_to_experiment_folder / "model.zip"
    if not path_to_model.is_file():
        raise ValueError(f"Experiment folder @ {path_to_experiment_folder} does not contain"
                         "a model.zip file!")


def find_experiment_config(path_to_experiment_folder: Path) -> Path:
    print(f"Searching for experiment configuration file in {path_to_experiment_folder}")

    all_files: [str] = os.listdir(path_to_experiment_folder)
    all_ini_files: [str] = list(filter(lambda f: f.endswith(".ini"), all_files))
    nr_ini_files: int = len(all_ini_files)

    if nr_ini_files == 0:
        raise ValueError(f"Unable to find any configuration file in {path_to_experiment_folder}!")

    elif nr_ini_files > 1:
        raise ValueError(f"Found more then one configuration file in {path_to_experiment_folder}!")

    path_to_experiment_config: Path = path_to_experiment_folder / all_ini_files.pop()
    print(f"Found configuration file @ {path_to_experiment_config}")

    return path_to_experiment_config


def find_model(path_to_experiment_folder: Path) -> Path:
    print(f"Searching for model.zip file in {path_to_experiment_folder}")
    path_to_model: Path = path_to_experiment_folder / "model.zip"

    if not path_to_model.is_file():
        raise ValueError(f"Experiment folder {path_to_experiment_folder} does not contain a 'model.zip' file!")
    else:
        print(f"Found model @ {path_to_model}")
        return path_to_model


def prepare_config(path_to_config: Path, storage_dir: Optional[Path]) -> ConfigParser:
    print(f"Asserting configuration file @ {path_to_config}")
    path_to_config: Path = Path(path_to_config)

    if not path_to_config.exists():
        raise ValueError(f"Configuration file {path_to_config} does not exist!")

    if not path_to_config.is_file():
        print(f"Path {path_to_config} does not lead to a configuration file."
              f"Required is a single file of type '.ini'.")

    if not path_to_config.suffix.lower() == '.ini':
        raise ValueError(f"Configuration file is required to be of type '.ini'."
                         f"Given file {path_to_config} has suffix '{path_to_config.suffix}'!")

    if storage_dir is not None:
        path_to_config_copy: Path = storage_dir / path_to_config.name
        print(f"Copying configuration file @ {path_to_config} to {path_to_config_copy}")
        shutil.copy(path_to_config, path_to_config_copy)
        path_to_config = path_to_config_copy

    print(f"Reading configuration file @ {path_to_config}")
    config: ConfigParser = ConfigParser()
    config.read(path_to_config)

    return config


def log_level(path_to_experiment_folder: Path or None, level: int):
    if path_to_experiment_folder is not None:
        path_to_level_file: Path = path_to_experiment_folder / "level"
        print(f"Saving level @ {path_to_level_file}")

        msg: str = f"This agent was trained on level {level}"
        with open(path_to_level_file, "w") as file:
            file.write(msg)


def build_agent(environment: UnityToPythonWrapper, configuration: ConfigParser) -> Learner:
    algorithm_type: str = get_value(configuration, "Algorithm", "type", str)
    print(f"Building the agent of type {algorithm_type}")

    if algorithm_type == "Reinforce":
        return build_reinforce(environment, configuration)

    elif algorithm_type == "BayesianReinforce":
        return build_bayesian_reinforce(environment, configuration)

    elif algorithm_type == "ActorCritic":
        return build_actor_critic(environment, configuration)

    elif algorithm_type == "BayesianActorCritic":
        return build_bayesian_actor_critic(environment, configuration)

    else:
        raise NotImplemented(f"Algorithm of type {algorithm_type} is not Implemented!")


def finish_experiment(path_to_experiment_folder: Path, duration: timedelta):
    print(f"Marking the experiment @ {path_to_experiment_folder} as done")
    path_to_done_flag_file: Path = path_to_experiment_folder / "done"
    with open(path_to_done_flag_file, "w") as file:
        file.write("This file marks the experiment as success")

    path_to_duration_file: Path = path_to_experiment_folder / "duration"
    print(f"Writing experiment duration to {path_to_duration_file}")
    with open(path_to_duration_file, "w") as file:
        file.write(str(duration))


def live_line_plot(queue: mp.Queue, frame_size: int):
    frame_size: int = int(frame_size)
    if frame_size <= 0:
        raise ValueError(f"Argument frame_size has to been an integer greater Zero!")

    plt.figure(num="Uncertainty in Reinforcement Learning")
    plt.ion()
    axis = plt.gca()
    plt.xlabel("Step")
    plt.ylabel("Uncertainty")

    y_vals = deque([0] * frame_size, maxlen=frame_size)
    x_vals = deque(list(range(frame_size)), maxlen=frame_size)

    line, = plt.plot(x_vals, y_vals)
    counter = 0

    while True:
        val = queue.get()
        counter += 1

        if counter < frame_size:
            y_vals[counter] = val
        else:
            y_vals.append(val)
            x_vals.append(counter)

        line.set_ydata(y_vals)
        line.set_xdata(x_vals)
        axis.relim()
        axis.autoscale_view(True, True, True)
        plt.draw()
        plt.pause(0.001)


def train(args: Namespace):
    t_start: datetime = datetime.now().replace(microsecond=0)
    print(f"Starting training @ {t_start}")

    storage_folder: Optional[Path] = None
    if not args.dry_run:
        storage_folder = make_folder()

    level: int = int(args.level)
    log_level(storage_folder, level)

    path_to_config: Path = Path(args.path)
    config = prepare_config(path_to_config, storage_folder)

    seed: int = get_value(config, "Global", "seed", int)
    set_global_seed(seed)

    render: bool = args.render
    env: UnityToPythonWrapper = build_environment(level=level, seed=seed, render=render)

    agent: Learner = build_agent(env, config)
    agent.training_mode()

    total_timesteps = get_value(config, "Hyperparameter", "time_steps", int)
    print(f"Training the agent for {total_timesteps} time steps \n")
    training_history: TrainingHistory = agent.learn(total_timesteps=total_timesteps)

    print(f"\nClosing the Unity sub-process")
    env.close()

    delta_t: timedelta = datetime.now() - t_start
    print(f"Experiment took {delta_t}")

    if storage_folder is not None:

        with IO_ACCESS_LOCK:
            path_to_training_history: Path = storage_folder / "training_history.csv"
            print(f"Writing training-history data to {path_to_training_history}")
            training_history.to_csv(path_to_training_history, index=False)

            path_to_training_history_plot: Path = storage_folder / "training_history.pdf"
            print(f"Plotting training-history data @ {path_to_training_history_plot}")
            path_to_training_history_plot_str: str = str(path_to_training_history_plot)
            plot_training_history(training_history, path_to_training_history_plot_str)

            path_to_model: Path = storage_folder / "model.zip"
            print(f"Saving model @ {path_to_model}")
            agent.save(path_to_model)

            finish_experiment(storage_folder, delta_t)

    print("Done")


def evaluate(args: Namespace):
    path_to_experiment_folder: Path = Path(args.path)
    assert_experiment_folder(path_to_experiment_folder)

    path_to_experiment_config: Path = find_experiment_config(path_to_experiment_folder)
    config: ConfigParser = prepare_config(path_to_experiment_config, storage_dir=None)

    render: bool = args.render
    level: int = int(args.level)
    env: UnityToPythonWrapper = build_environment(level=level, render=render, seed=EVALUATION_SEED)

    path_to_model: Path = find_model(path_to_experiment_folder)
    path_to_model_str = str(path_to_model)

    agent: Learner = build_agent(env, config)
    agent.load(path_to_model_str)
    agent.evaluation_mode()

    print("Evaluating the model for 50 episodes")

    render: bool = bool(args.render)
    plot_subprocess: None or mp.Process = None
    plot_data_queue: None or mp.Queue = None

    if render:
        plot_data_queue = mp.Queue()
        args: (mp.Queue, int) = (plot_data_queue, LIVE_PLOT_FRAME_SIZE)
        plot_subprocess = mp.Process(target=live_line_plot, args=args)
        plot_subprocess.start()

    collect_per_step_data: dict = {
        "episode": [],
        "step": [],
        "danger_zone": [],
        "uncertainty.rotation": [],
        "uncertainty.acceleration": [],
        "uncertainty.action_mse": [],
        "action.rotation": [],
        "action.acceleration": []
    }

    collect_per_episode_data: dict = {
       "episode_reward": [],
       "episode_length": [],
       "checkpoint_count": [],
       "start_checkpoint": [],
       "has_collided": [],
    }

    for ep in range(50):

        state, info = env.reset()
        episode_reward: int = 0
        step_count: int = 0
        done: bool = False

        while not done:
            action, uncertainties, action_mse = agent.predict(state)
            rotation, acceleration = action

            state, reward, done, info = env.step(action)
            in_danger: int = info["DangerZone"]

            collect_per_step_data["episode"].append(ep)
            collect_per_step_data["step"].append(step_count)
            collect_per_step_data["danger_zone"].append(in_danger)
            collect_per_step_data["uncertainty.rotation"].append(uncertainties[0])
            collect_per_step_data["uncertainty.acceleration"].append(uncertainties[1])
            collect_per_step_data["uncertainty.action_mse"].append(action_mse)
            collect_per_step_data["action.rotation"].append(rotation)
            collect_per_step_data["action.acceleration"].append(acceleration)

            episode_reward += reward
            step_count += 1

            if render:
                max_uncertainty: float = max(uncertainties)
                plot_data_queue.put(max_uncertainty)

        start_checkpoint: int = info["StartCheckpoint"]
        checkpoint_count: int = info["CheckpointCount"]
        has_collided: int = info["Collision"]

        collect_per_episode_data["episode_reward"].append(episode_reward)
        collect_per_episode_data["episode_length"].append(step_count)
        collect_per_episode_data["checkpoint_count"].append(checkpoint_count)
        collect_per_episode_data["start_checkpoint"].append(start_checkpoint)
        collect_per_episode_data["has_collided"].append(has_collided)

    print(f"Closing the Unity sub-process")
    env.close()

    print(f'\navg. episode reward: {np.mean(collect_per_episode_data["episode_reward"]):>5} \n'
          f'avg. checkpoint count: {np.mean(collect_per_episode_data["checkpoint_count"]):>5} \n'
          f'collision rate: {np.mean(collect_per_episode_data["has_collided"]):>5} \n')

    if render:
        plot_subprocess.terminate()
        plot_data_queue.close()

    if not args.dry_run:

        with IO_ACCESS_LOCK:
            path_to_evaluation_folder: Path = path_to_experiment_folder / "evaluation"
            if not path_to_evaluation_folder.is_dir():
                print(f"Creating evaluation folder @ {path_to_evaluation_folder}")
                path_to_evaluation_folder.mkdir()

            path_to_per_step_evaluation_data: Path = path_to_evaluation_folder / f"level_{level}_per_step_data.csv"
            print(f"Saving per step evaluation data to {path_to_per_step_evaluation_data}")
            pd.DataFrame(collect_per_step_data).to_csv(path_to_per_step_evaluation_data, index=False)

            path_to_per_episode_evaluation_data: Path = path_to_evaluation_folder / f"level_{level}_per_episode_data.csv"
            print(f"Saving the evaluation dat to {path_to_per_episode_evaluation_data}")
            pd.DataFrame(collect_per_episode_data).to_csv(path_to_per_episode_evaluation_data, index=False)

    print("Done")


def main():
    args = parse_cmd_args()
    if args.mode == "train":
        train(args)
    elif args.mode == "eval":
        evaluate(args)
    else:
        raise NotImplemented(f"Modus {args.mode} is not implemented!")


if __name__ == '__main__':
    main()








