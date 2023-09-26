import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


WIDTH: float = 8.27
HEIGHT: float = 5.84
FONT_SIZE = 1.5 * plt.rcParams["font.size"]

BAR_WIDTH: float = 0.4
WINDOW_SIZE: int = 50
INTERPOLATION_STEPS_SIZE: int = 50

RESULTS_FOLDER: str = "./results"


def assert_save_path(path: str):
    valid_file_types: (str,) = ("pdf", "png", "svg")
    for file_extension in valid_file_types:
        if path.endswith(file_extension):
            break
    else:
        raise ValueError(f"Invalid file extension {path}! "
                         f"Valid extension are {valid_file_types}.")


def assert_dataframe_columns(data: pd.DataFrame, required_keys: (str,)):
    data_keys: {str} = set(data.keys())
    for key in required_keys:
        if key not in data_keys:
            raise ValueError(f"DataFrame is missing required key {key}!")


def plot_training_history(data: pd.DataFrame, save_here: str) -> None:
    assert_save_path(save_here)

    required_keys: (str, ) = ("episode_length", "episode_reward")
    assert_dataframe_columns(data, required_keys)

    episode_reward: pd.Series = data["episode_reward"]
    rolling_reward_mean: pd.Series = data["episode_reward"].rolling(window=WINDOW_SIZE, min_periods=1).mean()
    steps: pd.Series = data["episode_length"].cumsum()

    plt.figure(figsize=(WIDTH, HEIGHT))
    plt.plot(steps, rolling_reward_mean, label="rolling mean")
    plt.scatter(steps, episode_reward, label="reward")
    plt.ylabel('Episode Reward', fontsize=FONT_SIZE, labelpad=10)
    plt.xlabel('Step', fontsize=FONT_SIZE, labelpad=10)
    plt.legend()
    plt.tight_layout()
    plt.grid()
    plt.savefig(save_here, transparent=True, dpi=300)
    plt.close()


def plot_training_history_smooth_average(list_of_paths: [str], save_here: str) -> None:
    assert_save_path(save_here)

    interpolation_min: float = 0
    interpolation_max: float = np.inf

    all_dataframes: [pd.DataFrame] = []
    required_keys: (str, ) = ("episode_length", "episode_reward")

    for path in list_of_paths:
        data: pd.DataFrame = pd.read_csv(path)
        assert_dataframe_columns(data, required_keys)

        data["episode_ended_at_step"]: pd.Series = data["episode_length"].cumsum()
        min_episode_length: float = data["episode_ended_at_step"].min()
        max_episode_length: float = data["episode_ended_at_step"].max()
        interpolation_min: float = max(min_episode_length, interpolation_min)
        interpolation_max: float = min(max_episode_length, interpolation_max)
        all_dataframes.append(data)

    # reset the x-axis such that is equal for all data-frames
    new_x: np.ndarray = np.arange(interpolation_min, interpolation_max, INTERPOLATION_STEPS_SIZE)
    # calculate the new y values accordingly
    interpolated_rewards: np.ndarray = np.empty((len(all_dataframes), len(new_x)))

    for i, data in enumerate(all_dataframes):
        episode_end: pd.Series = data["episode_ended_at_step"].to_numpy()
        reward: pd.Series = data["episode_reward"].to_numpy()
        reward_interpolated = np.interp(new_x, episode_end, reward)
        interpolated_rewards[i] = reward_interpolated

    average_reward = np.mean(interpolated_rewards, axis=0)
    smoothed_reward = pd.Series(average_reward).rolling(window=WINDOW_SIZE, min_periods=1).mean()
    smoothed_reward_std = pd.Series(average_reward).rolling(window=WINDOW_SIZE, min_periods=1).std()
    lower_ci_bound = smoothed_reward - smoothed_reward_std / np.sqrt(len(interpolated_rewards))
    upper_ci_bound = smoothed_reward + smoothed_reward_std / np.sqrt(len(interpolated_rewards))

    plt.figure(figsize=(WIDTH, HEIGHT))
    plt.plot(new_x, smoothed_reward, label="rolling mean")
    plt.ylabel('Episode Reward', fontsize=FONT_SIZE, labelpad=10)
    plt.fill_between(new_x, lower_ci_bound, upper_ci_bound, color='lightblue', alpha=1, label='conf. int.')
    plt.xlabel('Step', fontsize=FONT_SIZE, labelpad=10)
    plt.axhline(y=np.pi, color='black', linestyle='--', label='random')
    plt.axhline(y=271.15, color='black', linestyle=':', label='heuristic')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(save_here, transparent=True, dpi=300)
    plt.close()


def plot_uncertainty_of_a_single_episode(data: pd.DataFrame, episode: int, uncertainty_type: str, save_here: str):
    assert_save_path(save_here)

    required_keys: (str,) = ("episode", "step", "danger_zone", "uncertainty.rotation",
                             "uncertainty.acceleration", "uncertainty.action_mse")

    assert_dataframe_columns(data, required_keys)

    if not any(data["episode"] == int(episode)):
        raise ValueError(f"Episode {episode} is not present in the data!")

    uncertainty_types: (str,) = ("action_mse", "max_single_action_std")
    if uncertainty_type not in uncertainty_types:
        raise ValueError(f"Unknown uncertainty type {uncertainty_types}! "
                         f"Available types are {uncertainty_types}.")

    episode_data: pd.Series = data.loc[data["episode"] == episode]
    danger_zone_indicator: pd.Series = episode_data["danger_zone"]
    steps: pd.Series = episode_data["step"]
    x_ticks: np.ndarray = np.arange(len(steps))

    if uncertainty_type == "action_mse":
        data = episode_data["uncertainty.action_mse"]
    elif uncertainty_type == "max_single_action_std":
        data = np.maximum(episode_data["uncertainty.rotation"],
                          episode_data["uncertainty.acceleration"])
    else:
        raise ValueError("unable to select the data that matches the "
                         f"uncertainty type {uncertainty_type}.")

    plt.figure(figsize=(WIDTH, HEIGHT))

    plt.bar(x_ticks[danger_zone_indicator == 0], data[danger_zone_indicator == 0],
            width=BAR_WIDTH, color='black', label='no danger')

    plt.bar(x_ticks[danger_zone_indicator == 1], data[danger_zone_indicator == 1],
            width=BAR_WIDTH, color='darkblue', label='in danger')

    plt.xlabel('Step', fontsize=FONT_SIZE, labelpad=10)
    plt.ylabel('Uncertainty', fontsize=FONT_SIZE, labelpad=10)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(save_here, transparent=True, dpi=300)
    plt.close()


def load_and_plot_training_history(path_to_data_file: str, save_here: str):
    if not os.path.isfile(path_to_data_file):
        raise ValueError(f"File {path_to_data_file} does not exist!")

    if not path_to_data_file.endswith(".csv"):
        raise ValueError(f"File {path_to_data_file} is not a csv-file!")

    if os.path.isdir(save_here):
        default_name: str = path_to_data_file.replace(".csv", ".pdf")
        save_here = os.path.join(save_here, default_name)

    data: pd.DataFrame = pd.read_csv(path_to_data_file)
    plot_training_history(data, save_here)


def scan_folder(path_to_folder: str) -> [str]:
    if not os.path.isdir(path_to_folder):
        raise ValueError(f"The given path {path_to_folder} either does not lead to directory "
                         f"or does not exist!")

    print(f"Scanning folder: {path_to_folder}")
    collect_training_history_file_paths: [str] = []
    for root, _, files in os.walk(path_to_folder):
        for file in files:
            if file == "training_history.csv":
                path: str = os.path.join(root, file)
                print(f"Collecting {path}")
                collect_training_history_file_paths.append(path)

    return collect_training_history_file_paths


def main():
    training_history_file_paths: [str] = scan_folder(RESULTS_FOLDER)
    n_experiments: int = len(training_history_file_paths)
    print(f"Found {n_experiments} experiments")

    if n_experiments == 0:
        print("Nothing to do here")

    else:
        for path_to_csv in training_history_file_paths:
            save_here: str = path_to_csv.replace(".csv", ".pdf")
            load_and_plot_training_history(path_to_csv, save_here)


if __name__ == '__main__':
    main()
