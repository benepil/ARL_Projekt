import os
import re

import argparse
from argparse import Namespace

import pandas as pd

from termcolor import colored

from collections import OrderedDict

from configparser import ConfigParser


SUMMARY = OrderedDict


def parse_cmd_args() -> str:
    print("Parsing Command line arguments")
    parser = argparse.ArgumentParser(description="Summary Script")

    parser.add_argument("folder", type=str,
                        help="path/to/a/folder/containing/experiments/")

    args = parser.parse_args()

    msg: str = "Command line arguments: "
    msg += " , ".join((f"{arg} = {getattr(args, arg)}" for arg in vars(args)))
    print(msg)

    if not os.path.isdir(args.folder):
        raise ValueError(f"Given path {args.folder} is not a directory!")
    else:
        return args.folder


def collect_experiment_data(root_folder: str) -> [SUMMARY]:
    if not os.path.isdir(root_folder):
        raise ValueError(f"Given experiment root-folder {root_folder} is not a directory!")

    collected_summaries: [SUMMARY] = []
    experiment_folders = next(os.walk(root_folder))[1]

    print(f"Scanning directory: {root_folder}")
    for folder_name in experiment_folders:
        path_to_experiment_folder: str = os.path.join(root_folder, folder_name)
        experiment_data: OrderedDict or False = summarize_experiment(path_to_experiment_folder)
        if experiment_data:
            collected_summaries.append(experiment_data)

    return collected_summaries


def summarize_experiment(path_to_folder: str) -> OrderedDict or False:
    print(f"Summarizing experiment folder: {path_to_folder}")

    if not os.path.isdir(path_to_folder):
        msg: str = f"Unable to scan directory: {path_to_folder}. Directory does not exists!"
        print(colored(msg, "red"))
        return False

    configuration_file_names: [str] = \
        list(filter(lambda file: file.endswith(".ini"), os.listdir(path_to_folder)))

    if len(configuration_file_names) == 0:
        msg: str = f"Unable to find a configuration file in the directory at {path_to_folder}!"
        print(colored(msg, "red"))
        return False

    if len(configuration_file_names) > 1:
        msg: str = f"Found directory with more than on configuration file at {path_to_folder}!"
        print(colored(msg, "red"))
        return False

    file_name: str = configuration_file_names.pop()
    path_to_configuration_file = os.path.join(path_to_folder, file_name)
    if not os.path.isfile(path_to_configuration_file):
        msg: str = f"Configuration file {path_to_configuration_file} does not exist!"
        print(colored(msg, "red"))
        return False

    path_to_done_flag_file: str = os.path.join(path_to_folder, "done")
    is_done: bool = os.path.isfile(path_to_done_flag_file)
    if not is_done:
        msg: str = f"Experiment {path_to_folder} has not yet been marked as done!"
        print(colored(msg, "red"))

    collect_data: OrderedDict = OrderedDict()
    collect_data["meta.done"] = str(is_done)

    parser: ConfigParser = ConfigParser()
    parser.read(path_to_configuration_file)

    hash_str: str = ""
    for section in sorted(parser.sections()):
        for option in sorted(parser.options(section)):
            value: str = parser.get(section, option)
            value = value.replace(",", ":")
            collect_data[f"{section.lower()}.{option.lower()}"] = value
            hash_str += f"{section.lower()}.{option.lower()}.{value}."

    # helps identify duplicates
    collect_data["meta.hash"] = hash(hash_str)

    path_to_duration_file: str = os.path.join(path_to_folder, "duration")
    if os.path.isfile(path_to_duration_file):
        with open(path_to_duration_file, "r") as file:
            collect_data["meta.duration"] = file.read()
    else:
        collect_data["meta.duration"] = pd.NA
        msg: str = f"Experiment folder {path_to_folder} has no duration file!"
        print(colored(msg, "red"))

    collect_data["meta.folder"] = path_to_folder
    collect_data["meta.configuration"] = path_to_configuration_file

    path_to_evaluation_folder: str = os.path.join(path_to_folder, "evaluation")
    if not os.path.isdir(path_to_evaluation_folder):
        msg: str = f"Experiment folder {path_to_folder} has no evaluation sub-folder!"
        print(colored(msg, "red"))
        return False

    level_2_evaluation_file_paths: {int: str} = {}
    per_episode_evaluation_file_pattern: re.Pattern = (
        re.compile(r"level_(\d+)_per_episode_data.csv"))

    for file_name in os.listdir(path_to_evaluation_folder):
        match = per_episode_evaluation_file_pattern.match(file_name)
        if match:
            level_id: int = match.group(1)
            path: str = os.path.join(path_to_evaluation_folder, file_name)
            level_2_evaluation_file_paths[level_id] = path

    summarized_at_least_one: bool = False
    for level, path in level_2_evaluation_file_paths.items():
        if os.path.isfile(path):
            data: pd.DataFrame = pd.read_csv(path)
            collect_data[f"level_{level}.episode_reward.mean"] = data["episode_reward"].mean()
            collect_data[f"level_{level}.checkpoint_count.mean"] = data["checkpoint_count"].mean()
            collect_data[f"level_{level}.collision.freq"] = data["has_collided"].mean()
            summarized_at_least_one = True

        else:
            msg: str = f"Evaluation file {path} does not exist!"
            print(colored(msg, "red"))
            return False

    if summarized_at_least_one:
        msg: str = f"All necessary data is present."
        print(colored(msg, "green"))
        return collect_data
    else:
        msg: str = f"Zero evaluation files were summarized!"
        print(colored(msg, "red"))
        return False


def merge_data(summarized_experiments: [OrderedDict]) -> pd.DataFrame:
    all_keys: set = set()
    for summarized_experiment in summarized_experiments:
        for key in sorted(summarized_experiment.keys()):
            all_keys.add(key)

    joint_dictionaries = OrderedDict()
    for key in sorted(all_keys):
        joint_dictionaries[key] = []
        for summary in summarized_experiments:
            if key in summary:
                joint_dictionaries[key].append(summary[key])
            else:
                joint_dictionaries[key].append(pd.NA)

    joint_data_frame: pd.DataFrame = pd.DataFrame(joint_dictionaries)
    return joint_data_frame


def sort_columns(experiment_data: pd.DataFrame) -> pd.DataFrame:
    print("Sorting data-frame columns")
    column_names: [str] = experiment_data.columns
    fst: [str] = ["algorithm.type"]
    snd: [str] = [col for col in column_names if col.startswith("meta")]
    thr3: [str] = [col for col in column_names if col.startswith("global")]
    thr4: [str] = [col for col in column_names if col.startswith("hyperparameter")]
    thr5: [str] = [col for col in column_names if col.startswith("level")]
    sorted_sections: [str] = fst + snd + thr3 + thr4 + thr5
    rest: [str] = [col for col in column_names if col not in set(sorted_sections)]
    column_names_sorted: [str] = sorted_sections + rest
    experiment_data = experiment_data.reindex(column_names_sorted, axis=1)
    return experiment_data


def save_data_frame(experiment_data: pd.DataFrame):
    store_data_here: str = os.path.join(os.getcwd(), "summary.csv")
    print(f"Writing summary to {store_data_here}")
    experiment_data.to_csv(store_data_here, index=False)


def main():
    path_to_root_experiment_folder: str = parse_cmd_args()
    collected_data: [OrderedDict] = collect_experiment_data(path_to_root_experiment_folder)
    nr_summaries: int = len(collected_data)
    print(f"Collected data for {nr_summaries} experiments")

    if nr_summaries:
        joint_experiment_data: pd.DataFrame = merge_data(collected_data)
        joint_experiment_data = sort_columns(joint_experiment_data)
        save_data_frame(joint_experiment_data)
    else:
        print("Nothing to do here")


if __name__ == '__main__':
    main()