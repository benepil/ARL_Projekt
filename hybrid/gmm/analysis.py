import os
import pickle
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


SEED: int = 42
N_COMPONENTS: int = 6

THIS_FOLDER: str = os.path.abspath(os.path.dirname(__file__))
OBSERVATION_FOLDER: str = os.path.join(THIS_FOLDER, "observations")
MODEL_FOLDER: str = os.path.join(THIS_FOLDER, "models")

TRAINING_DATASET_AT: str = os.path.join(THIS_FOLDER, OBSERVATION_FOLDER, "level_0_training_observations.csv")
LEVEL_2_DATASET_AT: str = os.path.join(THIS_FOLDER, OBSERVATION_FOLDER, "level_2_evaluation_observations.csv")
LEVEL_4_DATASET_AT: str = os.path.join(THIS_FOLDER, OBSERVATION_FOLDER, "level_4_evaluation_observations.csv")


def save_model(model: GaussianMixture, save_to: str):
    with open(save_to, "wb") as file_handler:
        pickle.dump(model, file_handler)


def load_model(path: str) -> GaussianMixture:
    if os.path.isfile(path):
        with open(path, "rb") as file_handler:
            model: GaussianMixture = pickle.load(file_handler)
        return model

    else:
        raise FileNotFoundError(f"File {path} does not exist!")


def extract_observations(data_frame: pd.DataFrame) -> np.ndarray:
    collect_observations: [pd.Series] = []
    for idx in range(48):
        key: str = f"obs.idx.{idx}"
        if key in data_frame:
            observation_segment: np.ndarray = data_frame[key].to_numpy()
            collect_observations.append(observation_segment)
        else:
            raise KeyError(f"DataFrame is miss observation column {key}!")

    observations = np.column_stack(collect_observations)
    return observations


def training_model(dataset: pd.DataFrame, n_components: int) -> GaussianMixture:
    dataset: np.ndarray = extract_observations(dataset)
    gmm_full = GaussianMixture(n_components=n_components, random_state=SEED, max_iter=1000)
    gmm_full.fit(dataset)
    return gmm_full


def evaluate(model: GaussianMixture):
    level2path: dict = {
        "level 2": LEVEL_2_DATASET_AT,
        "level 4": LEVEL_4_DATASET_AT,
    }

    collect_eval_data: dict = {
        "level 2": {"tpr": [], "fpr": [], "youden": [], "theta": np.empty(0),
                    "mean.ep.reward": [], "std.ep.reward": [], "n.episodes": 0},

        "level 4": {"tpr": [], "fpr": [], "youden": [], "theta": np.empty(0),
                    "mean.ep.reward": [], "std.ep.reward": [], "n.episodes": 0},
    }

    for lvl, path in level2path.items():
        # load and predict
        level_data: pd.DataFrame = pd.read_csv(path)
        observations: np.ndarray = extract_observations(level_data)
        log_likelihood: np.ndarray = model.score_samples(observations)
        is_ood_data: np.ndarray = level_data["danger_zone"].to_numpy().astype(int)
        n_episodes: int = len(level_data["episode"].unique())
        collect_eval_data[lvl]["n.episodes"] = n_episodes

        # find optimal threshold
        max_log_likelihood: float = max(log_likelihood)
        min_log_likelihood: float = min(log_likelihood)
        thresholds: np.ndarray = np.linspace(min_log_likelihood, max_log_likelihood, num=100)
        collect_eval_data[lvl]["theta"] = thresholds

        for theta in reversed(thresholds):
            # ROC values
            predict_ood: np.ndarray = (log_likelihood <= theta).astype(int)
            tn, fp, fn, tp = confusion_matrix(is_ood_data, predict_ood).ravel()
            tpr: float = tp / (tp + fn)
            fpr: float = fp / (tn + fp)
            youden_j_value: float = tpr - fpr

            collect_eval_data[lvl]["tpr"].append(tpr)
            collect_eval_data[lvl]["fpr"].append(fpr)
            collect_eval_data[lvl]["youden"].append(youden_j_value)

            # modified reward
            predict_no_ood = np.invert(predict_ood.astype(bool))
            level_data_filtered: pd.DataFrame = level_data[predict_no_ood]
            grouped: pd.DataFrame = level_data_filtered.groupby('episode')['reward'].sum()
            episode_reward_mean: float = grouped.mean()
            episode_reward_mean = 0.0 if pd.isna(episode_reward_mean) else episode_reward_mean
            episode_reward_std: float = grouped.std()
            episode_reward_std = 0.0 if pd.isna(episode_reward_std) else episode_reward_std
            collect_eval_data[lvl]["mean.ep.reward"].append(episode_reward_mean)
            collect_eval_data[lvl]["std.ep.reward"].append(episode_reward_std)

        # plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 16))

    level: str
    eval_results: dict
    for level, eval_results in collect_eval_data.items():
        # ROC curve
        ax1.plot(eval_results["fpr"], eval_results["tpr"], label=f"{level}", marker="o")
        optimal_theta_at: int = np.argmax(eval_results["youden"])
        optimal_theta: float = eval_results["theta"][optimal_theta_at]
        print("Penis", eval_results["fpr"][optimal_theta_at], eval_results["tpr"][optimal_theta_at], optimal_theta)
        ax1.annotate(f'theta {optimal_theta:.2f}',
                     (eval_results["fpr"][optimal_theta_at], eval_results["tpr"][optimal_theta_at]),
                     textcoords="offset points", xytext=(0, 10), ha='center')

        # modified Reward curve
        n = eval_results["n.episodes"]
        lower_ci = [mu - sig/n for (mu, sig) in zip(eval_results["mean.ep.reward"], eval_results["std.ep.reward"])]
        upper_ci = [mu + sig/n for (mu, sig) in zip(eval_results["mean.ep.reward"], eval_results["std.ep.reward"])]
        ax2.plot(eval_results["theta"], eval_results["mean.ep.reward"], label=f"{level}")
        ax2.fill_between(eval_results["theta"], lower_ci, upper_ci, alpha=0.2)

    ax1.set_xlabel('False Positive Rate', fontsize=10, labelpad=10)
    ax1.set_ylabel('True Positive Rate', fontsize=10, labelpad=10)
    ax2.set_xlabel('Theta', fontsize=10, labelpad=10)
    ax2.set_ylabel('Episode Reward', fontsize=10, labelpad=10)
    ax2.set_yscale('log', base=2)
    ax1.legend(loc=4)
    ax2.legend(loc=4)
    plt.show()


def main():
    print("loading data")
    training_data: pd.DataFrame = pd.read_csv(TRAINING_DATASET_AT)

    print("training model")
    model: GaussianMixture = training_model(training_data, n_components=N_COMPONENTS)

    save_to: str = os.path.join(THIS_FOLDER, MODEL_FOLDER, "gmm_model.pkl")
    print("saving the model to", save_to)
    save_model(model, save_to)

    print("evaluating the model")
    evaluate(model)


if __name__ == '__main__':
    main()





