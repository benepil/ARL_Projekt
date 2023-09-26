import argparse
from argparse import Namespace, RawTextHelpFormatter

from termcolor import colored

from environment.wrapper import build_environment
from hybrid.gmm.core import model_and_heuristic_combined


OPTIMAL_THETA: [float] = [0, 0, 40, 0, 100]


def parse_cmd_args() -> Namespace:
    description: str = "Uncertainty in Reinforcement Learning Demo"
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=RawTextHelpFormatter)

    parser.add_argument("--level", type=int, default=0,
                        help="select a track [0, 1, 2, 3, 4]")

    args = parser.parse_args()
    msg: str = "Command line arguments: "
    msg += " , ".join((f"{arg} = {getattr(args, arg)}" for arg in vars(args)))
    print(msg)

    return args


def main():
    args = parse_cmd_args()
    level: int = max(0, min(args.level, 4))

    env = build_environment(level=level, seed=42, render=True)
    theta = OPTIMAL_THETA[level]

    observation, info = env.reset()
    done = False
    collect_reward = 0

    while not done:
        action = model_and_heuristic_combined(observation, theta=theta)
        new_observation, reward, done, info = env.step(action)
        observation = new_observation
        collect_reward += reward

    print(colored(f"Your scores is {collect_reward}", "green"))
    env.close()


if __name__ == '__main__':
    main()
