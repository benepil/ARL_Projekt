from termcolor import colored

import argparse
from argparse import Namespace

from environment.wrapper import build_environment
from heuristics.core import follow_the_center_line


def parse_cmd_args() -> Namespace:
    description: str = "Uncertainty in Reinforcement Learning Demo"
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("--level", type=int, default=0,
                        help="select a track [0, 1, 2, 3, 4]")

    args = parser.parse_args()
    msg: str = "Command line arguments: "
    msg += " , ".join((f"{arg} = {getattr(args, arg)}" for arg in vars(args)))
    print(msg)

    return args


def main():
    args = parse_cmd_args()
    level: int = args.level

    env = build_environment(level=level, seed=42, render=True)

    collect_reward = 0
    observation, info = env.reset()
    done = False

    while not done:
        action = follow_the_center_line(observation)
        new_observation, reward, done, info = env.step(action)
        observation = new_observation
        collect_reward += reward

    print(colored(f"Your scores is {collect_reward}", "green"))
    env.close()


if __name__ == '__main__':
    main()
