import numpy as np

from pynput import keyboard

from termcolor import colored

import argparse
from argparse import Namespace

from environment.wrapper import build_environment


rotation = 0
acceleration = 0
braking = 0

KEY_LEFT = keyboard.Key.left
KEY_RIGHT = keyboard.Key.right
KEY_UP = keyboard.Key.up
KEY_DOWN = keyboard.Key.down


def on_key_press(key):
    global rotation, acceleration, braking

    if key == KEY_LEFT:
        rotation = -1
    elif key == KEY_RIGHT:
        rotation = 1
    elif key == KEY_UP:
        acceleration = 1
        braking = 0
    elif key == KEY_DOWN:
        acceleration = 0
        braking = 1


def on_key_release(key):
    global rotation, acceleration, braking

    if key == KEY_LEFT or key == KEY_RIGHT:
        rotation = 0
    elif key == KEY_UP or key == KEY_DOWN:
        acceleration = 0
        braking = 0


def parse_cmd_args() -> Namespace:
    description: str = "Uncertainty in Reinforcement Learning Demo"
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("--level", type=int, default=0,
                        help="select a track [0, 1, 2, 3, 4]")

    args = parser.parse_args()

    return args


def main():
    args = parse_cmd_args()
    level: int = args.level

    env = build_environment(level=level, seed=42, render=True)

    listener = keyboard.Listener(on_press=on_key_press,
                                 on_release=on_key_release)
    listener.start()

    collect_reward: float = 0

    try:
        observation, info = env.reset()
        done = False

        while not done:
            action = np.array([rotation, acceleration, braking])
            new_observation, reward, done, info = env.step(action)
            collect_reward += reward
            observation = new_observation

    except KeyboardInterrupt:
        pass
    finally:
        listener.stop()
        listener.join()

    env.close()

    print(colored(f"Your scores is {collect_reward}", "green"))


if __name__ == '__main__':
    main()
