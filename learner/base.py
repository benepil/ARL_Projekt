import numpy as np
import pandas as pd

from pathlib import Path
from abc import ABC, abstractmethod

from environment.wrapper import UnityToPythonWrapper


TrainingHistory = pd.DataFrame


class Learner(ABC):

    def __init__(self, environment: UnityToPythonWrapper):
        self.env: UnityToPythonWrapper = environment

    @abstractmethod
    def evaluation_mode(self):
        pass

    @abstractmethod
    def training_mode(self):
        pass

    @abstractmethod
    def learn(self, total_timesteps: int) -> TrainingHistory:
        pass

    @abstractmethod
    def predict(self, observation: np.ndarray):
        pass

    @abstractmethod
    def load(self, path: str):
        pass

    @abstractmethod
    def save(self, path: str or Path):
        pass
