import numpy as np
from gymnasium import spaces
import gymnasium as gym
import fastfiz as ff
from typing import Callable
from utils.fastfiz import create_table_state, get_ball_positions, shot_params_from_action
from utils.env import RewardFunction


class BaseFastFiz(gym.Env):
    """Base class for FastFiz environments."""
    EPSILON_THETA = 0.001  # To avoid max theta (from FastFiz.h)

    def __init__(self, reward_function: RewardFunction, num_balls: int = 15) -> None:
        super().__init__()
        self._reward_function = reward_function
        self.NUM_BALLS = num_balls
        self.table_state = create_table_state(num_balls)
        self.observation_space = self._observation_space()
        self.action_space = self._action_space()

    def reset(self, seed: int = None):
        super().reset(seed=seed)

        self.table_state = create_table_state(self.NUM_BALLS)

        observation = self._compute_observation()
        info = self._get_info()

        return observation, info

    def step(self, action):
        prev_table_state = self.table_state
        shot_params = shot_params_from_action(action)

        possible_shot = self.table_state.isPhysicallyPossible(
            shot_params) == ff.TableState.OK_PRECONDITION

        if possible_shot:
            self.table_state.executeShot(shot_params)

        reward = self._reward_function(
            self.table_state, prev_table_state, possible_shot)

        observation = self._compute_observation()

        info = self._get_info()
        terminated = self._is_terminal_state()

        return observation, reward, terminated, False, info

    def render():
        """
        Render the environment
        """
        raise NotImplementedError("This method must be implemented")

    def _compute_observation(self):
        """
        Get the observation of the environment
        """
        raise NotImplementedError("This method must be implemented")

    def _get_info(self):
        return {}

    def _is_terminal_state(self):
        """
        Check if the state is terminal
        """
        raise NotImplementedError("This method must be implemented")

    def _observation_space(self):
        """
        Get the observation space
        """
        raise NotImplementedError("This method must be implemented")

    def _action_space(self):
        """
        Get the action space
        """
        raise NotImplementedError("This method must be implemented")
