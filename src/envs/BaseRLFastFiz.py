import numpy as np
from gymnasium import spaces
import gymnasium as gym
import fastfiz as ff
from typing import Callable
from utils.fastfiz import create_random_table_state, get_ball_positions, shot_params_from_action
from . import BaseFastFiz


class BaseRLFastFiz(BaseFastFiz):
    """FastFiz environment with random initial state, used for reinforcemet learning."""
    EPSILON_THETA = 0.001  # To avoid max theta (from FastFiz.h)

    def __init__(self, reward_function: Callable[[ff.TableState, ff.TableState, bool], float], num_balls: int = 15, ) -> None:
        super().__init__()
        self._reward_function = reward_function
        self.num_balls = num_balls
        self.table_state = create_random_table_state(num_balls)
        self.table_state.randomize()

        self.observation_space = self._observation_space()

        self.action_space = self._action_space()

    def reset(self, seed: int = None):
        super().reset(seed=seed)

        self.table_state = create_random_table_state(self.num_balls, seed=seed)
        self.table_state.randomize()

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

    def _compute_observation(self):
        ball_positions = get_ball_positions(self.table_state)

        observation = []
        for i, ball_pos in enumerate(ball_positions):
            if self.table_state.getBall(i).isInPlay():
                observation.append(ball_pos)
            else:
                observation.append([-1, -1])

        return np.array(observation)

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
        table = self.table_state.getTable()
        lower = np.full((self.num_balls, 2), [0, 0])
        upper = np.full((self.num_balls, 2), [
                        table.TABLE_WIDTH, table.TABLE_LENGTH])
        return spaces.Box(
            low=lower, high=upper, shape=lower.shape, dtype=np.float64)

    def _action_space(self):
        """
        Get the action space
        """
        return spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            high=np.array([0.0, 0.0, 1, 1, 1]),
            dtype=np.float64,
        )
