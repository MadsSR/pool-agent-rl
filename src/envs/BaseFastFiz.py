import numpy as np
from gymnasium import spaces
import gymnasium as gym
import fastfiz as ff
from typing import Callable
from utils.fastfiz import create_table_state, get_ball_positions, get_shot_params_from_action


class BaseFastFiz(gym.Env):
    EPSILON_THETA = 0.001  # To avoid max theta (from FastFiz.h)

    def __init__(self, reward_function: Callable[[ff.TableState, ff.TableState, bool], float], n_balls: int = 15, ) -> None:
        super().__init__()
        self._reward_function = reward_function
        self.n_balls = n_balls
        self.table_state = create_table_state(n_balls)
        self.table_state.randomize()

        table = self.table_state.getTable()
        lower = np.full((n_balls, 2), [0, 0])
        upper = np.full((n_balls, 2), [table.TABLE_WIDTH, table.TABLE_LENGTH])

        self.observation_space = spaces.Box(
            low=lower, high=upper, shape=lower.shape, dtype=np.float64)

        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            high=np.array([0.0, 0.0, 1, 1, 1]),
            dtype=np.float64,
        )

    def reset(self, seed: int = None):
        super().reset(seed=seed)

        self.table_state = create_table_state(self.n_balls)
        self.table_state.randomize()

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action):
        prev_table_state = self.table_state
        shot_params = get_shot_params_from_action(action)

        possible_shot = self.table_state.isPhysicallyPossible(
            shot_params) == ff.TableState.OK_PRECONDITION

        if possible_shot:
            self.table_state.executeShot(shot_params)

        reward = self._reward_function(
            self.table_state, prev_table_state, possible_shot)

        observation = self._get_observation()

        info = self._get_info()
        terminated = self._is_terminal_state()

        return observation, reward, terminated, False, info

    def _get_observation(self):
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
        return False
