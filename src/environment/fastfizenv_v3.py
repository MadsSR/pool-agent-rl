import fastfiz as ff
from gymnasium import spaces, Env
from numpy import interp
import numpy as np


class FastFizEnv(Env):
    metadata = {"render_modes": ["human", "rgb_array"]}
    # Total number of balls (including cue ball and balls not in play)
    N_BALLS = 16
    EPSILON_THETA = 0.001  # To avoid max theta (from FastFiz.h)

    def __init__(
        self,
        render_mode="human",
        n_balls_train=2,
    ):
        super().__init__()
        self.n_balls_train = n_balls_train
        self.table_state = self.create_table_state(n_balls_train)

        table: ff.Table = self.table_state.getTable()
        shape = (self.N_BALLS, 2)

        # Initialize ranges
        self.offset_range = [0, 0]
        self.theta_range = [
            self.table_state.MIN_THETA,
            self.table_state.MAX_THETA - self.EPSILON_THETA,
        ]
        self.phi_range = [0, 360]
        self.velocity_range = [0, self.table_state.MAX_VELOCITY]

        self.observation_space = spaces.Box(
            low=np.full(shape, [0, 0]),
            high=np.full(shape, [table.TABLE_WIDTH, table.TABLE_LENGTH]),
            shape=shape,
            dtype=np.float64,
        )

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Shot params
        # a - offset
        # b - offset
        # theta - vertical angle
        # phi - horizontal angle
        # v - velocity

        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            high=np.array([0.0, 0.0, 1, 1, 1]),
            dtype=np.float64,
        )

    def _interpolate_action(self, action):
        return self.interpolate_action(self.table_state, action)

    @staticmethod
    def interpolate_action(table_state: ff.TableState, action):
        a = interp(action[0], [0, 0], [0, 0])
        b = interp(action[1], [0, 0], [0, 0])
        theta = interp(
            action[2], [0, 1], [table_state.MIN_THETA,
                                table_state.MAX_THETA - 0.001]
        )
        phi = interp(action[3], [0, 1], [0, 360])
        v = interp(action[4], [0, 1], [0, table_state.MAX_VELOCITY])
        return [a, b, theta, phi, v]

    @staticmethod
    def get_ball_positions(table_state: ff.TableState):
        balls = []

        for i in range(FastFizEnv.N_BALLS):
            b = table_state.getBall(i)
            pos = b.getPos()
            balls.append((pos.x, pos.y))

        balls = np.array(balls)
        return balls

    @staticmethod
    def num_balls_in_play(table_state: ff.TableState):
        num = 0
        for i in range(table_state.getNumBalls()):
            if table_state.getBall(i).isInPlay():
                num += 1
        return num

    @staticmethod
    def get_observation(table_state: ff.TableState):
        observation = FastFizEnv.get_ball_positions(table_state)

        for i, _ in enumerate(observation):
            if not table_state.getBall(i).isInPlay():
                observation[i] = [-1, -1]

        return observation

    def _get_observation(self):
        return FastFizEnv.get_observation(self.table_state)

    def _get_ball_positions(self):
        return self.get_ball_positions(self.table_state)

    def _get_reward(self, prev_ball_positions):

        if self.table_state.getBall(0).isPocketed():
            return -1

        # Check if is won
        if not any(
            not self.table_state.getBall(i).isPocketed()
            for i in range(1, self.n_balls_train)
        ):
            return 1000

        step_pocketed = self._get_step_pocketed()
        reward = step_pocketed

        ball_positions = self._get_ball_positions()

        min_dist = self._min_dist_pocket(prev_ball_positions)[1:]
        new_min_dist = self._min_dist_pocket(ball_positions)[1:]

        for i in range(len(min_dist)):
            if new_min_dist[i] < min_dist[i]:
                reward += (min_dist[i] - new_min_dist[i])

        n_balls_in_play = self.num_balls_in_play(self.table_state)

        if not self._any_ball_has_moved(prev_ball_positions, ball_positions):
            reward = (1/(self.N_BALLS - n_balls_in_play)*self.N_BALLS*-1)/2
            # print(f"Not hit reward on {n_balls_in_play}: {reward}")

        # Scale reward by number of balls in play
        if reward > 0 and n_balls_in_play > 0:
            reward = reward / n_balls_in_play

        return reward

    def _any_ball_has_moved(self, prev_ball_positions, ball_positions) -> bool:
        return not np.all(ball_positions[1:, :] == prev_ball_positions[1:, :])

    def _get_step_pocketed(self):
        pocketed = 0
        for i in range(1, self.n_balls_train):
            if self.table_state.getBall(i).isPocketed():
                pocketed += 1

        step_pocketed = pocketed - self.n_pocketed
        return step_pocketed

    def _min_dist_pocket(self, ball_positions: np.ndarray) -> np.ndarray:
        pocket_positions = self._get_pocket_positions()
        min_dist = np.array(
            [
                min(
                    np.linalg.norm(pocket_pos - ball_positions[i])
                    for pocket_pos in pocket_positions
                )
                for i in range(self.N_BALLS)
                if not self.table_state.getBall(i).isPocketed()
            ]
        )
        return min_dist

    @staticmethod
    def create_table_state(n_balls: int) -> ff.TableState:
        game_state: ff.GameState = ff.GameState.RackedState(ff.GT_EIGHTBALL)
        table_state: ff.TableState = game_state.tableState()

        # Remove balls from table state
        # for i in range(n_balls, 16):
        for i in range(n_balls, FastFizEnv.N_BALLS):
            table_state.setBall(i, ff.Ball.NOTINPLAY, ff.Point(0.0, 0.0))

        table_state.randomize()
        return table_state

    def _get_pocket_positions(self):
        pockets = [
            ff.Table.SW,
            ff.Table.SE,
            ff.Table.W,
            ff.Table.E,
            ff.Table.NW,
            ff.Table.NE,
        ]
        table: ff.Table = self.table_state.getTable()
        pocket_positions = np.array(
            [
                np.array([table.getPocketCenter(p).x,
                         table.getPocketCenter(p).y])
                for p in pockets
            ]
        )

        return pocket_positions

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Reset info
        self.n_pocketed = 0

        self.table_state = self.create_table_state(self.n_balls_train)
        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def _get_info(self):
        return {
            "n_pocketed": self.n_pocketed,
            "is_success": False,
        }

    def step(self, action):
        interpolated_action = self._interpolate_action(action)
        sp = ff.ShotParams(*interpolated_action)
        ball_positions = self._get_ball_positions()

        if self.table_state.isPhysicallyPossible(sp) == ff.TableState.OK_PRECONDITION:
            self.table_state.executeShot(sp)
            reward = self._get_reward(ball_positions)
        else:
            reward = -1

        terminated = self._is_terminate_state()
        observation = self._get_observation()
        info = self._get_info()
        info["is_success"] = reward == 1000

        return observation, reward, terminated, False, info  # False = truncated

    def _is_terminate_state(self):
        # Terminate if cue ball is pocketed
        if self.table_state.getBall(0).isPocketed():
            return True

        # Terminate if all balls are pocketed (except cue ball)
        for i in range(1, self.n_balls_train):
            if not self.table_state.getBall(i).isPocketed():
                return False

        return True

    def render(self):
        raise NotImplementedError("Render not implemented")


def main():
    pass


if __name__ == "__main__":
    main()
