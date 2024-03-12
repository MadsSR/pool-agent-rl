import numpy as np
from typing import Optional
import fastfiz as ff


POCKETS = [
    ff.Table.SW,
    ff.Table.SE,
    ff.Table.W,
    ff.Table.E,
    ff.Table.NW,
    ff.Table.NE,
]


def get_ball_positions(table_state: ff.TableState) -> np.ndarray:
    balls = []
    for i in range(table_state.getNumBalls()):
        pos = table_state.getBall(i).getPos()
        balls.append((pos.x, pos.y))
    balls = np.array(balls)
    return balls


def num_balls_in_play(table_state: ff.TableState) -> int:
    return len([i for i in range(table_state.getNumBalls()) if table_state.getBall(i).isInPlay()])


def num_balls_pocketed(table_state: ff.TableState) -> int:
    return len([i for i in range(table_state.getNumBalls()) if table_state.getBall(i).isPocketed()])


def any_ball_has_moved(prev_ball_positions: np.ndarray, ball_positions: np.ndarray) -> bool:
    return not np.array_equal(prev_ball_positions, ball_positions)


def pocket_centers(table_state: ff.TableState) -> np.ndarray:
    table: ff.Table = table_state.getTable()
    pocket_positions = []
    for pocket in POCKETS:
        pocket_center = table.getPocketCenter(pocket)
        pocket_positions.append((pocket_center.x, pocket_center.y))

    return np.array(pocket_positions)


def distance_to_pocket(ball_position: np.ndarray, pocket: np.ndarray) -> float:
    return np.linalg.norm(pocket - ball_position)


def distance_to_pockets(ball_position: np.ndarray) -> np.ndarray:
    return np.array([distance_to_pocket(ball_position, pocket) for pocket in POCKETS])


def distance_to_closest_pocket(ball_position: np.ndarray) -> float:
    return np.min(distance_to_pockets(ball_position))


def distances_to_closest_pockets(ball_positions: np.ndarray) -> np.ndarray:
    return np.array([distance_to_closest_pocket(ball_position) for ball_position in ball_positions])


def create_table_state(n_balls: int) -> ff.TableState:
    assert 1 <= n_balls <= 15, "Number of balls must be between 1 and 15"

    game_state: ff.GameState = ff.GameState.RackedState(ff.GT_EIGHTBALL)
    table_state: ff.TableState = game_state.tableState()

    # Remove balls from table state
    for i in range(n_balls + 1, 16):
        table_state.setBall(i, ff.Ball.NOTINPLAY, ff.Point(0.0, 0.0))

    return table_state


def create_random_table_state(n_balls: int, seed: Optional[int] = None) -> ff.TableState:
    table_state = create_table_state(n_balls)
    table_state = randomize_table_state(table_state, seed)
    return table_state


def randomize_table_state(table_state: ff.TableState, seed: Optional[int] = None) -> None:
    # TODO: Implement randomize_table_state using a seed
    pass


def interpolate_action(table_state: ff.TableState, action: np.ndarray) -> np.ndarray:
    a = np.interp(action[0], [0, 0], [0, 0])
    b = np.interp(action[1], [0, 0], [0, 0])
    theta = np.interp(
        action[2], [0, 1], [table_state.MIN_THETA,
                            table_state.MAX_THETA - 0.001]
    )
    phi = np.interp(action[3], [0, 1], [0, 360])
    v = np.interp(action[4], [0, 1], [0, table_state.MAX_VELOCITY])
    return [a, b, theta, phi, v]


def shot_params_from_action(action: np.ndarray) -> ff.ShotParams:
    return ff.ShotParams(*interpolate_action(action))
