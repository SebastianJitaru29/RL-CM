import numpy as np
from typing import Tuple, Dict


def reward_EE(state: Dict):
    """Reward for positioning end-effector in line with ball and goal."""

    ee, ball, goal_pos = [state[key] for key in ('ee_info', 'ball_info', 'goal_info')]
    
    diff = goal_pos - ball[0]
    
    diff_plane_norm = unit2D(diff)

    offset = 0.2    # HYPERPARAMETER

    target_pos = ball[0] + offset * diff_plane_norm

    dist = np.linalg.norm(ee[0] - target_pos)

    # TODO return 0 if btwn target and ball?
    if dist < 1.0:
        return 1 - dist

    return 0


def reward_kick(state: Dict):
    """Reward for kicking the ball sort of towards the goal."""

    ball, goal_pos = [state[key] for key in ('ball_info', 'goal_info')]


    if np.linalg.norm(ball[1]) <= 0.1:
        return 0

    b2g_unit = unit2D(goal_pos - ball[0])    
    bvel_unit = unit2D(ball[1])

    angle_diff = angle_between_vectors(b2g_unit, bvel_unit)

    return 1 - 2 * np.abs(angle_diff) / np.pi


def reward_score(state: Dict):
    """Reward for scoring, i.e. the ball goes in the goal."""

    return state['score']


def reward_effort(state: Dict):
    """Reward for the amount of movement."""
    joint_vel = state['joint_info'][1]

    # print(f'Effort: {-np.sum(np.abs(joint_vel) / 7)}')
    return -np.sum(np.abs(joint_vel) / 7)


def reward_time(state: Dict):
    """Reward for time."""
    return -1


def unit2D(vec) -> np.ndarray:
    """Calculates the unit vector on the horizontal plane."""
    unit = np.zeros_like(vec)
    unit[0:2] = vec[0:2] / np.linalg.norm(vec[0:2])
    return unit


def angle_between_vectors(lhs, rhs) -> float:
    """Calculates the angle between two vectors."""
    return np.arccos(np.clip(np.dot(lhs, rhs), -1.0, 1.0))

