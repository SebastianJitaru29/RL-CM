import numpy as np
from typing import Tuple


"""
IDEA

Schedule rewards:
start -> ee + kick + score
phase out rewards one by one after certain success
lower their influence on the overall reward

"""

def reward_EE(state: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]):
    """Reward for positioning end-effector in line with ball and goal."""

    _, ee, ball, _ = state


    # if the ball is moving this reward is pointless
    if np.linalg.norm(ball[1]) >= 0.01:
        return 0
    
    goal_pos = state[3][0]
    diff = goal_pos - ball[0]
    
    diff_plane_norm = unit2D(diff)

    offset = 0.1    # HYPERPARAMETER

    target_pos = ball[0] + offset * diff_plane_norm

    dist = np.linalg.norm(ee[0] - target_pos)

    # TODO return 0 if btwn target and ball?

    return -dist


def reward_kick(state: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]):
    """Reward for kicking the ball sort of towards the goal."""

    _, _, ball, goal = state

    if np.linalg.norm(ball[1]) <= 0.01:
        return 0

    b2g_unit = unit2D(goal[0] - ball[0])    
    bvel_unit = unit2D(ball[1])

    angle_diff = angle_between_vectors(b2g_unit, bvel_unit)

    # TODO or smth like this
    return 1 - angle_diff / np.pi


def reward_score(state: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]):
    """Reward for scoring, i.e. the ball goes in the goal."""
    joints, ee, ball, goal = state

    # TODO implement
    #      mostly decide the shape of the goal wrt its centre

    return 0


def unit2D(vec):
    unit = np.zeros_like(vec)
    unit[0:2] = vec[0:2] / np.linalg.norm(vec[0:2])
    return unit


def angle_between_vectors(lhs, rhs):
    return np.arccos(np.clip(np.dot(lhs, rhs), -1.0, 1.0))