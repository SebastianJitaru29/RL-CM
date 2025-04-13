import numpy as np
import numpy.random as npr

from typing import List

class Randomizer():
    """Class to handle randomization of selected actions.

    This class randomizes the actions of the agent. There are three
    randomizations:
    1. Pose randomization -> select a fully random pose.
    2. Any randomization -> randomize any action (selected randomly).
    3. Joint randomization -> fully randomize one joint.
    """
    def __init__(
            self,
            e_pose_start: float,
            e_random_start: float,
            e_single_start: float,
            e_pose_step: float,
            e_random_step: float,
            e_single_step: float,
            e_pose_min: float,
            e_random_min: float,
            e_single_min: float,
            e_pose_reset: float,
            e_random_reset: float,
            e_single_reset: float,
            max_steps: int,
    ):
        """Initializes the randomizer.

        @param e_pose_start (float): Chance of random complete pose.
        @param e_random_start (float): Chance of randomly randomizing.
        @param e_single_start (float): Chance of fully randomizing one joint
        @param e_pose_step (float): Reduction per step of pose chance.
        @param e_random_step (float): Reduction per step of random chance.
        @param e_single_step (float): Reduction per step of joint chance.
        @param e_pose_min (float): Minimum chance of random pose.
        @param e_random_min (float): Minumum chance of random chance.
        @param e_single_min (float): Minimum chance of random joint.
        @param e_pose_reset (float): Value to set pose chance to after
            curriculum change.
        @param e_random_reset (float): Value to set random chance to after
            curriculum change.
        @param e_single_reset (float): Value to set joint chance to after
            curriculum change.
        @param max_steps (int): Maximum number of steps to keep the
            randomized action.
        """
        self.e_pose_start = e_pose_start
        self.e_random_start = e_random_start
        self.e_single_start = e_single_start

        self.e_pose_step = e_pose_step
        self.e_random_step = e_random_step
        self.e_single_step = e_single_step

        self.e_pose_min = e_pose_min
        self.e_random_min = e_random_min
        self.e_single_min = e_single_min

        self.e_pose_reset = e_pose_reset
        self.e_random_reset = e_random_reset
        self.e_single_reset = e_single_reset

        self.e_pose_current = e_pose_start
        self.e_random_current = e_random_start
        self.e_single_current = e_single_start

        self.max_steps = max_steps
        self.timer = 0

        self.poses = [[None] * 7] * 3

    def randomize(self, actions: List[List[int]]) -> List[List[int]]:
        """Select a randomization and randomize actios.
        
        @param actions (List[List[int]]): the actions selected by the agent.

        @return (List[List[int]]): the actions with randomizations.
        """
        if self.timer != 0:
            self.timer -= 1
            return self._timed_randomness(actions)
        else:
            self.poses = [[None] * 7] * 3
            self.timer = npr.randint(0, self.max_steps-1)

        rand = npr.random()

        rand -= self.e_pose_current
        if rand <= 0:
            return self._randomize_pose(actions)
        rand -= self.e_random_current
        if rand <= 0:
            return self._randomize_random(actions)
        rand -= self.e_single_current
        if rand <= 0:
            return self._randomize_single_joint(actions)

        return actions
    
    def step(self):
        """Perform actions for after a training step."""
        self.e_pose_current -= self.e_pose_step
        self.e_random_current -= self.e_random_step
        self.e_single_current -= self.e_single_step

    def reset(self):
        """Reset values for the next curriculum."""
        self.e_pose_current = self.e_pose_reset
        self.e_random_current = self.e_random_reset
        self.e_single_current = self.e_single_reset

    def _randomize_pose(self, actions: List[List[int]]):
        """Randomize the entire pose."""
        for r_idx, row in enumerate(self.poses):
            for c_idx, _ in enumerate(row):
                rand_bin = npr.randint(0, 3)
                self.poses[r_idx][c_idx] = rand_bin
                actions[r_idx, c_idx] = rand_bin

        return actions

    def _randomize_random(self, actions: List[List[int]]):
        """Randomize random bins."""
        for r_idx, row in enumerate(self.poses):
            for c_idx, _ in enumerate(row):
                if npr.random() <= 0.1:
                    rand_bin = npr.randint(0, 3)
                    self.poses[r_idx][c_idx] = rand_bin
                    actions[r_idx, c_idx] = rand_bin
        
        return actions

    def _randomize_single_joint(self, actions: List[List[int]]):
        """Fully randomize 1 joint."""

        joint = npr.randint(0, 7)
        for l_idx, _ in enumerate(self.poses):
            rand_bin = npr.randint(0, 3)
            self.poses[l_idx][joint] = rand_bin
            actions[l_idx, joint] = rand_bin

        return actions
        

    def _timed_randomness(self, actions: List[List[int]]):
        """Return randomized action of prev step.
        
        Ensures some consistency in the randomization.
        """        
        for r_idx, row in enumerate(self.poses):
            for c_idx, _ in enumerate(row):
                if self.poses[r_idx][c_idx] is not None:
                    actions[r_idx, c_idx] = self.poses[r_idx][c_idx]

        return actions
        