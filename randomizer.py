import numpy as np
import numpy.random as npr


class Randomizer():
    
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

    def randomize(self, actions):

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
        self.e_pose_current -= self.e_pose_step
        self.e_random_current -= self.e_random_step
        self.e_single_current -= self.e_single_step

    def reset(self):
        self.e_pose_current = self.e_pose_reset
        self.e_random_current = self.e_random_reset
        self.e_single_current = self.e_single_current

    def _randomize_pose(self, actions):
        """Randomize the entire pose."""
        for r_idx, row in enumerate(self.poses):
            for c_idx, _ in enumerate(row):
                rand_bin = npr.randint(0, 3)
                self.poses[r_idx][c_idx] = rand_bin
                actions[r_idx, c_idx] = rand_bin

        return actions

    def _randomize_random(self, actions):
        """Randomize random bins."""
        for r_idx, row in enumerate(self.poses):
            for c_idx, _ in enumerate(row):
                if npr.random() <= 0.1:
                    rand_bin = npr.randint(0, 3)
                    self.poses[r_idx][c_idx] = rand_bin
                    actions[r_idx, c_idx] = rand_bin
        
        return actions

    def _randomize_single_joint(self, actions):
        """Fully randomize 1 joint."""

        joint = npr.randint(0, 7)
        for l_idx, _ in enumerate(self.poses):
            rand_bin = npr.randint(0, 3)
            self.poses[l_idx][joint] = rand_bin
            actions[l_idx, joint] = rand_bin

        return actions
        

    def _timed_randomness(self, actions):
        """Return randomized action of prev step.
        
        Ensures some consistency in the randomization.
        """        
        for r_idx, row in enumerate(self.poses):
            for c_idx, _ in enumerate(row):
                if self.poses[r_idx][c_idx] is not None:
                    actions[r_idx, c_idx] = self.poses[r_idx][c_idx]

        return actions
        