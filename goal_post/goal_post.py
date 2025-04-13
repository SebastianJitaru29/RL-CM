import pybullet as pb
import pybullet_data
import os
import numpy as np

from typing import Tuple

class GoalPost:
    def __init__(
            self,
            position: Tuple[float, float, float],
            scaling: float
    ):
        
        pb.setAdditionalSearchPath(
            os.path.dirname(__file__) + '/model_description'
        )

        self.goal_id = pb.loadURDF('goal_post_nocol.urdf', position, 
                                    globalScaling=scaling, useFixedBase=True)
        
        self.dims = np.array([2, 0.7, 1.2]) * scaling

        #self.corners = None
        self.rot_z = 0
        self.centre = np.array(position)
        self.reset_state(position)
    
    def get_position_and_orientation(self):
        position, orientation = pb.getBasePositionAndOrientation(self.goal_id)
        return (position, orientation)
        
    def reset_state(
            self,
            position: Tuple[float, float, float],
            orientation: Tuple[float, float, float, float] = [0, 0, 0, 1],
    ):
        pb.resetBasePositionAndOrientation(self.goal_id, position, 
                                           orientation)
        pb.resetBaseVelocity(self.goal_id, [0,0,0], [0,0,0])
        pos, orient = self.get_position_and_orientation()
        rot_z = pb.getEulerFromQuaternion(orient)[2]
        self.rot_z = rot_z
        self.centre = np.array(pos)

        #self.corners = self._get_corners(pos, rot_z)

    def get_score(self, ball_pos: Tuple[float, float, float]):
        """Calculate whether the given position is within the goalframe."""

        if ball_pos[2] > self.dims[2]:
            return False

        gTb = np.array(ball_pos) - self.centre
        gTb[0] = gTb[0] * np.cos(self.rot_z) + gTb[1] * np.sin(self.rot_z)
        gTb[1] = -gTb[0] * np.sin(-self.rot_z) + gTb[1] * np.cos(self.rot_z)

        x_offset = self.dims[0] / 2

        # Check if point falls within dimensions
        if (
            gTb[0] <= x_offset and gTb[0] >= -x_offset 
            and gTb[1] <= self.dims[1] and gTb[1] > 0
        ):
            return True

        # if not we default to returning False
        return False
    