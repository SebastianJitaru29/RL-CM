import pybullet as pb
import pybullet_data

from typing import List, Tuple

class Ball:
    """Class to handle the ball object in PyBullet."""

    def __init__(self, position: List[float], scaling: float):
        """Initialize the ball.
        
        This function loads the ball urdf and sets the physics dynamics
        for the object.

        @param position (List[float]): the position to spawn the ball.
        @param scaling (float): the scaling of the ball object.

        """
        
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.ball_id = pb.loadURDF("soccerball.urdf", position,
                                   globalScaling=scaling)
        
        pb.changeDynamics(
            bodyUniqueId=self.ball_id,
            linkIndex=-1,
            lateralFriction=0.025,
            spinningFriction=0.005,
            rollingFriction=0.005,
        )
        
    
    def get_position_and_velocity(self) -> Tuple[List[float], List[float]]:
        """Get the position and velocity of the ball.
        
        @return (Tuple[List[float], List[float]]): the position and
            velocity of the ball.
        """
        position, _ = pb.getBasePositionAndOrientation(self.ball_id)
        velocity, _ = pb.getBaseVelocity(self.ball_id)

        return (position, velocity)
        
    def reset_state(self, position: List[float]):
        """Reset the ball physics and place it at the given position.
        
        @param position (List[float]): the position to place the ball.
        """
        pb.resetBasePositionAndOrientation(self.ball_id, position, [0,0,0,1])
        pb.resetBaseVelocity(self.ball_id, [0,0,0], [0,0,0])

    