import pybullet as pb
import pybullet_data
import os

class GoalPost:
    def __init__(self, position: list, scaling:float):
        
        pb.setAdditionalSearchPath(os.path.dirname(__file__) + '/model_description')
        self.goal_id = pb.loadURDF('goal_post.urdf', position, 
                                    globalScaling=scaling, useFixedBase=True)
        
    
    def get_position_and_orientation(self):
        position, orientation = pb.getBasePositionAndOrientation(self.goal_id)
        return (position, orientation)
        
    def reset_state(self, position:list, orientation:list=[0, 0, 0, 1]):
        pb.resetBasePositionAndOrientation(self.goal_id, position, 
                                           orientation)
        pb.resetBaseVelocity(self.goal_id, [0,0,0], [0,0,0])
