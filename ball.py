import pybullet as pb
import pybullet_data

class Ball:
    def __init__(self, position: list, scaling:float):
        
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.ball_id = pb.loadURDF("soccerball.urdf", position,
                                   globalScaling=scaling)
        
    def reset_state(self, position:list):
        pb.resetBasePositionAndOrientation(self.ball_id,position,[0,0,0,1])
        pb.resetBaseVelocity(self.ball_id, [0,0,0],[0,0,0])

    