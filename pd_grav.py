from panda_robot import PandaRobot
import numpy as np
from typing import List

class PDGravController:
    """A PD with gravity compensation controller.
    
    This class implements or PDG controller as implemented in our
    ROS version and written in the report.
    """

    def __init__(self, panda: PandaRobot, kp: List[float], kd: List[float]):
        """Initialize the controller.

        @param panda (PandaRobot): The panda robot to control.
        @param kp (List[float]): The diagonal values of the Kp matrix.
        @param kd (List[float]): The diagonal values of the Kd matrix.
        """
        self.panda = panda
        self.kp = np.diag(kp)
        self.kd = np.diag(kd)
        self.q_desired = np.array([0. for _ in range(panda.get_dof())])
        self.grav_acc = [0. for _ in range(panda.get_dof())]

    
    def step(self) -> None:
        """Sets panda's torques based on current and desired joint position.
        """
        pos, vel = self.panda.get_position_and_velocity()
        grav_comp = self.panda.calculate_inverse_dynamics(pos, vel, self.grav_acc)

        pos = np.array(pos)
        vel = np.array(vel)
        q_error = self.q_desired - pos
        torques = np.matmul(self.kp, q_error) - np.matmul(self.kd, vel) + grav_comp
        self.panda.set_torques(torques.tolist())


    def set_joint_desired(self, q_desired: List[float]) -> None:
        """Set the desired joint positions.
        
        @param q_desired (List[float]): The desired joint positions.
        """
        assert len(self.q_desired) == len(q_desired)
        self.q_desired = np.array(q_desired)

