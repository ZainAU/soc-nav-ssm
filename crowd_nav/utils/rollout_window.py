import numpy as np
from crowd_sim.envs.utils.state import JointState

class RolloutWindow():
    def __init__(self, window_size:int) -> None:
        self.window_size  = window_size
        self.rollout_window = np.zeros([window_size], dtype = JointState) 
    def push_to_rollout(self, x:JointState):
        self.rollout_window =  np.concatenate((np.array([x]),self.rollout_window[:-1]))
