import numpy as np
from crowd_sim.envs.utils.state import JointState,Zero_JointState

class RolloutWindow():
    def __init__(self, window_size:int) -> None:
        self.window_size  = window_size
        self.rollout_window = np.zeros([window_size], dtype = JointState) 
        # for i in range(self.window_size):
        #      self.rollout_window[i] = Zero_JointState()
        # The below attribute is the number of Joint States in the rollout window
        self.non_zero_states = 0
    def push_to_rollout(self, x:JointState):
        self.rollout_window =  np.concatenate((np.array([x]),self.rollout_window[:-1]))

    def next_state_window(self, x:JointState):
        return np.concatenate((np.array([x]),self.rollout_window[:-1]))
    def sort_distance(self):
        for i in range(self.non_zero_states):
            state =  self.rollout_window[i]
            def dist(human):
                    # sort human order by decreasing distance to the robot
                    return np.linalg.norm(np.array(human.position) - np.array(state.self_state.position))
            state.human_states = sorted(state.human_states, key=dist, reverse=True)
            self.rollout_window[i]  = state