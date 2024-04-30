from crowd_sim.envs.utils.agent import Agent
from crowd_sim.envs.utils.state import JointState
from crowd_nav.utils.rollout_window import RolloutWindow


class Robot(Agent):
    def __init__(self, config, section):
        super().__init__(config, section)
        """
        == From `Agent` ==

        Base class for robot and human. Have the physical attributes of an agent.
        config: info from config file
        section: section from config file -- i.e. 'human' or 'robot'
        """

    def act(self, rollout_window:RolloutWindow):
        '''
        Would need to change this, now the robot should accept a
        rollout window and pass that in predict -> next stop change cadrls predict
        '''
        if self.policy is None:
            raise AttributeError('Policy attribute has to be set!')
        # state = JointState(self.get_full_state(), ob)
        # print(f"policy {self.policy.name}")
        action = self.policy.predict(rollout_window)
        return action
