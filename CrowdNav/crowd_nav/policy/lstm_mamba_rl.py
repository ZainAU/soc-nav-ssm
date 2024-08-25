import torch
import torch.nn as nn
from torch.nn.functional import softmax
import logging
from crowd_nav.policy.cadrl import mlp
from crowd_nav.policy.multi_human_rl import MultiHumanRL
from mamba_ssm import Mamba
import numpy as np
from mamba_ssm.utils.generation import InferenceParams
from crowd_nav.utils.rollout_window import RolloutWindow
''''
So from my understanding the lstm_mamba class wont be the class where I define the model, 
Instead I should make a valuenetwork class to define the value network and just store it in self.model
'''

class ValueNetork(nn.Module):
    def __init__(self, 
                 input_dim,
                 self_state_dim, 
                 d_state, 
                 d_conv,  
                 lstm_hidden_dim,
                 device = 'cuda'
                 ):
        super().__init__()
        self.device = device
        if self.device !='cuda':
            raise AttributeError(f'Mamba requires device cuda, {self.device} is given')
        # print(f'input dimension = {input_dim}')
        self.self_state_dim = self_state_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm = nn.LSTM(input_dim, lstm_hidden_dim, batch_first=True)
        # logging.info(f'dmodel =  input + lstm hidden = {input_dim + lstm_hidden_dim}')
        # print()
        self.mambaLayer = Mamba(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=self_state_dim + lstm_hidden_dim, # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,    # Local convolution width
            expand=1,    # Block expansion factor--- HAVE HARDCODED this
            layer_idx=1,
            out_dim=1
        )


    def forward(self,state):
        # change the state shape similar to LSTM-RL
        '''
        The reason why it is not trainig properly is because of the way states are shaped maybe
        understand:
         1.  states
         2. Joint states
         Think: 
            - The state of the robot should be bigger since it is completely observable
            - Maybe the state of the observer 
        '''
        size = state.shape
        self_state = state[:, 0, :self.self_state_dim]
        

        h0 = torch.zeros(1, size[0], self.lstm_hidden_dim,device= self.device)
        c0 = torch.zeros(1, size[0], self.lstm_hidden_dim,device= self.device)
        output, (hn, cn) = self.lstm(state, (h0, c0))
        hn = hn.squeeze(0)
        # print(self.self_state_dim)
        joint_state = torch.cat([self_state, hn], dim=1)
        # print(f'the size of joint_state = {joint_state.shape}')
        joint_state = joint_state.unsqueeze(1)
        # print(f'the size of the js unsq {joint_state.shape}')
        value,_ = self.mambaLayer(joint_state)
        # print(value.shape)
        value = value.squeeze(1)
        # print(value.shape)
        ''''
        Maybe just see how CADRL implemented, and paste as it is
        '''

        
        return value

class LstmMamba(MultiHumanRL):
    def __init__(self):
        super().__init__()
        self.name = 'lstm_mamba'

    def configure(self,config):
        

        '''
        Eventually set everything in this one'
        Need to read MLP/Value network dimensions 

        Need to understand the parameters of 
        mamba, i.e. d_state, d_conv,expand

        '''
        

        self.set_common_parameters(config) #check what this does
        mlp_dims = [int(x) for x in config.get('lstm_mamba', 'mlp_dims').split(', ')]
        d_state =config.getint('lstm_mamba','d_state')
        d_conv =config.getint('lstm_mamba','d_conv') 
        global_state_dim = config.getint('lstm_mamba', 'global_state_dim')
        self.multiagent_training = config.getboolean('lstm_mamba', 'multiagent_training')
        # print(f'Multi agent=  {self.multiagent_training}')
        self.device = 'cuda'
        self.model = ValueNetork(input_dim =   self.input_dim(),
                                 self_state_dim =self.self_state_dim ,  #initialized in cadrl
                                 d_state = d_state, 
                                 d_conv  = d_conv, 
                                 lstm_hidden_dim = global_state_dim,
                                 device = self.device)


        print(self.model.eval())
        logging.info(f'Policy:{self.name}')
        return
    '''
    Check if something different would need to be done to change the MultiHumanRL functions. For example, in LstmRL they implement predict because they want to set the state in a particular order
    '''
    def predict(self, Rollout:RolloutWindow):
        """
        Input state is the joint state of robot concatenated with the observable state of other agents

        To predict the best action, agent samples actions and propagates one step to see how good the next state is
        thus the reward function is needed

        """

        # def dist(human):
        #     # sort human order by decreasing distance to the robot
        #     return np.linalg.norm(np.array(human.position) - np.array(state.self_state.position))

        # state.human_states = sorted(state.human_states, key=dist, reverse=True)
        Rollout.sort_distance()
        return super().predict(Rollout)