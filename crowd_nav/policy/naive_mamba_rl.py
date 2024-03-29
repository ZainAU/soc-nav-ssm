import torch
import torch.nn as nn
import logging
from crowd_nav.policy.multi_human_rl import MultiHumanRL
from mamba_ssm import Mamba
import numpy as np
from mamba_ssm.utils.generation import InferenceParams


class ValueNetork(nn.Module):
    def __init__(self, 
                 input_dim, 
                 d_state, 
                 d_conv,  
                 lstm_hidden_dim,
                 device = 'cuda',
                 phase = 'Train'
                 ):
        self.phase = phase
        self.done = False
        self.done_is_true = False
        super().__init__()
        self.device = device
        if self.device !='cuda':
            raise AttributeError(f'Mamba requires device cuda, {self.device} is given')
        # print(f'input dimension = {input_dim}')
        self.self_state_dim = input_dim
        self.inference = InferenceParams(max_seqlen=None, max_batch_size=None)
        self.mambaLayer = Mamba(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=input_dim, # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,    # Local convolution width
            expand=1,    # Block expansion factor--- HAVE HARDCODED this
            layer_idx=1,
            out_dim=1
        )


    def forward(self,state):
        # change the state shape similar to LSTM-RL
        # print(f'state type {type(state)}')
        '''
        The reason why it is not trainig properly is because of the way states are shaped maybe
        understand:
         1.  states
         2. Joint states
         Think: 
            - The state of the robot should be bigger since it is completely observable
            - Maybe the state of the observer 
        '''
        # if self.done == True and self.done_is_true == False:
        #     self.done_is_true = True
        #     print(f'done is {self.done}') 
        # elif self.done_is_true == True:
        #     self.done_is_true = False
        #     print(f'done is false')
        size = state.shape
        # print(f"The size of the input is {size}")
        values, _ = self.mambaLayer(state)
        '''
        The below condition is only applicable when we have states from multiple 
        '''
        # if self.phase in ['train', 'val']:
        #     values, _ = self.mambaLayer(state)
        # else: #inference in the case of 
        #     values, _ = self.mambaLayer(state, self.inference, episode_reset = self.done)
        value = values[:,-1,:]
        # print(f"value = {value.shape}")
  
        ''''
        Maybe just see how CADRL implemented, and paste as it is
        '''

        
        return value

class MambaRL(MultiHumanRL):
    def __init__(self):
        super().__init__()
        self.name = 'Naive-MambaRL'

    def configure(self,config):
        self.set_common_parameters(config) #check what this does
        d_state =config.getint('Naive-MambaRL','d_state')
        d_conv =config.getint('Naive-MambaRL','d_conv') 
        global_state_dim = config.getint('Naive-MambaRL', 'global_state_dim')
        self.multiagent_training = config.getboolean('Naive-MambaRL', 'multiagent_training')
        # print(f'Multi agent=  {self.multiagent_training}')
        self.device = 'cuda'
        self.model = ValueNetork(input_dim =   self.input_dim(),
                                #  self_state_dim =self.self_state_dim ,  #initialized in cadrl what are they anyways?
                                 d_state = d_state, 
                                 d_conv  = d_conv, 
                                 lstm_hidden_dim = global_state_dim,
                                 device = self.device,
                                 phase = self.phase)


        print(self.model.eval())
        logging.info(f'Policy:{self.name}')
        return
    
    def predict(self, state):
        """
        Input state is the joint state of robot concatenated with the observable state of other agents

        To predict the best action, agent samples actions and propagates one step to see how good the next state is
        thus the reward function is needed

        """

        def dist(human):
            # sort human order by decreasing distance to the robot
            return np.linalg.norm(np.array(human.position) - np.array(state.self_state.position))

        state.human_states = sorted(state.human_states, key=dist, reverse=True)
        
        return super().predict(state)
