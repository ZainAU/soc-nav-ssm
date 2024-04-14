import math
from functools import partial
import json
import os
import numpy as np

from collections import namedtuple

import torch
import torch.nn as nn

from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.modules.mamba_simple import Mamba, Block
from mamba_ssm.utils.generation import GenerationMixin
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf

from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn


import logging
from crowd_nav.policy.multi_human_rl import MultiHumanRL
from mamba_ssm.utils.generation import InferenceParams
from crowd_nav.utils.rollout_window import RolloutWindow


class ValueNetork(nn.Module):
    def __init__(self, 
                 input_dim      :int, 
                 self_state_dim :int,
                 d_state        :int = 16,
                 device         :str = 'cuda',
                 phase          :str = 'Train', 
                 n_layers     :int = 1,
                 norm_epsilon: float = 1e-5,
                 rms_norm: bool = False,
                 fused_add_norm=False,
                 norm_f = False,
                 dtype=None,
                 window_size = None,
                 latent_state_dim = 16,
                 ):
        if not window_size:
            raise ValueError("Window size should not be none") 
        #######
        # NotImplementedError("Need to add embeddings to get shape from (Batch, T, #agents, dim) -> (Batch, T, join_state_dim)")
        
        ###### Generating joint states: Look into methods that use pretrained embedding generators like word2vec rightnow we will use
         
        #######
        super().__init__()
        self.window_size = window_size
        self.self_state_dim = self_state_dim
        self.latent_state_dim = latent_state_dim
        self.encoder_layer = nn.GRU(input_dim,latent_state_dim,batch_first= True)
        self.norm_f = norm_f
        self.fused_add_norm = fused_add_norm
        self.phase = phase
        self.done = False
        self.done_is_true = False
        self.device = device
        factory_kwargs = {"device": device, "dtype": dtype}
        mamba_model_dim = self_state_dim + latent_state_dim
        self.config = MambaConfig(mamba_model_dim, n_layer=n_layers, rms_norm=rms_norm, fused_add_norm=fused_add_norm)
        self.layers = nn.ModuleList(
            [
                self.create_block(
                    mamba_model_dim,
                    d_state = d_state,
                    ssm_cfg=None,
                    norm_epsilon=norm_epsilon,
                    rms_norm=False,
                    fused_add_norm=False,
                    residual_in_fp32=False,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(n_layers)
            ]
        )
        
        self.value_layer = nn.Linear(in_features=mamba_model_dim,out_features=1, bias = True, device=device,dtype=dtype)
        self.flatten_layer = nn.Flatten(start_dim=1, end_dim=2)
    def forward(self, state:torch.Tensor, inference_params=None):
        shape = state.shape
        if len(shape) == 4:
            
            for i in range(shape[1]):  
                # print(f'self state shape {state.shape} and i ={i}')
                self_state = state[:,i, 0, :self.self_state_dim]
                h0 = torch.zeros(1, shape[0], self.latent_state_dim,device= self.device)
                spatial_feature = state[:,i,:,:]
                output,hn = self.encoder_layer(spatial_feature,h0)
                # (B, T, latents space)
                hn = hn.squeeze(0)
                joint_state = torch.cat([self_state, hn], dim=1)
                # print(joint_state.shape)
                # print(f"This is the shape {hn.shape, joint_state.shape}")
                if i == 0:
                    hidden_states = joint_state.unsqueeze(0)
                else:
                    hidden_states = torch.cat((hidden_states, joint_state.unsqueeze(0)), dim = 0)
                # print('before',hidden_states.shape)
            _shape = hidden_states.shape
            # print(_shape)
            hidden_states = hidden_states.view(_shape[1],_shape[0],_shape[2])
            # print('after',hidden_states.shape)
            # if 1:
            #     hidden_states = state.view(shape[0],shape[1]*shape[2],shape[3])
            # else:
            #     pass
            # print(hidden_states.shape, shape) if hidden_states.shape[0] != 1 else 0 

        else:
           
            hidden_states = state
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )
        value = hidden_states
        value = self.value_layer(hidden_states)
        value = self.flatten_layer( value)

        # print(f'shape before entering value {value.shape}')       
        # print(f'shape of state {hidden_states[1,-1,:]}\nshape of value{value.shape}')
        return value    

    def create_block(
                    self,
                    d_model,
                    d_state = 100,
                    ssm_cfg=None,
                    norm_epsilon=1e-5,
                    rms_norm=False,
                    residual_in_fp32=False,
                    fused_add_norm=False,
                    layer_idx=None,
                    device=None,
                    dtype=None,
                ):
        if ssm_cfg is None:
            ssm_cfg = {}
        factory_kwargs = {"device": device, "dtype": dtype}
        mixer_cls = partial(Mamba, layer_idx=layer_idx, d_state = d_state, **ssm_cfg, **factory_kwargs)
        norm_cls = partial(
            nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
        )
        block = Block(
            d_model,
            mixer_cls,
            norm_cls=norm_cls,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
        )
        block.layer_idx = layer_idx
        return block    
    
   

class MambaRL(MultiHumanRL):
    def __init__(self):
        super().__init__()
        self.name = 'MambaRL'

    def configure(self,config):
        self.set_common_parameters(config) #check what this does
        self.multiagent_training = config.getboolean(self.name, 'multiagent_training')
        d_state =config.getint(self.name,'d_state')
        # print(f'Multi agent=  {self.multiagent_training}')
        self.device = 'cuda'
        self.model = ValueNetork(input_dim =   self.input_dim(),
                                 self_state_dim =self.self_state_dim ,
                                 device = self.device,
                                 phase = self.phase,
                                 n_layers=4,
                                 window_size = self.window_size,
                                 d_state=d_state)


        print(self.model.eval())
        logging.info(f'Policy:{self.name}')
        return
    
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
