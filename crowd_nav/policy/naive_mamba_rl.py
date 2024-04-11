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



class ValueNetork(nn.Module):
    def __init__(self, 
                 input_dim      :int, 
                 device         :str = 'cuda',
                 phase          :str = 'Train', 
                 n_layers     :int = 1,
                 norm_epsilon: float = 1e-5,
                 rms_norm: bool = False,
                 fused_add_norm=False,
                 norm_f = False,
                 dtype=None
                 ):
        self.config = MambaConfig(input_dim, n_layer=n_layers, rms_norm=rms_norm, fused_add_norm=fused_add_norm)
        self.norm_f = norm_f
        self.fused_add_norm = fused_add_norm
        self.phase = phase
        self.done = False
        self.done_is_true = False
        super().__init__()
        self.device = device
        factory_kwargs = {"device": device, "dtype": dtype}
        
        self.layers = nn.ModuleList(
            [
                self.create_block(
                    input_dim,
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
        
        self.value_layer = nn.Linear(in_features=input_dim,out_features=1, bias = True, device=device,dtype=dtype)

    def forward(self, state, inference_params=None):
        hidden_states = state
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )
      
       
        value = self.value_layer(hidden_states[:,-1,:])
        # print(f'shape of state {hidden_states[1,-1,:]}\nshape of value{value.shape}')
        return value    

    def create_block(
                    self,
                    d_model,
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
        mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
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
        self.name = 'Naive-MambaRL'

    def configure(self,config):
        self.set_common_parameters(config) #check what this does
        self.multiagent_training = config.getboolean('Naive-MambaRL', 'multiagent_training')
        # print(f'Multi agent=  {self.multiagent_training}')
        self.device = 'cuda'
        self.model = ValueNetork(input_dim =   self.input_dim(),
                                 device = self.device,
                                 phase = self.phase,
                                 n_layers=4)


        print(self.model.eval())
        logging.info(f'Policy:{self.name}')
        return
    
    def predict(self, rollout):
        """
        Input state is the joint state of robot concatenated with the observable state of other agents

        To predict the best action, agent samples actions and propagates one step to see how good the next state is
        thus the reward function is needed

        """

        def dist(human):
            # sort human order by decreasing distance to the robot
            return np.linalg.norm(np.array(human.position) - np.array(state.self_state.position))

        # state.human_states = sorted(state.human_states, key=dist, reverse=True)
        
        return super().predict(rollout)
