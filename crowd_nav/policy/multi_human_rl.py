import torch
import numpy as np
from crowd_sim.envs.utils.action import ActionRot, ActionXY
from crowd_nav.policy.cadrl import CADRL
from crowd_nav.utils.rollout_window import RolloutWindow
from crowd_sim.envs.utils.state import JointState
from typing import List


class MultiHumanRL(CADRL):
    def __init__(self):
        super().__init__()

    def next_state(self, Rollout:RolloutWindow, action, occupancy_maps):

        '''
        
        returns:
            - Reward of the next state
            - rotated_NextStates: Tensor shpae (T, batch, num_humans, dimension )
        '''
        # rotated_NextStates = list()
        
        current_state = Rollout.rollout_window[0]

        next_self_state = self.propagate(current_state.self_state, action) #This was one of the difference between GA3C CADRL paper and crowdnav
        if self.query_env:
            next_human_states, reward, done, info = self.env.onestep_lookahead(action)
        else:
            next_human_states = [self.propagate(human_state, ActionXY(human_state.vx, human_state.vy))
                                for human_state in current_state.human_states]
            reward = self.compute_reward(next_self_state, next_human_states)
        next_state = JointState(next_self_state,next_human_states)
        next_state_window = Rollout.next_state_window(next_state)
        for i in range(Rollout.non_zero_states):
            next_self_state = next_state_window[i].self_state
            next_human_states = next_state_window[i].human_states
            batch_next_states = torch.cat([torch.Tensor([next_self_state + next_human_state]).to(self.device)
                                            for next_human_state in next_human_states], dim=0)
            # I don't understand why the authors added the two states? Maybe Chen et al didn't do this and the diverse 4 paper did this
            rotated_batch_input = self.rotate(batch_next_states).unsqueeze(0)
            if self.with_om:
                if occupancy_maps is None:
                    occupancy_maps = self.build_occupancy_maps(next_human_states).unsqueeze(0)
                rotated_batch_input = torch.cat([rotated_batch_input, occupancy_maps.to(self.device)], dim=2)
            
            if i == 0:
                rotated_NextStates = rotated_batch_input.unsqueeze(0)
            else:
                rotated_NextStates = torch.cat((rotated_NextStates,rotated_batch_input.unsqueeze(0)), dim = 1).to(self.device)
                # tensor should be (#batch,Time, agents, dim )
            
            # print(f'shape of the tensor{rotated_NextStates.shape}\nThe non zero {Rollout.non_zero_states}')
            # rotated_NextStates.append(rotated_batch_input)
        # print(i)
        return rotated_NextStates,reward
    
    def predict(self, Rollout : RolloutWindow):
        """
        A base class for all methods that takes pairwise joint state as input to value network.
        The input to the value network is always of shape (batch_size, # humans, rotated joint state length)

        """
        '''
        Proposed changes:
        Accpts a Rollout window,
        instead of state.self_state, make a current state 
        maybe make a Rollout window class
        
        '''
        # state represents the current state
        state = Rollout.rollout_window[0]
        # print(self.rollout_transform(Rollout).shape)
   
        
        if self.phase is None or self.device is None:
            raise AttributeError('Phase, device attributes have to be set!')
        if self.phase == 'train' and self.epsilon is None:
            raise AttributeError('Epsilon attribute has to be set in training phase')

        if self.reach_destination(state):
            return ActionXY(0, 0) if self.kinematics == 'holonomic' else ActionRot(0, 0)
        if self.action_space is None:
            self.build_action_space(state.self_state.v_pref)

        occupancy_maps = None
        probability = np.random.random()
        if self.phase == 'train' and probability < self.epsilon:
            max_action = self.action_space[np.random.choice(len(self.action_space))]
        else:
            self.action_values = list()
            max_value = float('-inf')
            max_action = None
            for action in self.action_space:
                rotated_batch_input_window,reward = self.next_state(Rollout=Rollout,action=action, occupancy_maps=occupancy_maps)
                # print(f'shape of rotated batch {rotated_batch_input_window.shape}, and expected shape = 2,#agents,dim')
                # VALUE UPDATE
                if Rollout.window_size == 1:
                    rotated_batch_input= rotated_batch_input_window[0] # Do this if temporal window is false or equal to one 
                    # print(f'shape of input = {rotated_batch_input.shape}')
                    next_state_value = self.model(rotated_batch_input).data.item() #quering the NN here
                else:
                    next_state_value = self.model(rotated_batch_input_window).data.item()
                # print(rotated_batch_input.shape)
                value = reward + pow(self.gamma, self.time_step * state.self_state.v_pref) * next_state_value
                self.action_values.append(value)
                if value > max_value:
                    max_value = value
                    max_action = action
            if max_action is None:
                raise ValueError('Value network is not well trained. ')

        if self.phase == 'train':
            if self.window_size == 1:
                self.last_state = self.transform(state)
            else:
                self.last_state = self.rollout_transform(Rollout)

        return max_action

    def compute_reward(self, nav, humans):
        # collision detection
        dmin = float('inf')
        collision = False
        for i, human in enumerate(humans):
            dist = np.linalg.norm((nav.px - human.px, nav.py - human.py)) - nav.radius - human.radius
            if dist < 0:
                collision = True
                break
            if dist < dmin:
                dmin = dist

        # check if reaching the goal
        reaching_goal = np.linalg.norm((nav.px - nav.gx, nav.py - nav.gy)) < nav.radius
        if collision:
            reward = -0.25
        elif reaching_goal:
            reward = 1
        elif dmin < 0.2:
            reward = (dmin - 0.2) * 0.5 * self.time_step
        else:
            reward = 0

        return reward
    def rollout_transform(self,Rollout:RolloutWindow):
        """
        Take the state passed from agent and transform it to the input of value network

        :param state:
        :return: tensor of shape (T,# of humans, len(state))
        """
        occupancy_maps = None
        for i in range(Rollout.non_zero_states):
            next_self_state = Rollout.rollout_window[i].self_state
            next_human_states = Rollout.rollout_window[i].human_states
            batch_next_states = torch.cat([torch.Tensor([next_self_state + next_human_state]).to(self.device)
                                            for next_human_state in next_human_states], dim=0)
            # I don't understand why the authors added the two states? Maybe Chen et al didn't do this and the diverse 4 paper did this
            rotated_batch_input = self.rotate(batch_next_states).unsqueeze(0)
            # print(f"In rollout transform {rotated_batch_input.shape}")

            if self.with_om:
                if occupancy_maps is None:
                    occupancy_maps = self.build_occupancy_maps(next_human_states).unsqueeze(0)
                rotated_batch_input = torch.cat([rotated_batch_input, occupancy_maps.to(self.device)], dim=2)
            if i == 0:
                rotated_state = rotated_batch_input
            else:
                rotated_state = torch.cat((rotated_state,rotated_batch_input), dim = 0).to(self.device)
        # print(f"In rollout transform {rotated_state.shape}")
        return rotated_state
    def transform(self, state):
        """
        Take the state passed from agent and transform it to the input of value network

        :param state:
        :return: tensor of shape (# of humans, len(state))
        """
        state_tensor = torch.cat([torch.Tensor([state.self_state + human_state]).to(self.device)
                                  for human_state in state.human_states], dim=0)
        if self.with_om:
            occupancy_maps = self.build_occupancy_maps(state.human_states)
            state_tensor = torch.cat([self.rotate(state_tensor), occupancy_maps.to(self.device)], dim=1)
        else:
            state_tensor = self.rotate(state_tensor)
        return state_tensor

    def input_dim(self):
        return self.joint_state_dim + (self.cell_num ** 2 * self.om_channel_size if self.with_om else 0)

    def build_occupancy_maps(self, human_states):
        """

        :param human_states:
        :return: tensor of shape (# human - 1, self.cell_num ** 2)
        """
        occupancy_maps = []
        for human in human_states:
            other_humans = np.concatenate([np.array([(other_human.px, other_human.py, other_human.vx, other_human.vy)])
                                         for other_human in human_states if other_human != human], axis=0)
            other_px = other_humans[:, 0] - human.px
            other_py = other_humans[:, 1] - human.py
            # new x-axis is in the direction of human's velocity
            human_velocity_angle = np.arctan2(human.vy, human.vx)
            other_human_orientation = np.arctan2(other_py, other_px)
            rotation = other_human_orientation - human_velocity_angle
            distance = np.linalg.norm([other_px, other_py], axis=0)
            other_px = np.cos(rotation) * distance
            other_py = np.sin(rotation) * distance

            # compute indices of humans in the grid
            other_x_index = np.floor(other_px / self.cell_size + self.cell_num / 2)
            other_y_index = np.floor(other_py / self.cell_size + self.cell_num / 2)
            other_x_index[other_x_index < 0] = float('-inf')
            other_x_index[other_x_index >= self.cell_num] = float('-inf')
            other_y_index[other_y_index < 0] = float('-inf')
            other_y_index[other_y_index >= self.cell_num] = float('-inf')
            grid_indices = self.cell_num * other_y_index + other_x_index
            occupancy_map = np.isin(range(self.cell_num ** 2), grid_indices)
            if self.om_channel_size == 1:
                occupancy_maps.append([occupancy_map.astype(int)])
            else:
                # calculate relative velocity for other agents
                other_human_velocity_angles = np.arctan2(other_humans[:, 3], other_humans[:, 2])
                rotation = other_human_velocity_angles - human_velocity_angle
                speed = np.linalg.norm(other_humans[:, 2:4], axis=1)
                other_vx = np.cos(rotation) * speed
                other_vy = np.sin(rotation) * speed
                dm = [list() for _ in range(self.cell_num ** 2 * self.om_channel_size)]
                for i, index in np.ndenumerate(grid_indices):
                    if index in range(self.cell_num ** 2):
                        if self.om_channel_size == 2:
                            dm[2 * int(index)].append(other_vx[i])
                            dm[2 * int(index) + 1].append(other_vy[i])
                        elif self.om_channel_size == 3:
                            dm[3 * int(index)].append(1)
                            dm[3 * int(index) + 1].append(other_vx[i])
                            dm[3 * int(index) + 2].append(other_vy[i])
                        else:
                            raise NotImplementedError
                for i, cell in enumerate(dm):
                    dm[i] = sum(dm[i]) / len(dm[i]) if len(dm[i]) != 0 else 0
                occupancy_maps.append([dm])

        return torch.from_numpy(np.concatenate(occupancy_maps, axis=0)).float()

