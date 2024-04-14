import logging
import copy
import torch
import torch.nn.functional as F 
from crowd_sim.envs.utils.info import *
from crowd_sim.envs.utils.utils import write_results, get_env_code
import numpy as np
import time
import os
from tqdm import tqdm
import numpy as np
from crowd_sim.envs.utils.state import JointState #chang
from crowd_nav.utils.rollout_window import RolloutWindow
np.seterr(all='raise')

class Explorer(object):
    def __init__(self, env, robot, device, memory=None, gamma=None, target_policy=None, window_size = 10):
        self.env = env
        self.robot = robot
        self.device = device
        self.memory = memory
        self.gamma = gamma
        self.target_policy = target_policy
        self.target_model = None
        self.window_size = window_size
        

    def update_target_model(self, target_model):
        self.target_model = copy.deepcopy(target_model)

    def push_to_tensor(self,tensor, x):
        '''
        input:
        tensor: np.array
        x: some element
        The function implements a FIFO queue
        '''
       
        return np.concatenate((np.array([x]),tensor[:-1]))
    # @profile
    def run_k_episodes(self, k, phase, update_memory=False, imitation_learning=False, episode=None,
                       print_test=True, results_dir=None):
        self.robot.policy.set_phase(phase)
        success_times = []
        collision_times = []
        timeout_times = []
        success = 0
        collision = 0
        timeout = 0
        too_close = 0
        min_dist = []
        cumulative_rewards = []
        collision_cases = []
        timeout_cases = []

        for i in tqdm(range(k)):
            epoch_start = time.perf_counter()
            ob = self.env.reset(phase)
            state = JointState(self.robot.get_full_state(), ob)
            Rollout = RolloutWindow(window_size = self.window_size)
            # rollout_window = np.zeros([window_size], dtype = JointState) # Shape should be window_size x joint state_size however join state is a single variable/data struct
           
            
            done = False
            states = []
            actions = []
            rewards = []


            while not done:
                
                # Get the current state
                state = JointState(self.robot.get_full_state(), ob)
                # Append the current state to rollout_window
                Rollout.push_to_rollout(state)
                if Rollout.non_zero_states < Rollout.window_size:
                    Rollout.non_zero_states += 1
              
                

                action = self.robot.act(Rollout)
                ob, reward, done, info = self.env.step(action)
                self.robot.policy.done = done # Doing this to reset the internal SSM state when  episode gets done
                states.append(self.robot.policy.last_state)
                actions.append(action)
                rewards.append(reward)

                
                
                if isinstance(info, Danger):
                    too_close += 1
                    min_dist.append(info.min_dist)

            if isinstance(info, ReachGoal):
                success += 1
                success_times.append(self.env.global_time)
            elif isinstance(info, Collision):
                collision += 1
                collision_cases.append(i)
                collision_times.append(self.env.global_time)
            elif isinstance(info, Timeout):
                timeout += 1
                timeout_cases.append(i)
                timeout_times.append(self.env.time_limit)
            else:
                raise ValueError('Invalid end signal from environment')

            if update_memory:
                if isinstance(info, ReachGoal) or isinstance(info, Collision):
                    # only add positive(success) or negative(collision) experience in experience set
                    self.update_memory(states, actions, rewards, imitation_learning)

            cumulative_rewards.append(sum([pow(self.gamma, t * self.robot.time_step * self.robot.v_pref)
                                           * reward for t, reward in enumerate(rewards)]))

            # print first 10, then every 10% milestone, then last 3
            if (
                    print_test and
                    phase.lower() == 'test' and
                    (
                        i < 100 or
                        i % (k//10) == 0 or
                        i > k-4
                    )
                ):
                ep_str = str(i).zfill(int(np.log10(k-1))+1)
                epoch_diff = time.perf_counter() - epoch_start
                logging.debug(f'Test Episode {ep_str}/{k-1}    Time: [{time.strftime("%H:%M:%S", time.gmtime(epoch_diff))}]')

        success_rate = success / k

        collision_rate = collision / k

        assert success + collision + timeout == k

        avg_nav_time = sum(success_times) / len(success_times) if success_times else self.env.time_limit

        # was checking why this wasn't working
        # logging.debug(f"success times: {success_times}")
        # Turns out, there might be *no* success times, so we condition:
        if len(success_times) > 0:
            std_nav_time = np.std(success_times)
        else:
            std_nav_time = 0


        std_cumul_rewards = np.std(cumulative_rewards)


        extra_info = '' if episode is None else 'in episode {} '.format(episode)
        logging.info('{:<5} {}has success rate: {:.2f}, collision rate: {:.2f}, nav time: {:.2f}±{:1f}, total reward: {:.4f})'.
                     format(phase.upper(), extra_info, success_rate, collision_rate, avg_nav_time, std_nav_time,
                            average(cumulative_rewards)))


        if phase in ['val', 'test']:
            num_step = sum(success_times + collision_times + timeout_times) / self.robot.time_step
            try:
                logging.info('Frequency of being in danger: %.2f and average min separate distance in danger: %.2f±%.1f', too_close / num_step, average(min_dist), np.std(min_dist))
            except:
                print(f'min_distance = {min_dist}')
                logging.info('some error occured while calcutating the std ')
        # array of [env, succ, coll, time, std_time, rew, std_rew, disc_freq,
        #           danger_d_min, std_danger, d_min_overall, std_overall]
        if results_dir is not None:
            results_path = os.path.join(results_dir, 'results.csv')
            logging.info(f"results_path: {results_path}")
            env_code = get_env_code(self.env, phase)
            results = [
                env_code,
                success_rate,
                collision_rate,
                avg_nav_time,
                std_nav_time,
                average(cumulative_rewards),
                std_cumul_rewards,
                too_close / num_step,
                average(min_dist),
                np.std(min_dist)
            ]
            write_results(results_path, results)

    def update_memory(self, states, actions, rewards, imitation_learning=False):
        '''
        rotated_NextStates: Tensor shpae (T, batch, num_humans, dimension )
        The memory should have a tuple of (rotated_NextStates, value)

        plan:
        1. print the state_stack.shape in run k episode to see the shape, and see if enumerate can be done
        2. implement a function of transform that takes in a rollout window


        OR MAYBE THE memory should have a rollout window
        '''
        # print("AHHHHHHHHHHHHHHHHHHHH")
        # print(type(states[0]))
        if self.memory is None or self.gamma is None:
            raise ValueError('Memory or gamma value is not set!')
        k = self.window_size  # making this variable cause its easier to write k instead of typing it multiple times
        for i, state in enumerate(states):
            # i is essentially the time step here
            reward = rewards[i]
            # value = []
            # VALUE UPDATE
            values = []
            if imitation_learning:
                # define the value of states in IL as cumulative discounted rewards, which is the same in RL
                '''
                Currently we are not taking temporal information for imitation learning, the 
                '''
                if self.window_size == 1:
                    state = self.target_policy.transform(state)#.unsqueeze(0)
                else:
                    # Later add .rollout_transform i.e. take rollout window as input for IL as well
                    state = self.target_policy.transform(state).unsqueeze(0)
                for j in range(np.min([k-1, i])+1):
                    values.append(sum([pow(self.gamma, max(t - i - j, 0) * self.robot.time_step * self.robot.v_pref) * reward
                             * (1 if t >= i - j else 0) for t, reward in enumerate(rewards)]))
                # value = sum([pow(self.gamma, max(t - i, 0) * self.robot.time_step * self.robot.v_pref) * reward
                #              * (1 if t >= i else 0) for t, reward in enumerate(rewards)])
                # x = np.zeros(self.window_size)
                # x[self.window_size-1:] = value
                # value = x
                # values.reverse()
                # value = torch.zeros(self.window_size)
                # value[k-1-np.min([k-1, i]):] = torch.Tensor(values)

            else:
                # state should be of the shape (T, #agents, dim) if agents is window size >= 1, because in multi human RL we implemented the predict method such that it outputs shape (#agents, dim) if window size == 1 other wise (T, #agents, dim) 
                
                # if i == len(states) - 1:
                #     # terminal state
                #     value = reward
                # else:
                #     next_state = states[i + 1]
                #     gamma_bar = pow(self.gamma, self.robot.time_step * self.robot.v_pref)
                #     value = reward + gamma_bar * self.target_model(next_state.unsqueeze(0)).data.item()
                    
                gamma_bar = pow(self.gamma, self.robot.time_step * self.robot.v_pref)
                
                for j in range(np.min([k-1, i])+1 ):
                    if i-j== len(states) - 1:
                        values.append(reward)
                    else:
                        reward = rewards[i-j]
                        state = states[i-j + 1]
                        values.append(reward + gamma_bar * self.target_model(state.unsqueeze(0)).data[-1,-1].item())
            values.reverse()
            value = torch.zeros(self.window_size)
            value[k-1-np.min([k-1, i]):] = torch.Tensor(values)



            # Make a function to pad 
            if self.window_size != 1:
                state = self.pad_state(state)
                # print(f'state shape in update memory {state.shape}')

            value = torch.Tensor(value).to(self.device)
            # print(value.shape)
            # print("AKDMSKLSKN")
            # value = value.view[]
            self.memory.push((state, value))
    
    def pad_state(self, state):
        if len(state) != self.window_size:
            padd = (0, 0, 0, 0, 0, self.window_size -state.shape[0])
            state = F.pad(state, padd, "constant", 0)
            # print(f'window size {self.window_size}')
            # print(f'{state.shape}')
        return state

def average(input_list):
    #logging.debug(f'Input list:\n{input_list}')
    if input_list:
        return sum(input_list) / len(input_list)
    else:
        return 0
