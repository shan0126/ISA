import numpy as np
import torch as th
from sklearn.metrics import normalized_mutual_info_score
import random
import math
from sklearn.neighbors import NearestNeighbors
import time
from collections import Counter

from utils.goal_utils import find_common_elements, find_union, find_difference
from utils.goal_utils import multiply_hash_array, min_count_value_and_index

class GoalManager(object):
    def __init__(self, args):
        # hyper-parameters for buffers of curriculum goal and MDP goal state
        self.max_size = args.goal_buffer_size  # the buffer size of curriculum goal during exploring the goal state
        
        # parameters about the environment
        self.n_agent = args.n_agents           # the number of agent in environment
        self.n_actions = args.n_actions        # the number of action
        self.state_dim = args.state_shape      # state dimension
        self.reserved_indg_dim = args.goal_shape      # Dimensions reserved for input of individual goal information
        self.batch_size_run = args.batch_size_run     # Number of threads collecting data from environment in parallel
        self.map_name = args.env_args['map_name']
        
        # hyper-parameters about the proposed GCA method
        self.threshold = args.threshold_phi
        self.fac_hanming = args.fac_hanming
        self.fac_ciec = args.fac_ciec
        
        # ----about exploration
        self.exp_alg = args.exp_alg
        self.discrete_scale_count = args.discrete_scale_count
        self.arrival_ratio = args.arrival_ratio
        
        
        
        # initialize buffers of curriculum goal
        self.curr_ptr = 0
        self.curr_size = 0
        self.curr_goal_global = np.zeros((self.max_size, self.state_dim))
        
        # initialize buffers of MDP goal state
        self.ptr = 0
        self.size = 0
        self.goal_global = np.zeros((self.max_size, self.state_dim))
        
        # initialize lists (as set) for goal decomposition
        self.D_i = [[] for _ in range(self.n_agent)]           # self.D_i[i] denote agent i's index set to project a state into its goal 
        self.D_i_li = [[] for _ in range(self.n_agent)]        # self.D_i_li[i][j] denote which elements of state the j-th action of agent i can affect
        self.D_c = []                                          # self.D_c denote the common part of state that all agents have affects on
        self.D_iec = [[] for _ in range(self.n_agent)]         # self.D_iec denote the special part, where (self.D_c[i] = self.D_i[i] - self.D_c)
        
        # initialize other variables
        self.not_build = True                                  # a variable indicating if the above 4 lists have been built or not
        
        self.v_goal_shape = None                               # TRUE size of individual goal (instead of the size reserved via hyper-parameter)
        self.c_goal_shape = None                               # size of common part
        self.vec_goal_shape = None                             # size of special part
        
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
        
        self.counter = dict()
        self.c_state = dict()
        
        self.iec_counter = [dict() for _ in range(self.n_agent)]
        self.iec_state = [dict() for _ in range(self.n_agent)]
        
        
        # document for debug
        self.file_path = 'output_{}_{}.txt'.format(self.map_name, args.seed)
        open(self.file_path, 'w').close()
        
    
    def inquiry(self, listener):
        self.D_i = listener.D_i          
        self.D_i_li = listener.D_i_li      
        self.D_c = listener.D_c                              
        self.D_iec = listener.D_iec        

        self.not_build = False
        
        self.v_goal_shape = listener.v_goal_shape                
        self.c_goal_shape = listener.c_goal_shape                      
        self.vec_goal_shape = listener.vec_goal_shape                     
        
        
    
    def explore_goal_state(self, episode_batch):
        if self.exp_alg == "count":
            state_runbatch = episode_batch["state"]    # (ep_batch, ep_length, state_size)
            state_runbatch = state_runbatch.reshape(-1, state_runbatch.shape[-1])
            if self.map_name == "3m" or self.map_name == "8m" or self.map_name == "2s_vs_1sc":
                self.build_goal_count(state_runbatch)
            elif self.map_name == "shooting" or self.map_name == "navigation" or self.map_name == "unlock":
                self.build_goal_count(state_runbatch)
                self.build_iecgoal_count(state_runbatch)
        else:
            raise NotImplementedError
        
        
        
    def build_iecgoal_count(self, states):
        # Preprocess states and remove meaningless ones
        states = states.reshape(-1, self.state_dim)
        states = states.numpy()
        
        non_zero_mask = np.any(states != 0, axis=1)
        states = states[non_zero_mask]
        
        # define the slices to project state
        for agent_idx in range(self.n_agent):
            index_list_slices = self.D_iec[agent_idx]
            hashed_state_slices = multiply_hash_array(states[:, index_list_slices], self.discrete_scale_count)
            state_to_store, stored_slices, min_count  = self.min_count_value_and_index_iec(agent_idx, hashed_state_slices, states, self.arrival_ratio)
            
            with open(self.file_path, 'a') as f:
                f.write('-------------------------------\n')
                f.write(f'iec for agent {agent_idx}:')
                f.write('state_to_store: {}.\n'.format(stored_slices))
                f.write('{}.\n'.format(min_count))
                # f.write('{}.\n'.format(self.curr_size)) 
                # f.write('{}.\n'.format(self.size)) 
    
    
    
    def build_goal_count(self, states):
        """
            pick valuable state to learn, store them into buffer of curriculum goal
        """
        # Preprocess states and remove meaningless ones
        states = states.reshape(-1, self.state_dim)
        states = states.numpy()
        
        non_zero_mask = np.any(states != 0, axis=1)
        states = states[non_zero_mask]
        
        
        # define the slices to project state
        index_list_slices = self.D_c
        
        hashed_state_slices = multiply_hash_array(states[:, index_list_slices], self.discrete_scale_count)
        state_to_store, stored_slices, min_count  = self.min_count_value_and_index(hashed_state_slices, states, self.arrival_ratio)
        
        with open(self.file_path, 'a') as f:
            f.write('-------------------------------\n')
            f.write('state_to_store: {}.\n'.format(stored_slices))
            f.write('{}.\n'.format(min_count))
            # f.write('{}.\n'.format(self.curr_size)) 
            # f.write('{}.\n'.format(self.size)) 
            
        for i in range(state_to_store.shape[0]):
            self.add_curr_goal(state_to_store[i])
        
        
        
        
    
    def build_Dcec(self, predefined = None):
        if predefined is None:
            self.D_c = find_common_elements(self.D_i)
            for agent_idx in range(self.n_agent):
                self.D_iec[agent_idx] = find_difference(self.D_i[agent_idx], self.D_c)
                self.D_iec[agent_idx] = self.D_iec[agent_idx] + [agent_idx]
        else:
            self.curr_ptr = (self.curr_ptr + 1) % self.max_size
            self.curr_size = min(self.curr_size + 1, self.max_size)
            self.D_c = predefined
            self.D_i = [find_union([l, self.D_c]) for l in self.D_i]
            for agent_idx in range(self.n_agent):
                self.D_iec[agent_idx] = find_difference(self.D_i[agent_idx], self.D_c)
                self.D_iec[agent_idx] = self.D_iec[agent_idx] + [agent_idx]
        self.not_build = False
        self.v_goal_shape = max([len(l) for l in self.D_i])
        self.c_goal_shape = len(self.D_c)
        self.vec_goal_shape = self.v_goal_shape - self.c_goal_shape
        D_i_s = [sorted(l) for l in self.D_i]
        # self.D_i_li = [sorted(l) for l in self.D_i_li]
        self.D_iec = [sorted(l) for l in self.D_iec]
        self.D_c = sorted(self.D_c)
        self.D_i = [self.D_c + l for l in self.D_iec] 
        print(D_i_s)
        print(self.D_i)
    
    def print_goal_info(self):
        print("D^i: ", self.D_i)
        print("common part: ", self.D_c)
        print("D^(i-c): ", self.D_iec)
    
    
    def global2indiv(self, global_state_batch, bs):
        indi_goals = th.zeros((bs, self.n_agent, self.reserved_indg_dim))
        for bs_idx in range(bs):
            for agent_idx in range(self.n_agent):
                if self.fac_ciec == 0.0:
                    v_indi_goals_i = global_state_batch[bs_idx, self.D_c]
                else:
                    v_indi_goals_i = global_state_batch[bs_idx, self.D_i[agent_idx]]
                indi_goals[bs_idx, agent_idx, :v_indi_goals_i.shape[0]] = v_indi_goals_i
        return indi_goals
        

    def global2indiv_np(self, global_state_batch, bs):
        indi_goals = np.zeros((bs, self.n_agent, self.reserved_indg_dim))
        for bs_idx in range(bs):
            for agent_idx in range(self.n_agent):
                if self.fac_ciec == 0.0:
                    v_indi_goals_i = global_state_batch[bs_idx, self.D_c]
                else:
                    v_indi_goals_i = global_state_batch[bs_idx, self.D_i[agent_idx]]
                indi_goals[bs_idx, agent_idx, :v_indi_goals_i.shape[0]] = v_indi_goals_i
        return indi_goals
        
    def sample_curr_goal(self, batch_size=1):
        ind = np.random.randint(0, self.curr_size, size=batch_size)
        return (
                th.FloatTensor(self.curr_goal_global[ind]).to(self.device)
        )
        
        
    def sample_goal(self, batch_size=1):
        """
        sample MDP goal state (win state)
        """
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
                th.FloatTensor(self.goal_global[ind]).to(self.device)
        )
        
    def add_curr_goal(self, state):
        self.curr_goal_global[self.curr_ptr] = state
        
        self.curr_ptr = (self.curr_ptr + 1) % self.max_size
        self.curr_size = min(self.curr_size + 1, self.max_size)
        
        
    def add_goal(self, state):
        self.goal_global[self.ptr] = state
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        
        
        # global_goals = self.obtain_global_goals(1)
        # indi_goals = self.global2indiv(global_goals, 1)
        # print(indi_goals)
        # exit(0)
        
    def empty_goals(self, batch_size):
        return th.zeros((batch_size, self.n_agent, self.reserved_indg_dim))
        
    
    
    def obtain_indi_goals(self, batch_size):
        global_goals = self.obtain_global_goals(batch_size)
        if len(self.D_i[0])==0:
            return self.empty_goals(batch_size)
        else:
            indi_goals = self.global2indiv(global_goals, batch_size)
            return indi_goals
        
        
    def obtain_global_goals(self, batch_size):
        if self.size > 0:
            return self.sample_goal(batch_size)
        elif self.curr_size > 0:
            return self.sample_curr_goal(batch_size)
        else:
            return th.zeros((batch_size, self.state_dim))
            
            
    def dual_dis(self, tensor1, tensor2):
        return (self.fac_hanming * self.hamming_distance(tensor1, tensor2) + self.euclidean_distance(tensor1, tensor2))
            
    def hamming_distance(self, tensor1, tensor2):
        # Ensure tensors have the same shape
        assert tensor1.shape == tensor2.shape, "Tensors must have the same shape when computing hamming_distance"
    
        diff_tensor = tensor1 - tensor2

        # Calculate Hamming distance
        distance = th.count_nonzero(diff_tensor, dim=-1)

        return distance

    def euclidean_distance(self, tensor1, tensor2):
        distance = th.norm(tensor1 - tensor2, dim=-1)
        return distance
        
    
    
    def min_count_value_and_index_iec(self, agent_idx, array, states, arrival_ratio=0.5):
        new_counter = Counter(map(tuple, array))
        for key, value in new_counter.items():
            if key in self.counter:
                self.iec_counter[agent_idx][key] += value
            else:
                self.iec_counter[agent_idx][key] = value
                
                index = [i for i in range(array.shape[0]) if np.array_equal(array[i], key)]
                self.iec_state[agent_idx][key] = states[index[0]]
                
        min_count = min(self.iec_counter[agent_idx].values())
        min_subarrays = [subarray for subarray, count in self.iec_counter[agent_idx].items() if count == min_count]
        
        return_state = np.zeros((len(min_subarrays), self.state_dim))
        for i in range(len(min_subarrays)):
            return_state[i] =  self.iec_state[agent_idx][min_subarrays[i]]
            
        return return_state, min_subarrays, min_count
    
    
    
    
    
    def min_count_value_and_index(self, array, states, arrival_ratio=0.5):
        new_counter = Counter(map(tuple, array))

        for key, value in new_counter.items():
            if key in self.counter:
                self.counter[key] += value
            else:
                self.counter[key] = value
                
                index = [i for i in range(array.shape[0]) if np.array_equal(array[i], key)]
                self.c_state[key] = states[index[0]]
                
        min_count = min(self.counter.values())
        min_subarrays = [subarray for subarray, count in self.counter.items() if count == min_count]
        
        return_state = np.zeros((len(min_subarrays), self.state_dim))
        for i in range(len(min_subarrays)):
            return_state[i] =  self.c_state[min_subarrays[i]]
            
        
        for k in min_subarrays:
            if k in self.counter:
                self.counter[k] += arrival_ratio
            else:
                raise KeyError(f"The key '{key}' does not exist in self.counter.")


        return return_state, min_subarrays, min_count
          
        