import numpy as np
import torch as th

from utils.goal_utils import find_union, find_common_elements, find_difference

from utils.goal_utils import get_d4actions_yanchi

class MutIListener(object):
    def __init__(self, args):
        self.args = args
        self.init_availact = None
        self.fix_actions = None
        
        self.action2count = None
        self.need_exp4mi = None
        self.is_listenning = False
        
        self.state_buffer_win = [[None for _ in range(self.args.n_agents)] for _ in range(args.max_window)]
        self.action_buffer_win = [[None for _ in range(self.args.n_agents)] for _ in range(args.max_window)]
        
        self.D_i = [[] for _ in range(self.args.n_agents)]           # self.D_i[i] denote agent i's index set to project a state into its goal 
        self.D_i_li = [[] for _ in range(self.args.n_agents)]        # self.D_i_li[i][j] denote which elements of state the j-th action of agent i can affect
        self.D_c = []                                          # self.D_c denote the common part of state that all agents have affects on
        self.D_iec = [[] for _ in range(self.args.n_agents)]         # self.D_iec denote the special part, where (self.D_c[i] = self.D_i[i] - self.D_c)
        
        self.v_goal_shape = None                               # TRUE size of individual goal (instead of the size reserved via hyper-parameter)
        self.c_goal_shape = None                               # size of common part
        self.vec_goal_shape = None                             # size of special part
        
        self.listen_part_state = {"state": [], "actions": []}
        
        # document for debug
        self.file_path = 'listener_{}.txt'.format(self.args.seed)
        open(self.file_path, 'w').close()
        
    def start_listenning(self):
        self.is_listenning = True
        self.need_exp4mi = np.ones(self.args.n_agents)
        self.action2count = np.ones((self.args.n_agents, self.args.n_actions)) * 100000
        
    def suggest_action2take(self, ava_action):
        self.explore_agent = -1
        for agent_idx in range(self.args.n_agents):
            if self.need_exp4mi[agent_idx] == 1:
                self.explore_agent = agent_idx
                break
        
        ava_action_curagent = ava_action[:, self.explore_agent, :]   
        assert not(th.all(ava_action_curagent == 0).item()), "cannot suggest action because no action is avaliable"
        ava_action_curagent = ava_action_curagent.numpy()
        
        action2take = self.fix_actions
        
        
        for i in range(self.args.batch_size_run):
            ava_action_curagent_repinf = np.where(ava_action_curagent == 0, -np.inf, ava_action_curagent)

            ava_count = np.multiply(ava_action_curagent_repinf[i], self.action2count[self.explore_agent])
                    
            valid_indices = np.where(ava_count != -np.inf)[0]
                    
            if len(valid_indices) == 0: 
                a = 0
            else:
                a = np.random.choice(valid_indices)

            action2take[i, self.explore_agent] = a
                    
            self.action2count[self.explore_agent, a] = self.action2count[self.explore_agent, a] - 1
            
            for j in range(self.args.n_agents):
                if ava_action[i, j, action2take[i, j]].item()==0:
                    if th.all(ava_action[i,j,:] == 0).item():
                        action2take[i, j] = 0
                    else:
                        non_zero_th = th.nonzero(ava_action[i,j,:]).reshape(-1)
                        action2take[i, j] = non_zero_th[0].item()
                        
        return action2take
        
        
    def insert_expdata(self, episode_batch):
        max_ep_t = episode_batch.max_t_filled()
        episode_batch = episode_batch[:, :max_ep_t]
        # res_state = episode_batch["state"].reshape(-1, episode_batch["state"].shape[-1])
        # res_action = episode_batch["actions"].reshape(-1, episode_batch["actions"].shape[-1])
        
        if self.state_buffer[self.explore_agent] is None:
            self.state_buffer[self.explore_agent] = episode_batch["state"]
            self.action_buffer[self.explore_agent] = episode_batch["actions"]
        else:
            self.state_buffer[self.explore_agent] = th.cat((self.state_buffer[self.explore_agent], episode_batch["state"]), dim=1)
            self.action_buffer[self.explore_agent] = th.cat((self.action_buffer[self.explore_agent], episode_batch["actions"]), dim=1)
            

        
        
    def insert_expdata(self, episode_batch):
        # obtain all state and actions in episode_batch
        max_ep_t = episode_batch.max_t_filled()
        episode_batch = episode_batch[:, :max_ep_t]
        state = episode_batch['state']
        actions = episode_batch['actions']
        
        # clean the state where all element is 0
        state_list = []
        action_list = []
        for ba_id in range(state.shape[0]):
            state_ep = state[ba_id]
            actions_ep = actions[ba_id]
            zero_rows = (state_ep.sum(dim=1) == 0).nonzero(as_tuple=True)[0]
            state_ep_cleaned = state_ep[~(state_ep.sum(dim=1) == 0)]
            actions_ep_cleaned = actions_ep[~(state_ep.sum(dim=1) == 0)]
            # calculate state change with different window and the caused actions
            for w in range(1, self.args.max_window+1):
                # try:
                #     state_ep_diff = state_ep_cleaned[w:] - state_ep_cleaned[:-w] 
                #     print(f"window: {w}; shapes: a: {state_ep_cleaned[w:].shape}; b: {state_ep_cleaned[:-w]}")
                # except RuntimeError:
                #     print(f"window: {w}; shapes: a: {state_ep_cleaned[w:].shape}; b: {state_ep_cleaned[:-w]}")
                
                state_ep_diff = state_ep_cleaned[w:] - state_ep_cleaned[:-w] 
                actions_ep_eff = actions_ep_cleaned[:-w]
                if self.state_buffer_win[w-1][self.explore_agent] is None:
                    self.state_buffer_win[w-1][self.explore_agent] = state_ep_diff
                    self.action_buffer_win[w-1][self.explore_agent] = actions_ep_eff
                else:
                    self.state_buffer_win[w-1][self.explore_agent] = th.cat((self.state_buffer_win[w-1][self.explore_agent], state_ep_diff), dim=0)
                    self.action_buffer_win[w-1][self.explore_agent] = th.cat((self.action_buffer_win[w-1][self.explore_agent], actions_ep_eff), dim=0)
    
    
    
        

                
        
    
    def check_progress(self):
        is_finish_all = False
        if self.state_buffer_win[0][self.explore_agent].shape[0] > self.args.N_trans_effect:
            self.build_Di_win()
            self.need_exp4mi[self.explore_agent] = 0
        
        if self.need_exp4mi[-1]==0:
            self.build_Dcec()
            is_finish_all = True
            return is_finish_all
        else:
            return is_finish_all
            
            
            
    def build_Dcec(self, predefined = None):
        if predefined is None:
            self.D_c = find_common_elements(self.D_i)
            for agent_idx in range(self.args.n_agents):
                self.D_iec[agent_idx] = find_difference(self.D_i[agent_idx], self.D_c)
                # self.D_iec[agent_idx] = self.D_iec[agent_idx] + [agent_idx]
        self.v_goal_shape = max([len(l) for l in self.D_i])
        self.c_goal_shape = len(self.D_c)
        self.vec_goal_shape = self.v_goal_shape - self.c_goal_shape
        D_i_s = [sorted(l) for l in self.D_i]
        # self.D_i_li = [sorted(l) for l in self.D_i_li]
        self.D_iec = [sorted(l) for l in self.D_iec]
        self.D_c = sorted(self.D_c)
        self.D_i = [self.D_c + l for l in self.D_iec] 
        # self.print_goal_info()
            
    
    
    def build_Di_win(self):
        # build di across windows
        D_i_li_list = []
        for w in range(self.args.max_window):
            state = self.state_buffer_win[w][self.explore_agent]
            actions = self.action_buffer_win[w][self.explore_agent]
            D_i_li, v_i_li = get_d4actions_yanchi(state, 
                                               actions, 
                                               self.args.threshold_phi, 
                                               agent_id=self.explore_agent, 
                                               num_classes=self.args.n_actions,
                                               scaler=self.args.scaler)
            print(f"window {w+1} for agent {self.explore_agent}, action influence scope is {D_i_li}")
            D_i_li_list.append(D_i_li)
            
        self.D_i_li[self.explore_agent] = [find_union([D_i_li_list[j][i] for j in range(len(D_i_li_list))]) for i in range(len(D_i_li_list[0]))]
        print(f"action influence scope for agent {self.explore_agent} is {self.D_i_li[self.explore_agent]}")
        self.D_i[self.explore_agent] = find_union(self.D_i_li[self.explore_agent])
        print("build D for agent {}: {}".format(self.explore_agent, self.D_i[self.explore_agent]))
        with open(self.file_path, 'a') as f:
            f.write("build D for agent {}: {}".format(self.explore_agent, self.D_i[self.explore_agent]))
            
            

        
    def print_goal_info(self, remark_spring=""):
        with open(self.file_path, 'a') as f:
            f.write("D^i: {}".format(self.D_i))
            f.write("each i: {}".format(self.D_i_li))
            f.write("common part: {}".format(self.D_c))
            f.write("D^(i-c): {}".format(self.D_iec))
            f.write("remark_spring: {}".format(remark_spring))
            
    
    def end_listenning(self):
        self.is_listenning = False
        self.action2count = None
        self.need_exp4mi = None
        
    def _suggest_fix_actions(self):
        if not (self.init_availact.shape[0]==self.args.n_agents and self.init_availact.shape[1]==self.args.n_actions):
            print("the shape of available is wrong")
            exit(0)
        np_availact = self.init_availact.numpy()
        actions = []
        for agent_idx in range(self.args.n_agents):
            ava_index = np.where(np_availact[agent_idx]==1)[0]
            actions.append(np.random.choice(ava_index))
        actions = np.array(actions)
        self.fix_actions = np.tile(actions, (self.args.batch_size_run, 1))

        

            



        
