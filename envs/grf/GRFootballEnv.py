import gym
import numpy as np
from envs.multiagentenv import MultiAgentEnv

try:
    import gfootball.env as football_env
except ImportError as e:
    raise e(
        "Please install Google football evironment before use: https://github.com/google-research/football"
    ) from None
    


class GRFootballEnv(MultiAgentEnv):
    def __init__(self, map_name, seed=None):
        if map_name == "academy_pass_and_shoot_with_keeper":
            num_controlled_lagents = 2
            num_controlled_ragents = 0
            self.n_agents = 2
            self.episode_limit = 400
            self.action_num = 19
        elif map_name == "academy_run_pass_and_shoot_with_keeper":
            num_controlled_lagents = 2
            num_controlled_ragents = 0
            self.n_agents = 2
            self.episode_limit = 400
            self.action_num = 19
        elif map_name == "academy_3_vs_1_with_keeper":
            num_controlled_lagents = 3
            num_controlled_ragents = 0
            self.n_agents = 3
            self.episode_limit = 400
            self.action_num = 19
        # elif map_name == "academy_corner":
        #     num_controlled_lagents = 2
        #     num_controlled_ragents = 0
        #     self.n_agents = 2
        #     self.episode_limit = 400
        #     self.action_num = 19
        elif map_name == "academy_counterattack_easy":
            num_controlled_lagents = 4
            num_controlled_ragents = 0
            self.n_agents = 4
            self.episode_limit = 400
            self.action_num = 19
        elif map_name == "academy_counterattack_hard":
            num_controlled_lagents = 4
            num_controlled_ragents = 0
            self.n_agents = 4
            self.episode_limit = 400
            self.action_num = 19
        else:
            raise ValueError("map_name {} not accepted".format(map_name))
            
        self.env = football_env.create_environment(env_name=map_name,
                                                   stacked=False,
                                                   representation='simple115',
                                                   rewards='scoring',
                                                   write_goal_dumps=False,
                                                   write_full_episode_dumps=False,
                                                   render=False,
                                                   dump_frequency=0,
                                                   logdir='/tmp/test',
                                                   extra_players=None,
                                                   number_of_left_players_agent_controls=num_controlled_lagents,
                                                   number_of_right_players_agent_controls=num_controlled_ragents,
                                                   channel_dimensions=(3, 3))
        self.n_agents = num_controlled_lagents + num_controlled_ragents
        self.episode_limit = 400
        # 'action_space', 'class_name', 'close', 'compute_reward', 'env', 'get_state', 'metadata', 'observation_space', 'render', 'reset', 'reward_range', 'seed', 'set_state', 'spec', 'step', 'unwrapped']
        # print(dir(self.env))
        # print(self.env.game_duration)
        
        self.num_controlled_agents = num_controlled_lagents + num_controlled_ragents
        if self.num_controlled_agents > 1:
            action_space = gym.spaces.Discrete(self.env.action_space.nvec[1])
        else:
            action_space = self.env.action_space
            
        if self.num_controlled_agents > 1:
            observation_space = gym.spaces.Box(
                low=self.env.observation_space.low[0],
                high=self.env.observation_space.high[0],
                dtype=self.env.observation_space.dtype)
        else:
            observation_space = gym.spaces.Box(
                low=self.env.observation_space.low,
                high=self.env.observation_space.high,
                dtype=self.env.observation_space.dtype)
                
        self.action_space = action_space
        self.observation_space = observation_space 
        self.observation_size  = len(self.env.observation_space.high[0])
        
        self.cur_obs = None
        self.t = 0
        self.battles_won = False
        
        self.battles_won = 0
        self.battles_game = 0
        self.timeouts = 0
        self.force_restarts = 0
        
        
        
    def reset(self):
        self.stat = dict()
        obs = self.env.reset()
        self.t = 0
        self.battles_won = False
        if self.num_controlled_agents == 1:
            obs = obs.reshape(1, -1)
        self.cur_obs = obs
        return obs
        
    def step(self, actions):
        o, r, d, i = self.env.step(actions)
        if self.num_controlled_agents == 1:
            o = o.reshape(1, -1)
            r = r.reshape(1, -1)
        obs = o
        self.cur_obs = obs
        infos = i
        reward = infos['score_reward']
        done = d  # [False]
        
        if reward > 0.1:
            self.battles_won = True
        
        self.t = self.t + 1
        if self.t >= self.episode_limit: 
            done = True
            # self.battles_game += 1
            
        
        
        # self.stat['success'] = infos['score_reward']
        
        return reward, done, infos


       
    def get_state(self):
        state = np.array(self.cur_obs[0])
        state = state.flatten()
        state = state.astype(dtype=np.float32)
        return state

    
    def get_state_size(self):
        # state is obs?
        return self.observation_size
    
    
    def get_obs_size(self):
        return self.observation_size
    
    
    def get_total_actions(self):
        return self.action_num
        
        
    def get_avail_actions(self):
        """Returns the available actions of all agents in a list."""
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = [1 for _ in range(self.action_num)]
            avail_actions.append(avail_agent)
        return avail_actions
        
    def get_obs(self):
        """ Returns all agent observations in a list """
        return self.cur_obs
        
        
    # def get_stats(self):
    #     stats = {
    #         "battles_won": self.battles_won,
    #         "battles_game": self.battles_game,
    #         "battles_draw": self.timeouts,
    #         "win_rate": self.battles_won / self.battles_game,
    #         "timeouts": self.timeouts,
    #         "restarts": self.force_restarts,
    #     }
    #     return stats