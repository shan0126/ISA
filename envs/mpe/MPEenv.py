import gym
import numpy as np
import copy
# from envs.multiagentenv import MultiAgentEnv
from envs.multiagentenv import MultiAgentEnv

coordinates2 = [
    (6, 15), (4, 3), (9, 2), (9, 5), (3, 19),
    (19, 9), (4, 18), (9, 8), (11, 2), (8, 18),
    (1, 6), (8, 15), (1, 3), (10, 12), (15, 5),
    (18, 7), (16, 13), (7, 19), (12, 6), (12, 12),
    (18, 8), (3, 6), (3, 18), (8, 2), (8, 5),
    (11, 7), (19, 14), (13, 4), (11, 13), (7, 9),
    (18, 9), (14, 2), (3, 2), (4, 1), (12, 8),
    (14, 11), (17, 1), (3, 8), (4, 13), (9, 3),
    (19, 7), (8, 10), (5, 17), (9, 15), (19, 13),
    (13, 9), (7, 11), (15, 18), (7, 8), (18, 14)
]

coordinates3 = [
    (12, 5), (12, 14), (13, 5), (6, 1), (11, 13), (12, 8), (2, 6), (11, 6), 
    (2, 8), (10, 13), (9, 11), (3, 4), (3, 3), (5, 3), (10, 2), (9, 4), 
    (1, 4), (5, 8), (4, 1), (7, 1), (14, 2), (6, 3), (2, 12), (7, 11), 
    (13, 8), (3, 8), (1, 14), (14, 3), (3, 2), (4, 10), (6, 7), (8, 10), 
    (9, 5), (13, 3), (10, 4), (4, 13), (6, 4), (8, 13), (5, 14), (5, 1), 
    (9, 13), (2, 5), (1, 9), (12, 11), (8, 12), (8, 7), (4, 11), (4, 4), 
    (5, 5), (14, 4)
]

coordinates1 = [
    (8, 7), (4, 9), (3, 7), (5, 4), (4, 6), (5, 1), (0, 2), (8, 9), (1, 0), (1, 6),
    (0, 8), (1, 3), (1, 9), (2, 8), (6, 2), (7, 1), (6, 5), (4, 2), (4, 5), (3, 3),
    (3, 9), (4, 8), (3, 6), (5, 3), (8, 2), (9, 1), (0, 1), (8, 8), (1, 2), (2, 1),
    (6, 1), (6, 4), (7, 9), (6, 7), (4, 7), (3, 5), (3, 8), (9, 3), (9, 9), (0, 0),
    (0, 9), (1, 4), (0, 6), (2, 3), (2, 9), (2, 6), (7, 2), (6, 0), (7, 5), (7, 8)
]

positions = [(5, 7), (3, 7), (7, 7), (1, 7), (9, 7)]



class MPEenv(MultiAgentEnv):
    def __init__(self, map_name, n_agents, grid_size=20, H=60, seed=None):
        self.n_agents = n_agents
        self.agents = [(0, 0) for _ in range(self.n_agents)] 
        self.n_actions = 5
        self.grid_size = grid_size
        self.episode_limit = H
        self.step_count = 0
        self.done = False
        self.map_name = map_name
        
        
        if self.map_name == "navigation":
            self.landmarks = coordinates3[:self.n_agents]
            # remark the landmarks that are not occupied
            self.left_occupy = copy.deepcopy(self.landmarks)
            # remark which landmark is occupied by which agent
            self.occupy_map = [-1 for _ in range(self.n_agents)]
            
            # observation: self.x, self.y, {landmark.x, landmark.y}*N
            self.observation_space = gym.spaces.MultiDiscrete([self.grid_size for _ in range((self.n_agents + 1) * 2)])
            self.obs_size = (self.n_agents + 1) * 2
            
            # state: {agent.x, agent.y}*N, {landmark.x, landmark.y}*N
            self.state_space = gym.spaces.MultiDiscrete([self.grid_size for _ in range((self.n_agents + self.n_agents) * 2)])
            self.state_size = (self.n_agents + self.n_agents) * 2
            # action: N
            self.action_space = gym.spaces.MultiDiscrete([self.n_actions] * self.n_agents)
        elif self.map_name == "unlock":
            self.locks = coordinates3[:self.n_agents]
            # remark if each lock is unlocked, lock: -1, unlock: 1
            self.is_unlock = [-1 for _ in range(self.n_agents)]
            # observation: self.x, self.y, lock.x, lock.y
            self.observation_space = gym.spaces.MultiDiscrete([self.grid_size for _ in range(4)])
            self.obs_size = 4
            # state: {agent.x, agent.y}*N, {lock.x, lock.y}*N
            self.state_space = gym.spaces.MultiDiscrete([self.grid_size for _ in range((self.n_agents + self.n_agents) * 2)])
            self.state_size = (self.n_agents + self.n_agents) * 2
            # action: N
            self.action_space = gym.spaces.MultiDiscrete([self.n_actions] * self.n_agents)
        elif self.map_name == "shooting":
            self.episode_limit = H
            self.grid_size = 10
            self.agents = [(1, 1) for _ in range(self.n_agents)]
            self.n_actions = 6
            self.positions = positions[:self.n_agents]
            
            # remark the landmarks that are not occupied
            self.left_occupy = copy.deepcopy(self.positions)
            # remark which landmark is occupied by which agent
            self.occupy_map = [-1 for _ in range(self.n_agents)]
            
            self.num_bullets = 0
            # observation: self.x, self.y, bullets
            self.observation_space = gym.spaces.MultiDiscrete([self.grid_size for _ in range(2)] + [100])
            self.obs_size = 3
            # observation: {self.x, self.y} * N, bullets
            self.state_space = gym.spaces.MultiDiscrete([self.grid_size for _ in range(self.n_agents * 2)] + [100])
            self.state_size = self.n_agents * 2 + 1
            # action: N
            self.action_space = gym.spaces.MultiDiscrete([self.n_actions] * self.n_agents)
        else:
            raise NotImplementedError
            
    def reset(self):
        self.agents = [(0, 0) for _ in range(self.n_agents)] 
        self.step_count = 0
        self.done = False
        if self.map_name == "navigation":
            self.left_occupy = copy.deepcopy(self.landmarks)
            self.occupy_map = [-1 for _ in range(self.n_agents)]
        elif self.map_name == "unlock":
            self.is_unlock = [-1 for _ in range(self.n_agents)]
        elif self.map_name == "shooting":
            self.agents = [(1, 1) for _ in range(self.n_agents)]
            self.num_bullets = 0
            self.left_occupy = copy.deepcopy(self.positions)
            self.occupy_map = [-1 for _ in range(self.n_agents)]
        return self.get_obs(), self.get_state()
        
        
    def step(self, actions):
        assert not self.done, "error: Trying to call step() after an episode is done"
        # perform actions
        # reward = [0.0 for _ in range(self.n_agents)]
        reward = 0
        for agent_id, agent in enumerate(self.agents):
            x, y = agent[0], agent[1]
            # if map_name == "navigation":
            #     if agent_id in self.occupied:
            #         action = 0
            #     else:
                    
            # elif self.map_name == "unlock":
            action = actions[agent_id]
            avail_action = self.get_avail_agent_actions(agent_id)
            if avail_action[action] == 0:
                action = 0
            if action == 1 and y > 0:
                self.agents[agent_id] = (self.agents[agent_id][0], self.agents[agent_id][1]-1)
                # self.agents[agent_id][1] -= 1
            elif action == 2 and y < self.grid_size - 1:
                self.agents[agent_id] = (self.agents[agent_id][0], self.agents[agent_id][1]+1)
                # self.agents[agent_id][1] += 1
            elif action == 3 and x > 0:
                # self.agents[agent_id][0] -= 1
                self.agents[agent_id] = (self.agents[agent_id][0]-1, self.agents[agent_id][1])
            elif action == 4 and x < self.grid_size - 1:
                # self.agents[agent_id][0] += 1
                self.agents[agent_id] = (self.agents[agent_id][0]+1, self.agents[agent_id][1])
            elif action == 0:
                self.agents[agent_id] = (self.agents[agent_id][0], self.agents[agent_id][1])
            elif action == 5 and self.map_name == "shooting":
                self.num_bullets += 1
            else:
                raise NotImplementedError
            
            if self.map_name == "navigation":
                if self.agents[agent_id] in self.left_occupy:
                    current_occu_index = self.landmarks.index(self.agents[agent_id])
                    self.occupy_map[current_occu_index] = agent_id
                    self.left_occupy.remove(self.agents[agent_id])
                    # reward[agent_id] = 1.0
                    reward = reward + 1
            elif self.map_name == "unlock":
                if self.agents[agent_id][0] == self.locks[agent_id][0] and self.agents[agent_id][1] == self.locks[agent_id][1]:
                    if self.is_unlock[agent_id] == -1:
                        self.is_unlock[agent_id] = 1
                        # reward[agent_id] = 1.0
                        reward = reward + 1
            elif self.map_name == "shooting":
                if self.agents[agent_id] in self.left_occupy:
                    current_occu_index = self.positions.index(self.agents[agent_id])
                    self.occupy_map[current_occu_index] = agent_id
                    self.left_occupy.remove(self.agents[agent_id])
            # elif self.map_name == "unlock":
            
        
        self.step_count += 1
        success = self._success()
        self.done = True if self.step_count == self.episode_limit or success else False
        info = {"battle_won": False}
        if success:
            info["battle_won"] = True
            # bonus for win
            # reward = [r + 1.0 for r in reward]
        
        return reward, self.done, info
        
    
    def _success(self):
        if self.map_name == "navigation":
            if self.left_occupy == []:
                return True
            else:
                return False
        elif self.map_name == "unlock":
            return all(x == 1 for x in self.is_unlock)
        elif self.map_name == "shooting":
            # pos = all(agent[0]==self.positions[agent_id][0] and agent[1]==self.positions[agent_id][1] for agent_id, agent in enumerate(self.agents))
            if self.num_bullets > 50 and self.left_occupy == []:
                return True
            else:
                return False
        else:
            raise NotImplementedError
        
        
    
    def get_obs(self):
        group_obs = []
        if self.map_name == "navigation":
            for agent_id, agent in enumerate(self.agents):
                obs = []
                obs.extend(list(agent))
                for landmark in self.landmarks:
                    obs.extend(list(landmark))
                group_obs.append(obs)
        elif self.map_name == "unlock":
            for agent_id, agent in enumerate(self.agents):
                obs = []
                obs.extend(list(agent))
                obs.extend(list(self.locks[agent_id]))
                group_obs.append(obs)
        elif self.map_name == "shooting":
            for agent_id, agent in enumerate(self.agents):
                obs = []
                obs.extend(list(agent))
                obs.extend([self.num_bullets])
                group_obs.append(obs)
        else:
            raise NotImplementedError
        return np.array(group_obs)
    
    
    
    def get_state(self):
        state = []
        if self.map_name == "navigation":
            for agent in self.agents:
                state.extend(list(agent))
            for landmark in self.landmarks:
                state.extend(list(landmark))
        elif self.map_name == "unlock":
            for agent in self.agents:
                state.extend(list(agent))
            for lock in self.locks:
                state.extend(list(lock))
        elif self.map_name == "shooting":
            for agent in self.agents:
                state.extend(list(agent))
            state.extend([self.num_bullets])
        else:
            raise NotImplementedError
        return np.array(state)
        
        
        
    def get_avail_agent_actions(self, agent_id):
        avail_actions = [0] * self.n_actions
        avail_actions[0] = 1
        if self.map_name == "navigation":
            if agent_id not in self.occupy_map:
                # see if we can move
                x, y = self.agents[agent_id][0], self.agents[agent_id][1]
                if y > 0: avail_actions[1] = 1
                if y < self.grid_size - 1: avail_actions[2] = 1
                if x > 0: avail_actions[3] = 1
                if x < self.grid_size - 1: avail_actions[4] = 1
        elif self.map_name == "unlock":
            if not (self.is_unlock[agent_id] == 1):
                x, y = self.agents[agent_id][0], self.agents[agent_id][1]
                if y > 0: avail_actions[1] = 1
                if y < self.grid_size - 1: avail_actions[2] = 1
                if x > 0: avail_actions[3] = 1
                if x < self.grid_size - 1: avail_actions[4] = 1
        elif self.map_name == "shooting":
            x, y = self.agents[agent_id][0], self.agents[agent_id][1]
            if agent_id not in self.occupy_map:
                if y > 0: avail_actions[1] = 1
                if y < self.grid_size - 1: avail_actions[2] = 1
                if x > 0: avail_actions[3] = 1
                if x < self.grid_size - 1: avail_actions[4] = 1
            if y > 5: avail_actions[5] = 1
        else:
            raise NotImplementedError
        return avail_actions
        
        
        
    def get_avail_actions(self):
        """Returns the available actions of all agents in a list."""
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions
        
        
    def get_state_size(self):
        # state is obs?
        return self.state_size
    
    
    def get_obs_size(self):
        return self.obs_size
    
    
    def get_total_actions(self):
        return self.n_actions
        
    def close(self):
        pass
            
                
            
            
            
            
            
            
            
            
            
            
            
            
            