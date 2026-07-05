from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
from multiprocessing import Pipe, Process
import numpy as np
import torch as th
import os, wandb, csv
import math

from utils.goal_utils import multiply_hash_array


# Based (very) heavily on SubprocVecEnv from OpenAI Baselines
class ParallelRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run

        # Make subprocesses for the envs
        self.parent_conns, self.worker_conns = zip(*[Pipe() for _ in range(self.batch_size)])
        env_fn = env_REGISTRY[self.args.env]
        self.ps = []
        for i, worker_conn in enumerate(self.worker_conns):
            ps = Process(target=env_worker,
                         args=(worker_conn, CloudpickleWrapper(partial(env_fn, **self.args.env_args))))
            self.ps.append(ps)

        for p in self.ps:
            p.daemon = True
            p.start()

        self.parent_conns[0].send(("get_env_info", None))
        self.env_info = self.parent_conns[0].recv()
        self.episode_limit = self.env_info["episode_limit"]

        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        self.log_train_stats_t = -100000
        map_name = self.args.env_args['map_name']
        seed = self.args.env_args['seed']
        self.csv_dir = f'./csv_files/{map_name}/{args.mixer}/{args.label}/'
        self.csv_path = f'{self.csv_dir}seed_{seed}_{args.label}.csv'
        if not os.path.exists(self.csv_dir):
            os.makedirs(self.csv_dir)
        if args.wandb:
            job_type = args.label
            wandb_name = f'seed_{args.seed}_{args.label}'
            # wandb.login(key="", relogin=True)
            wandb.init(project=args.project_name, name=wandb_name, group=map_name, job_type=job_type, config=args)

    def setup(self, scheme, groups, preprocess, mac, goal_manager):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac
        self.scheme = scheme
        self.groups = groups
        self.preprocess = preprocess
        self.goal_manager = goal_manager

    def get_env_info(self):
        return self.env_info

    def save_replay(self):
        pass

    def close_env(self):
        for parent_conn in self.parent_conns:
            parent_conn.send(("close", None))

    def reset(self, goals):
        self.batch = self.new_batch()

        # Reset the envs
        for parent_conn in self.parent_conns:
            parent_conn.send(("reset", None))

        pre_transition_data = {
            "state": [],
            "avail_actions": [],
            "obs": [],
            "goals": [],
            "curr_emb": []
        }
        # Get the obs, state and avail_actions back
        for idx, parent_conn in enumerate(self.parent_conns):
            data = parent_conn.recv()
            pre_transition_data["state"].append(data["state"])
            pre_transition_data["avail_actions"].append(data["avail_actions"])
            pre_transition_data["obs"].append(data["obs"])
            pre_transition_data["goals"].append(goals[idx].numpy())
            ce = self.goal_manager.global2indiv_np(np.expand_dims(data["state"], axis=0), 1)
            pre_transition_data["curr_emb"].append(ce)

        self.batch.update(pre_transition_data, ts=0)

        self.t = 0
        self.env_steps_this_run = 0

    def run(self, goals, test_mode=False, mutlistener=None, require_epsilon=False):
        self.reset(goals)
        
        if mutlistener.init_availact is None:
            ava_action = self.batch["avail_actions"][:, self.t]
            mutlistener.init_availact = ava_action[0]
            
        EXPLORE = False
        fixed_action = True
        if self.goal_manager.not_build:
            EXPLORE = True
            if fixed_action:
                mutlistener._suggest_fix_actions()
                action2take = mutlistener.fix_actions

        all_terminated = False
        episode_returns = [0] * self.batch_size
        episode_lengths = [0] * self.batch_size
        self.mac.init_hidden(batch_size=self.batch_size)
        terminated = [False] * self.batch_size
        envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
        final_env_infos = []  # may store extra stats like battle won. this is filled in ORDER OF TERMINATION
        
        # 创建列表，用于储存一个批量的state_goal, indivi_goal
        envs_terminated_state = [[] for _ in range(self.batch_size)]
        envs_terminated_obs = [[] for _ in range(self.batch_size)]
        envs_terminated_win = [False] * self.batch_size
        # 此三个列表值通过index一一对应，在run.py中向goal buffer插入目标时，将胜利的index插入goal，将没胜利的index插入curr_goal

        save_probs = getattr(self.args, "save_probs", False)
        while True:
            # print(save_probs) # True


            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch for each un-terminated env
            if save_probs:
                actions, probs = self.mac.select_actions(self.batch, 
                                                         goals, 
                                                         t_ep=self.t, 
                                                         t_env=self.t_env,
                                                         bs=envs_not_terminated, 
                                                         test_mode=test_mode, 
                                                         require_epsilon=require_epsilon)
            else:
                actions = self.mac.select_actions(self.batch, 
                                                  goals, 
                                                  t_ep=self.t, 
                                                  t_env=self.t_env, 
                                                  bs=envs_not_terminated, 
                                                  test_mode=test_mode, 
                                                  require_epsilon=require_epsilon)
                                                  

            if EXPLORE and fixed_action:
                ava_action = self.batch["avail_actions"][:, self.t]
                
                action2take = mutlistener.suggest_action2take(ava_action)
                
                action2take_t = th.tensor(action2take)
                # print(action2take)
                acs = [action2take_t[i].reshape(1, -1) for i in envs_not_terminated]
                # print(acs)
                action2take_t = th.cat(acs, dim = 0)
                # print(action2take)
                # exit(0)
                
                assert action2take_t.shape==actions.shape, "action2take: {}, actions: {}, envs_not_terminated: {}".format(action2take_t.shape, actions.shape, envs_not_terminated)
                actions = th.tensor(action2take_t)
                # actions = action2take_t.clone().detach()

            cpu_actions = actions.to("cpu").numpy()

            # Update the actions taken
            actions_chosen = {
                "actions": actions.unsqueeze(1).to("cpu"),
            }
            if save_probs:
                actions_chosen["probs"] = probs.unsqueeze(1).to("cpu")

            self.batch.update(actions_chosen, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            # Send actions to each env
            action_idx = 0
            for idx, parent_conn in enumerate(self.parent_conns):
                if idx in envs_not_terminated:  # We produced actions for this env
                    if not terminated[idx]:  # Only send the actions to the env if it hasn't terminated
                        parent_conn.send(("step", cpu_actions[action_idx]))
                    action_idx += 1  # actions is not a list over every env

            # Update envs_not_terminated
            envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
            all_terminated = all(terminated)
            if all_terminated:
                break

            # Post step data we will insert for the current timestep
            post_transition_data = {
                "reward": [],
                "terminated": [],
                "i_reward": []
            }
            # Data for the next step we will insert in order to select an action
            pre_transition_data = {
                "state": [],
                "avail_actions": [],
                # "extrinsic_state": [],
                # "visible_matrix": [],
                "obs": [],
                "goals": [],
                "curr_emb": []
            }

            # Receive data back for each unterminated env
            for idx, parent_conn in enumerate(self.parent_conns):
                if not terminated[idx]:
                    data = parent_conn.recv()

                    

                    if isinstance(data["reward"], list):  # navigation and unlock
                        post_transition_data["reward"].append((sum(data["reward"]) / len(data["reward"]),))
                        episode_returns[idx] += sum(data["reward"]) / len(data["reward"])
                    else: # others
                        # Remaining data for this current timestep
                        post_transition_data["reward"].append((data["reward"],))
                        episode_returns[idx] += data["reward"]
                        
                    episode_lengths[idx] += 1
                    if not test_mode:
                        self.env_steps_this_run += 1

                    env_terminated = False
                    if data["terminated"]:
                        final_env_infos.append(data["info"])
                    if data["terminated"] and not data["info"].get("episode_limit", False):
                        env_terminated = True
                    terminated[idx] = data["terminated"]
                    post_transition_data["terminated"].append((env_terminated,))
                    

                    if data["terminated"]:
                        envs_terminated_state[idx] = data["state"]
                        envs_terminated_obs[idx] = data["obs"]
                        if 'battle_won' in data["info"].keys():
                            envs_terminated_win[idx] = data["info"]["battle_won"]
                        # else:
                            # raise RuntimeError('Env dont have battle_won')
                            # sometimes smac donnot have 'battle_won' in data, check if win here.
                            # if self.args.env_args['map_name'] == '3m':
                            #     preg = [39, 42, 45] 
                            # elif self.args.env_args['map_name'] == '8m':
                            #     preg = [144, 147, 150, 153, 156, 159, 162, 165]
                            # elif self.args.env_args['map_name'] == '2s_vs_1sc':
                            #     preg = [24]
                            # else:
                            #     raise RuntimeError('Need battle_won key in info in env to check if win')
                            # is_win = np.sum(data["state"][preg]) == 0
                            # envs_terminated_win[idx] = True if is_win else False
                        
                        

                    ce = self.goal_manager.global2indiv_np(np.expand_dims(data["state"], axis=0), 1)
                    
                    
                    if self.t==0:
                        reward_i_sys = self.compute_gcrf(pre_goal_emb=self.batch["curr_emb"][idx, self.t],
                                                         curr_goal_emb=ce[0], 
                                                         ep_goals=goals.numpy()[idx], 
                                                         curr_ac=self.batch["actions"][idx, self.t], 
                                                         goal_manager=self.goal_manager, 
                                                         env_r = data["reward"]) # all to be zero
                    else:
                        if require_epsilon and (not len(self.goal_manager.counter) == 0):
                            reward_i_sys = self.compute_countrf(curr_state=np.expand_dims(data["state"], axis=0),
                                                                curr_ac=self.batch["actions"][idx, self.t],
                                                                goal_manager=self.goal_manager, 
                                                                env_r = data["reward"])
                            
                        else:
                            reward_i_sys = self.compute_gcrf(pre_goal_emb=self.batch["curr_emb"][idx, self.t],
                                                             curr_goal_emb=ce[0], 
                                                             ep_goals=goals.numpy()[idx], 
                                                             curr_ac=self.batch["actions"][idx, self.t],
                                                             goal_manager=self.goal_manager, env_r = data["reward"])
                        

                    post_transition_data["i_reward"].append(reward_i_sys)
                    pre_transition_data["state"].append(data["state"])
                    pre_transition_data["avail_actions"].append(data["avail_actions"])
                    pre_transition_data["obs"].append(data["obs"])
                    pre_transition_data["goals"].append(goals[idx].numpy())
                    
                    pre_transition_data["curr_emb"].append(ce)

            # Add post_transiton data into the batch
            self.batch.update(post_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            # Move onto the next timestep
            self.t += 1

            # Add the pre-transition data
            self.batch.update(pre_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=True)

        if not test_mode:
            self.t_env += self.env_steps_this_run


        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        infos = [cur_stats] + final_env_infos

        cur_stats.update({k: sum(d.get(k, 0) for d in infos) for k in set.union(*[set(d) for d in infos])})
        cur_stats["n_episodes"] = self.batch_size + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = sum(episode_lengths) + cur_stats.get("ep_length", 0)

        cur_returns.extend(episode_returns)
        
        
        
        
        
        # put goal info together
        goal_data = {
            "envs_terminated_state": envs_terminated_state,
            "envs_terminated_obs": envs_terminated_obs,
            "envs_terminated_win": envs_terminated_win
        }

        n_test_runs = max(1, self.args.test_nepisode // self.batch_size) * self.batch_size
        if test_mode and (len(self.test_returns) == n_test_runs):
            if 'battle_won' in cur_stats.keys():
                win_rate = cur_stats['battle_won'] / cur_stats['n_episodes']
            else:
                raise RuntimeError('Env dont have battle_won')
                win_rate = ((np.array(cur_returns) > 0).astype('int')).sum() / cur_stats['n_episodes']
            reward = np.mean(cur_returns)
            mean_steps = cur_stats["ep_length"] / cur_stats['n_episodes']
            self.writereward(self.csv_path, reward, win_rate, mean_steps, self.t_env)
            if self.args.wandb:
                wandb.log({'step': self.t_env, 'Test_win_rate': win_rate, log_prefix + "return_mean": reward,
                           log_prefix + "return_std": np.std(cur_returns), 'mean_steps:': mean_steps})
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        
            
            
        
        
        return self.batch, goal_data, None

    def writereward(self, path, reward, win_rate, mean_steps, step):
        if os.path.isfile(path):
            with open(path, 'a+') as f:
                csv_write = csv.writer(f)
                csv_write.writerow([step, reward, win_rate, mean_steps])
        else:
            with open(path, 'w') as f:
                csv_write = csv.writer(f)
                csv_write.writerow(['step', 'reward', 'win_rate', 'mean_steps'])
                csv_write.writerow([step, reward, win_rate, mean_steps])
                
    
    def compute_countrf(self, curr_state, curr_ac, goal_manager, env_r):
        reward_i_sys = [0.0 for _ in range(self.args.n_agents)]
        reward_i_sys = np.array(reward_i_sys)
        if isinstance(env_r, list):
            reward_ep = env_r 
        else:
            reward_ep = [env_r for _ in range(self.args.n_agents)]
        reward_ep = np.array(reward_ep)
        
        if self.args.env_args['map_name'] == "3m" or self.args.env_args['map_name'] == "8m" or self.args.env_args['map_name'] == "2s_vs_1sc":
            hashed_state_slices = multiply_hash_array(curr_state[:, goal_manager.D_c], goal_manager.discrete_scale_count)
            state_repr = tuple(hashed_state_slices[0])
            if state_repr in goal_manager.counter:
                count_value = goal_manager.counter[state_repr]
            else:
                count_value = 0.25
            
            reward_seg = 1.0/math.sqrt(count_value)
            
            for agent_idx in range(self.args.n_agents):
                action_id = curr_ac[agent_idx]
                set1 = set(goal_manager.D_i_li[agent_idx][action_id])
                set2 = set(goal_manager.D_c)
                if not set1.intersection(set2):
                    reward_i_sys[agent_idx] = 0.0 + self.args.fac_mdpr * reward_ep[agent_idx]
                else:
                    reward_i_sys[agent_idx] = reward_seg + self.args.fac_mdpr * reward_ep[agent_idx]
        elif self.args.env_args['map_name'] == "shooting" or self.args.env_args['map_name'] == "navigation" or self.args.env_args['map_name'] == "unlock":
            hashed_state_slices = multiply_hash_array(curr_state[:, goal_manager.D_c], goal_manager.discrete_scale_count)
            state_repr = tuple(hashed_state_slices[0])
            if state_repr in goal_manager.counter:
                count_value = goal_manager.counter[state_repr]
            else:
                count_value = 0.25
            
            reward_com = 1.0/math.sqrt(count_value)
            
            for agent_idx in range(self.args.n_agents):
                # calculate the special part
                hashed_state_slices = multiply_hash_array(curr_state[:, goal_manager.D_iec[agent_idx]], goal_manager.discrete_scale_count)
                state_repr = tuple(hashed_state_slices[0])
                if state_repr in goal_manager.counter:
                    count_value = goal_manager.counter[state_repr] 
                else:
                    count_value = 0.25
            
                reward_spe = 1.0/math.sqrt(count_value)
            
                action_id = curr_ac[agent_idx]
                set1 = set(goal_manager.D_i_li[agent_idx][action_id])
                set2 = set(goal_manager.D_c)
                if not set1.intersection(set2):
                    reward_i_sys[agent_idx] = self.args.beta *reward_spe + self.args.fac_mdpr * reward_ep[agent_idx]
                else:
                    reward_i_sys[agent_idx] = reward_com + self.args.beta * reward_spe + + self.args.fac_mdpr * reward_ep[agent_idx]
            
                    
        return reward_i_sys
    
    
    
    
    def compute_gcrf(self, pre_goal_emb, curr_goal_emb, ep_goals, curr_ac, goal_manager, env_r):
        reward_i_sys = [0.0 for _ in range(self.args.n_agents)]
        reward_i_sys = np.array(reward_i_sys)
        if isinstance(env_r, list):
            reward_ep = env_r 
        else:
            reward_ep = [env_r for _ in range(self.args.n_agents)]
        reward_ep = np.array(reward_ep)
        # get common part
        common_part_len = len(goal_manager.D_c)
        pre_common_goal = pre_goal_emb[:, :common_part_len]
        curr_common_goal = curr_goal_emb[:, :common_part_len]
        ep_common_goal = ep_goals[:, :common_part_len]
        
        # get special part
        pre_spec_goal = pre_goal_emb[:, common_part_len:]
        curr_spec_goal = curr_goal_emb[:, common_part_len:]
        ep_spec_goal = ep_goals[:, common_part_len:]
        
        
        
        if len(goal_manager.D_c)>0 or len(goal_manager.D_iec[0])>0:
            t_curr_common_goal = th.tensor(curr_common_goal, dtype=th.float32)
            t_ep_common_goal = th.tensor(ep_common_goal, dtype=th.float32)
            t_pre_common_goal = th.tensor(pre_common_goal, dtype=th.float32)
            curr_dis_sys = goal_manager.dual_dis(t_curr_common_goal, t_ep_common_goal)
            post_dis_sys = goal_manager.dual_dis(t_pre_common_goal, t_ep_common_goal)
            delta_sys_c = post_dis_sys - curr_dis_sys
            
            # compute distance between special parts
            t_curr_spec_goal = th.tensor(curr_spec_goal, dtype=th.float32)
            t_ep_spec_goal = th.tensor(ep_spec_goal, dtype=th.float32)
            t_pre_spec_goal = th.tensor(pre_spec_goal, dtype=th.float32)
            curr_dis_sys_s = goal_manager.dual_dis(t_curr_spec_goal, t_ep_spec_goal)
            post_dis_sys_s = goal_manager.dual_dis(t_pre_spec_goal, t_ep_spec_goal)
            delta_sys_s = post_dis_sys_s - curr_dis_sys_s
            
            delta_sys_c = delta_sys_c.numpy()
            delta_sys_s = delta_sys_s.numpy()
            
            delta_sys = delta_sys_c + delta_sys_s * self.args.fac_ciec


            reward_i_sys = delta_sys  + self.args.fac_mdpr * reward_ep

            
            for agent_idx in range(self.args.n_agents):
                action_id = curr_ac[agent_idx]
                set1 = set(goal_manager.D_i_li[agent_idx][action_id])
                set2 = set(goal_manager.D_c)
                if not set1.intersection(set2):
                    reward_i_sys[agent_idx] = reward_i_sys[agent_idx] - delta_sys_c[agent_idx]

            
        return reward_i_sys
            

    def _log(self, returns, stats, prefix):

        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean", v / stats["n_episodes"], self.t_env)
        stats.clear()


def env_worker(remote, env_fn):
    # Make environment
    env = env_fn.x()
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            actions = data
            # Take a step in the environment
            reward, terminated, env_info = env.step(actions)
            # Return the observations, avail_actions and state to make the next action
            state = env.get_state()
            avail_actions = env.get_avail_actions()
            obs = env.get_obs()
            remote.send({
                "state": state,
                "avail_actions": avail_actions,
                "obs": obs,
                "reward": reward,
                "terminated": terminated,
                "info": env_info
            })
        elif cmd == "reset":
            env.reset()
            remote.send({
                "state": env.get_state(),
                "avail_actions": env.get_avail_actions(),
                "obs": env.get_obs()
            })
        elif cmd == "close":
            env.close()
            remote.close()
            break
        elif cmd == "get_env_info":
            env_info = env.get_env_info()
            remote.send(env_info)
        elif cmd == "get_stats":
            remote.send(env.get_stats())
        else:
            raise NotImplementedError


class CloudpickleWrapper():
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)
