import copy
from components.episode_buffer import EpisodeBatch
from components.goal_manager import GoalManager
from controllers.n_controller import NMAC
from components.action_selectors import categorical_entropy
from utils.rl_utils import build_gae_targets
import torch as th
from torch.optim import Adam
from utils.value_norm import ValueNorm

class PPOGLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.mac = mac
        self.logger = logger

        self.last_target_update_step = 0
        self.critic_training_steps = 0

        self.log_stats_t = -self.args.learner_log_interval - 1

        # a trick to reuse mac
        dummy_args = copy.deepcopy(args)
        dummy_args.n_actions = 1
        self.critic = NMAC(scheme, None, dummy_args)
        self.params = list(mac.parameters()) + list(self.critic.parameters())

        self.optimiser = Adam(params=self.params, lr=args.lr)
        self.last_lr = args.lr

        self.use_value_norm = getattr(self.args, "use_value_norm", False)
        if self.use_value_norm:
            self.value_norm = ValueNorm(1, device=self.args.device)
        
    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, goal_manager: GoalManager):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"][:, :-1]
        
        old_probs = batch["probs"][:, :-1]
        old_probs[avail_actions == 0] = 1e-10
        old_logprob = th.log(th.gather(old_probs, dim=3, index=actions)).detach()
        mask_agent = mask.unsqueeze(2).repeat(1, 1, self.n_agents, 1)
        
        # print('original rewards')
        # print(rewards)
        # print(rewards.shape) # shape: (batch_size, episode_length, 1)
        # exit(0)
        
        # get states from batch
        # shape: (batch_size, episode_length, state_size)
        states = batch["state"][:, :-1]
        
        
        
        # build individual current 
        with th.no_grad():
            # each agent copies the state
            # shape: (batch_size, episode_length, agent_num, state_size)
            copied_state = states.unsqueeze(2).repeat(1, 1, self.n_agents, 1)
            
            # get individual goal mapping function
            phi = goal_manager.oracle_phi()
            
            # individual current state embedding on the goal space
            # shape: (batch_size, episode_length, agent_num, indi_goal_size)
            ind_current = th.zeros(copied_state.shape[0], copied_state.shape[1], copied_state.shape[2], 3)
            
            # individual goal conditioned rewards
            # shape: (batch_size, episode_length, agent_num, 1)
            ind_rewards = th.zeros(copied_state.shape[0], copied_state.shape[1], copied_state.shape[2], 1)
            ind_dist = th.zeros(copied_state.shape[0], copied_state.shape[1], copied_state.shape[2], 1)
            
            ind_goals = goal_manager.oracle_goal()  # n_agent * goal_size
            ind_goals = ind_goals.unsqueeze(0).repeat(copied_state.shape[1], 1, 1)  #  episode_length * n_agent * goal_size
            ind_goals = ind_goals.unsqueeze(0).repeat(copied_state.shape[0], 1, 1, 1) # (batch_size, episode_length, agent_num, indi_goal_size)
            
            # phi提供individual的切片，通过切片来获取individual goal
            for agent_i in range(self.n_agents):
               ind_current[:, :, agent_i] = states[:, :, phi[agent_i]]
               
            # ind_current和ind_goals都是(batch_size, episode_length, agent_num, indi_goal_size)的张量
            # 计算两张量最后一维的距离
            ind_dist = goal_manager.dual_dis(ind_current, ind_goals)
            # 取上张量的第0时间步的距离作为初始距离
            total_dist = ind_dist[:, 0, :].unsqueeze(1).repeat(1, copied_state.shape[1]-1, 1)
            
            # 沿着时间步一维求距离变化（后一个减去前一个），并取负
            ind_dist_diff = th.diff(ind_dist, dim = 1)
            ind_dist_diff = th.neg(ind_dist_diff)
            # ind_dist_diff_div = th.div(ind_dist_diff, total_dist)
            
            # 因为上述diff的计算过程中时间步一维丢失了一个长度，这里用环境reward补上
            reward_final = rewards[:, -1, :].unsqueeze(1).repeat(1, 1, self.n_agents)
            
            # 补完之后连接起来
            ind_rewards = th.cat((ind_dist_diff, reward_final), dim=1).unsqueeze(-1)
            
            # print(ind_dist_diff_div)
            # print(ind_dist_diff_div.shape)
            # exit(0)
            
            # for agent_i in range(self.n_agents):
            #     ind_current[:, :, agent_i] = states[:, :, phi[agent_i]]
            #     # print(ind_current[:, 0, agent_i].shape) 64,3
            #     # print(ind_goals[:, 0, agent_i].shape)  64,3
            #     # exit(0)
                
            #     initial_dis = goal_manager.dual_dis(ind_current[:, 0, agent_i], ind_goals[:, 0, agent_i])
                
            #     ind_dist[:, :, agent_i] = goal_manager.dual_dis(ind_current[:, :, agent_i], ind_goals[:, :, agent_i])
            #     print(initial_dis)
            #     print(ind_dist[:, :, agent_i])
            #     exit(0)
            #     ind_curr_diff_i = th.diff(ind_current)
            #     ind_rewards[:, :, agent_i] = goal_manager.negative_eucl_dis(states[:, :, phi[agent_i]], goal_manager.oracle_goal)
            # ind_current =  
            # print('state')
            # print(states.shape)
            # print('copied_state')
            # print(copied_state.shape)
            # print('ind_current')
            # print(ind_current.shape)
            # print('ind_rewards')
            # print(ind_rewards.shape)
            # exit(0)
        
        # targets and advantages
        with th.no_grad():
            old_values = []
            self.critic.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                agent_outs = self.critic.forward(batch, t=t)
                old_values.append(agent_outs)
            old_values = th.stack(old_values, dim=1) 

            if self.use_value_norm:
                value_shape = old_values.shape
                values = self.value_norm.denormalize(old_values.view(-1)).view(value_shape)

            advantages, targets = build_gae_targets(ind_rewards, 
                    mask_agent, values, self.args.gamma, self.args.gae_lambda)

            if self.use_value_norm:
                targets_shape = targets.shape
                targets = targets.reshape(-1)
                self.value_norm.update(targets)
                targets = self.value_norm.normalize(targets).view(targets_shape)
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)
        
        # PPO Loss
        for _ in range(self.args.mini_epochs):
            # Critic
            values = []
            self.critic.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length-1):
                agent_outs = self.critic.forward(batch, t=t)
                values.append(agent_outs)
            values = th.stack(values, dim=1) 

            # value clip
            values_clipped = old_values[:,:-1] + (values - old_values[:,:-1]).clamp(-self.args.eps_clip,
                                                                                self.args.eps_clip)

            # 0-out the targets that came from padded data
            td_error = th.max((values - targets.detach())** 2, (values_clipped - targets.detach())** 2)
            masked_td_error = td_error * mask_agent
            critic_loss = 0.5 * masked_td_error.sum() / mask_agent.sum()

            # Actor
            pi = []
            self.mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length-1):
                agent_outs = self.mac.forward(batch, t=t)
                pi.append(agent_outs)
            pi = th.stack(pi, dim=1)  # Concat over time

            pi[avail_actions == 0] = 1e-10
            pi_taken = th.gather(pi, dim=3, index=actions)
            log_pi_taken = th.log(pi_taken)
            
            ratios = th.exp(log_pi_taken - old_logprob)
            surr1 = ratios * advantages
            surr2 = th.clamp(ratios, 1-self.args.eps_clip, 1+self.args.eps_clip) * advantages
            actor_loss = -(th.min(surr1, surr2) * mask_agent).sum() / mask_agent.sum()
            
            # entropy
            entropy_loss = categorical_entropy(pi).mean(-1, keepdim=True) # mean over agents
            entropy_loss[mask == 0] = 0 # fill nan
            entropy_loss = (entropy_loss * mask).sum() / mask.sum()
            loss = actor_loss + self.args.critic_coef * critic_loss - self.args.entropy * entropy_loss / entropy_loss.item()

            # Optimise agents
            self.optimiser.zero_grad()
            loss.backward()
            grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
            self.optimiser.step()


        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            mask_elems = mask_agent.sum().item()
            self.logger.log_stat("advantage_mean", (advantages * mask_agent).sum().item() / mask_elems, t_env)
            self.logger.log_stat("actor_loss", actor_loss.item(), t_env)
            self.logger.log_stat("entropy_loss", entropy_loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            self.logger.log_stat("lr", self.last_lr, t_env)
            self.logger.log_stat("critic_loss", critic_loss.item(), t_env)
            self.logger.log_stat("target_mean", (targets * mask_agent).sum().item() / mask_elems, t_env)
            self.log_stats_t = t_env


    def cuda(self):
        self.mac.cuda()
        self.critic.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.optimiser.state_dict(), "{}/agent_opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.optimiser.load_state_dict(th.load("{}/agent_opt.th".format(path), map_location=lambda storage, loc: storage))
