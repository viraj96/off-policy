import os
import wandb
import numpy as np
from itertools import chain
from tensorboardX import SummaryWriter
import torch

from offpolicy.utils.mlp_buffer import MlpReplayBuffer, PrioritizedMlpReplayBuffer
from offpolicy.utils.util import is_discrete, is_multidiscrete, DecayThenFlatSchedule


class MlpRunner(object):

    def __init__(self, config):
        # non-tunable hyperparameters are in args
        self.args = config["args"]
        self.device = config["device"]

        # set tunable hyperparameters
        self.share_policy = self.args.share_policy
        self.algorithm_name = self.args.algorithm_name
        self.env_name = self.args.env_name
        self.num_env_steps = self.args.num_env_steps
        self.use_wandb = self.args.use_wandb
        self.use_reward_normalization = self.args.use_reward_normalization
        self.use_per = self.args.use_per
        self.per_alpha = self.args.per_alpha
        self.per_beta_start = self.args.per_beta_start
        self.buffer_size = self.args.buffer_size
        self.batch_size = self.args.batch_size
        self.hidden_size = self.args.hidden_size
        self.max_grad_norm = self.args.max_grad_norm
        self.use_soft_update = self.args.use_soft_update
        self.hard_update_interval = self.args.hard_update_interval
        self.actor_train_interval_step = self.args.actor_train_interval_step
        self.train_interval = self.args.train_interval
        self.use_eval = self.args.use_eval
        self.eval_interval = self.args.eval_interval
        self.save_interval = self.args.save_interval
        self.log_interval = self.args.log_interval

        self.total_env_steps = 0  # total environment interactions collected during training
        self.num_episodes_collected = 0  # total episodes collected during training
        self.total_train_steps = 0  # number of gradient updates performed
        self.last_train_T = 0
        self.last_eval_T = 0  # last episode after which a eval run was conducted
        self.last_save_T = 0  # last epsiode after which the models were saved
        self.last_log_T = 0
        self.last_hard_update_T = 0

        if config.__contains__("take_turn"):
            self.take_turn = config["take_turn"]
        else:
            self.take_turn = False

        if config.__contains__("use_same_share_obs"):
            self.use_same_share_obs = config["use_same_share_obs"]
        else:
            self.use_same_share_obs = False

        if config.__contains__("use_cent_agent_obs"):
            self.use_cent_agent_obs = config["use_cent_agent_obs"]
        else:
            self.use_cent_agent_obs = False

        if config.__contains__("use_available_actions"):
            self.use_avail_acts = config["use_available_actions"]
        else:
            self.use_avail_acts = False

        self.episode_length = self.args.episode_length

        self.policy_info = config["policy_info"]
        self.policy_ids = sorted(list(self.policy_info.keys()))
        self.policy_mapping_fn = config["policy_mapping_fn"]

        self.num_agents = config["num_agents"]
        self.agent_ids = [i for i in range(self.num_agents)]

        self.env = config["env"]
        self.eval_env = config["eval_env"]
        self.num_envs = self.env.num_envs
        self.num_eval_envs = self.eval_env.num_envs

        if self.share_policy:
            self.collecter = self.shared_collect_rollout
        else:
            self.collecter = self.separated_collect_rollout

        self.train = self.batch_train
        self.logger = self.log_stats
        self.saver = self.save
        if self.use_wandb:
            self.save_dir = str(wandb.run.dir)
        else:
            self.run_dir = config["run_dir"]
            self.log_dir = str(self.run_dir / 'logs')
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            self.writter = SummaryWriter(self.log_dir)
            self.save_dir = str(self.run_dir / 'models')
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

        # initialize all the policies and organize the agents corresponding to each policy
        if self.algorithm_name == "matd3":
            from offpolicy.algorithms.matd3.algorithm.MATD3Policy import MATD3Policy as Policy
            from offpolicy.algorithms.matd3.matd3 import MATD3 as TrainAlgo
        elif self.algorithm_name == "maddpg":
            assert self.actor_train_interval_step == 1, (
                "maddpg only support actor_train_interval_step=1.")
            from offpolicy.algorithms.maddpg.algorithm.MADDPGPolicy import MADDPGPolicy as Policy
            from offpolicy.algorithms.maddpg.maddpg import MADDPG as TrainAlgo
        elif self.algorithm_name == "masac":
            assert self.actor_train_interval_step == 1, (
                "masac only support actor_train_interval_step=1.")
            from offpolicy.algorithms.masac.algorithm.MASACPolicy import MASACPolicy as Policy
            from offpolicy.algorithms.masac.masac import MASAC as TrainAlgo
        elif self.algorithm_name == "mqmix":
            from offpolicy.algorithms.mqmix.algorithm.mQMixPolicy import M_QMixPolicy as Policy
            from offpolicy.algorithms.mqmix.mqmix import M_QMix as TrainAlgo
            self.saver = self.save_mq
            self.train = self.batch_train_mq
            self.logger = self.log_stats_mq
        elif self.algorithm_name == "mvdn":
            from offpolicy.algorithms.mvdn.algorithm.mVDNPolicy import M_VDNPolicy as Policy
            from offpolicy.algorithms.mvdn.mvdn import M_VDN as TrainAlgo
            self.saver = self.save_mq
            self.train = self.batch_train_mq
            self.logger = self.log_stats_mq
        else:
            raise NotImplementedError

        self.policies = {p_id: Policy(
            config, self.policy_info[p_id]) for p_id in self.policy_ids}

        if self.args.model_dir is not None:
            self.restore(self.args.model_dir)
        self.log_clear()

        # initialize class for updating policies
        self.trainer = TrainAlgo(self.args, self.num_agents, self.policies, self.policy_mapping_fn,
                                 device=self.device)

        self.policy_agents = {policy_id: sorted(
            [agent_id for agent_id in self.agent_ids if self.policy_mapping_fn(agent_id) == policy_id]) for policy_id in
            self.policies.keys()}

        self.policy_obs_dim = {
            policy_id: self.policies[policy_id].obs_dim for policy_id in self.policy_ids}
        self.policy_act_dim = {
            policy_id: self.policies[policy_id].act_dim for policy_id in self.policy_ids}
        self.policy_central_obs_dim = {
            policy_id: self.policies[policy_id].central_obs_dim for policy_id in self.policy_ids}

        num_train_iters = self.num_env_steps / self.train_interval
        self.beta_anneal = DecayThenFlatSchedule(
            self.per_beta_start, 1.0, num_train_iters, decay="linear")

        if self.use_per:
            self.buffer = PrioritizedMlpReplayBuffer(self.per_alpha,
                                                     self.policy_info,
                                                     self.policy_agents,
                                                     self.buffer_size,
                                                     self.use_same_share_obs,
                                                     self.use_avail_acts,
                                                     self.use_reward_normalization)
        else:
            self.buffer = MlpReplayBuffer(self.policy_info,
                                          self.policy_agents,
                                          self.buffer_size,
                                          self.use_same_share_obs,
                                          self.use_avail_acts,
                                          self.use_reward_normalization)

        # fill replay buffer with random actions
        self.finish_first_train_reset = False
        num_warmup_episodes = max(
            (self.batch_size/self.episode_length, self.args.num_random_episodes))
        self.warmup(num_warmup_episodes)

    def run(self):
        # collect data
        self.trainer.prep_rollout()
        train_step_reward, train_metric = self.collecter(
            explore=True, training_episode=True, warmup=False)

        self.train_step_rewards.append(train_step_reward)
        self.train_metrics.append(train_metric)

        # save
        if (self.total_env_steps - self.last_save_T) / self.save_interval >= 1:
            self.saver()
            self.last_save_T = self.total_env_steps

        # log
        if ((self.total_env_steps - self.last_log_T) / self.log_interval) >= 1:
            self.log()
            self.last_log_T = self.total_env_steps

        # eval
        if self.use_eval and ((self.total_env_steps - self.last_eval_T) / self.eval_interval) >= 1:
            self.eval()
            self.last_eval_T = self.total_env_steps

        return self.total_env_steps

    def batch_train(self):
        self.trainer.prep_training()
        # do a gradient update if the number of episodes collected since the last training update exceeds the specified amount
        update_actor = ((self.total_train_steps %
                         self.actor_train_interval_step) == 0)
        # gradient updates
        self.train_stats = []
        for p_id in self.policy_ids:
            if self.use_per:
                beta = self.beta_anneal.eval(self.total_train_steps)
                sample = self.buffer.sample(self.batch_size, beta, p_id)
            else:
                sample = self.buffer.sample(self.batch_size)

            if self.use_same_share_obs:
                stats, new_priorities, idxes = self.trainer.shared_train_policy_on_batch(
                    p_id, sample, update_actor)
            else:
                stats, new_priorities, idxes = self.trainer.cent_train_policy_on_batch(
                    p_id, sample, update_actor)

            if self.use_per:
                self.buffer.update_priorities(idxes, new_priorities, p_id)

            self.train_stats.append(stats)

        if self.use_soft_update and update_actor:
            for pid in self.policy_ids:
                self.policies[pid].soft_target_updates()
        else:
            if ((self.total_env_steps - self.last_hard_update_T) / self.hard_update_interval) >= 1:
                for pid in self.policy_ids:
                    self.policies[pid].hard_target_updates()
                self.last_hard_update_T = self.total_env_steps

    def batch_train_mq(self):
        self.trainer.prep_training()
        # gradient updates
        self.train_stats = []
        for p_id in self.policy_ids:
            if self.use_per:
                beta = self.beta_anneal.eval(self.total_train_steps)
                sample = self.buffer.sample(self.batch_size, beta, p_id)
            else:
                sample = self.buffer.sample(self.batch_size)

            stats, new_priorities, idxes = self.trainer.train_policy_on_batch(
                sample, self.use_same_share_obs)

            if self.use_per:
                self.buffer.update_priorities(idxes, new_priorities, p_id)

            self.train_stats.append(stats)

        if self.use_soft_update:
            self.trainer.soft_target_updates()
        else:
            if (self.total_env_steps - self.last_hard_update_T) / self.hard_update_interval >= 1:
                self.trainer.hard_target_updates()
                self.last_hard_update_T = self.total_env_steps

    def log(self):
        print("\n Env {} Algo {} Exp {} runs total num timesteps {}/{}.\n"
              .format(self.args.scenario_name,
                      self.algorithm_name,
                      self.args.experiment_name,
                      self.total_env_steps,
                      self.num_env_steps))
        for p_id, train_stat in zip(self.policy_ids, self.train_stats):
            self.logger(p_id, train_stat, self.total_env_steps)

        average_step_reward = np.mean(self.train_step_rewards)

        average_episode_reward = average_step_reward * self.episode_length
        print("train average episode rewards is " +
                str(average_episode_reward))
        if self.use_wandb:
            wandb.log(
                {'train_average_episode_rewards': average_episode_reward}, step=self.total_env_steps)
        else:
            self.writter.add_scalars("train_average_episode_rewards", {
                                        'train_average_episode_rewards': average_episode_reward}, self.total_env_steps)

        self.log_env(self.train_metrics, suffix="train")
        self.log_clear()

    def log_env(self, metrics, suffix="train"):
        pass
 
    def log_clear(self):
        self.train_step_rewards = []
        self.train_metrics = []

    def save(self):
        for pid in self.policy_ids:
            policy_critic = self.policies[pid].critic
            critic_save_path = self.save_dir + '/' + str(pid)
            if not os.path.exists(critic_save_path):
                os.makedirs(critic_save_path)
            torch.save(policy_critic.state_dict(),
                       critic_save_path + '/critic.pt')

            policy_actor = self.policies[pid].actor
            actor_save_path = self.save_dir + '/' + str(pid)
            if not os.path.exists(actor_save_path):
                os.makedirs(actor_save_path)
            torch.save(policy_actor.state_dict(),
                       actor_save_path + '/actor.pt')

    def save_mq(self):
        for pid in self.policy_ids:
            policy_Q = self.policies[pid].q_network
            p_save_path = self.save_dir + '/' + str(pid)
            if not os.path.exists(p_save_path):
                os.makedirs(p_save_path)
            torch.save(policy_Q.state_dict(), p_save_path + '/q_network.pt')

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        torch.save(self.trainer.mixer.state_dict(),
                   self.save_dir + '/mixer.pt')

    def restore(self, checkpoint):
        for pid in self.policy_ids:
            path = checkpoint + str(pid)
            policy_critic_state_dict = torch.load(path + '/critic.pt')
            policy_actor_state_dict = torch.load(path + '/actor.pt')

            self.policies[pid].critic.load_state_dict(policy_critic_state_dict)
            self.policies[pid].actor.load_state_dict(policy_actor_state_dict)

    def warmup(self, num_warmup_episodes):
        # fill replay buffer with enough episodes to begin training
        self.trainer.prep_rollout()
        warmup_rewards = []
        print("warm up...")
        for _ in range(int(num_warmup_episodes // self.num_envs) + 1):
            reward, _ = self.collecter(
                explore=True, training_episode=False, warmup=True)
            warmup_rewards.append(reward)
        warmup_reward = np.mean(warmup_rewards)
        print("warmup average step rewards: ", warmup_reward)

    def eval(self):
        self.trainer.prep_rollout()

        eval_step_reward, eval_metric = self.collecter(
            explore=False, training_episode=False, warmup=False)
        average_episode_reward = eval_step_reward * self.episode_length
        print("eval average episode rewards is " +
                str(average_episode_reward))
        if self.use_wandb:
            wandb.log(
                {'eval_average_episode_rewards': average_episode_reward}, step=self.total_env_steps)
        else:
            self.writter.add_scalars('eval_average_episode_rewards', {
                                        'eval_average_episode_rewards': average_episode_reward}, self.total_env_steps)
        
        self.log_env(eval_metrics, suffix="eval")

    # for mpe-simple_spread and mpe-simple_reference
    def shared_collect_rollout(self, explore=True, training_episode=True, warmup=False):
        p_id = "policy_0"
        policy = self.policies[p_id]

        env = self.env if explore else self.eval_env
        n_rollout_threads = self.num_envs if explore else self.num_eval_envs

        if not explore:
            obs = env.reset()
            share_obs = obs.reshape(n_rollout_threads, -1)
        else:
            if self.finish_first_train_reset:
                obs = self.obs
                share_obs = self.share_obs
            else:
                obs = env.reset()
                share_obs = obs.reshape(n_rollout_threads, -1)
                self.finish_first_train_reset = True

        # init
        episode_rewards = []
        step_obs = {}
        step_share_obs = {}
        step_acts = {}
        step_rewards = {}
        step_next_obs = {}
        step_next_share_obs = {}
        step_dones = {}
        step_dones_env = {}
        step_avail_acts = {}
        step_next_avail_acts = {}

        for step in range(self.episode_length):
            obs_batch = np.concatenate(obs)
            # get actions for all agents to step the env
            if warmup:
                # completely random actions in pre-training warmup phase
                acts_batch = policy.get_random_actions(obs_batch)
            else:
                # get actions with exploration noise (eps-greedy/Gaussian)
                if self.algorithm_name == "masac":
                    acts_batch, _ = policy.get_actions(
                        obs_batch, sample=explore)
                else:
                    acts_batch, _ = policy.get_actions(obs_batch,
                                                       t_env=self.total_env_steps,
                                                       explore=explore,
                                                       use_target=False,
                                                       use_gumbel=False)

            if not isinstance(acts_batch, np.ndarray):
                acts_batch = acts_batch.detach().numpy()
            env_acts = np.split(acts_batch, n_rollout_threads)

            # env step and store the relevant episode information
            next_obs, rewards, dones, infos = env.step(env_acts)

            episode_rewards.append(rewards)
            dones_env = np.all(dones, axis=1)

            if explore and n_rollout_threads == 1 and np.all(dones_env):
                next_obs = env.reset()

            if not explore and np.all(dones_env):
                average_step_reward = np.mean(episode_rewards)
                return average_step_reward, None

            next_share_obs = next_obs.reshape(n_rollout_threads, -1)

            step_obs[p_id] = obs
            step_share_obs[p_id] = share_obs
            step_acts[p_id] = env_acts
            step_rewards[p_id] = rewards
            step_next_obs[p_id] = next_obs
            step_next_share_obs[p_id] = next_share_obs
            step_dones[p_id] = np.zeros_like(dones)
            step_dones_env[p_id] = dones_env
            step_avail_acts[p_id] = None
            step_next_avail_acts[p_id] = None

            obs = next_obs
            share_obs = next_share_obs

            if explore:
                self.obs = obs
                self.share_obs = share_obs
                # push all episodes collected in this rollout step to the buffer
                self.buffer.insert(n_rollout_threads,
                                   step_obs,
                                   step_share_obs,
                                   step_acts,
                                   step_rewards,
                                   step_next_obs,
                                   step_next_share_obs,
                                   step_dones,
                                   step_dones_env,
                                   step_avail_acts,
                                   step_next_avail_acts)

            # train
            if training_episode:
                self.total_env_steps += n_rollout_threads
                if (self.last_train_T == 0 or ((self.total_env_steps - self.last_train_T) / self.train_interval) >= 1):
                    self.train()
                    self.total_train_steps += 1
                    self.last_train_T = self.total_env_steps

        average_step_reward = np.mean(episode_rewards)

        return average_step_reward, None

    # for mpe-simple_speaker_listener
    def separated_collect_rollout(self, explore=True, training_episode=True, warmup=False):
        env = self.env if explore else self.eval_env
        n_rollout_threads = self.num_envs if explore else self.num_eval_envs

        if not explore:
            obs = env.reset()
            share_obs = []
            for o in obs:
                share_obs.append(list(chain(*o)))
            share_obs = np.array(share_obs)
        else:
            if self.finish_first_train_reset:
                obs = self.obs
                share_obs = self.share_obs
            else:
                obs = env.reset()
                share_obs = []
                for o in obs:
                    share_obs.append(list(chain(*o)))
                share_obs = np.array(share_obs)
                self.finish_first_train_reset = True

        agent_obs = []
        for agent_id in range(self.num_agents):
            env_obs = []
            for o in obs:
                env_obs.append(o[agent_id])
            env_obs = np.array(env_obs)
            agent_obs.append(env_obs)

        # [agents, parallel envs, dim]
        episode_rewards = []
        step_obs = {}
        step_share_obs = {}
        step_acts = {}
        step_rewards = {}
        step_next_obs = {}
        step_next_share_obs = {}
        step_dones = {}
        step_dones_env = {}
        step_avail_acts = {}
        step_next_avail_acts = {}

        acts = []
        for p_id in self.policy_ids:
            if is_multidiscrete(self.policy_info[p_id]['act_space']):
                self.sum_act_dim = int(np.sum(self.policy_act_dim[p_id]))
            else:
                self.sum_act_dim = self.policy_act_dim[p_id]
            temp_act = np.zeros((n_rollout_threads, self.sum_act_dim))
            acts.append(temp_act)

        for step in range(self.episode_length):
            for agent_id, p_id in zip(self.agent_ids, self.policy_ids):
                policy = self.policies[p_id]
                # get actions for all agents to step the env
                if warmup:
                    # completely random actions in pre-training warmup phase
                    # [parallel envs, agents, dim]
                    act = policy.get_random_actions(agent_obs[agent_id])
                else:
                    # get actions with exploration noise (eps-greedy/Gaussian)
                    if self.algorithm_name == "masac":
                        act, _ = policy.get_actions(
                            agent_obs[agent_id], sample=explore)
                    else:
                        act, _ = policy.get_actions(agent_obs[agent_id],
                                                    t_env=self.total_env_steps,
                                                    explore=explore,
                                                    use_target=False,
                                                    use_gumbel=False)

                if not isinstance(act, np.ndarray):
                    act = act.detach().numpy()
                acts[agent_id] = act

            env_acts = []
            for i in range(n_rollout_threads):
                env_act = []
                for agent_id in range(self.num_agents):
                    env_act.append(acts[agent_id][i])
                env_acts.append(env_act)

            # env step and store the relevant episode information
            next_obs, rewards, dones, infos = env.step(env_acts)

            episode_rewards.append(rewards)
            dones_env = np.all(dones, axis=1)

            if explore and n_rollout_threads == 1 and np.all(dones_env):
                next_obs = env.reset()

            if not explore and np.all(dones_env):
                average_step_reward = np.mean(episode_rewards)
                return average_step_reward, None

            next_share_obs = []
            for no in next_obs:
                next_share_obs.append(list(chain(*no)))
            next_share_obs = np.array(next_share_obs)

            next_agent_obs = []
            for agent_id in range(n_rollout_threads):
                next_env_obs = []
                for no in next_obs:
                    next_env_obs.append(no[agent_id])
                next_env_obs = np.array(next_env_obs)
                next_agent_obs.append(next_env_obs)

            for agent_id, p_id in zip(self.agent_ids, self.policy_ids):
                step_obs[p_id] = np.expand_dims(agent_obs[agent_id], axis=1)
                step_share_obs[p_id] = share_obs
                step_acts[p_id] = np.expand_dims(acts[agent_id], axis=1)
                step_rewards[p_id] = np.expand_dims(
                    rewards[:, agent_id], axis=1)
                step_next_obs[p_id] = np.expand_dims(
                    next_agent_obs[agent_id], axis=1)
                step_next_share_obs[p_id] = next_share_obs
                step_dones[p_id] = np.zeros_like(
                    np.expand_dims(dones[:, agent_id], axis=1))
                step_dones_env[p_id] = dones_env
                step_avail_acts[p_id] = None
                step_next_avail_acts[p_id] = None

            obs = next_obs
            agent_obs = next_agent_obs
            share_obs = next_share_obs

            if self.explore:
                self.obs = obs
                self.share_obs = share_obs
                self.buffer.insert(n_rollout_threads,
                                   step_obs,
                                   step_share_obs,
                                   step_acts,
                                   step_rewards,
                                   step_next_obs,
                                   step_next_share_obs,
                                   step_dones,
                                   step_dones_env,
                                   step_avail_acts,
                                   step_next_avail_acts)

            # train
            if training_episode:
                self.total_env_steps += n_rollout_threads
                if (self.last_train_T == 0 or ((self.total_env_steps - self.last_train_T) / self.train_interval) >= 1):
                    self.train()
                    self.total_train_steps += 1
                    self.last_train_T = self.total_env_steps

        average_step_reward = np.mean(episode_rewards)

        return average_step_reward, None

    def log_stats(self, policy_id, stats, t_env):
        # unpack the statistics
        critic_loss, actor_loss, alpha_loss, critic_grad_norm, actor_grad_norm, alpha, ent_diff = stats

        if self.use_wandb:
            # log into wandb
            wandb.log({str(policy_id) + '/critic_loss': critic_loss}, step=t_env)
            wandb.log(
                {str(policy_id) + '/critic_grad_norm': critic_grad_norm}, step=t_env)
            if actor_loss is not None:
                wandb.log(
                    {str(policy_id) + '/actor_loss': actor_loss}, step=t_env)
            if actor_grad_norm is not None:
                wandb.log(
                    {str(policy_id) + '/actor_grad_norm': actor_grad_norm}, step=t_env)
            if alpha_loss is not None:
                wandb.log(
                    {str(policy_id) + '/alpha_loss': alpha_loss}, step=t_env)
            if alpha is not None:
                wandb.log({str(policy_id) + '/alpha': alpha}, step=t_env)
            if ent_diff is not None:
                wandb.log({str(policy_id) + '/ent_diff': ent_diff}, step=t_env)
        else:
            # log into tensorboardX
            self.writter.add_scalars(str(
                policy_id) + '/critic_loss', {str(policy_id) + '/critic_loss': critic_loss}, t_env)
            self.writter.add_scalars(str(policy_id) + '/critic_grad_norm',
                                     {str(policy_id) + '/critic_grad_norm': critic_grad_norm}, t_env)
            if actor_loss is not None:
                self.writter.add_scalars(str(
                    policy_id) + '/actor_loss', {str(policy_id) + '/actor_loss': actor_loss}, t_env)
            if actor_grad_norm is not None:
                self.writter.add_scalars(str(policy_id) + '/actor_grad_norm',
                                         {str(policy_id) + '/actor_grad_norm': actor_grad_norm}, t_env)
            if alpha_loss is not None:
                self.writter.add_scalars(str(
                    policy_id) + '/alpha_loss', {str(policy_id) + '/alpha_loss': alpha_loss}, t_env)
            if alpha is not None:
                self.writter.add_scalars(
                    str(policy_id) + '/alpha', {str(policy_id) + '/alpha': alpha}, t_env)
            if ent_diff is not None:
                self.writter.add_scalars(
                    str(policy_id) + '/ent_diff', {str(policy_id) + '/ent_diff': ent_diff}, t_env)

    def log_stats_mq(self, policy_id, stats, t_env):
        # unpack the statistics
        loss, grad_norm, mean_Qs = stats
        if self.use_wandb:
            # log into wandb
            wandb.log({str(policy_id) + '/loss': loss}, step=t_env)
            wandb.log({str(policy_id) + '/grad_norm': grad_norm}, step=t_env)
            wandb.log({str(policy_id) + '/mean_q': mean_Qs}, step=t_env)
        else:
            # log into tensorboardX
            self.writter.add_scalars(
                str(policy_id) + '/loss', {str(policy_id) + '/loss': loss}, t_env)
            self.writter.add_scalars(
                str(policy_id) + '/grad_norm', {str(policy_id) + '/grad_norm': grad_norm}, t_env)
            self.writter.add_scalars(
                str(policy_id) + '/mean_q', {str(policy_id) + '/mean_q': mean_Qs}, t_env)