#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
from .normalizer import *
import argparse
import torch


class Config(object):
    DEVICE = torch.device('cuda:0')

    def __init__(self, gpu_id=None):
        if gpu_id is not None:
            self.DEVICE = torch.device('cuda:'+str(gpu_id))
        self.parser = argparse.ArgumentParser()
        self.task_fn = None
        self.optimizer_fn = None
        self.actor_optimizer_fn = None
        self.critic_optimizer_fn = None
        self.network_fn = None
        self.actor_network_fn = None
        self.critic_network_fn = None
        self.replay_fn = None
        self.random_process_fn = None
        self.target_network_update_freq = None
        self.exploration_steps = None
        self.log_level = 0
        self.history_length = None
        self.double_q = False
        self.tag = 'vanilla'
        self.num_workers = 1
        self.gradient_clip = None
        self.entropy_weight = 0
        self.use_gae = False
        self.gae_tau = 1.0
        self.target_network_mix = 0.001
        self.state_normalizer = RescaleNormalizer()
        self.reward_normalizer = RescaleNormalizer()
        self.min_memory_size = None
        self.max_steps = 0
        self.rollout_length = None
        self.value_loss_weight = 1.0
        self.iteration_log_interval = 30
        self.categorical_v_min = None
        self.categorical_v_max = None
        self.categorical_n_atoms = 51
        self.num_quantiles = None
        self.optimization_epochs = 4
        self.mini_batch_size = 64
        self.termination_regularizer = 0
        self.sgd_update_frequency = None
        self.random_action_prob = None
        self.__eval_env = None
        self.log_interval = int(1e3)
        self.save_interval = 9000
        self.eval_interval = 0
        self.eval_episodes = 50
        self.async_actor = True
        self.EOT_eval = 0
        self.gpu_id = 0

        self.strike_price = None
        self.rl_method = None
        self.data_type = None
        self.hedging_task = None
        self.asset_model = None
        self.option_type = None

        self.rate_of_return = 0.05
        self.volatility = 0.2
        self.year = 250
        self.maturity = 60
        self.trading_freq_per_day = 5
        self.tau = self.maturity / self.year
        self.dt = 1 / (self.year * self.trading_freq_per_day)
        self.identical_path = False
        self.random_parameter = False
        self.interest = 0.0
        self.tick_size = 0.05
        self.discount = 0.99

    @property
    def eval_env(self):
        return self.__eval_env

    @eval_env.setter
    def eval_env(self, env):
        self.__eval_env = env
        self.state_dim = env.state_dim
        self.action_dim = env.action_dim
        self.task_name = env.name

    def add_argument(self, *args, **kwargs):
        self.parser.add_argument(*args, **kwargs)

    def merge(self, config_dict=None):
        if config_dict is None:
            args = self.parser.parse_args()
            config_dict = args.__dict__
        for key in config_dict.keys():
            setattr(self, key, config_dict[key])
