#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
import numpy as np

from network import *
from component import *
from .BaseAgent import *
from collections import deque


class MVPITD3Agent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.network.state_dict())
        self.replay = config.replay_fn()
        self.random_process = config.random_process_fn()
        self.total_steps = 0
        self.state = None
        self.online_rewards = deque(maxlen=int(1e4))
        self.history_dim = config.state_dim + config.action_dim
        if config.nn_model == 'lstm':
            self.history_reset(self.config)

    def soft_update(self, target, src):
        for target_param, param in zip(target.parameters(), src.parameters()):
            target_param.detach_()
            target_param.copy_(target_param * (1.0 - self.config.target_network_mix) +
                               param * self.config.target_network_mix)

    def eval_step(self, state, history=None):
        if history is None:
            self.config.state_normalizer.set_read_only()
            state = self.config.state_normalizer(state)
            action = self.network(state)
            self.config.state_normalizer.unset_read_only()

        else:
            self.config.state_normalizer.set_read_only()
            state = self.config.state_normalizer(state)
            action = self.network(state, history)
            self.config.state_normalizer.unset_read_only()
        return to_np(action)

    def step(self):
        config = self.config
        if self.state is None:
            self.random_process.reset_states()
            self.state = self.task.reset()
            self.state = config.state_normalizer(self.state)

        if self.total_steps < config.warm_up:
            action = [self.task.action_space.sample()]
        else:
            if config.nn_model == 'dnn':
                action = self.network(self.state)
            if config.nn_model == 'lstm':
                action = self.network(self.state, self.history)
            action = to_np(action)
            action += self.random_process.sample()
        action = np.clip(action, self.task.action_space.low, self.task.action_space.high)
        next_state, reward, done, info = self.task.step(action)
        next_state = self.config.state_normalizer(next_state)
        self.record_online_return(info)
        reward = self.config.reward_normalizer(reward)
        self.online_rewards.append(reward[0])

        if config.nn_model == 'dnn':
            experiences = list(zip(self.state, action, reward, next_state, done))
        if config.nn_model == 'lstm':
            self.history.append(np.hstack([self.state, action]).flatten())
            experiences = list(zip(self.state, action, reward, next_state, done, [list(self.history)]))
        self.replay.feed_batch(experiences)
        if done[0]:
            self.random_process.reset_states()
            self.history_reset(self.config)
        self.state = next_state
        self.total_steps += 1

        if self.replay.size() >= config.warm_up:
            y = np.mean(self.online_rewards)
            self.learn(config, y)

    def learn(self, config, y):
        experiences = self.replay.sample()

        if self.config.nn_model == 'dnn':
            states, actions, rewards, next_states, terminals = experiences
            states = tensor(states)
            actions = tensor(actions)
            rewards = tensor(rewards).unsqueeze(-1)
            rewards = rewards - config.lam * rewards.pow(2) + 2 * config.lam * rewards * y
            next_states = tensor(next_states)
            mask = tensor(1 - terminals).unsqueeze(-1)

            a_next = self.target_network(next_states)
            noise = torch.randn_like(a_next).mul(config.td3_noise)
            noise = noise.clamp(-config.td3_noise_clip, config.td3_noise_clip)

            min_a = float(self.task.action_space.low[0])
            max_a = float(self.task.action_space.high[0])
            a_next = (a_next + noise).clamp(min_a, max_a)

            q_1, q_2 = self.target_network.q(next_states, a_next)
            target = rewards + config.discount * mask * torch.min(q_1, q_2)
            target = target.detach()

            q_1, q_2 = self.network.q(states, actions)
            critic_loss = F.mse_loss(q_1, target) + F.mse_loss(q_2, target)

            self.network.zero_grad()
            critic_loss.backward()
            self.network.critic_opt.step()

            if self.total_steps % config.td3_delay:
                action = self.network(states)
                policy_loss = -self.network.q(states, action)[0].mean()  # only use q1

                self.network.zero_grad()
                policy_loss.backward()
                self.network.actor_opt.step()

                self.soft_update(self.target_network, self.network)

        if self.config.nn_model == 'lstm':
            states, actions, rewards, next_states, terminals, history = experiences
            states = tensor(states)
            actions = tensor(actions)
            rewards = tensor(rewards).unsqueeze(-1)
            rewards = rewards - config.lam * rewards.pow(2) + 2 * config.lam * rewards * y
            next_states = tensor(next_states)
            mask = tensor(1 - terminals).unsqueeze(-1)

            history = tensor(history)

            a_next = self.target_network(next_states, history)
            noise = torch.randn_like(a_next).mul(config.td3_noise)
            noise = noise.clamp(-config.td3_noise_clip, config.td3_noise_clip)

            min_a = float(self.task.action_space.low[0])
            max_a = float(self.task.action_space.high[0])
            a_next = (a_next + noise).clamp(min_a, max_a)

            q_1, q_2 = self.target_network.q(next_states, a_next, history)
            target = rewards + config.discount * mask * torch.min(q_1, q_2)
            target = target.detach()

            q_1, q_2 = self.network.q(states, actions, history)
            critic_loss = F.mse_loss(q_1, target) + F.mse_loss(q_2, target)

            self.network.zero_grad()
            critic_loss.backward()
            self.network.critic_opt.step()

            if self.total_steps % config.td3_delay:
                action = self.network(states, history)
                policy_loss = -self.network.q(states, action, history)[0].mean()  # only use q1

                self.network.zero_grad()
                policy_loss.backward()
                self.network.actor_opt.step()

                self.soft_update(self.target_network, self.network)
