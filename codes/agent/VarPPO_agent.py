#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from network import *
from component import *
from .BaseAgent import *


class VarPPOAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.network = config.network_fn()
        self.actor_opt = config.actor_opt_fn(self.network.actor_params)
        self.critic_opt = config.critic_opt_fn(self.network.critic_params)
        self.total_steps = 0
        self.states = self.task.reset()
        self.states = config.state_normalizer(self.states)

    def eval_step(self, state, is_eval=False):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        prediction = self.network(state, is_eval=is_eval)
        self.config.state_normalizer.unset_read_only()
        return to_np(prediction['a'])
# total dataset is load to holder
#
    def step(self):
        config = self.config
        storage = Storage(config.rollout_length)
        states = self.states
        for _ in range(config.rollout_length):
            prediction = self.network(states)
            next_states, rewards, terminals, info = self.task.step(to_np(prediction['a']))
            self.record_online_return(info)
            rewards = config.reward_normalizer(rewards)
            next_states = config.state_normalizer(next_states)
            storage.add(prediction)
            storage.add({'r_original': tensor(rewards).unsqueeze(-1),
                         'm': tensor(1 - terminals).unsqueeze(-1),
                         's': tensor(states),
                         })
            states = next_states
            self.total_steps += config.num_workers

        self.states = states
        prediction = self.network(states)
        storage.add(prediction)
        storage.placeholder()

        rewards = list(storage.cat(['r_original']))[0]
        y = rewards.mean()

        for i in range(config.rollout_length):
            r_original = storage.r_original[i]
            storage.r[i] = r_original - config.lam * r_original ** 2 + 2 * config.lam * r_original * y

        advantages = tensor(np.zeros((config.num_workers, 1)))
        returns = prediction['v'].detach()
        for i in reversed(range(config.rollout_length)):
            returns = storage.r[i] + config.discount * storage.m[i] * returns
            if not config.use_gae:
                advantages = returns - storage.v[i].detach()
            else:
                td_error = storage.r[i] + config.discount * storage.m[i] * storage.v[i + 1] - storage.v[i]
                advantages = advantages * config.gae_tau * config.discount * storage.m[i] + td_error
            storage.adv[i] = advantages.detach()
            storage.ret[i] = returns.detach()

        states, actions, log_probs_old, returns, advantages, rewards = storage.cat(['s', 'a', 'log_pi_a', 'ret', 'adv', 'r_original'])
        actions = actions.detach()
        log_probs_old = log_probs_old.detach()
        advantages = (advantages - advantages.mean()) / advantages.std()

        for _ in range(config.optimization_epochs):
            sampler = random_sample(np.arange(states.size(0)), config.mini_batch_size)
            for batch_indices in sampler:
                batch_indices = tensor(batch_indices).long()
                sampled_states = states[batch_indices]
                sampled_actions = actions[batch_indices]
                sampled_log_probs_old = log_probs_old[batch_indices]
                sampled_returns = returns[batch_indices]
                sampled_advantages = advantages[batch_indices]

                prediction = self.network(sampled_states, sampled_actions)
                ratio = (prediction['log_pi_a'] - sampled_log_probs_old).exp()
                obj = ratio * sampled_advantages
                obj_clipped = ratio.clamp(1.0 - self.config.ppo_ratio_clip,
                                          1.0 + self.config.ppo_ratio_clip) * sampled_advantages

                # policy_loss 에 KL을 사용할 때는 KL 계산 input 순서에 주의. 순서에 따라 부호가 바뀜.
                policy_loss = - torch.min(obj, obj_clipped).mean() - config.entropy_weight * prediction['ent'].mean()
                # policy_loss = policy_loss - config.y_square_weight * (- config.lam * y ** 2)
                value_loss = 0.5 * (sampled_returns - prediction['v']).pow(2).mean()

                # approx_kl = (sampled_log_probs_old - prediction['log_pi_a']).mean()
                # if approx_kl <= 1.5 * config.target_kl:
                    # self.actor_opt.zero_grad()
                    # policy_loss.backward()
                    # self.actor_opt.step()
                self.actor_opt.zero_grad()
                policy_loss.backward()
                self.actor_opt.step()

                self.critic_opt.zero_grad()
                value_loss.backward()
                self.critic_opt.step()
