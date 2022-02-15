#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from utils.logger import *
import torch.multiprocessing as mp
from collections import deque
from skimage.io import imsave


class BaseAgent:
    def __init__(self, config):
        self.config = config
        self.logger = get_logger(tag=config.tag, log_level=config.log_level)
        self.task_ind = 0
        # cuda_id = config.gpu_id
        # self._device = torch.device('cuda:' + str(cuda_id) if torch.cuda.is_available() else 'cpu')

    def close(self):
        close_obj(self.task)
        
    def history_reset(self, config):
        self.history = deque(tuple(np.zeros([config.history_len + config.burnin_len, config.state_dim + config.action_dim])),
                             maxlen=int(config.history_len + config.burnin_len))

    def save(self, filename):
        torch.save(self.network.state_dict(), '%s.model' % (filename))
        with open('%s.stats' % (filename), 'wb') as f:
            pickle.dump(self.config.state_normalizer.state_dict(), f)

    def load(self, filename):
        state_dict = torch.load('%s.model' % filename, map_location=lambda storage, loc: storage)
        self.network.load_state_dict(state_dict)
        with open('%s.stats' % (filename), 'rb') as f:
            self.config.state_normalizer.load_state_dict(pickle.load(f))

    def eval_step(self, state):
        raise NotImplementedError

    def eval_episode(self):
        env = self.config.eval_env
        state = env.reset()
        discount_t = 1
        discounted_rewards = []
        rewards = []
        if self.config.nn_model == 'lstm':
            self.history_reset(self.config)

        while True:
            if self.config.nn_model == 'dnn':
                action = self.eval_step(state)
            if self.config.nn_model == 'lstm':
                action = self.eval_step(state, history=self.history)
                self.history.append(np.hstack([state, action]).flatten())
            state, reward, done, info = env.step(action)
            discounted_rewards.append(reward * discount_t)
            rewards.append(reward)
            discount_t *= self.config.discount
            ret = info[0]['episodic_return']
            if ret is not None:
                break
        assert np.abs(np.sum(rewards) - np.sum(ret)) < 1e-5
        return dict(ret=ret, discounted_rewards=discounted_rewards, rewards=rewards)

    def eval_episodes(self):
        episodic_returns = []
        discounted_rewards = []
        rewards = []
        for ep in range(self.config.eval_episodes):
            info = self.eval_episode()
            episodic_returns.append(info['ret'])
            discounted_rewards.extend(info['discounted_rewards'])
            rewards.extend(info['rewards'])
        self.logger.info('steps %d, episodic_return_test %.2f(%.2f)' % (
            self.total_steps, np.mean(episodic_returns), np.std(episodic_returns) / np.sqrt(len(episodic_returns))
        ))
        self.logger.add_scalar('episodic_return_test', np.mean(episodic_returns), self.total_steps)
        self.logger.add_scalar('discounted_per_step_reward_test_mean', np.mean(discounted_rewards), self.total_steps)
        self.logger.add_scalar('discounted_per_step_reward_test_std', np.std(discounted_rewards), self.total_steps)
        self.logger.add_scalar('per_step_reward_test_mean', np.mean(rewards), self.total_steps)
        self.logger.add_scalar('per_step_reward_test_std', np.std(rewards), self.total_steps)
        return {
            'episodic_return_test': np.mean(episodic_returns),
        }

    def record_online_return(self, info, offset=0):
        if isinstance(info, dict):
            ret = info['episodic_return']
            if ret is not None:
                self.logger.add_scalar('episodic_return_train', ret, self.total_steps + offset)
                self.logger.info('steps %d, episodic_return_train %s' % (self.total_steps + offset, ret))
        elif isinstance(info, tuple):
            for i, info_ in enumerate(info):
                self.record_online_return(info_, i)
        else:
            raise NotImplementedError

    def record_episode(self, dir, env):
        mkdir(dir)
        steps = 0
        state = env.reset()
        while True:
            self.record_obs(env, dir, steps)
            action = self.record_step(state)
            state, reward, done, info = env.step(action)
            ret = info[0]['episodic_return']
            steps += 1
            if ret is not None:
                break

    def record_step(self, state):
        raise NotImplementedError

    # For DMControl
    def record_obs(self, env, dir, steps):
        env = env.env.envs[0]
        obs = env.render(mode='rgb_array')
        imsave('%s/%04d.png' % (dir, steps), obs)

    def end_of_training_evaluation(self):
        if not self.config.EOT_eval:
            pass
        else:
            print(self.config.EOT_eval)
            discounted_rewards = []
            rewards = []
            for i in range(self.config.EOT_eval):
                print(f'{i} - eval_episode')
                info = self.eval_episode()
                self.logger.add_scalar('EOT_eval', info['ret'])
                discounted_rewards.extend(info['discounted_rewards'])
                rewards.extend(info['rewards'])
            print(f'discounted_rewards => {discounted_rewards} ')
            print(f'rewards => {rewards} ')
            self.logger.add_scalar('EOT_discounted_per_step_reward_mean', np.mean(discounted_rewards),
                                       self.total_steps)
            self.logger.add_scalar('EOT_discounted_per_step_reward_std', np.std(discounted_rewards), self.total_steps)
            self.logger.add_scalar('EOT_per_step_reward_mean', np.mean(rewards), self.total_steps)
            self.logger.add_scalar('EOT_per_step_reward_std', np.std(rewards), self.total_steps)


class BaseActor(mp.Process):
    STEP = 0
    RESET = 1
    EXIT = 2
    SPECS = 3
    NETWORK = 4
    CACHE = 5

    def __init__(self, config):
        mp.Process.__init__(self)
        self.config = config
        self.__pipe, self.__worker_pipe = mp.Pipe()

        self._state = None
        self._task = None
        self._network = None
        self._total_steps = 0
        self.__cache_len = 2

        if not config.async_actor:
            self.start = lambda: None
            self.step = self._sample
            self.close = lambda: None
            self._set_up()
            self._task = config.task_fn()

    def _sample(self):
        transitions = []
        for _ in range(self.config.sgd_update_frequency):
            transitions.append(self._transition())
        return transitions

    def run(self):
        self._set_up()
        config = self.config
        self._task = config.task_fn()

        cache = deque([], maxlen=2)
        while True:
            op, data = self.__worker_pipe.recv()
            if op == self.STEP:
                if not len(cache):
                    cache.append(self._sample())
                    cache.append(self._sample())
                self.__worker_pipe.send(cache.popleft())
                cache.append(self._sample())
            elif op == self.EXIT:
                self.__worker_pipe.close()
                return
            elif op == self.NETWORK:
                self._network = data
            else:
                raise NotImplementedError

    def _transition(self):
        raise NotImplementedError

    def _set_up(self):
        pass

    def step(self):
        self.__pipe.send([self.STEP, None])
        return self.__pipe.recv()

    def close(self):
        self.__pipe.send([self.EXIT, None])
        self.__pipe.close()

    def set_network(self, net):
        if not self.config.async_actor:
            self._network = net
        else:
            self.__pipe.send([self.NETWORK, net])
