#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
import pickle
import os
import datetime
import torch
import time
from .torch_utils import *
from pathlib import Path


def run_steps(agent):
    config = agent.config
    agent_name = agent.__class__.__name__
    t0 = time.time()
    prev_steps = 0
    steps_since_last_eval = 0
    steps_since_last_save = 0
    while True:
        if config.save_interval and steps_since_last_save >= config.save_interval:
            agent.save('model/%s-%s-%s-%d' % (config.trial_num, agent_name, config.tag, agent.total_steps))
            steps_since_last_save = 0
        if config.log_interval and not agent.total_steps % config.log_interval:
            agent.logger.info('steps %d, %.2f steps/s' % (agent.total_steps, config.log_interval / (time.time() - t0)))
            t0 = time.time()
        if config.eval_interval and steps_since_last_eval >= config.eval_interval:
            agent.eval_episodes()
            steps_since_last_eval = 0
        if config.max_steps and agent.total_steps >= config.max_steps:
            agent.close()
            agent.end_of_training_evaluation()
            print('Training Over')
            break
        agent.step()
        steps_since_last_eval += agent.total_steps - prev_steps
        steps_since_last_save += agent.total_steps - prev_steps
        prev_steps = agent.total_steps


def get_time_str():
    return datetime.datetime.now().strftime("%y%m%d-%H%M%S")


def get_default_log_dir(name):
    return './log/%s-%s' % (name, get_time_str())


def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def close_obj(obj):
    if hasattr(obj, 'close'):
        obj.close()


def random_sample(indices, batch_size):
    indices = np.asarray(np.random.permutation(indices))
    batches = indices[:len(indices) // batch_size * batch_size].reshape(-1, batch_size)
    for batch in batches:
        yield batch
    r = len(indices) % batch_size
    if r:
        yield indices[-r:]


def generate_tag(params):
    if 'tag' in params.keys():
        return
    hedging_task = params['hedging_task']
    params.setdefault('run', 0)
    run = params['run']
    trial_num = params['trial_num']
    del params['hedging_task']
    del params['run']
    del params['trial_num']

    str = ['%s_%s' % (k, v) for k, v in sorted(params.items())]
    tag = '%s-%s-run-%d' % (hedging_task, '-'.join(str), run)
    params['tag'] = tag
    params['hedging_task'] = hedging_task
    params['run'] = run
    params['trial_num'] = trial_num

def translate(pattern):
    groups = pattern.split('.')
    pattern = ('\.').join(groups)
    return pattern


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))
