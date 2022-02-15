import os
import sys
# sys.path.append(os.getcwd() + '/codes')
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from agent import MVPITD3Agent
from utils import Config
from Envs import DeltaHedgingEnv
from component import Task
from component.replay import *
from component.random_process import *
from network import RTD3Net, OneDenseLSTM, multiRTD3Net, TD3Net, FCBody

import argparse

class Checking_mvpitd3():
    '''
    Checking_mvpitd3 tests experimental results and visualizes them.
    - model_dir can be an absolute path
    '''
    def __init__(self, model_dir='./model/'):
        self.model_dir = model_dir
        
    def create_config(self, file, gpu_no=0):
        # search existing model name
        last_dash = file.rfind('-')
        model_name = file[:last_dash+1]
        parsers = model_name.split('-')
        
        config = Config(gpu_id = gpu_no)
        config.DEVICE = torch.device(f'cuda:{gpu_no}')
        config.tag = None
    
        # lstm model config
        if parsers[11].split('_')[-1] == 'lstm':
            # config - from model name
            config_dict = {'data_type':'simulation',
                        'hedging_task':parsers[2] + '-' + parsers[3],
                        'asset_model':parsers[4].split('_')[-1],
                        'burnin_len':int(parsers[5].split('_')[-1]),
                        'history_len':int(parsers[7].split('_')[-1]),
                        'lam':float(parsers[8].split('_')[-1]),
                        'lstm_hiddensize':int(parsers[9].split('_')[-1]),
                        'lstm_inputsize':int(parsers[10].split('_')[-1]),
                        'nn_model':parsers[11].split('_')[-1],
                        'option_type':parsers[12].split('_')[-1],
                        'strike_price':float(parsers[13].split('_')[-1])
                        }
            config_dict.setdefault('log_level', 0)
            config_dict.setdefault('action_noise', 0)
            config.merge(config_dict)
            
            # config - others
            config.task = parsers[2]
            config.task_fn = lambda: Task(config.hedging_task, action_noise=config.action_noise, config=config)
            config.eval_env = config.task_fn()
            config.eval_interval = int(5e4)
            config.eval_episodes = 1000
            config.actor_encoding_size = 6
            config.critic_encoding_size = 7
            config.lstm_encoding_size = 6
            config.hidden_size = 24
            # itemlist = config.__dict__.items()
            # for item in itemlist:
            #     print(item)
            config.network_fn = self.get_network_fn(config)
            
        # dnn model config
        elif parsers[11].split('_')[-1] == 'dnn':
            # config - from model name
            config_dict = {'data_type':'simulation',
                        'hedging_task':parsers[2] + '-' + parsers[3],
                        'asset_model':parsers[4].split('_')[-1],
                        'burnin_len':int(parsers[5].split('_')[-1]),
                        'history_len':int(parsers[7].split('_')[-1]),
                        'lam':float(parsers[8].split('_')[-1]),
                        'nn_model':parsers[9].split('_')[-1],
                        'option_type':parsers[10].split('_')[-1],
                        'strike_price':float(parsers[11].split('_')[-1])
                        }
            config_dict.setdefault('log_level', 0)
            config_dict.setdefault('action_noise', 0)
            config.merge(config_dict)
            
            # config - others
            config.task = parsers[2]
            config.task_fn = lambda: Task(config.hedging_task, action_noise=config.action_noise, config=config)
            config.eval_env = config.task_fn()
            config.eval_interval = int(5e4)
            config.eval_episodes = 1000

            config.network_fn = lambda: TD3Net(
                    config.action_dim,
                    actor_body_fn=lambda: FCBody(config.state_dim, (400, 300), gate=F.relu),
                    critic_body_fn=lambda: FCBody(
                        config.state_dim+config.action_dim, (400, 300), gate=F.relu),
                    actor_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-3),
                    critic_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-3),
                config=config)
        
        
        config.discount = 0.99
        config.td3_delay = 2
        config.warm_up = int(1e4)
        config.target_network_mix = 5e-3
        config.replay_fn = lambda: Replay(memory_size=int(5e4), batch_size=100)
        config.random_process_fn = lambda: GaussianProcess(
            size=(config.action_dim,), std=LinearSchedule(0.1))
        config.td3_noise = 0.2
        config.td3_noise_clip = 0.5
        config.td3_delay = 2
        return config

    def get_network_fn(self, config):
        task = config.task
        if task == 'DeltaHedging':
            network_fn = lambda: RTD3Net(
                config.state_dim,
                config.action_dim,
                config.actor_encoding_size,
                config.critic_encoding_size,
                actor_body_fn=lambda: OneDenseLSTM(config.state_dim + config.action_dim, config.lstm_encoding_size,
                                                config.hidden_size, config=config, gate=F.relu),
                critic_body_fn=lambda: OneDenseLSTM(
                    config.state_dim + config.action_dim, config.lstm_encoding_size, config.hidden_size,
                    config=config, gate=F.relu),
                actor_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-3),
                critic_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-3),
            config=config)

        if task == 'DeltaHedgingTiming':
            network_fn = lambda: RTD3Net(
                config.state_dim,
                config.action_dim,
                config.actor_encoding_size,
                config.critic_encoding_size,
                actor_body_fn=lambda: OneDenseLSTM(config.state_dim + config.action_dim, config.lstm_encoding_size,
                                                config.hidden_size, config=config, gate=F.relu),
                critic_body_fn=lambda: OneDenseLSTM(
                    config.state_dim + config.action_dim, config.lstm_encoding_size, config.hidden_size,
                    config=config, gate=F.relu),
                actor_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-3),
                critic_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-3),
            config=config)

        if task == 'DeltaHedgingMulti':
            network_fn1 = lambda: RTD3Net(
                config.state_dim,
                int(config.action_dim),
                config.actor_encoding_size,
                config.critic_encoding_size,
                actor_body_fn=lambda: OneDenseLSTM(config.state_dim + int(config.action_dim), config.lstm_encoding_size,
                                                config.hidden_size, config=config, gate=F.relu),
                critic_body_fn=lambda: OneDenseLSTM(
                    config.state_dim + config.action_dim, config.lstm_encoding_size, config.hidden_size,
                    config=config, gate=F.relu),
                actor_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-3),
                critic_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-3),
            config=config)

            network_fn2 = lambda: RTD3Net(
                config.state_dim,
                int(config.action_dim),
                config.actor_encoding_size,
                config.critic_encoding_size,
                actor_body_fn=lambda: OneDenseLSTM(config.state_dim + int(config.action_dim), config.lstm_encoding_size,
                                                config.hidden_size, config=config, gate=F.relu),
                critic_body_fn=lambda: OneDenseLSTM(
                    config.state_dim + config.action_dim, config.lstm_encoding_size, config.hidden_size,
                    config=config, gate=F.relu),
                actor_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-3),
                critic_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-3),
            config=config)

            network_fn = lambda: multiRTD3Net(
                network_fn1(),
                network_fn2()
            )

        return network_fn    
    
    def visualize(self, reward_list, total_sum_array, total_mean_array, total_std_array, total_delta_pnl_array):
        # to do: plt.savefig() 이용해서 ./visualization/ 에다가 visualize 하기
        pass
    
    def check(self, trial_list, vis=True):
        print('check starts')
        for trial_no in trial_list:
            file_list = [x for x in os.listdir(self.model_dir) if x.endswith('.model') and x.startswith(f't{trial_no}-')]
            file_list.sort()
            previous_cfg = ''
            for file in file_list:
                last_dash = file.rfind('-')+1
                current_cfg = file[:last_dash+1]
                if not previous_cfg or previous_cfg != current_cfg:
                    print('#'*50, 'set new config', '#'*50)
                    # init ckpFile_list, std_list, previous_cfg
                    ckpFile_list = [x for x in file_list if x.startswith(current_cfg)]
                    std_list = []  # for select best ckp
                    sum_list = []
                    mean_list = []
                    ckp_list = []
                    previous_cfg = current_cfg
                    config = self.create_config(file)
                    print('trial_no: ', trial_no, config.hedging_task, config.nn_model, config.history_len, config.burnin_len, config.lstm_inputsize, config.lstm_hiddensize)
                    print('lambda: ', config.lam)
                else:
                    continue
                
                for ckpFile in ckpFile_list:
                    ckp = ckpFile[last_dash:]
                    td3_agent = MVPITD3Agent(config)
                    reward_list = []
                    total_sum_array = np.zeros(0)
                    total_mean_array = np.zeros(0)
                    total_std_array = np.zeros(0)
                    total_delta_pnl_array = np.zeros(0)                
                    filename = self.model_dir + ckpFile
                    state_dict = torch.load(filename)
                    td3_agent.network.load_state_dict(state_dict)

                    state_list = []
                    action_array = np.array([])
                    action_mean_array = np.array([])
                    delta_array = np.array([])
                    reward_array = np.array([])
                    delta_pnl_array = np.array([])
                    
                    deltahedging_env = DeltaHedgingEnv(config, seed=0)
                    
                    for i in range(100):
                        done = False
                        state = deltahedging_env.reset()
                        while not done:
                            state = state.reshape(1,-1)
                            action = td3_agent.eval_step(state, history=td3_agent.history)
                            td3_agent.history.append(np.hstack([state, action]).flatten())
                            delta = deltahedging_env.env_params.delta.copy()
                            next_state, hedging_performance, done, _ = deltahedging_env.step(action, delta_check=True)
                            if not done:
                                state_list.append(state)
                                action_array = np.append(action_array, action)
                                reward_array = np.append(reward_array, hedging_performance[0])
                                delta_pnl_array = np.append(delta_pnl_array, hedging_performance[1])
                                delta_array = np.append(delta_array, delta)
                                state = next_state
                                
                        td3_agent.history_reset(config)         
                        reward_list.append([reward_array.sum(), reward_array.mean(), reward_array.std()])
                    
                    total_delta_pnl_array = np.append(total_delta_pnl_array, delta_pnl_array)
                    total_sum_array = np.append(total_sum_array, np.array(reward_list)[:,0].mean())
                    total_mean_array = np.append(total_mean_array, np.array(reward_list)[:,1].mean())
                    total_std_array = np.append(total_std_array, np.array(reward_list)[:,2].mean())
                    print('='*100)
                    print(ckp)
                    print('-'*100)
                    print('sum: ', total_sum_array.mean(), '\n',
                        'mean: ', total_mean_array.mean(), '\n',
                        'std: ', total_std_array.mean(), '\n',
                        'delta_pnl:', total_delta_pnl_array.mean())
                    print('='*100)
                    std_list.append(total_std_array.mean())
                    sum_list.append(total_sum_array.mean())
                    mean_list.append(total_mean_array.mean())
                    ckp_list.append(ckp)
                # select best model
                lowest_std = min(std_list)
                best_index = std_list.index(lowest_std)
                print('trial_no: ', trial_no, config.hedging_task, config.nn_model, config.history_len, config.burnin_len, config.lstm_inputsize, config.lstm_hiddensize)
                print('lambda: ', config.lam)
                print('lowest std: ', lowest_std)
                print('best model: ', ckp_list[best_index])
                print('sum, mean, std: ', sum_list[best_index], mean_list[best_index], std_list[best_index])
                print('finished this config', '\n', '\n', '\n')
        
                # visualization - savefig
                # if vis:
                #     self.visualize(reward_list, total_sum_array, total_mean_array, total_std_array, total_delta_pnl_array)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--trial_num', type=str, default='0', help='trial_num to test ex) 1,2,3')
    args = parser.parse_args()

    trial_num = args.trial_num
    # put trial numbers to test in trial_list 
    trial_list = trial_num.split(',')
    print('cheking trial list is ', trial_list)
    
    checking_mvpitd3 = Checking_mvpitd3(model_dir='./model/')
    checking_mvpitd3.check(trial_list, vis=True)
    
    # to do: 결과 csv 등으로 저장해도 될듯