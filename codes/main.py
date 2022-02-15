import argparse

from component.envs import *
from agent import *
from component.replay import *
from component.random_process import *
from utils import *
from network import *


parser = argparse.ArgumentParser(description='RL setting')


parser.add_argument('--trial_num', type=str, help='RL algorithm')
parser.add_argument('--rl', type=str, help='RL algorithm')
parser.add_argument('--data_type', type=str, help='simulation or real data')
parser.add_argument('--hedging_task', type=str, help='delta or timing or delta_timing')
parser.add_argument('--asset_model', type=str, help='GBM or heston')
parser.add_argument('--strike_price', type=float, help='strike price of exposed option')
parser.add_argument('--option_type', type=str, help='call or put option')
parser.add_argument('--lam', type=float, help='lambda')
parser.add_argument('--nn_model', type=str, default='dnn', help='dnn or lstm')
parser.add_argument('--history_len', type=int, default=10, help='history length')
parser.add_argument('--burnin_len', type=int, default=5, help='history length')
parser.add_argument('--lstm_inputsize', type=int, default=6, help='lstm input size')
parser.add_argument('--lstm_hiddensize', type=int, default=24, help='lstm hidden size')
parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')


args = parser.parse_args()


trial_num = args.trial_num
rl_method = args.rl
data_type = args.data_type
hedging_task = args.hedging_task
asset_model = args.asset_model
strike_price = args.strike_price
option_type = args.option_type
lam = args.lam
nn_model = args.nn_model
history_len = args.history_len
burnin_len = args.burnin_len
lstm_inputsize = args.lstm_inputsize
lstm_hiddensize = args.lstm_hiddensize
gpu_id = args.gpu_id

def var_ppo(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    kwargs.setdefault('action_noise', 0)
    config = Config(gpu_id=gpu_id)
    config.merge(kwargs)

    config.task_fn = lambda: Task(config.hedging_task, single_process=True, action_noise=config.action_noise, config=config)
    config.eval_env = config.task_fn()

    config.network_fn = lambda: GaussianActorCriticNet(
        config.state_dim, config.action_dim, actor_body=FCBody(config.state_dim, gate=torch.tanh),
        critic_body=TwoLayerFCBodyWithAction(config.state_dim, config.action_dim, gate=torch.relu))
    config.actor_opt_fn = lambda params: torch.optim.Adam(params, 5e-3)
    config.critic_opt_fn = lambda params: torch.optim.Adam(params, 5e-3)
    config.discount = 0.999
    config.use_gae = False
    config.gae_tau = 0.95
    config.gradient_clip = 0.5
    config.rollout_length = 300 * 1 # 300 * 120
    config.optimization_epochs = 50
    config.mini_batch_size = 300 * 1
    config.ppo_ratio_clip = 0.3
    config.log_interval = 2048
    set_max_steps(config)
    config.target_kl = 0.001
    config.state_normalizer = RescaleNormalizer()
    config.eval_interval = 300 # int(3e3)
    config.eval_episodes = 50
    config.entropy_weight = 0.001
    config.y_square_weight = 0.1
    run_steps(VarPPOAgent(config))

def mvpi_td3(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    kwargs.setdefault('action_noise', 0)
    config = Config(gpu_id=gpu_id)
    config.merge(kwargs)

    config.task_fn = lambda: Task(config.hedging_task, action_noise=config.action_noise, config=config)
    config.task = hedging_task.split('-')[0]
    config.eval_env = config.task_fn()
    set_max_steps(config)
    config.eval_interval = int(5e4)
    config.eval_episodes = 1000
    config.lstm_encoding_size = 6
    config.hidden_size = 24
    config.actor_encoding_size = 6
    config.critic_encoding_size = 7

    if nn_model == 'dnn':
        config.network_fn = lambda: TD3Net(
            config.action_dim,
            actor_body_fn=lambda: FCBody(config.state_dim, (400, 300), gate=F.relu),
            critic_body_fn=lambda: FCBody(
                config.state_dim+config.action_dim, (400, 300), gate=F.relu),
            actor_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-3),
            critic_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-3),
        config=config)

    if nn_model == 'lstm':
        config.network_fn = get_network_fn(config)

    config.replay_fn = lambda: Replay(memory_size=int(5e4), batch_size=100)
    config.discount = 0.99
    config.random_process_fn = lambda: GaussianProcess(
        size=(config.action_dim,), std=LinearSchedule(0.1))
    config.td3_noise = 0.2
    config.td3_noise_clip = 0.5
    config.td3_delay = 2
    config.warm_up = int(1e4)
    config.target_network_mix = 5e-3
    run_steps(MVPITD3Agent(config))


def get_network_fn(config):
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


def set_max_steps(config):
    config.max_steps = int(3.5e5)


rl_method_dict = {'var_ppo': var_ppo,
                  'mvpi_td3': mvpi_td3}


if __name__ == '__main__':
    mkdir('log')
    mkdir('model')
    random_seed()

    exp_model = rl_method_dict[rl_method]
    exp_model(trial_num=trial_num,
              data_type=data_type,
              hedging_task=hedging_task,
              asset_model=asset_model,
              strike_price=strike_price,
              option_type=option_type,
              lam=lam,
              nn_model=nn_model,
              history_len=history_len,
              burnin_len=burnin_len,
              lstm_inputsize=args.lstm_inputsize,
              lstm_hiddensize=args.lstm_hiddensize)
