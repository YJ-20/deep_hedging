import numpy as np
import gym
from gym import spaces
from gym.utils import seeding

from pricing_model import *


# delta, gamma를 나눠야겠다.

# simulation data는 S와 C, t가 계산되어야함.
# 반면 real data는 가져다 사용하면 된다.
# -> S와 C, t는 Env 밖에서 계산이 돼야함.

# HedgingEnv는 데이터를 받아올지, 아니면 simulation data를 사용할 지 먼저 선택
# 내부적으로 S와 C, t는 계산되지 않고 가지고온 데이터로 사용.

# simulation data를 사용한다면,
# T, sigma, K, ir 미리 설정 (sigma를 volatility하게 가져가면 S와 함께 return 필요)
# S와 C, t를 계산할 모듈 필요
# t까지 미리 계산해놓고,
# S는 미리 계산해놓고 한개씩 가져오는 기존의 방법 그대로 사용
# C는 S와 t를 계산한 것을 가지고 오고, 미리 설정한 K와 sigma 를 가져와 계산
# 계산한 값으로 state, reward 계산하기

# real data를 사용한다면,
# S, C, t, K, ir 가지고 오기
# t=T가 될 때까지 가지고 온 값으로 state, reward 계산하기

class DeltaHedgingEnv_Fromdataset(gym.Env):
    def __init__(self, config):
        self.config = config
        self.env_params = EnvironmentParameterGenerator(config)
        self.seed()
        self.pre_action = None

        self.min_action = 0.0
        self.max_action = 1.0
        self.min_underlying = 0.0
        self.max_underlying = 400.0
        self.min_option = 0.0
        self.max_option = 100.0
        self.max_delta = 1.0
        self.min_hold_num = 0.0
        self.max_hold_num = 1.0
        self.min_vol = 0.0
        self.max_vol = 1.0

        low_state_list = [self.min_underlying, self.min_option, -self.max_delta, self.min_hold_num]
        high_state_list = [self.max_underlying, self.max_option, self.max_delta, self.max_hold_num]

        if config.asset_model == 'Heston':
            low_state_list.append(self.min_vol)
            high_state_list.append(self.max_vol)

        self.low_state = np.array(
            low_state_list,
            dtype=np.float32
        )
        self.high_state = np.array(
            high_state_list,
            dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=self.min_action,
            high=self.max_action,
            shape=(1,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=self.low_state,
            high=self.high_state,
            dtype=np.float32
        )

        self.s = None
        self.c = None

        self.asset_model = config.asset_model

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action, delta_check=False):
        action = float(action)
        underlying_transaction_num = action - self.pre_action

        # next time step
        self.env_params.update()

        S0 = self.s
        S1 = self.env_params.s

        C0 = self.c
        C1 = self.env_params.c

        hedging_performance = self.get_hedging_performance(C0, C1, S0, S1, action, underlying_transaction_num)

        if delta_check is True:
            delta_hedging_performance = self.get_delta_hedging_performance(C0, C1, S0, S1)
            hedging_performance = [hedging_performance, delta_hedging_performance]

        next_state = np.array(
            [S1/self.env_params.k, - C1, self.env_params.delta, action],
            dtype=np.float32)  # TODO: rolling vol

        self.pre_action = action
        self.state = next_state
        self.s = S1
        self.c = C1

        return next_state, hedging_performance, self.done, {}

    def get_delta_hedging_performance(self,V0, V1, S0, S1):
        pre_delta_position = getattr(self, 'pre_delta_position', 0)
        delta_position = self.state[2].copy()
        delta_transaction_num = delta_position - pre_delta_position
        delta_hedging_performance = self.get_hedging_performance(V0, V1, S0, S1, delta_position, delta_transaction_num)

        self.pre_delta_position = delta_position

        return delta_hedging_performance

    def get_hedging_performance(self, V0, V1, S0, S1, new_underlying_num, transaction_num):

        transaction_cost = self.get_transaction_cost(transaction_num)
        inventory_pnl = V1 - V0 + new_underlying_num * (S1 - S0)

        # hedging_performance = inventory_pnl - transaction_cost
        hedging_performance = inventory_pnl
        if self.env_params.tau < 1.00e-06:  # at maturity
            # terminal_transaction_cost = self.get_transaction_cost(new_underlying_num)
            # hedging_performance = hedging_performance - self.config.discount * \
            #                       terminal_transaction_cost
            self.done = True

        return hedging_performance

    def get_transaction_cost(self, transaction_num):
        transaction_cost = self.config.tick_size * (abs(transaction_num) + 0.01 * (transaction_num ** 2))
        return transaction_cost

    def reset(self):
        self.done = False

        start_hold_num = 0
        self.env_params.first_generate()
        s = self.env_params.s
        c = self.env_params.c
        k = self.env_params.k
        delta = self.env_params.delta
        self.state = np.array([s/k, - c, delta, start_hold_num])
        self.pre_action = start_hold_num

        self.s = s
        self.c = c

        return np.array(self.state)

    def render(self, mode='human'):
        return


class EnvironmentParameterGenerator(object):
    def __init__(self, config):
        self.s = None
        self.k = None
        self.tau = None
        self.ir = None
        self.vol = None
        self.c = None
        self.delta = None

        if config.data_type == 'simulation':
            self.config = config
            ror = config.rate_of_return
            vol = config.volatility
            dt = config.dt

            self.ror = ror
            self.vol = vol
            self.dt = dt # dt 나 tau에 대해 trade_freq를 고려해서 변화할 수 있게 config에 고려해야함.

            identical_path = config.identical_path
            random_parameter = config.random_parameter
            asset_model = config.asset_model
            self.path_maker = AssetPathMaker(ror, vol, dt, identical_path, random_parameter, asset_model)

        elif config.data_type == 'real data':
            pass

    def first_generate(self):
        if self.config.data_type == 'simulation':
            self.path_maker.reset(self.config.maturity * self.config.trading_freq_per_day)
            self.s = self.path_maker.get_underlying_price()
            self.k = self.config.strike_price
            self.tau = self.config.tau
            self.ir = self.config.interest
            self.vol = self.path_maker.get_volatility()
            self.c = - BS(self.config.option_type, self.s, self.k, self.tau, self.ir, self.vol) # option config function을 만드는 것이 더 좋을 것 같은데. type을 위한?
            self.delta = get_delta(self.config.option_type, self.s, self.k, self.tau, self.ir, self.vol)

        elif self.config.data_type == 'real data':
            pass

    def update(self):
        if self.config.data_type == 'simulation':
            self.s = self.path_maker.get_underlying_price()
            self.k = self.config.strike_price
            self.tau = self.tau - self.dt
            self.ir = self.config.interest
            self.vol = self.path_maker.get_volatility()
            self.c = - BS(self.config.option_type, self.s, self.k, self.tau, self.ir, self.vol)
            self.delta = get_delta(self.config.option_type, self.s, self.k, self.tau, self.ir, self.vol)


class AssetPathMaker(object):

    def __init__(self, ror, vol, dt, identical_path=None, random_parameter=None, asset_model=None):
        self.identical_path = identical_path
        self.random_parameter = random_parameter
        # self._ror_array = np.linspace(-0.1, 0.1, 20)
        # self._vol_list = np.linspace(0.1, 0.3, 20)

        self._ror_list = ror
        self._vol_list = vol
        self._dt = dt
        self._S = 100

        self._rand_num_array = None
        self.path_array = None

        self.only_one = False if identical_path is True else None

        if random_parameter is False:
            self.ror = 0.05
            self.vol = 0.2

        self.asset_model = asset_model

    def reset(self, T):
        self.path_array_index = 0

        if self.asset_model == 'GBM':
            if self.identical_path is True:
                if not self.only_one:
                    self.make_asset_path(T)
                    self.only_one = True
                    self.first_vol = self.vol

                else:
                    pass

            else:
                if self.random_parameter is True:
                    self.ror = np.random.uniform(*self._ror_list)
                    self.vol = np.random.uniform(*self._vol_list)
                    self.first_vol = self.vol
                    self.make_asset_path(T)
                else:
                    self.make_asset_path(T)

        elif self.asset_model == 'Heston':
            self.vol_array_index = 0
            if self.random_parameter is True:
                self.ror = np.random.uniform(*self._ror_list)
                self.first_ror = self.ror
                self.vol = np.random.uniform(*self._vol_list)
                self.first_vol = self.vol
                self.rho = - 0.5
                self.kappa = 0.8
                self.long_term_vol = 0.2
                self.sig = 0.9
                self.vol_array = np.array([])
                self.vol = 0.3
                self.vol_array = np.append(self.vol_array, self.vol)
                self.make_asset_path(T)
            else:
                self.rho = - 0.5
                self.kappa = 0.6
                self.long_term_vol = 0.2
                self.sig = 0.2
                self.vol_array = np.array([])
                self.vol = 0.3
                self.first_vol = self.vol
                self.vol_array = np.append(self.vol_array, self.vol)
                self.make_asset_path(T)

    def make_asset_path(self, T):
        self._S = 100
        self._rand_num_array = np.random.randn(T)
        self.path_array = np.array([self._S])
        S0 = self._S
        vol1 = self.vol
        for i in range(len(self._rand_num_array)):
            if self.asset_model == 'GBM':
                S1 = S0 * np.exp(
                    (self.ror - (self.vol ** 2) / 2) * self._dt + self.vol * self._rand_num_array[i] * np.sqrt(self._dt))
                self.path_array = np.append(self.path_array, S1)
                S0 = S1

            elif self.asset_model == 'Heston':
                self._rand_num_array2 = np.random.randn(T)
                chol_matrix = np.array([[1, 0], [self.rho, np.sqrt(1 - self.rho ** 2)]])
                self._rand_num_array, self._rand_num_array2 = np.matmul(chol_matrix, np.array([self._rand_num_array, self._rand_num_array2]))
                self.vol = abs(vol1 + self.kappa * (self.long_term_vol - vol1) * self._dt +
                               self.sig * self._rand_num_array2[i] * np.sqrt(self._dt))
                S1 = S0 * np.exp(
                    (self.ror - (self.vol ** 2) / 2) * self._dt + self.vol * self._rand_num_array[i] * np.sqrt(self._dt))
                self.path_array = np.append(self.path_array, S1)
                self.vol_array = np.append(self.vol_array, self.vol)
                vol1 = self.vol
                S0 = S1

    def get_underlying_price(self):
        underlying_price = self.path_array[self.path_array_index]
        self.path_array_index += 1

        return underlying_price

    def get_volatility(self):
        if self.asset_model == 'GBM':
            return self.vol

        elif self.asset_model == 'Heston':

            underlying_vol = self.path_array[self.vol_array_index]
            self.path_array_index += 1

            return underlying_vol


class DeltaGammaHedgingEnv(gym.Env):
    def __init__(self):
        return