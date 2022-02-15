'''
S1 = S*mu*dt+S*sig*dw*np.sqrt(dt)

T = T-dt

K = constant

ir = constant

ATM_Call(S1,T,[sig],ir,K )
OTM_Call
ITM_Call
ATM_PUt


Inventory = 1 ATM_Call

action = hedge_instrument = [S1,..,]

state = [Inventory, S, Calls, Puts, sig]

ewma garch = vol
'''
import argparse
import numpy as np
import gym
from option_generator import option_wrapper
import scipy.stats as spst

from pricing_model import BS, calculate_d1
from utils import DifferentialSharpeRatio, RollingVolatility

class HedgingSimulation(gym.Env):

    WORKDAY = 250
    dt = 1/WORKDAY

    ir = 0.01

    DISCOUNT_FACTOR = np.exp(-ir * dt)

    def __init__(self, asset_model, **kwargs):
        ### 수정 필요
        '''
        if kwargs is not None:

            # self._ror_list = [0.05]
            # self._vol_list = [0.2]
            # self._ror_array = np.linspace(-0.1, 0.1, 20)
            # self._vol_list = np.linspace(0.1, 0.3, 20)
            # self._use_bs = True
            # self.reward_approach = 'P&L'
            # self.reward_approach = 'Cashflow'


        else:
            self._kwargs = kwargs
            self._ror_list = self._kwargs['ror_list']
            self._vol_list = self._kwargs['vol_list']
            self._use_bs = self._kwargs['use_bs']
            self.reward_approach = 'P&L'
        '''
        self._ror = 0.05
        self._vol = 0.2

        self._ror_list = [-0.1, 0.2]
        self._vol_list = [0.05, 0.4]

        self.identical_path = False
        self.reward_approach = 'P&L'
        self._exposed_option = None
        self._hedge_instrument_option = None
        self.rolling_vol = None
        self._inventory = None
        self._trading_num = 0
        self._eta = 0.04

        self.is_first = True
        self.Done = False

        self.tick_size = 0.05

        # self.asset_path = AssetPathMaker(self._ror, self._vol, HedgingSimulation.dt, self.identical_path,
        #                                  random_parameter=False)
        self.asset_path = AssetPathMaker(self._ror_list, self._vol_list, HedgingSimulation.dt, self.identical_path,
                                         random_parameter=True, asset_model=asset_model)
        self.ds_reward_maker = DifferentialSharpeRatio(self._eta)

    def get_transaction_cost(self, tick_size, transaction_amount):
        cost = tick_size * (abs(transaction_amount) + 0.01 * transaction_amount ** 2)
        return cost


    def step(self, action):
        new_asset_num = action[0]

        self._trading_num += 1

        transaction_asset = new_asset_num - self._inventory['holding_num_asset']

        S0 = self._inventory['asset_price'] * self._exposed_option['K']
        S1 = self.asset_path.update_underlying_asset(self._trading_num)

        self._exposed_option = self.update_option(self._exposed_option, S1)

        V0 = self._inventory['exposed_option_price']
        V1 = -BS(**self._exposed_option)

        hedging_performance = self.return_hedging_performance(V0, V1, S0, S1, new_asset_num, transaction_asset)
        self.update_inventory(new_asset_num, S1, V1)

        self.rolling_vol.update(np.log(S1/S0))

        next_state = np.array(list(self._inventory.values()) + [self._exposed_option['tau']] + [self._exposed_option['vol']],
                              dtype=np.float32) # TODO: rolling vol

        reward = self.ds_reward_maker.return_reward(hedging_performance)
        # reward = hedging_performance

        if self.is_first is True:
            self.is_first = False

        return next_state, reward, self.Done, hedging_performance

    def test_step(self, action, holding_num, S0, S1, K, V0, V1, next_tau, implied_vol, done):
        new_asset_num = action[0]
        transaction_asset = new_asset_num - holding_num
        hedging_performance = self.return_hedging_performance(V0, V1, S0, S1, new_asset_num, transaction_asset,
                                                              is_test=True, done=done)
        reward = self.ds_reward_maker.return_reward(hedging_performance)
        next_state = np.array([new_asset_num, S1/K, V1, next_tau, implied_vol])

        if self.is_first is True:
            self.is_first = False

        return next_state, reward, hedging_performance


    def return_hedging_performance(self, V0, V1, S0, S1, new_asset_num, transaction_asset, is_test=False, done=False):

         transaction_cost = self.get_transaction_cost(self.tick_size, transaction_asset)
         # TODO: reward 부분 전부 수정

        if is_test is False:
            if self.reward_approach == 'P&L':  # should make function to track easily
                inventory_pnl = V1 - V0 + new_asset_num * (S1 - S0)
                transaction_cost = (abs(transaction_asset) * S0) * HedgingSimulation.TRANSACTION_FEE

                hedging_performance = inventory_pnl - transaction_cost
                if self._exposed_option['tau'] == 0:  # at maturity
                    #reward = reward - HedgingSimulation.DISCOUNT_FACTOR * \
                    #         (abs(S1 * new_asset_num)) * HedgingSimulation.TRANSACTION_FEE
                    hedging_performance = hedging_performance - HedgingSimulation.DISCOUNT_FACTOR * \
                             (abs(S1 * new_asset_num)) * HedgingSimulation.TRANSACTION_FEE
                    self.Done = True
                else:
                    pass

            elif self.reward_approach == 'Cashflow':
                inventory_cashflow = - transaction_asset * S0
                if self.is_first is True:
                    inventory_cashflow += - V0
                transaction_cost = (abs(transaction_asset) * S0) * HedgingSimulation.TRANSACTION_FEE

                hedging_performance = inventory_cashflow - transaction_cost
                if self._exposed_option['tau'] == 0:  # at maturity
                    final_inventory_cashflow = S1 * new_asset_num + V1
                    hedging_performance = hedging_performance + HedgingSimulation.DISCOUNT_FACTOR * (
                             final_inventory_cashflow -
                             (abs(S1 * new_asset_num)) * HedgingSimulation.TRANSACTION_FEE)
                    self.Done = True

                else:
                    pass

                hedging_performance = hedging_performance / 100

            else:
                print('Type P&L or Cashflow')
                raise NameError

        else:
            if self.reward_approach == 'P&L':  # should make function to track easily
                inventory_pnl = V1 - V0 + new_asset_num * (S1 - S0)
                transaction_cost = (abs(transaction_asset) * S0) * HedgingSimulation.TRANSACTION_FEE

                hedging_performance = inventory_pnl - transaction_cost
                if done is True:  # at maturity
                    # reward = reward - HedgingSimulation.DISCOUNT_FACTOR * \
                    #         (abs(S1 * new_asset_num)) * HedgingSimulation.TRANSACTION_FEE
                    hedging_performance = hedging_performance - HedgingSimulation.DISCOUNT_FACTOR * \
                                          (abs(S1 * new_asset_num)) * HedgingSimulation.TRANSACTION_FEE
                    self.Done = True
                else:
                    pass

            elif self.reward_approach == 'Cashflow':
                inventory_cashflow = - transaction_asset * S0
                if self.is_first is True:
                    inventory_cashflow += - V0
                transaction_cost = (abs(transaction_asset) * S0) * HedgingSimulation.TRANSACTION_FEE

                hedging_performance = inventory_cashflow - transaction_cost
                if done is True:  # at maturity
                    final_inventory_cashflow = S1 * new_asset_num + V1
                    hedging_performance = hedging_performance + HedgingSimulation.DISCOUNT_FACTOR * (
                            final_inventory_cashflow -
                            (abs(S1 * new_asset_num)) * HedgingSimulation.TRANSACTION_FEE)
                    self.Done = True

                else:
                    pass

                hedging_performance = hedging_performance / 100

            else:
                print('Type P&L or Cashflow')
                raise NameError
        return hedging_performance

    def update_inventory(self, *args):
        new_asset_num, S1, V1 = args
        self._inventory.update({'holding_num_asset': new_asset_num, 'asset_price': S1/self._exposed_option['K'],
                                'exposed_option_price': V1})

    def update_option(self, option, S1):
        option['S'] = S1
        option['tau'] = option['tau'] - HedgingSimulation.dt

        if abs(option['tau']) < 0.0004:  # at maturity
            option['tau'] = 0

        return option

    def reset(self, asset_path=None, test_vol=None):
        self.Done = False
        self._trading_num = 0

        self.ds_reward_maker.reset()

        assert isinstance(option_wrapper, object)
        self._exposed_option = option_wrapper.option_list[0].parameters_dict.copy() # dictionary 형으로 받아놓음 (__getitem__ 사용 필요)

        self.asset_path.reset(self._exposed_option['T'])

        self._exposed_option['tau'] = self._exposed_option['T'] * HedgingSimulation.dt

        self._exposed_option['vol'] = self.asset_path.first_vol
        if asset_path is not None and test_vol is not None:
            self.asset_path.path_array = asset_path
            self._exposed_option['vol'] = test_vol
        self._exposed_option.pop('T')

        new_asset_price = option_wrapper.option_list[0]['S']
        exposed_option_price = -BS(**self._exposed_option)

        self._inventory = {'holding_num_asset': 0, 'asset_price': new_asset_price/self._exposed_option['K'],
                           'exposed_option_price': exposed_option_price}

        self.rolling_vol = RollingVolatility(self._exposed_option['vol'], 0.25)
        state = np.array(list(self._inventory.values())+[self._exposed_option['tau']]+[self._exposed_option['vol']]) # TODO: rolling vol
        # state = np.array(list(self._inventory.values())+self._exposed_option['tau'], vol)

        return state

    def delta_hedging(self):
        """
        Taking short position on call, position on asset is + delta
        hedging_performance = - dV + (+)delta x dS
        """
        all_path = self.asset_path.path_array
        tau_array = np.arange(self._exposed_option['tau'], 0, -HedgingSimulation.dt)
        tau_to_zero_array = np.append(tau_array, 0)
        d1_array = calculate_d1(all_path, self._exposed_option['K'], tau_to_zero_array,
                                HedgingSimulation.ir, self._exposed_option['vol'])
        delta_array = spst.norm.cdf(d1_array)[:-1] # the last delta isn't used because it will be zero at final step

        if self._exposed_option['option_type'] == 'P':
            delta_array = delta_array - 1

        V_array = - BS(self._exposed_option['option_type'], all_path, self._exposed_option['K'], tau_to_zero_array,
                       HedgingSimulation.ir, self._exposed_option['vol'])  # short position => -1
        tracking_error_array = np.diff(V_array) + delta_array * np.diff(all_path)  # V1 - V0 + delta * (S1 - S0)

        transaction = np.diff(np.append(0, delta_array))
        buy_sell_amount_array = all_path[:-1] * transaction

        fee_array = abs(buy_sell_amount_array) * HedgingSimulation.TRANSACTION_FEE
        clear_amount = abs(all_path[-1] * delta_array[-1])

        fee_array[-1] = fee_array[-1] + clear_amount * HedgingSimulation.TRANSACTION_FEE * HedgingSimulation.DISCOUNT_FACTOR
        days = np.flip(tau_to_zero_array) * HedgingSimulation.WORKDAY
        discount_array = np.exp(- HedgingSimulation.ir * HedgingSimulation.dt * days[:-1])
        total_tracking_error = (tracking_error_array - fee_array) * discount_array

        return total_tracking_error, delta_array, buy_sell_amount_array

class AssetPathMaker(object):

    def __init__(self, ror, vol, dt, identical_path=False, random_parameter=False, asset_model='GBM'):
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

    def update_underlying_asset(self, trading_num):
        updated_price = self.path_array[trading_num]

        return updated_price






