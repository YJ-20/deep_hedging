import pickle
import tqdm
import argparse

import numpy as np
from pricing_model import *

parser = argparse.ArgumentParser(description='RL setting')

parser.add_argument('--asset_model', type=str, help='GBM or heston')

args = parser.parse_args()

asset_model = args.asset_model

path_number = 10000
test_number = 2000


T = 60 * 5
S = 100
vol = 0.2
ror = 0.0
dt = 1 / (250 * 5)
tau = 60/250
ir = 0.0

rho = - 0.5
kappa = 0.6
long_term_vol = 0.2
sig = 0.2

k = 100

def generate_data(T, S, vol, ror, dt, tau, ir, rho, kappa, long_term_vol, sig, k, data_length):

    path_lists_list = []

    for _ in tqdm.tqdm(range(data_length)):

        rand_num_array = np.random.randn(T)
        S0 = S
        first_vol = vol

        c = - BS('C', S0, k, tau, ir, vol)
        delta = get_delta('c', S0, k, tau, ir, vol)
        path_list = [[S0, k, tau, ir, vol, c, delta]]

        if asset_model == 'GBM':
            for i in range(len(rand_num_array)):
                S1 = S0 * np.exp(
                    (ror - (vol ** 2) / 2) * dt + vol * rand_num_array[i] * np.sqrt(dt))
                S0 = S1
                tau = tau - dt
                c = - BS('C', S0, k, tau, ir, vol)
                delta = get_delta('c', S0, k, tau, ir, vol)

                path_list.append([S0, k, tau, ir, vol, c, delta])

        elif asset_model == 'Heston':
            vol1 = first_vol

            rand_num_array2 = np.random.randn(T)
            chol_matrix = np.array([[1, 0], [rho, np.sqrt(1 - rho ** 2)]])
            rand_num_array, rand_num_array2 = np.matmul(chol_matrix, np.array([rand_num_array, rand_num_array2]))
            for i in range(len(rand_num_array)):
                vol = abs(vol1 + kappa * (long_term_vol - vol1) * dt +
                          sig * rand_num_array2[i] * np.sqrt(dt))
                S1 = S0 * np.exp(
                    (ror - (vol ** 2) / 2) * dt + vol * rand_num_array[i] * np.sqrt(dt))
                vol1 = vol
                S0 = S1

                tau = tau - dt
                c = - BS('C', S0, k, tau, ir, vol)
                delta = get_delta('c', S0, k, tau, ir, vol)

                path_list.append([S0, k, tau, ir, vol, c, delta])

        path_lists_list.append(path_list)

    return path_lists_list


path_lists_list = generate_data(T, S, vol, ror, dt, tau, ir, rho, kappa, long_term_vol, sig, k, path_number)
test_path_lists_list = generate_data(T, S, vol, ror, dt, tau, ir, rho, kappa, long_term_vol, sig, k, test_number)

with open(asset_model + '_train_simulation_data.pkl', 'wb') as f:
    pickle.dump(path_lists_list, f)

with open(asset_model + '_test_simulation_data.pkl', 'wb') as f:
    pickle.dump(test_path_lists_list, f)
