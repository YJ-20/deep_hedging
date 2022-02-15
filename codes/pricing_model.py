def BS(option_type, S, K, tau, r, vol):
    import numpy as np
    import scipy.stats as spst

    d1 = calculate_d1(S, K, tau, r, vol)
    d2 = d1 - vol * np.sqrt(tau)

    # Black-Scholes Equation
    # call option
    if option_type == 'C':
        return S * spst.norm.cdf(d1, 0, 1) - K * np.exp(-r * tau) * spst.norm.cdf(d2, 0, 1)

    # put option
    elif option_type == 'P':
        return -S * spst.norm.cdf(-d1, 0, 1) + K * np.exp(-r * tau) * spst.norm.cdf(-d2, 0, 1)

    # exception handling
    else:
        print(option_type)
        print(" error: 1st parameter(option_type) should be either 'C' or 'P' ")
        raise AssertionError


def calculate_d1(S, K, tau, r, vol):
    import numpy as np
    d1 = (np.log(S / K) + (r + 0.5 * vol ** 2) * tau) / (vol * np.sqrt(tau))

    if np.isnan(d1) is True:
        d1 = 0
    return d1

def get_prime_cdf(d1):
    import numpy as np
    result = np.exp((- d1 ** 2) / 2) / np.sqrt(2 * np.pi)
    return result

def get_delta(option_type, S, K, tau, r, vol):
    import scipy.stats as spst

    d1 = calculate_d1(S, K, tau, r, vol)
    delta = spst.norm.cdf(d1)
    if option_type is 'C':
        pass
    elif option_type is 'P':
        delta = delta - 1

    return delta

def get_gamma(S, K, tau, r, vol):
    import numpy as np
    d1 = calculate_d1(S, K, tau, r, vol)
    N_prime = get_prime_cdf(d1)
    gamma = N_prime / (S * vol * np.sqrt(tau))

    return gamma

def get_vega(S, K, tau, r, vol):
    import numpy as np
    d1 = calculate_d1(S, K, tau, r, vol)
    N_prime = get_prime_cdf(d1)
    vega = S * N_prime * np.sqrt(tau)

    return vega
