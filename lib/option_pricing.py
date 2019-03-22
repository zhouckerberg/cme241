import numpy as np
from scipy.stats import norm


class Option_pricing():
    def __init__(self, S_t, K, r, sigma, delta_t):
        self.S_t = S_t
        self.K = K
        self.r = r
        self.sigma = sigma
        self.delta_t = delta_t

    def eur_call_price(self):
        return

    def eur_put_price(self):
        return


class Black_Scholes(Option_pricing):
    def __init__(self, S_t, K, r, sigma, delta_t):
        super().__init__(S_t, K, r, sigma, delta_t)
        self.d1 = np.log(self.S_t / self.K) + (self.r + self.sigma * self.sigma / 2) * self.delta_t
        self.d2 = np.log(self.S_t / self.K) + (self.r - self.sigma * self.sigma / 2) * self.delta_t
        self.d1 /= self.sigma * np.sqrt(self.delta_t)
        self.d2 /= self.sigma * np.sqrt(self.delta_t)

    def eur_call_price(self):
        return self.S_t * norm.cdf(self.d1) - self.K * np.exp(-self.r * self.delta_t) * norm.cdf(self.d2)

    def eur_put_price(self):
        return -self.S_t * norm.cdf(-self.d1) + self.K * np.exp(-self.r * self.delta_t) * norm.cdf(-self.d2)


class Binomial_lattice(Option_pricing):
    def __init__(self, S_t, K, r, sigma, delta_t, num_steps):
        super().__init__(S_t, K, r, sigma, delta_t)
        self.num_steps = num_steps
        self.R = np.exp(self.r * self.delta_t / self.num_steps)
        self.u = np.exp(self.sigma * np.sqrt(self.delta_t / self.num_steps))
        self.d = np.exp(-self.sigma * np.sqrt(self.delta_t / self.num_steps))
        self.p = (self.R - self.d) / (self.u - self.d)  # risk-neutral probabilities
        self.q = (self.u - self.R) / (self.u - self.d)
        self.underlyer = np.zeros((self.num_steps + 1, self.num_steps + 1))
        for i in range(self.num_steps + 1):
            for j in range(i + 1):
                self.underlyer[j, i] = self.S_t * (self.u ** (i - j)) * (self.d ** j)

    def eur_call_price(self):
        option = np.zeros((self.num_steps + 1, self.num_steps + 1))
        option[:, -1] = np.maximum(np.zeros(self.num_steps + 1), (self.underlyer[:, -1] - self.K))
        for i in range(self.num_steps - 1, -1, -1):
            for j in range(i + 1):
                option[j, i] = (self.p * option[j, i + 1] + self.q * option[j + 1, i + 1]) / self.R
        return option[0, 0]

    def am_call_price(self):
        option = np.zeros((self.num_steps + 1, self.num_steps + 1))
        option[:, -1] = np.maximum(np.zeros(self.num_steps + 1), (self.underlyer[:, -1] - self.K))
        for i in range(self.num_steps - 1, -1, -1):
            for j in range(i + 1):
                option[j, i] = np.maximum(
                    (self.p * option[j, i + 1] + self.q * option[j + 1, i + 1]) / self.R, self.underlyer[j, i] - self.K)
        return option[0, 0]

    def eur_put_price(self):
        option = np.zeros((self.num_steps + 1, self.num_steps + 1))
        option[:, -1] = np.maximum(np.zeros(self.num_steps + 1), (self.K - self.underlyer[:, -1]))
        for i in range(self.num_steps - 1, -1, -1):
            for j in range(i + 1):
                option[j, i] = (self.p * option[j, i + 1] + self.q * option[j + 1, i + 1]) / self.R
        return option[0, 0]

    def am_put_price(self):
        option = np.zeros((self.num_steps + 1, self.num_steps + 1))
        option[:, -1] = np.maximum(np.zeros(self.num_steps + 1), (self.K - self.underlyer[:, -1]))
        for i in range(self.num_steps - 1, -1, -1):
            for j in range(i + 1):
                option[j, i] = (self.p * option[j, i + 1] + self.q * option[j + 1, i + 1]) / self.R
                option[j, i] = np.maximum(
                    (self.p * option[j, i + 1] + self.q * option[j + 1, i + 1]) / self.R, self.K - self.underlyer[j, i])
        return option[0, 0]

    def am_put_early_exercise(self):
        return bool(1 - np.isclose(self.am_put_price(), self.eur_put_price(), 0.01))

    def am_call_early_exercise(self):
        return bool(1 - np.isclose(self.am_call_price(), self.eur_call_price(), 0.01))
