import numpy as np
from numba import f8, jitclass, njit, prange, i8


@njit(parallel=True, nogil=True, fastmath=True)
def getconf(N, rho=0.4, trials=10):
    L = (N / rho)**(1 / 3)
    conf = np.random.rand(trials, N, 3) * L
    n_min, count = N, 0
    for trial in range(trials):
        overlapping_couples = 0
        for i in range(N):
            for j in range(i):
                r = conf[trial, i] - conf[trial, j]
                if np.linalg.norm(r - L * np.rint(r / L)) < 1:
                    overlapping_couples += 1
        if overlapping_couples < n_min:
            n_min, chosen_conf = overlapping_couples, trial
    print('Starting with', n_min, 'overlaps')
    mc_temp = MC(conf[chosen_conf], rho, 1, 1)
    while mc_temp.overlap():
        mc_temp.update(1)
        count += 1
    print('Configuration ready in', count, 'iterations')
    return conf[chosen_conf]


@njit(parallel=True, nogil=True, fastmath=True)
def autocorr(data, corr_len):
    autocorr_func = np.empty(corr_len)
    autocorr_func[0], res, var = 1, data - data.mean(), data.var()
    for t in prange(1, corr_len):
        autocorr_func[t] = (res[t:] * res[:-t]).mean() / var
    return autocorr_func


params = [('delta', f8), ('r_cut', f8), ('eps', f8), ('N', i8), ('L', f8),
          ('pos', f8[:, :])]


@jitclass(params)
class MC:
    def __init__(self, pos, rho=0.4, delta=1, r_cut=1):
        self.pos = pos
        self.delta = delta
        self.r_cut = r_cut
        self.eps = 0.5
        self.N = pos.shape[0]
        self.L = (self.N / rho)**(1 / 3)

    def _dist(self, a, b):
        r = a - b
        return np.linalg.norm(r - self.L * np.rint(r / self.L))

    def _E(self, r, upd=True):
        if r < 1:
            return np.inf
        else:
            if upd:
                return -self.eps * (r**(-6) - self.r_cut**(-6))
            else:
                return -self.eps * r**(-6)

    def overlap(self):
        for i in range(self.N):
            for j in range(i):
                if self._dist(self.pos[i], self.pos[j]) < 1:
                    return True
        return False

    def update(self, steps):
        accepted_moves, E = 0, np.zeros(steps + 1)
        for i in range(self.N):
            for j in range(i):
                E[0] += self._E(self._dist(self.pos[i, :], self.pos[j, :]),
                                upd=False)
        distances = np.empty((self.N * (self.N - 1) // 2, steps))
        for t in range(steps):
            for part in range(self.N):
                trial_pos = (self.pos[part, :] + self.delta *
                             (np.random.ranf(3) - 0.5)) % self.L
                E_prev, E_last = 0, 0
                for neighbours in np.hstack(  # yapf: disble
                    (np.arange(part), np.arange(part + 1, self.N))):
                    dist_prev = self._dist(self.pos[part, :],
                                           self.pos[neighbours, :])
                    dist_last = self._dist(trial_pos, self.pos[neighbours, :])
                    if dist_prev <= self.r_cut:
                        E_prev += self._E(dist_prev)
                    if dist_last <= self.r_cut:
                        E_last += self._E(dist_last)
                delta_E = E_last - E_prev
                boltzmann_factor = np.exp(-delta_E)
                if (boltzmann_factor >= 1
                        or np.random.ranf() < boltzmann_factor
                        or delta_E == np.nan):
                    self.pos[part, :] = trial_pos
                    accepted_moves += 1

            k = 0
            for i in range(self.N):
                for j in range(i):
                    r = self._dist(self.pos[i], self.pos[j])
                    distances[k + j, t] = r
                    if r < self.r_cut:
                        E[t + 1] += self._E(r, upd=False)
                k += i

        n_in_bin, R = np.histogram(distances,
                                   bins=300,
                                   range=(0, distances.max()))
        density = n_in_bin / (2 * np.pi * R[1] * R[1:]**2 * 0.4 *
                              (self.N - 1) * steps)
        return E / self.N, accepted_moves / (steps * self.N), density, R[1:]


@njit(parallel=True, nogil=True, fastmath=True)
def point_b(conf, delta_values):
    steps, corr_len, n_sim = 5000, 500, len(delta_values)
    E, E_corr = np.empty((n_sim, steps + 1)), np.empty((n_sim, corr_len))
    acc_rate, tau, err = np.empty(n_sim), np.empty(n_sim), np.empty(n_sim)
    mc = [MC(conf.copy(), 0.4, delta, 1) for delta in delta_values]
    for i in prange(n_sim):
        mc[i].r_cut = mc[i].L / 2
        E[i], acc_rate[i], _, _ = mc[i].update(steps)
        E[i] -= 2 * np.pi * mc[i].eps / (3 * (mc[i].L * mc[i].r_cut)**3)
        E_corr[i] = autocorr(E[i], corr_len)
        tau[i] = E_corr[:(E_corr[i] <= 0).nonzero()[0][0], i].sum(0)
        err[i] = E[i].std() * np.sqrt(2 * tau[i] / E.shape[0])
    return E, E_corr, err, acc_rate


@njit(parallel=True, nogil=True, fastmath=True)
def point_c(conf, cutoff_values):
    steps, rho, n_sim = 5000, 0.4, len(cutoff_values)
    E = np.empty((n_sim, steps + 1))
    E_mean, E_tail = np.empty(n_sim), np.empty(n_sim)
    mc = [MC(conf.copy(), rho, 1, cutoff) for cutoff in cutoff_values]
    for i in range(n_sim):
        mc[i].r_cut *= mc[i].L
        E[i], _, _, _ = mc[i].update(steps)
        E_mean[i] = E[i].mean()
        E_tail[i] = (E_mean[i] - 2 * np.pi * rho * mc[i].eps /
                     (3 * mc[i].r_cut**3))
    return E_mean, E_tail


@njit(parallel=True, nogil=True, fastmath=True)
def point_d(conf):
    steps, hist_len, rho, n_sim = 5000, 300, 0.4, len(conf)
    E, density = np.empty((n_sim, steps + 1)), np.empty((n_sim, hist_len))
    R, pressure = np.empty_like(density), np.empty(n_sim)
    mc = [MC(c, rho, 1, 0.5) for c in conf]
    for i in prange(n_sim):
        mc[i].r_cut = mc[i].L / 2
        E[i], _, density[i], R[i] = mc[i].update(steps)
        pressure[i] = (rho + ((2 / 3) * np.pi * rho**2) *
                       (density[i].max() - 2 * mc[i].eps / mc[i].r_cut**3))
    return E, density, R, pressure
