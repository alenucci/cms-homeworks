import numpy as np
from numba import f8, i8, jitclass, njit, prange


def getconf(N, thermalize=True):
    rho = 0.7
    L = np.cbrt(N / rho)
    pos = np.random.rand(N, 3) * L
    vel = np.random.rand(N, 3) - 0.5
    vel -= vel.mean(0, keepdims=True)
    vel *= np.sqrt(2 * N / (vel**2).sum())
    if thermalize:
        MD(pos, vel).update(0.002, 1)
        vel *= np.sqrt(2 * N / (vel**2).sum())
    return pos, vel


@njit(parallel=True, nogil=True, fastmath=True)
def autocorr(data):
    res, var = data - data.mean(), data.var()
    tau, c_t, t = 0.5, 0, 1
    while c_t >= 0:
        tau += c_t
        c_t = (res[t:] * res[:-t]).mean() / var
        t += 1
    return tau


types = [
    ("N", i8),
    ("pos", f8[:, :]),
    ("vel", f8[:, :]),
    ("dt", f8),
    ("L", f8),
    ("r_cut", f8),
    ("V_rc", f8),
    ("V_rc_prime", f8),
]


@jitclass(types)
class MD:
    def __init__(self, pos, vel):
        self.N = pos.shape[0]
        self.pos, self.vel = pos, vel
        rho = 0.7
        self.L = (self.N / rho)**(1 / 3)
        self.r_cut = self.L / 2
        self.V_rc = (np.exp(-self.r_cut) / self.r_cut**2 +
                     np.exp(-2 * self.r_cut) / 2)
        self.V_rc_prime = -(np.exp(-2 * self.r_cut) + np.exp(-self.r_cut) *
                            (1 + 2 / self.r_cut) / self.r_cut**2)

    def _calc_F_U(self, forces, force_indices, part_forces):
        U, k = 0.0, 0
        for i in range(self.N):
            for j in range(i):
                vect_r = self.pos[i] - self.pos[j]
                vect_r -= self.L * np.rint(vect_r / self.L)
                norm_r = np.linalg.norm(vect_r)
                if norm_r < self.r_cut:
                    e = np.exp(-norm_r)
                    forces[k + j] = -(vect_r / norm_r) * (
                        e * (e + (1 + 2 / norm_r) / norm_r ** 2)
                        + self.V_rc_prime)  # yapf: disable
                    U += (e * (1 / norm_r**2 + e / 2) - self.V_rc -
                          (norm_r - self.r_cut) * self.V_rc_prime)
                else:
                    forces[k + j] = 0
            k += i
        for part in range(self.N):
            part_forces[part] = (
                forces[force_indices[part, part:]].sum(axis=0) -
                forces[force_indices[part, :part]].sum(axis=0))
        return U

    def update(self, dt, t_max):
        force_indices = np.empty((self.N, self.N - 1), np.int64)
        rng = np.arange(self.N - 1)
        cum_rng = rng.cumsum()
        for part in range(self.N):
            force_indices[part] = np.concatenate(
                (cum_rng[part - 1] + rng[:part], cum_rng[part:] + part))
        forces = np.empty((self.N * (self.N - 1) // 2, 3))
        part_forces = np.empty((2, self.N, 3))

        steps = np.int(t_max / dt)
        U, P = np.empty(steps), np.empty(steps)
        E, T = np.empty(steps), np.empty(steps)

        self._calc_F_U(forces, force_indices, part_forces[1])
        for t in range(steps):
            self.pos += self.vel * dt + part_forces[(t + 1) % 2] * dt**2 / 2
            # self.pos %= self.L
            U[t] = self._calc_F_U(forces, force_indices, part_forces[t % 2])
            self.vel += part_forces.sum(axis=0) * dt / 2
            K = np.sum(self.vel**2) / 2
            E[t] = U[t] + K
            T[t] = 2 * K / (3 * self.N)
            P[t] = (self.N * T[t] + (self.pos * part_forces[t % 2]).sum() /
                    (3 * self.N)) / self.L**3
        return U, P, E, T


@njit(parallel=True, nogil=True, fastmath=True, cache=True)
def point(pos, vel, dt_values, t_max):
    dt_values = np.array(dt_values)
    dt_max, n_obs = dt_values.max(), 4
    rec_len, n_sim = np.int(t_max / dt_max), len(dt_values)
    obs = np.empty((n_sim, n_obs, rec_len))
    tau, err = np.empty((n_sim, n_obs)), np.empty((n_sim, n_obs))
    for i in prange(n_sim):
        for j, a in enumerate(
                MD(pos.copy(), vel.copy()).update(dt_values[i], t_max)):
            time_scale = np.int(dt_max / dt_values[i])
            obs[i, j] = a[::time_scale][:rec_len]
            tau[i, j] = autocorr(a[time_scale * 30:])
            err[i, j] = a.std() * np.sqrt(
                2 * tau[i, j] * dt_values[i] / rec_len)
    return obs[:, 0], obs[:, 1], obs[:, 2], obs[:, 3], tau, err


@njit(parallel=True, nogil=True, fastmath=True)
def unstable(pos, vel, dt_values, t_max):
    dt_max, n_sim = dt_values.max(), len(dt_values)
    E = np.empty((n_sim, np.int(t_max / dt_max)))
    for i in prange(n_sim):
        E[i] = MD(pos.copy(), vel.copy()).update(  # yapf: disable
            dt_values[i], t_max)[2, ::np.int(dt_max / dt_values[i])]
    return E


if __name__ == "__main__":
    point(*getconf(70), [0.002, 0.006], 10)
