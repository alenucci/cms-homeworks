# %%
import time
from importlib import reload

import holoviews as hv
import numpy as np
from holoviews import opts

import hw3

hv.extension("bokeh", "matplotlib")
opts.defaults(opts.Curve(width=650))

conf = hw3.getconf(70)

# %%
start = time.time()
hw3.MD(*conf).update(0.002, 30.0)
print(time.time() - start)

# %%
dt_values = np.array([0.002, 0.006, 0.018, 0.054, 0.162])
start = time.time()
U, P, E, T, tau, err = hw3.point(*hw3.getconf(70), dt_values[:3])
print(time.time() - start)

time_axe = dt_values[:3].max() * np.arange(U.shape[1])
plot_E = hv.NdOverlay({
    dt_values[i]: hv.Curve((time_axe, E[i]), "time", "energy")
    for i in range(3)
})
plot_T = hv.NdOverlay({
    dt_values[i]: hv.Curve((time_axe, T[i]), "time", "temperature")
    for i in range(3)
})
plot_P = hv.NdOverlay({
    dt_values[i]: hv.Curve((time_axe, P[i]), "time", "pressure")
    for i in range(3)
})
plot_U = hv.NdOverlay({
    dt_values[i]: hv.Curve((time_axe, U[i]), "time", "potential")
    for i in range(3)
})

divergence_E = hv.NdOverlay({
    dt_values[i]: hv.Curve((time_axe, E[i] - E[0]), "time", "energy")
    for i in range(1, 3)
})
divergence_T = hv.NdOverlay({
    dt_values[i]: hv.Curve((time_axe, T[i] - T[0]), "time", "temperature")
    for i in range(1, 3)
})
divergence_P = hv.NdOverlay({
    dt_values[i]: hv.Curve((time_axe, P[i] - P[0]), "time", "pressure")
    for i in range(1, 3)
})

# %%
E_dump = hw3.unstable(*conf, dt_values[3:])
time_axe2 = dt_values[3:].max() * np.arange(E_dump.shape[1])
dump = hv.Curve((time_axe2, E_dump[0])) + hv.Curve((time_axe2, E_dump[1]))


# %%
def V(r):
    return np.exp(-r) / r**2 + np.exp(-2 * r) / 2


def F(r):
    return -np.exp(-r) * (np.exp(-r) + (1 + 2 / r) / r**2)


r, N, rho = np.linspace(0.1, 0.8, 100), 70, 0.7
r_c = np.cbrt(N / rho) / 2
hv.Curve((r, V(r) - V(r_c))) * hv.Curve((r, F(r) - F(r_c)))

# %%
reload(hw3)

# %%
