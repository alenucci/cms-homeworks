# %%
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set()
sns.set_palette("husl")
sns.set_context("paper")
sns.set_style("whitegrid")
if not os.path.exists("plot"):
    os.mkdir("plot")

data = np.hstack(
    (np.genfromtxt("data1.txt")[:, 1:], np.genfromtxt("data2.txt")[:, 1:])
)
N_max, i_max = data.shape

means = data.mean(0)
var_unbiased = data.var(0) * N_max / (N_max - 1)
print(means, np.sqrt(var_unbiased / N_max), sep="\n")

# %%
colors = sns.color_palette(n_colors=4)
for u in range(i_max):
    plt.xlim(0, N_max)
    sns.scatterplot(
        x=range(N_max), y=data[:, u], marker="|", s=1, color=colors[u]
    )
    plt.savefig("plot/scatter_{}.png".format(u), facecolor="white")
    plt.show()

# %% blocking analysis
blocking_n = 15
blocking_errs = np.empty((blocking_n, i_max))
blocking_errs[0, :] = np.sqrt(var_unbiased / N_max)
U = data
for i in range(1, blocking_n):
    if U.shape[0] % 2 == 0:
        U = 0.5 * (U[::2, :] + U[1::2, :])
    else:
        U = 0.5 * (U[:-1:2, :] + U[1::2, :])
    blocking_errs[i, :] = U.std(0) / np.sqrt(U.shape[0] - 1)
print(blocking_errs[10, :])

plt.xlim(0, blocking_n - 1)
for i in range(i_max):
    sns.lineplot(y=blocking_errs[:, i], x=np.arange(blocking_n))
plt.savefig("plot/blocking.png", facecolor="white")
plt.show()

plt.xlim(0, blocking_n - 1)
for i in range(i_max):
    sns.lineplot(
        y=blocking_errs[:, i] / blocking_errs[0, i], x=np.arange(blocking_n)
    )
plt.savefig("plot/blocking_norm.png", facecolor="white")
plt.show()

# %% autocorrelation analysis
corr_len = 120
res = data - means[np.newaxis, :]
autocorr_funcs = np.empty((corr_len, i_max))
autocorr_funcs[0, :] = 1
for i in range(1, corr_len):
    autocorr_funcs[i, :] = (res[:-i, :] * res[i:, :]).sum(0) / (
        var_unbiased * (N_max - i)
    )
tau = 0.5 + autocorr_funcs[1:100, :].sum(0)
autocorr_errs = np.sqrt(2 * var_unbiased * tau / N_max)
print("tau:", tau, "errors w/ autocorrelation:", autocorr_errs, sep="\n")

plt.xlim(0, corr_len)
for i in range(i_max):
    sns.lineplot(y=autocorr_funcs[:, i], x=np.arange(corr_len))
plt.savefig("plot/autocorr.png", facecolor="white")
plt.show()

plt.xlim(0, 40)
plt.ylim(-3, 0)
tau_plot = 40
sns.regplot(y=np.log(autocorr_funcs[:tau_plot, 0]), x=list(range(tau_plot)))
sns.regplot(y=np.log(autocorr_funcs[:tau_plot, 1]), x=list(range(tau_plot)))
tau_plot = 20
sns.regplot(y=np.log(autocorr_funcs[:tau_plot, 2]), x=list(range(tau_plot)))
sns.regplot(y=np.log(autocorr_funcs[:tau_plot, 3]), x=list(range(tau_plot)))
plt.savefig("plot/log_autocorr.png", facecolor="white")
plt.show()

plt.xlim(0, corr_len)
for i in range(i_max):
    sns.lineplot(
        y=autocorr_funcs[:, i].cumsum(), x=np.arange(autocorr_funcs.shape[0])
    )
plt.savefig("plot/tau", facecolor="white")
plt.show()

# %% jackknife
block_len = 1500
n_blocks = N_max // block_len
jackknife_blocks = np.empty((n_blocks, i_max))
for l in range(n_blocks):
    jackknife_blocks[l, :] = (
        data[l * block_len : (l + 1) * block_len - 1, :].sum(0) / block_len
    )

jackknife_avgs = np.empty_like(jackknife_blocks)
for y in range(n_blocks):
    jackknife_avgs[y, :] = np.delete(jackknife_blocks, y, 0).sum(0) / (
        n_blocks - 1
    )

Kest = means[1:] / means[0]
K_J = jackknife_avgs[:, 1:] / jackknife_avgs[:, 0, np.newaxis]
unbias = 1 - 1 / n_blocks
R = n_blocks * Kest - K_J.sum(0) * unbias
err_K = np.sqrt(((K_J - R) ** 2).sum(0) * unbias)

ind_err = (
    np.sqrt(
        (autocorr_errs[1:] / means[1:]) ** 2
        + (autocorr_errs[0] / means[0]) ** 2
    )
    * R
)
worst_err = (
    np.sqrt((autocorr_errs[1:] / means[1:] + autocorr_errs[0] / means[0]) ** 2)
    * R
)

print(
    f"R:\n{R}",
    f"error with JK:\n{err_K}",
    f"error neglecting correlations:\n{ind_err}",
    f"worst case error:\n{worst_err}",
    sep="\n",
)

n_worst = (
    Kest
    * (np.sqrt(var_unbiased[1:] / means[1:] + var_unbiased[0] / means[0]))
    / np.sqrt(N_max)
)
n_ind = (
    Kest
    * np.sqrt(
        var_unbiased[1:] / means[1:] ** 2 + var_unbiased[0] / means[0] ** 2
    )
    / np.sqrt(N_max)
)

# %%
