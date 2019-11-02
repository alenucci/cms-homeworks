# %%
from fractions import Fraction
from importlib import reload

import holoviews as hv
import numpy as np
from holoviews import opts

import hw2

hv.extension('bokeh', 'matplotlib')
opts.defaults(opts.Curve(width=650))

conf = np.loadtxt('60.txt')
conf100, conf200 = np.loadtxt('100.txt'), np.loadtxt('200.txt')

# %%
deltas = [Fraction(1, 6), Fraction(1, 3), 1, 2, 3]
E, E_corr, err, acc_rate = hw2.point_b(conf, np.array(deltas, np.float))
# yapf: disable
print('delta values (in units of d):', [str(d) for d in deltas],
      'means and errors:', E.mean(1), err, 'acceptance ratios:', acc_rate,
      sep='\n')
E_plot = hv.NdOverlay({delta: hv.Curve(E[i, ::40])
                       for i, delta in enumerate(deltas)}).redim(x='t', y='E')
acc_plot = hv.Curve((np.array(deltas, np.float), acc_rate))
E_corr_plot = hv.NdOverlay({delta: hv.Curve(E_corr[i, :300])
                            for i, delta in enumerate(deltas)})
err_plot = hv.Curve((list(map(float, deltas)), err))
# yapf: enable

# %%
cutoffs = [Fraction(3, 8), Fraction(1, 4)]
E_no_tail, E_tail = hw2.point_c(conf, np.array(cutoffs, np.float))
print('Cutoff values (in units of L):', [str(c) for c in cutoffs],
      'Energies with and without tail correction:', E_tail, E_no_tail,
      sep='\n')  # yapf: disable

# %%
E, density, R, pressure = hw2.point_d([conf100, conf200])
E_plot = hv.NdOverlay({n: hv.Curve(E[i]) for i, n in enumerate([100, 200])})
g_R = hv.NdOverlay({n: hv.Curve((R[i], density[i]))
                    for i, n in enumerate([100, 200])})  # yapf: disable

# %%
reload(hw2)
