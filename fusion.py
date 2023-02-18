import numpy as np
from matplotlib import rc
import matplotlib.pyplot as plt
from scipy.constants import e, k as kB
rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 14})
rc('text', usetex=True)

# To plot using centre-of-mass energies instead of lab-fixed energies, set True
COFM = True

# Reactant masses in atomic mass units (u).
masses = {'D': 2.014, 'T': 3.016, '3He': 3.016}

# Energy grid, 1 – 1000 keV, evenly spaced in log-space.
Egrid = np.logspace(0, 3, 100)

def read_xsec(filename):
    """Read in cross section from filename and interpolate to energy grid."""

    E, xs = np.genfromtxt(filename, comments='#', skip_footer=2, unpack=True)
    if COFM:
        collider, target = filename.split('_')[:2]
        m1, m2 = masses[target], masses[collider]
        E *= m1 / (m1 + m2)

    xs = np.interp(Egrid, E*1.e3, xs*1.e-28)
    return xs

# D + T -> α + n
DT_xs = read_xsec("D_T_-_a_n.txt")

# D + D -> T + p
DDa_xs = read_xsec("D_D_-_T_p.txt")
# D + D -> 3He + n
DDb_xs = read_xsec("D_D_-_3He_n.txt")
# Total D + D fusion cross section is due to equal contributions from the
# above two processes.
DD_xs = DDa_xs + DDb_xs

# D + 3He -> α + p
DHe_xs = read_xsec('D_3He_-_4He_p.txt')

fig, ax = plt.subplots()

ax.grid()
ax.set_xlim(1, 1000)
xticks= np.array([1, 10, 100, 1000])
ax.set_xticks(xticks)
ax.set_xticklabels([str(x) for x in xticks])


ax.set_xlabel('E /keV')

ax.set_ylabel('$\sigma\;/\mathrm{m^2}$')
ax.set_ylim(1.e-32, 1.e-27)

ax.legend()
plt.savefig('fusion-xsecs.png')
plt.show()