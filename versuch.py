import math
import cmath
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation
import scipy.integrate
import sigma
import scipy

messwerte_v = np.array([5.8, 7.3, 8.9, 10.6, 11.2], dtype=float)

messwerte_F = np.array([0.10, 0.15, 0.22, 0.33,  0.36], dtype=float)

fehlerwerte_v = np.array([0.3, 0.3, 0.2, 0.2, 0.1], dtype=float )

fehlerwerte_F = np.array([0.02, 0.02, 0.02, 0.02, 0.02], dtype=float)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('Geschwindigkeit $v$ [m/s]')
ax.set_ylabel('Strömungswiderstandskraft $F$ [N]')
ax.set_yscale('log')
ax.grid()

def fitfunktion(v, b, n):
    return b*(abs(v)**(n))


popt, pcov = scipy.optimize.curve_fit(fitfunktion, messwerte_v, messwerte_F, [5,2], sigma=fehlerwerte_v)

fitwert_b, fitwert_n = popt
fehler_b, fehler_n = np.sqrt(np.diag(pcov))

print('Ergebnis der Kurvenanpassung:')
print(f'b = ({fitwert_b:.8f} +- {fehler_b:.8f}) kg/m')
print(f'n = ({fitwert_n:.2f} +- {fehler_n:.2f})')
v = np.linspace(np.min(messwerte_v), np.max(messwerte_v), 500)
F = fitfunktion(v, fitwert_b, fitwert_n)
ax.plot(v, F, '-')


ax.errorbar(messwerte_v, messwerte_F, yerr=fehlerwerte_F, fmt='.', capsize=2)
plt.show()