import math
import cmath
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation
import sigma

#Sinüs Fehler
"""r = np.array([1,2,3,4,5,6,7,8,9])
p = r.reshape(9,1)
r=r*p
print(r)

x = np.linspace(0.01, 45, 500)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlim(np.min(0), np.max(50))
ax.set_ylim(np.min(0), np.max(200))
ax.set_xlabel('winkel')
ax.set_ylabel('fehler')
ax.grid()
plot, = ax.plot([], [])
text = ax.text(0.5, 1.05, '')

sin_x = np.sin(np.degrees(x))
fehler =  (sin_x/x)
plot.set_data(x, fehler)

print(fehler)


plt.show()"""



#Fourier Reihe


"""x = np.linspace(-np.pi, np.pi, 500)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlim(np.min(x), np.max(x))
ax.set_ylim(-2, 2)

ax.grid()
text = ax.text(0.5, 1.05, '')
plot, = ax.plot([], [])

pre_sum = x * 0
def update(n):
    global pre_sum
    global plot
    total = np.sin(np.radians(((2*n)+1)*x))/((2*n)+1)
    pre_sum = total+pre_sum
    sum = (4/np.pi)*pre_sum
    text.set_text(n)
    plot.set_data(x,sum)

    return plot, text

ani = mpl.animation.FuncAnimation(fig, update,interval=3, blit=True)
plt.show()"""


#Takagi Funktion

"""def s(x):
    
    d = x - np.floor(x)
    return np.minimum(d, 1 - d)


def fn(N):

    
    def f(x):
        return sum(s(x * 2 ** n) / 2 ** n for n in range(N))
    return f




x = np.linspace(0, 1, 1000)
y = fn(10)(x)
plt.plot(x, y)
plt.show()"""

#Decimal
"""messwerte = np.array([2.05, 1.99, 2.06, 1.97, 2.01, 2.00, 2.03, 1.97, 2.02, 1.96])
n = messwerte.size

mittelwert = 0
for x in messwerte:
    mittelwert += x
mittelwert /= n

standardabw = 0
for x in messwerte:
    standardabw += (x - mittelwert) ** 2
standardabw = math.sqrt(standardabw / (n - 1))

fehler = standardabw / math.sqrt(n)
print(f'Mittelwert: T = {mittelwert: .5f}s')
print(f'Standardabweichung: sigma = {standardabw:.5f} s')
print(f'Mittlerer Fehler: Delta T = {fehler:.5f} s')"""




"""messwerte_v = np.array([5.8, 7.3, 8.9, 10.6, 11.2], dtype=float)

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
plt.show()"""