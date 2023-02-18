import math
import cmath
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation
import sigma
import random
import scipy.interpolate
import scipy.integrate
import scipy


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



#Strömungswiderstandskraft
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



# Kinematik

"""alpha = (np.linspace(0, 90, 500))

g = 9.81

v0 = 30.0

x = (v0**2 * np.sin(np.radians(2*alpha)))/g

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

ax.set_xlabel('$alpha$ [°]')
ax.set_ylabel('$x$ [m]')
ax.set_aspect('equal')
ax.grid()
print(alpha)
print(np.arcsin((max(x)*g/(v0**2)))*57.3/2)
ax.plot(alpha, x)
plt.show()"""

#Hundkurve nach Funktion
"""r0_hund = np.array([0.0, 10.0])

# Startposition (x, y) des Menschen [m].
r0_mensch = np.array([0.0, 0.0])

# Vektor der Geschwindigkeit (vx, vy) des Menschen [m/s].
v0_mensch = np.array([2.0, 0.0])

# Betrag der Geschwindigkeit des Hundes [m/s].
v0_hund = 3.0

# Maximale Simulationsdauer [s].
t_max = 500

# Zeitschrittweite [s].
dt = 0.01

# Breche die Simulation ab, wenn der Abstand von Hund und
# Mensch kleiner als epsilon ist.
epsilon = v0_hund * dt

# Wir legen Listen an, um die Simulationsergebnisse zu speichern.
t = [0]
r_hund = [r0_hund]
r_mensch = [r0_mensch]
v_hund = []

# Schleife der Simulation
while True:
    # Berechne den neuen Geschwindigkeitsvektor des Hundes.
    delta_r = r_mensch[-1] - r_hund[-1]
    v = v0_hund * delta_r / np.linalg.norm(delta_r)
    v_hund.append(v)

    # Beende die Simulation, wenn der Abstand von Hund und
    # Mensch klein genug ist oder die maximale Simulationszeit
    # überschritten ist.
    if (np.linalg.norm(delta_r) < epsilon) or (t[-1] > t_max):
        break

    # Berechne die neue Position von Hund und Mensch und die
    # neue Zeit.
    r_hund.append(r_hund[-1] + dt * v)
    r_mensch.append(r_mensch[-1] + dt * v0_mensch)
    t.append(t[-1] + dt)

# Wandele die Listen in Arrays um. Die Zeilen entsprechen den
# Zeitpunkten und die Spalten entsprechen den Koordinaten.
t = np.array(t)
r_hund = np.array(r_hund)
v_hund = np.array(v_hund)
r_mensch = np.array(r_mensch)

# Berechne den Beschleunigungsvektor des Hundes für alle
# Zeitpunkte. Achtung das Array a_hund hat eine Zeile weniger,
# als es Zeitpunkte gibt.
a_hund = (v_hund[1:, :] - v_hund[:-1, :]) / dt

# Erzeuge eine Figure und ein Axes-Objekt.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_xlim(-0.2, 15)
ax.set_ylim(-0.2, 10)
ax.set_aspect('equal')
ax.grid()
k = np.linalg.norm(v0_mensch)/v0_hund
y0 = r0_hund[1]

y = np.linspace(0, y0, 100)

x = (y0/2)*(
        (1 - (y / y0) ** (1 - k)) / (1 - k) -
        (1 - (y / y0) ** (1 + k)) / (1 + k))
ax.plot(x, y, '--k')
plot_bahn_hund, = ax.plot([], [])
plot_hund, = ax.plot([], [], 'o', color='blue')
plot_mensch, = ax.plot([], [], 'o', color='red')
style = mpl.patches.ArrowStyle.Simple(head_length=6, head_width=3)
pfeil_v = mpl.patches.FancyArrowPatch((0, 0), (0, 0), color='red', arrowstyle=style)
pfeil_a = mpl.patches.FancyArrowPatch((0, 0), (0, 0), color='black', arrowstyle=style)
ax.add_patch(pfeil_v)
ax.add_patch(pfeil_a)
def update(n):
    pfeil_v.set_positions(r_hund[n], r_hund[n] + v_hund[n])
    if n < len(a_hund):
        pfeil_a.set_positions(r_hund[n], r_hund[n] + a_hund[n])
    plot_hund.set_data(r_hund[n])
    plot_mensch.set_data(r_mensch[n])
    plot_bahn_hund.set_data(r_hund[:n + 1, 0], r_hund[:n + 1, 1])
    return plot_bahn_hund, plot_hund, plot_mensch, pfeil_v, pfeil_a
ani = mpl.animation.FuncAnimation(fig, update, frames=t.size, interval=30, blit=True)


plt.show()"""


#Handkurve auf Kreis
"""R = 3.0
omega = 1.0

r0_hund = np.array([25.0, 0.0])

# Startposition (x, y) des Menschen [m].
r0_mensch = np.array([3.0, 0.0])

# Vektor der Geschwindigkeit (vx, vy) des Menschen [m/s].
v0_mensch = np.array([2.0, 0.0])

# Betrag der Geschwindigkeit des Hundes [m/s].
v0_hund = 3.0

# Maximale Simulationsdauer [s].
t_max = 500

# Zeitschrittweite [s].
dt = 0.01

# Breche die Simulation ab, wenn der Abstand von Hund und
# Mensch kleiner als epsilon ist.
epsilon = v0_hund * dt

# Wir legen Listen an, um die Simulationsergebnisse zu speichern.
t = [0]
r_hund = [r0_hund]
r_mensch = [r0_mensch]
v_hund = []

# Schleife der Simulation
while True:
    # Berechne den neuen Geschwindigkeitsvektor des Hundes.
    delta_r = r_mensch[-1] - r_hund[-1]
    v = v0_hund * delta_r / np.linalg.norm(delta_r)
    v_hund.append(v)

    # Beende die Simulation, wenn der Abstand von Hund und
    # Mensch klein genug ist oder die maximale Simulationszeit
    # überschritten ist.
    if (np.linalg.norm(delta_r) < epsilon) or (t[-1] > t_max):
        break

    # Berechne die neue Position von Hund und Mensch und die
    # neue Zeit.
    r_hund.append(r_hund[-1] + dt * v)
    r_mensch.append([R*np.cos(omega*t[-1]), R*np.sin(omega*t[-1])])
    t.append(t[-1] + dt)

# Wandele die Listen in Arrays um. Die Zeilen entsprechen den
# Zeitpunkten und die Spalten entsprechen den Koordinaten.
t = np.array(t)
r_hund = np.array(r_hund)
v_hund = np.array(v_hund)
r_mensch = np.array(r_mensch)

# Berechne den Beschleunigungsvektor des Hundes für alle
# Zeitpunkte. Achtung das Array a_hund hat eine Zeile weniger,
# als es Zeitpunkte gibt.
a_hund = (v_hund[1:, :] - v_hund[:-1, :]) / dt

# Erzeuge eine Figure und ein Axes-Objekt.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_xlim(-10, 30)
ax.set_ylim(-10, 10)
ax.set_aspect('equal')
ax.grid()

plot_bahn_hund, = ax.plot([], [])
plot_bahn_mensch, = ax.plot([], [])
plot_hund, = ax.plot([], [], 'o', color='blue')
plot_mensch, = ax.plot([], [], 'o', color='red')
style = mpl.patches.ArrowStyle.Simple(head_length=6, head_width=3)
pfeil_v = mpl.patches.FancyArrowPatch((0, 0), (0, 0), color='red', arrowstyle=style)
pfeil_a = mpl.patches.FancyArrowPatch((0, 0), (0, 0), color='black', arrowstyle=style)
ax.add_patch(pfeil_v)
ax.add_patch(pfeil_a)
def update(n):
    pfeil_v.set_positions(r_hund[n], r_hund[n] + v_hund[n])
    if n < len(a_hund):
        pfeil_a.set_positions(r_hund[n], r_hund[n] + a_hund[n])
    plot_hund.set_data(r_hund[n])
    plot_mensch.set_data(r_mensch[n])
    plot_bahn_hund.set_data(r_hund[:n + 1, 0], r_hund[:n + 1, 1])
    plot_bahn_mensch.set_data(r_mensch[:n + 1, 0], r_mensch[:n + 1, 1])
    return plot_bahn_hund, plot_hund, plot_mensch, pfeil_v, pfeil_a, plot_bahn_mensch
ani = mpl.animation.FuncAnimation(fig, update, frames=t.size, interval=30, blit=True)


plt.show()"""



#Boot
"""v_betrag = 5.0

v_stroemung = np.array([0.0, -15.0])

r0_boot = np.array([-20.0, -20.0])
r0_ziel = np.array([10, 3.0])
v_boot = []
d_t = 0.01
t_max = 500
mindestabstand = v_betrag * d_t

r_boot = [r0_boot]
r_ziel = [r0_ziel]

t = [0]

while True :
    r_boot_ziel = r_boot[-1] - r_ziel[-1]
    abstand = np.linalg.norm(r_boot_ziel)
    v = (v_betrag * r_boot_ziel / (-abstand))
    v_boot.append(v)
    if (abstand < mindestabstand) or (t[-1] > t_max):
        break
    v_vector = v_boot[-1] - v_stroemung
    r_boot.append(r_boot[-1] + v * d_t)
    t.append(t[-1] + d_t)

t = np.array(t)
r_boot = np.array(r_boot)
v_boot = np.array(v_boot)
v_vector = np.array(v_vector)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('$x$ [m]')
ax.set_ylabel('$y$ [m]')
ax.set_xlim(-40, 40)
ax.set_ylim(-40, 40)
ax.set_aspect('equal')
ax.grid()



plot_bahn_boot, = ax.plot([], [])
plot_boot, = ax.plot([], [], 'o', color='blue')
plot_ziel, = ax.plot([], [], 'o', color='red')
style = mpl.patches.ArrowStyle.Simple(head_length=6, head_width=3)
pfeil_v_boot = mpl.patches.FancyArrowPatch((0, 0), (0, 0), color='black', arrowstyle=style)
pfeil_v_vector = mpl.patches.FancyArrowPatch((0, 0), (0, 0), color='red', arrowstyle=style)
pfeil_v_stroemung = mpl.patches.FancyArrowPatch((0, 0), (0, 0), color='blue', arrowstyle=style)

ax.add_patch(pfeil_v_boot)
ax.add_patch(pfeil_v_stroemung)
ax.add_patch(pfeil_v_vector)

def update(n):
    pfeil_v_boot.set_positions(r_boot[n], r_boot[n] + v_boot[n])
    pfeil_v_stroemung.set_positions(r_boot[n], r_boot[n] + v_stroemung)
    pfeil_v_vector.set_positions(r_boot[n], r_boot[n] + v_vector)
    plot_boot.set_data(r_boot[n])
    plot_ziel.set_data(r_ziel[0])
    plot_bahn_boot.set_data(r_boot[:n + 1, 0], r_boot[:n + 1, 1])
    return pfeil_v_boot, pfeil_v_stroemung, pfeil_v_vector, plot_boot, plot_ziel, plot_bahn_boot

ani = mpl.animation.FuncAnimation(fig, update, frames=t.size, interval = 30, blit=True)

plt.show()"""



#Interpolate

"""messwerte_h = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                        11.02, 15, 20.06, 25, 32.16, 40])
messwerte_rho = np.array([1.225, 1.112, 1.007, 0.909, 0.819, 0.736,
                          0.660, 0.590, 0.526, 0.467, 0.414, 0.364,
                          0.195, 0.0880, 0.0401, 0.0132, 0.004])

# Erzeuge die Interpolationsfunktionen.
interp_cubic = scipy.interpolate.interp1d(messwerte_h,
                                          messwerte_rho,
                                          kind='cubic')
interp_linear = scipy.interpolate.interp1d(messwerte_h,
                                           messwerte_rho,
                                           kind='linear')
interp_nearest = scipy.interpolate.interp1d(messwerte_h,
                                            messwerte_rho,
                                            kind='nearest')

# Erzeuge ein fein aufgelöstes Array von Höhen.
h = np.linspace(0, np.max(messwerte_h), 1000)

# Erzeuge eine Figure und eine Axes.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('$h$ [km]')
ax.set_ylabel('$\\rho$ [kg/m³]')
ax.set_xlim(10, 40)
ax.set_ylim(0, 0.4)
ax.grid()

# Plotte die Messdaten und die drei Interpolationsfunktionen.
ax.plot(messwerte_h, messwerte_rho, 'or', label='Messung', zorder=5)
ax.plot(h, interp_nearest(h), '-k', label='nearest')
ax.plot(h, interp_linear(h), '-b', label='linear')
ax.plot(h, interp_cubic(h), '-r', label='cubic')
ax.legend()

# Zeige die Grafik an.
plt.show()"""



#Luftauftrieb
"""scal_v = 0.1
scal_a = 0.1
h = 2.0
teta = math.radians(30)
betrag_v0 = 30
cwA = 0.45 * math.pi * 20e-3 ** 2
m = 2.7e-3
g = 9.81
rho = 1.225
hs = 8.4e3

r0 = np.array([0, h])
v0 = np.array([betrag_v0 * math.cos(teta), betrag_v0 * math.sin(teta)])

y0 = h 



def F(y, v):
    Fr = -0.5 * rho * np.e**(-y/hs) * cwA * np.linalg.norm(v) * v
    Fg = m * g * np.array([0, -1])
    return Fg + Fr


def dgl(t, u):
    r, v = np.split(u, 2)
    y = r[1]
    return np.concatenate([v, F(y, v)/m])

u0 = np.concatenate((r0, v0))

def aufprall(t, u):
    r, v = np.split(u, 2)
    return r[1]

aufprall.terminal = True

result = scipy.integrate.solve_ivp(dgl, [0, np.inf], u0, events=aufprall, dense_output=True)

t_stuetz = result.t

r_stuetz, v_stuetz = np.split(result.y, 2)

t_interp = np.linspace(0, max(t_stuetz), 1000)
r_interp, v_interp = np.split(result.sol(t_interp), 2)
fig = plt.figure(figsize=(9, 4))

# Plotte die Bahnkurve.
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('$x$ [m]')
ax.set_ylabel('$y$ [m]')
ax.set_aspect('equal')
ax.set_xlim(0, 20)
ax.set_ylim(0, 10)
ax.grid()
ax.plot(r_stuetz[0], r_stuetz[1], '.b')
ax.plot(r_interp[0], r_interp[1], '-b')

# Erzeuge einen Punktplot für die Position des Balles.
plot_ball, = ax.plot([], [], 'o', color='red', zorder=4)

# Erzeuge Pfeile für die Geschwindigkeit und die Beschleunigung.
style = mpl.patches.ArrowStyle.Simple(head_length=6, head_width=3)
pfeil_v = mpl.patches.FancyArrowPatch((0, 0), (0, 0),
                                      color='red',
                                      arrowstyle=style, zorder=3)
pfeil_a = mpl.patches.FancyArrowPatch((0, 0), (0, 0),
                                      color='black',
                                      arrowstyle=style, zorder=3)
ax.add_patch(pfeil_v)
ax.add_patch(pfeil_a)

# Erzeuge Textfelder für die Anzeige des aktuellen
# Geschwindigkeits- und Beschleunigungsbetrags.
text_t = ax.text(2.1, 1.5, '', color='blue')
text_v = ax.text(2.1, 1.1, '', color='red')
text_a = ax.text(2.1, 0.7, '', color='black')

def update(n):
    t = t_interp[n]
    r = r_interp[:, n]
    v = v_interp[:, n]
    y = r[1]
    a = F(y, v_interp[:, n]) / m
    plot_ball.set_data(r)
    pfeil_v.set_positions(r, r + scal_v * v)
    pfeil_a.set_positions(r, r + scal_a * a)
    text_t.set_text(f'$t$ = {t:.2f} s')
    text_v.set_text(f'$v$ = {np.linalg.norm(v):.1f} m/s')
    text_a.set_text(f'$a$ = {np.linalg.norm(a):.1f} m/s²')
    print(y)
    return plot_ball, pfeil_v, pfeil_a, text_v, text_a, text_t


ani = mpl.animation.FuncAnimation(fig, update, frames=t_interp.size,
                                  interval=30, blit=True)
plt.show()"""




#Schiefer Wurf

"""h = 100.0
scal_v = 0.1
scal_a = 0.1
cwA = 0.45 * math.pi * 20e-3 ** 2
m = 2.7e-3
g = 9.81
rho = 1.225
hs = 8.4e3

r0 = np.array([0, 1])
v0 = np.array([0, 44.3])

def F(v):
    Fg = m * g * np.array([0, -1])
    #Fr = -0.5 * rho * cwA * np.linalg.norm(v) * v
    Fl = np.array([-0.0005, 0])
    return Fg + Fl

def dgl(t, u):
    r, v = np.split(u, 2)
    return np.concatenate([v, F(v)/m])

u0 = np.concatenate((r0, v0))

def aufprall(t, u):
    r, v = np.split(u, 2)
    return r[1]

aufprall.terminal = True


result = scipy.integrate.solve_ivp(dgl, [0, np.inf], u0, events=aufprall, dense_output=True)

t_stuetz = result.t

r_stuetz, v_stuetz = np.split(result.y, 2)

t_interp = np.linspace(0, max(t_stuetz), 1000)

r_interp, v_interp = np.split(result.sol(t_interp), 2)

fig = plt.figure(figsize=(9, 4))

# Plotte die Bahnkurve.
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('$x$ [m]')
ax.set_ylabel('$y$ [m]')
ax.set_aspect('equal')
ax.set_xlim(-20, 120)
ax.set_ylim(0, 120)
ax.grid()
ax.plot(r_stuetz[0], r_stuetz[1], '.b')
ax.plot(r_interp[0], r_interp[1], '-b')

# Erzeuge einen Punktplot für die Position des Balles.
plot_ball, = ax.plot([], [], 'o', color='red', zorder=4)

# Erzeuge Pfeile für die Geschwindigkeit und die Beschleunigung.
style = mpl.patches.ArrowStyle.Simple(head_length=6, head_width=3)
pfeil_v = mpl.patches.FancyArrowPatch((0, 0), (0, 0),
                                      color='red',
                                      arrowstyle=style, zorder=3)
pfeil_a = mpl.patches.FancyArrowPatch((0, 0), (0, 0),
                                      color='black',
                                      arrowstyle=style, zorder=3)
ax.add_patch(pfeil_v)
ax.add_patch(pfeil_a)

# Erzeuge Textfelder für die Anzeige des aktuellen
# Geschwindigkeits- und Beschleunigungsbetrags.
text_t = ax.text(70, 100.5, '', color='blue')
text_v = ax.text(70, 80.5, '', color='red')
text_a = ax.text(70, 60.7, '', color='black')

def update(n):
    t = t_interp[n]
    r = r_interp[:, n]
    v = v_interp[:, n]
    a = F(v_interp[:, n]) / m
    plot_ball.set_data(r)
    pfeil_v.set_positions(r, r + scal_v * v)
    pfeil_a.set_positions(r, r + scal_a * a)
    text_t.set_text(f'$t$ = {t:.2f} s')
    text_v.set_text(f'$v$ = {np.linalg.norm(v):.1f} m/s')
    text_a.set_text(f'$a$ = {np.linalg.norm(a):.1f} m/s²')
    return plot_ball, pfeil_v, pfeil_a, text_v, text_a, text_t


ani = mpl.animation.FuncAnimation(fig, update, frames=t_interp.size,
                                  interval=30, blit=True)
plt.show()"""



"""T = 5.0
dt = 0.005
# Anfangsort [m].
r0 = np.array([0.1, 0, 0])
# Anfangsgeschwindigkeit [m/s].
v0 = np.array([0.0, 0.0, 0.0])
# Drehzahl der Scheibe [1/s].
f = 1.0

# Vektor der Winkelgeschwindigkeit [1/s].
omega = np.array([0, 0, 2 * math.pi * f])


def dgl(t, u):
    r, v = np.split(u, 2)

    # Berechne die Coriolisbeschleunigung.
    a_c = - 2 * np.cross(omega, v)

    # Berechne die Zentrifugalbeschleunigung.
    a_z = - np.cross(omega, np.cross(omega, r))

    # Berechne die Gesamtbeschleunigung.
    a = a_c + a_z

    return np.concatenate([v, a])


# Lege den Zustandsvektor zum Zeitpunkt t=0 fest.
u0 = np.concatenate((r0, v0))

# Löse die Bewegungsgleichung numerisch.
result = scipy.integrate.solve_ivp(dgl, [0, T], u0,
                                   t_eval=np.arange(0, T, dt))
t = result.t
r, v = np.split(result.y, 2)

# Erzeuge eine Figure und eine Axes.
fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(1, 1, 1)
ax.set_aspect('equal')
ax.set_xlabel('$x$ [m]')
ax.set_ylabel('$y$ [m]')
ax.grid()

# Plotte die Bahnkurve in der Aufsicht.
plot_bahn, = ax.plot(r[0], r[1], '-b', zorder=3)

# Erzeuge eine Punktplot für die Position des Körpers.
plot_punkt, = ax.plot([], [], 'o', zorder=5,
                      color='red', markersize=10)


def update(n):
    # Aktualisiere die Position des Körpers.
    plot_punkt.set_data(r[0:2, n])

    # Plotte die Bahnkurve bis zum aktuellen Zeitpunkt.
    plot_bahn.set_data(r[0:2, :n + 1])

    return plot_punkt, plot_bahn


# Erzeuge das Animationsobjekt und starte die Animation.
ani = mpl.animation.FuncAnimation(fig, update, frames=t.size,
                                  interval=30, blit=True)
plt.show()"""





#Das keplersche Gesetz
"""tag = 24 * 60 *60
jahr = tag * 365.25
AE = 1.495978707e11
scal_a = 20
scal_v = 1e-5
t_max = 10 * jahr
dt = 0.5 * tag
r0 = np.array([152.10e9, 0.0])
v0 = np.array([0.0, 15e3])
r1 = np.array([170.10e9, 0.0])
v1 = np.array([0.0, 13.415e3])
dN = 40
M = 1.9885e30
G = 6.6743e-11
T = 3 * jahr
def dgl(t, u):

    r, v = np.split(u, 2)
    a = - G * M * r / np.linalg.norm(r) ** 3
    return np.concatenate([v, a])

u0 = np.concatenate((r0, v0))
u1 = np.concatenate((r1, v1))

result0 = scipy.integrate.solve_ivp(dgl, [0, T], u0, rtol=1e-9,
                                   t_eval=np.arange(0, T, dt))
result1 = scipy.integrate.solve_ivp(dgl, [0, T], u1, rtol=1e-9,
                                   t_eval=np.arange(0, T, dt))

t = result0.t

t1 = result1.t
r, v = np.split(result0.y, 2)
rp, vp = np.split(result1.y, 2)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_aspect('equal')
ax.set_xlabel('$x$ [AE]')
ax.set_ylabel('$y$ [AE]')
ax.set_xlim(-0.2, 1.1)
ax.set_ylim(-0.6, 0.6)
ax.grid()




ax.plot(r[0] / AE, r[1] / AE, '-b')


ax.plot(rp[0] / AE, rp[1] / AE, '-r')


plot_planet, = ax.plot([], [], 'o', color='red')
plot_sonne, = ax.plot([0], [0], 'o', color='gold')
plot_planet1, = ax.plot([], [], 'o', color='green')

style = mpl.patches.ArrowStyle.Simple(head_length=6, head_width=3)
pfeil_v = mpl.patches.FancyArrowPatch((0, 0), (0, 0), color='red',
                                      arrowstyle=style)
pfeil_a = mpl.patches.FancyArrowPatch((0, 0), (0, 0), color='black',
                                      arrowstyle=style)
pfeil_vp = mpl.patches.FancyArrowPatch((0, 0), (0, 0), color='red',
                                      arrowstyle=style)
pfeil_ap = mpl.patches.FancyArrowPatch((0, 0), (0, 0), color='black',
                                      arrowstyle=style)




text_t = ax.text(0.01, 0.95, '', color='blue',
                 transform=ax.transAxes)
text_x = ax.text(0.01, 0.90, '', color='blue',
                 transform=ax.transAxes)
text_y = ax.text(0.01, 0.85, '', color='blue',
                 transform=ax.transAxes)

flaeche = mpl.patches.Polygon([[0, 0], [0, 0]], closed=True,
                              alpha=0.5, facecolor='red')

flaeche1 = mpl.patches.Polygon([[0, 0], [0, 0]], closed=True,
                              alpha=0.5, facecolor='green')
ax.add_artist(flaeche)
ax.add_artist(flaeche1)

def polygon_flaeche(x, y):
    return 0.5 * np.abs((y + np.roll(y, 1)) @ (x - np.roll(x, 1)))



def update(n):
   





    plot_planet.set_data(r[:, n] / AE)
    
    plot_planet1.set_data(rp[:, n] / AE)
    
    
    if n >= dN:
            # Erzeuge ein (dN + 2) x 2 - Array. Als ersten Punkt
            # enthält dies die Position (0, 0) der Sonne und die
            # weiteren Punkte sind die dN + 1 Punkte der Bahnkurve
            # des Planeten.
        xy = np.zeros((dN + 2, 2))
        xy[1:, :] = r[:, (n - dN):(n + 1)].T / AE
        flaeche.set_xy(xy)

        xy1 = np.zeros((dN + 2, 2))
        xy1[1:, :] = rp[:, (n - dN):(n + 1)].T / AE
        flaeche1.set_xy(xy1)

        A = polygon_flaeche(xy[:, 0], xy[:, 1])
        A1 = polygon_flaeche(xy1[:, 0], xy1[:, 1])
        text_x.set_text(f'$Ax$ = {A} AE**2')
        text_y.set_text(f'$Ay$ = {A1} AE**2')
    else:

        flaeche.set_xy([[0, 0], [0, 0]])
        flaeche1.set_xy([[0, 0], [0, 0]])
        text_x.set_text(f'')
        
        text_y.set_text(f'')

    
    text_t.set_text(f't = {t[n] / tag:.0f} d')


    

    return plot_planet,  text_t, plot_planet1, text_x, text_y, flaeche

ani = mpl.animation.FuncAnimation(fig, update,
                                  interval=30, frames=t.size)

plt.show()"""