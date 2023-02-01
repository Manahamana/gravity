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