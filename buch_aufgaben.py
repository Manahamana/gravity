
import math
import cmath
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation
import sigma
import random
import scipy

"""messwerte_d = np.array([0.000, 0.029, 0.039, 0.064, 0.136, 0.198, 0.247, 0.319, 0.419, 0.511, 0.611, 0.719, 0.800, 0.900, 1.000, 1.100, 1.189])

messwerte_n = np.array([2193, 1691, 1544, 1244, 706, 466, 318, 202, 108, 80, 52, 47, 45, 46, 47, 42, 43], dtype=float)

fehlerwerte_n = np.array([47, 41, 39, 35, 26, 22, 18, 14, 10, 9, 7, 7, 7, 7, 7, 7, 7], dtype=float)



fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('Filterdicke $d$ [mm]')
ax.set_ylabel('Intensität $n$ [1/min]')
ax.set_yscale('log')
ax.grid()
def fitfunktion(x, nu, n0, alpha):
    return nu + n0 * np.exp(-alpha*x)

popt, pcov = scipy.optimize.curve_fit(fitfunktion, messwerte_d, messwerte_n, [40, 2200, 10], sigma=fehlerwerte_n)

fitwert_nu, fitwert_n0, fitwert_alpha = popt
fehler_nu, fehler_n0, fehler_alpha = np.sqrt(np.diag(pcov))
print('Ergebnis der Kurvenanpassung:')
print(f' n_u = ({fitwert_nu:4.0f} +- {fehler_nu:2.0f}) 1/min.')
print(f' n_0 = ({fitwert_n0:4.0f} +- {fehler_n0:2.0f}) 1/min.')
print(f'alpha = ({fitwert_alpha:.2f} +- {fehler_alpha:.2f}) 1/mm.')
d = np.linspace(np.min(messwerte_d), np.max(messwerte_d), 500)
n = fitfunktion(d, fitwert_nu, fitwert_n0, fitwert_alpha)
ax.plot(d, n, '-')


ax.errorbar(messwerte_d, messwerte_n, yerr=fehlerwerte_n, fmt='.', capsize=2)
plt.show()


h = 10.0

betrag_v0 = 9.0

alpha = math.radians(25.0)
g = 9.81

r0 = np.array([0, h])
v0 = betrag_v0 * np.array([math.cos(alpha), math.sin(alpha)])
a = np.array([0, -g])
t_ende = v0[1] / g + math.sqrt((v0[1] / g) ** 2 + 2 * r0[1] / g)

t = np.linspace(0, t_ende, 1000)
t = t.reshape(-1, 1)

r = np.empty((t.size, r0.size))
for i in range(t.size):
    for j in range(r0.size):
        r[i, j] = r0[j] + v0[j] * t[i] + 0.5 * a[j] * t[i]**2

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('$x$ [m]')
ax.set_ylabel('$y$ [m]')
ax.set_aspect('equal')
ax.grid()

ax.plot(r[:, 0], r[:, 1])
plt.show()"""


"""r0_hund = np.array([0.0, 10.0])

r0_mensch = np.array([10.0, 5.0])

v_mensch = np.array([0.0, 0.0])

betrag_v_hund = 3.0

t_max = 500

dt = 0.01
mindestabstand = betrag_v_hund * dt
t = [0]
r_hund = [r0_hund]
r_mensch = [r0_mensch]
v_hund = []
                      
while True:
    r_hund_mensch = r_mensch[-1] - r_hund[-1]
    abstand = np.linalg.norm(r_hund_mensch)
    v_hund.append(betrag_v_hund * r_hund_mensch / abstand)
    if (abstand < mindestabstand) or (t[-1] > t_max):
        break
    r_hund.append(r_hund[-1] + dt * v_hund[-1])
    r_mensch.append(r_mensch[-1] + dt * v_mensch)
    t.append(t[-1] + dt)


t = np.array(t)
r_hund = np.array(r_hund)
v_hund = np.array(v_hund)
r_mensch = np.array(r_mensch)
a_hund = (v_hund[1:] - v_hund[:-1]) / dt
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('$x$ [m]')
ax.set_ylabel('$y$ [m]')
ax.set_xlim(-0.2, 15)
ax.set_ylim(-0.2, 10)
ax.set_aspect('equal')
ax.grid()
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

"""import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation

# Parameter der Simulation.
R = 3.0                      # Radius der Kreisbahn [m].
T = 12.0                     # Umlaufdauer [s].
dt = 0.02                    # Zeitschrittweite [s].
omega = 2 * np.pi / T        # Winkelgeschwindigkeit [1/s].

# Gib das analytische Ergebnis aus.
print(f'Bahngeschwindigkeit:       {R*omega:.3f} m/s')
print(f'Zentripetalbeschleunigung: {R*omega**2:.3f} m/s²')

# Erzeuge ein Array von Zeitpunkten für einen Umlauf.
t = np.arange(0, T, dt)

# Erzeuge ein leeres n x 2 - Arrray für die Ortsvektoren.
r = np.empty((t.size, 2))

# Berechne die Position des Massenpunktes für jeden Zeitpunkt.
r[:, 0] = R * np.cos(omega * t)
r[:, 1] = R * np.sin(omega * t)

# Berechne den Geschwindigkeits- und Beschleunigungsvektor.
v = (r[1:, :] - r[:-1, :]) / dt
a = (v[1:, :] - v[:-1, :]) / dt

# Erzeuge eine Figure und ein Axes-Objekt.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_xlim(-1.2 * R, 1.2 * R)
ax.set_ylim(-1.2 * R, 1.2 * R)
ax.set_aspect('equal')
ax.grid()

# Plotte die Kreisbahn.
plot, = ax.plot(r[:, 0], r[:, 1])

# Erzeuge einen Kreis, der die Position der Masse darstellt.
punkt, = ax.plot([0], [0], 'o', color='blue')
sun, = ax.plot([0],[0], 'o', color='yellow')
# Erzeuge zwei Textfelder für die Anzeige des aktuellen
# Geschwindigkeits- und Beschleunigungsbetrags.
text_v = ax.text(0, 0.2, '', color='red')
text_a = ax.text(0, -0.2, '', color='black')

# Erzeuge Pfeile für die Gewschwindigkeit und die Beschleunigung.
style = mpl.patches.ArrowStyle.Simple(head_length=6,
                                      head_width=3)
arrow_v = mpl.patches.FancyArrowPatch((0, 0), (0, 0),
                                      color='red',
                                      arrowstyle=style)
arrow_a = mpl.patches.FancyArrowPatch((0, 0), (0, 0),
                                      color='black',
                                      arrowstyle=style)

# Füge die Grafikobjekte zur Axes hinzu.
ax.add_artist(arrow_v)
ax.add_artist(arrow_a)


def update(n):
    # Aktualisiere den Geschwindigkeitspfeil und zeige den
    # Geschwindigkeitsbetrag an.
    if n < v.shape[0]:
        arrow_v.set_positions(r[n], r[n] + v[n])
        text_v.set_text(f'v = {np.linalg.norm(v[n]):.3f} m/s')

    # Aktualisiere den Beschleunigungspfeil und zeige den
    # Beschleunigungssbetrag an.
    if n < a.shape[0]:
        arrow_a.set_positions(r[n], r[n] + a[n])
        text_a.set_text(f'a = {np.linalg.norm(a[n]):.3f} m/s²')

    # Aktualisiere die Position des Punktes.
    punkt.set_data(r[n])

    return punkt, arrow_v, arrow_a, text_a, text_v


# Erzeuge das Animationsobjekt und starte die Animation.
ani = mpl.animation.FuncAnimation(fig, update, interval=30,
                                  frames=t.size, blit=True)
plt.show()"""



scal_kraft = 0.002

punkte = np.array([[0, 0], [0, 1.1], [1.2, 0], [2.5, 1.1]])
indizes_stuetz = [0, 1]

staebe = np.array([[0, 2], [1, 2], [2, 3], [1, 3]])
F_ext = np.array([[0, 0], [0, 0], [0, -147.15], [0, -98.1]])
n_punkte, n_dim = punkte.shape
n_staebe = len(staebe)
n_stuetz = len(indizes_stuetz)
n_knoten = n_punkte - n_stuetz
indizes_knoten = list(set(range(n_punkte)) - set(indizes_stuetz))

def ev(i_pkt, i_stb, koord=punkte):
    stb = staebe[i_stb]
    if i_pkt not in stb:
        return np.zeros(n_dim)
    if i_pkt == stb[0]:
        vektor = koord[stb[1]] - koord[i_pkt]
    else:
        vektor = koord[stb[0]] - koord[i_pkt]
    return vektor / np.linalg.norm(vektor)

A = np.empty((n_knoten, n_dim, n_staebe))
for n, k in enumerate(indizes_knoten):
    for i in range(n_staebe):
        A[n, :, i] = ev(k, i)

A = A.reshape(n_knoten * n_dim, n_staebe)
b = -F_ext[indizes_knoten].reshape(-1)
F = np.linalg.solve(A, b)
for i_stuetz in indizes_stuetz:
    for i_stab in range(n_staebe):
        F_ext[i_stuetz] -= F[i_stab] * ev(i_stuetz, i_stab)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(punkte[indizes_knoten, 0], punkte[indizes_knoten, 1], 'bo')
ax.plot(punkte[indizes_stuetz, 0], punkte[indizes_stuetz, 1], 'ro')
for stab, kraft in zip(staebe, F):
    ax.plot(punkte[stab, 0], punkte[stab, 1], color='black')
    position = np.mean(punkte[stab], axis=0)
    annot = ax.annotate(f'{kraft:+.1f} N', position, color='blue')
    annot.draggable(True)

style = mpl.patches.ArrowStyle.Simple(head_length=10, head_width=5)
for p1, kraft in zip(punkte, F_ext):
    p2 = p1 + scal_kraft * kraft
    pfeil = mpl.patches.FancyArrowPatch(p1, p2, color='red',arrowstyle=style,zorder=2)
    ax.add_patch(pfeil)
    annot = ax.annotate(f'{np.linalg.norm(kraft):.1f} N',(p1 + p2) / 2, color='red')
    annot.draggable(True)


for i_stab, stab in enumerate(staebe):
    for i_punkt in stab:
        p1 = punkte[i_punkt]
        p2 = p1 + scal_kraft * F[i_stab] * ev(i_punkt, i_stab)
        pfeil = mpl.patches.FancyArrowPatch(p1, p2, color='blue', arrowstyle=style, zorder=2)
        ax.add_patch(pfeil)

plt.show()