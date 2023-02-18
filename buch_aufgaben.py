
import math
import cmath
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation
import sigma
import random
import scipy.interpolate
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




#Statisch
"""scal_kraft = 0.002

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

plt.show()"""




#Elastisch
"""punkte = np.array([[0, 0], [1.2, 0], [1.2, 4.1], [0, 4.1],[0.6, 0.0]])
n_punkte, n_dim = punkte.shape
scal_kraft = 0.002

indizes_stuetz = [0, 1]
staebe = np.array([[1, 2], [2, 3], [3, 0], [0, 4], [1, 4], [3, 4], [2, 4]])
steifigkeiten = np.array([5.6e6, 5.6e6, 5.6e6, 7.1e3, 7.1e3, 7.1e3, 7.1e3])
n_stuetz = len(indizes_stuetz)
n_knoten = n_punkte - n_stuetz
n_staebe = len(staebe)
F_ext = np.array([[0, 0], [0, 0], [400.0, 0.0], [0, 0], [0, 0]])
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
def laenge(i_stb, koord=punkte):
    i1, i2 = staebe[i_stb]
    return np.linalg.norm(koord[i2] - koord[i1])

def stabkraft(i_stb, koord):
    l0 = laenge(i_stb)
    return steifigkeiten[i_stb] * (laenge(i_stb, koord) - l0) / l0

def gesamtkraft(koord):
    F_ges = F_ext.copy()
    for i_stb, stb in enumerate(staebe):
        for i_pkt in stb:
            F_ges[i_pkt] += (stabkraft(i_stb, koord) * ev(i_pkt, i_stb, koord))
    return F_ges

def funktion_opti(x):
    p = punkte.copy()
    p[indizes_knoten] = x.reshape(n_knoten, n_dim)
    F_ges = gesamtkraft(p)
    F_knoten = F_ges[indizes_knoten]
    return F_knoten.reshape(-1)
result = scipy.optimize.root(funktion_opti, punkte[indizes_knoten])
print(result.message)
print(f'Die Funktion wurde {result.nfev}-mal ausgewertet.')
punkte_neu = punkte.copy()
punkte_neu[indizes_knoten] = result.x.reshape(n_knoten, n_dim)
F = np.zeros(n_staebe)
for i_stab in range(n_staebe):
    F[i_stab] = stabkraft(i_stab, punkte_neu)
    print(F[i_stab])
F_ext[indizes_stuetz] = -gesamtkraft(punkte_neu)[indizes_stuetz]


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(punkte[indizes_knoten, 0], punkte[indizes_knoten, 1], 'bo')
ax.plot(punkte[indizes_stuetz, 0], punkte[indizes_stuetz, 1], 'ro')
for stab, kraft in zip(staebe, F):
    ax.plot(punkte[stab, 0], punkte[stab, 1], color='black')
    position = np.mean(punkte[stab], axis=0)
    annot = ax.annotate(f'{kraft:+.1f} N', position, color='blue')
    annot.draggable(True)

style = mpl.patches.ArrowStyle.Simple(head_length=8, head_width=6)
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

plt.show()"""



#Dynamisch
"""t_max = 20
dt = 0.2

m = 15.0

b = 2.5
x0 = 0
v0 = 10 


t = np.arange(0, t_max, dt)

def F(v):
    return -b * v * np.abs(v)

x = np.empty(t.size)
v = np.empty(t.size)
x[0] = x0
v[0] = v0

for i in range(t.size-1):
    x[i+1] = x[i] + v[i] * dt
    v[i+1] = v[i] + (F(v[i])/m)*dt

erwartet_w =  m / b * np.log(1 + v0 * b / m * t)


fig = plt.figure(figsize=(8, 4))
fig.set_tight_layout(True)

ax_geschw = fig.add_subplot(1, 2, 1)
ax_geschw.set_xlabel('$t$ [s]')
ax_geschw.set_ylabel('$v$ [m/s]')
ax_geschw.grid()
ax_geschw.plot(t, v0 / (1 + v0 * b / m * t), '-b', label='analytisch')
ax_geschw.plot(t, v, '.r', label='simuliert')
ax_geschw.legend()
ax_ort = fig.add_subplot(1, 2, 1)
ax_ort.set_xlabel('$t$ [s]')
ax_ort.set_ylabel('$x$ [m]')
ax_ort.grid()
ax_ort.plot(t, m / b * np.log(1 + v0 * b / m * t),'-b', label='analytisch')
ax_ort.plot(t, x, '.r', label='simuliert')
ax_ort.legend()
ax_abweichung = fig.add_subplot(1, 2, 2)
ax_abweichung.set_ylabel('$x-<x>$ [m]')
ax_abweichung.set_xlabel('$t$ [s]')
ax_abweichung.grid()
ax_abweichung.plot(t, abs(x-erwartet_w))
ax_abweichung.legend()
plt.show()"""


"""t_max = 20
dt = 0.02

m = 15.0

b = 2.5
x0 = 0
v0 = 10 
t = np.arange(0, t_max, dt)
x = np.empty(t.size)
v = np.empty(t.size)
x[0] = x0
v[0] = v0
t = np.arange(0, t_max, dt)
t = np.linspace(0, t_max, 1000)
def F(v):
    return -b * v * np.abs(v)


def dgl(t, u):
    x, v = u
    return np.array([v, F(v) / m])


u0 = np.array([x0, v0])
result = scipy.integrate.solve_ivp(dgl, [0, t_max], u0, t_eval=t)
result = scipy.integrate.solve_ivp(dgl, [0, t_max], u0, dense_output= True)
t_stuetz = result.t
x_stuetz, v_stuetz = result.y
print(result.message)
t = result.t
x, v = result.y
t_interp = np.linspace(0, t_max, 15)
x_interp, v_interp = result.sol(t_interp)
fig = plt.figure(figsize=(8, 4))
fig.set_tight_layout(True)
ax_geschw = fig.add_subplot(1, 2, 1)
ax_geschw.set_xlabel('$t$ [s]')
ax_geschw.set_ylabel('$v$ [m/s]')
ax_geschw.grid()
ax_geschw.plot(t, v0 / (1 + v0 * b / m * t), '-b', label='analytisch')
ax_geschw.plot(t, v_interp, '.r', label='simuliert')
ax_geschw.legend()
ax_ort = fig.add_subplot(1, 2, 2)
ax_ort.set_xlabel('$t$ [s]')
ax_ort.set_ylabel('$x$ [m]')
ax_ort.grid()
ax_ort.plot(t, m / b * np.log(1 + v0 * b / m * t),'-b', label='analytisch')
ax_ort.plot(t, x_interp, '.r', label='simuliert')
ax_ort.legend()

plt.show()"""



#Freier Fall
"""m = 90.0

# Erdbeschleunigung [m/s²].
g = 9.81

# Produkt aus c_w-Wert und Stirnfläche [m²].
cwA = 0.47

# Anfangshöhe [m].
y0 = 39.045e3

# Anfangsgeschwindigkeit [m/s].
v0 = 0.0

# Messwerte: Luftdichte [kg/m³] in Abhängigkeit von der Höhe [m].
h_mess = 1e3 * np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                         11.02, 15, 20.06, 25, 32.16, 40])
rho_mess = np.array([1.225, 1.112, 1.007, 0.909, 0.819, 0.736,
                     0.660, 0.590, 0.526, 0.467, 0.414, 0.364,
                     0.195, 0.0880, 0.0401, 0.0132, 0.004])

# Erzeuge eine Interpolationsfunktion für die Luftdichte.
fill = (rho_mess[0], rho_mess[-1])
rho = scipy.interpolate.interp1d(h_mess, rho_mess, kind='cubic',
                                 bounds_error=False,
                                 fill_value=fill)


def F(y, v):
    Fg = -m * g
    Fr = -0.5 * rho(y) * cwA * v * np.abs(v)
    return Fg + Fr


def dgl(t, u):
    y, v = u
    return np.array([v, F(y, v) / m])


def aufprall(t, u):
    y, v = u
    return y


# Bende die Simulation beim Auftreten des Ereignisses.
aufprall.terminal = True

# Lege den Zustandsvektor zum Zeitpunkt t=0 fest.
u0 = np.array([y0, v0])

# Löse die Bewegungsgleichung bis zum Auftreffen auf der Erde.
result = scipy.integrate.solve_ivp(dgl, [0, np.inf], u0,
                                   events=aufprall,
                                   dense_output=True)
t_s = result.t
y_s, v_s = result.y

# Berechne die Interpolation auf einem feinen Raster.
t = np.linspace(0, np.max(t_s), 1000)
y, v = result.sol(t)

# Erzeuge eine Figure.
fig = plt.figure(figsize=(9, 4))
fig.set_tight_layout(True)

# Plotte das Geschwindigkeits-Zeit-Diagramm.
ax1 = fig.add_subplot(1, 2, 1)
ax1.set_xlabel('t [s]')
ax1.set_ylabel('v [m/s]')
ax1.grid()
ax1.plot(t_s, v_s, '.b')
ax1.plot(t, v, '-b')

# Plotte das Orts-Zeit-Diagramm.
ax2 = fig.add_subplot(1, 2, 2)
ax2.set_xlabel('t [s]')
ax2.set_ylabel('y [m]')
ax2.grid()
ax2.plot(t_s, y_s, '.b')
ax2.plot(t, y, '-b')

# Zeige die Grafik an.
plt.show()"""




"""tag = 24 * 60 * 60
jahr = 365.25 * tag
AE = 1.495978707e11
scal_a = 20
scal_v = 1e-5
t_max = 100 * jahr
dt = 1 * tag
r0 = np.array([149.10e9, 0.0])
v0 = np.array([0.0, 29.29e3])
r_Mond0 = np.array([149.40e9, 0])
r_s0 = np.array([38.4e7, 0])
v_mond0 = np.array([13e1, 33e3])
M = 1.9885e30
m = 6e24
G = 6.6743e-11
def dgl(t, u):

    r, v = np.split(u, 2)
    a = - G * M * r / np.linalg.norm(r) ** 3
    
    return np.concatenate([v, a]) 
u0 = np.concatenate((r0, v0))


result = scipy.integrate.solve_ivp(dgl, [0, t_max], u0, rtol=1e-9, dense_output=True)

def dgl_mond(t, u):
    r_s, v = np.split(u, 2)
    r_mond, v_mond = np.split(u, 2)
    a_monda = -G * M * (r_mond) / np.linalg.norm(r_mond) ** 3
    a_erde = -G * m * (r_s) / np.linalg.norm(r_s) ** 3
    a_mond = a_monda + a_erde
    return np.concatenate([v_mond, a_mond])
print(r_Mond0, v_mond0)
u_mond0 = np.concatenate((r_Mond0, v_mond0))

result_mond = scipy.integrate.solve_ivp(dgl_mond, [0, t_max], u_mond0, rtol=1e-9, dense_output=True)

t_stuetz = result.t
r_stuetz, v_stuetz = np.split(result.y, 2)
t_interp = np.arange(0, np.max(t_stuetz), dt)
r_interp, v_interp = np.split(result.sol(t_interp), 2)


r_s_mond, v_s_mond = np.split(result_mond.y, 2)
r_interp_mond, v_interp_mond = np.split(result_mond.sol(t_interp), 2)
def update(n):
    t = t_interp[n]
    r = r_interp[:, n]
    v = v_interp[:, n]
    r_moon = r_interp_mond[:, n]
    v_moon = v_interp_mond[:, n]
    # Berechne die aktuelle Beschleunigung.
    u_punkt = dgl(t, np.concatenate([r, v]))
    u_moon = dgl_mond(t, np.concatenate([r_moon, v_moon]))
    a = np.split(u_punkt, 2)[1]
    a_moon = np.split(u_moon, 2)[1]
    # Aktualisiere die Position des Himmelskörpers und die Pfeile.
    plot_planet.set_data(r / AE)
    pfeil_a.set_positions(r / AE, r / AE + scal_a * a)
    pfeil_v.set_positions(r / AE, r / AE + scal_v * v)
    plot_moon.set_data(r_moon / AE)
    # Aktualisiere das Textfeld für die Zeit.
    text_t.set_text(f'$t$ = {t / tag:.0f} d')

    return plot_planet, pfeil_v, pfeil_a, text_t, plot_moon

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_aspect('equal')
ax.set_xlabel('$x$ [AE]')
ax.set_ylabel('$y$ [AE]')
ax.grid()

# Plotte die Bahnkurve der Himmelskörper.
ax.plot(r_stuetz[0] / AE, r_stuetz[1] / AE, '.b')
ax.plot(r_interp[0] / AE, r_interp[1] / AE, '-b')

# Erzeuge Punktplots für die Positionen der Himmelskörper.
plot_planet, = ax.plot([], [], 'o', color='red')
plot_sonne, = ax.plot([0], [0], 'o', color='gold')
plot_moon, = ax.plot([], [], 'o', color ='green')
# Erzeuge Pfeile für die Geschwindigkeit und die Beschleunigung.
style = mpl.patches.ArrowStyle.Simple(head_length=6, head_width=3)
pfeil_v = mpl.patches.FancyArrowPatch((0, 0), (0, 0), color='red',
                                      arrowstyle=style)
pfeil_a = mpl.patches.FancyArrowPatch((0, 0), (0, 0), color='black',
                                      arrowstyle=style)

# Füge die Pfeile zur Axes hinzu.
ax.add_patch(pfeil_a)
ax.add_patch(pfeil_v)

text_t = ax.text(0.01, 0.95, '', color='blue',
                 transform=ax.transAxes)


ani = mpl.animation.FuncAnimation(fig, update, frames=t_interp.size,
                                  interval=30, blit=True)
plt.show()"""






#Doppelsternsystem
"""tag = 24 * 60 * 60
jahr = 365.25 * tag
AE = 1.495978707e11

# Skalierungsfaktor für die Darstellung der Beschleunigung
# [AE / (m/s²)].
scal_a = 20

# Simulationszeit und Zeitschrittweite [s].
t_max = 2 * jahr
dt = 1 * tag

# Massen der beiden Sterne [kg].
m1 = 2.0e30
m2 = 4.0e29

# Anfangspositionen der Sterne [m].
r0_1 = AE * np.array([0.0, 0.0])
r0_2 = AE * np.array([0.0, 1.0])

# Anfangsgeschwindigkeiten der Sterne [m/s].
v0_1 = np.array([0.0, 0.0])
v0_2 = np.array([25e3, 0.0])

# Gravitationskonstante [m³ / (kg * s²)].
G = 6.6743e-11


def dgl(t, u):

    r1, r2, v1, v2 = np.split(u, 4)
    a1 = G * m2 / np.linalg.norm(r2 - r1)**3 * (r2 - r1)
    a2 = G * m1 / np.linalg.norm(r1 - r2)**3 * (r1 - r2)
    return np.concatenate([v1, v2, a1, a2])


# Lege den Zustandsvektor zum Zeitpunkt t=0 fest.
u0 = np.concatenate((r0_1, r0_2, v0_1, v0_2))

# Löse die Bewegungsgleichung bis zum Zeitpunkt t_max.
result = scipy.integrate.solve_ivp(dgl, [0, t_max], u0, rtol=1e-9,
                                   t_eval=np.arange(0, t_max, dt))
t = result.t
r1, r2, v1, v2 = np.split(result.y, 4)

# Berechne die verschiedenen Energiebeiträge.
E_kin1 = 1/2 * m1 * np.sum(v1 ** 2, axis=0)
E_kin2 = 1/2 * m2 * np.sum(v2 ** 2, axis=0)
E_pot = - G * m1 * m2 / np.linalg.norm(r1 - r2, axis=0)

# Berechne den Gesamtimpuls.
impuls = m1 * v1 + m2 * v2

# Berechne den Drehimpuls.
drehimpuls = (m1 * np.cross(r1, v1, axis=0) +
              m2 * np.cross(r2, v2, axis=0))

# Erzeuge eine Figure.
fig = plt.figure(figsize=(10, 7))
fig.set_tight_layout(True)

# Erzeuge eine Axes für die Bahnkurve der Sterne.
ax_bahn = fig.add_subplot(2, 2, 1)
ax_bahn.set_xlabel('$x$ [AE]')
ax_bahn.set_ylabel('$y$ [AE]')
ax_bahn.set_aspect('equal')
ax_bahn.grid()

# Plotte die Bahnkurven der Sterne.
ax_bahn.plot(r1[0] / AE, r1[1] / AE, '-r')
ax_bahn.plot(r2[0] / AE, r2[1] / AE, '-b')

# Erzeuge Punktplots für die Positionen der Sterne.
plot_stern1, = ax_bahn.plot([], [], 'o', color='red')
plot_stern2, = ax_bahn.plot([], [], 'o', color='blue')

# Erzeuge zwei Pfeile für die Beschleunigungsvektoren.
style = mpl.patches.ArrowStyle.Simple(head_length=6, head_width=3)
pfeil_a1 = mpl.patches.FancyArrowPatch((0, 0), (0, 0), color='red',
                                       arrowstyle=style)
pfeil_a2 = mpl.patches.FancyArrowPatch((0, 0), (0, 0), color='blue',
                                       arrowstyle=style)

# Füge die Pfeile zur Axes hinzu.
ax_bahn.add_patch(pfeil_a1)
ax_bahn.add_patch(pfeil_a2)

# Erzeuge eine Axes und plotte die Energie.
ax_energ = fig.add_subplot(2, 2, 2)
ax_energ.set_title('Energie')
ax_energ.set_xlabel('$t$ [d]')
ax_energ.set_ylabel('$E$ [J]')
ax_energ.grid()
ax_energ.plot(t / tag, E_kin1, '-r', label='$E_{kin,1}$')
ax_energ.plot(t / tag, E_kin2, '-b', label='$E_{kin,2}$')
ax_energ.plot(t / tag, E_pot, '-c', label='$E_{pot}$')
ax_energ.plot(t / tag, E_pot + E_kin1 + E_kin2,
              '-k', label='$E_{ges}$')
ax_energ.legend()

# Erzeuge eine Axes und plotte den Drehimpuls.
ax_drehimpuls = fig.add_subplot(2, 2, 3)
ax_drehimpuls.set_title('Drehimpuls')
ax_drehimpuls.set_xlabel('$t$ [d]')
ax_drehimpuls.set_ylabel('$L$ [kg m² / s]')
ax_drehimpuls.grid()
ax_drehimpuls.plot(t / tag, drehimpuls)

# Erzeuge eine Axes und plotte den Impuls.
ax_impuls = fig.add_subplot(2, 2, 4)
ax_impuls.set_title('Impuls')
ax_impuls.set_xlabel('$t$ [d]')
ax_impuls.set_ylabel('$p$ [kg m / s]')
ax_impuls.grid()
ax_impuls.plot(t / tag, impuls[0, :], label='$p_x$')
ax_impuls.plot(t / tag, impuls[1, :], label='$p_y$')
ax_impuls.legend()

# Sorge dafür, dass die nachfolgenden Linien nicht mehr die
# y-Skalierung verändern.
ax_energ.set_ylim(auto=False)
ax_drehimpuls.set_ylim(auto=False)
ax_impuls.set_ylim(auto=False)

# Erzeuge drei schwarze Linien, die die aktuelle Zeit in den
# Plots für Energie, Impuls und Drehimpuls darstellen.
linie_t_energ, = ax_energ.plot([], [], '-k')
linie_t_drehimp, = ax_drehimpuls.plot([], [], '-k')
linie_t_impuls, = ax_impuls.plot([], [], '-k')


def update(n):

    # Aktualisiere die Position der Sterne.
    plot_stern1.set_data(r1[:, n] / AE)
    plot_stern2.set_data(r2[:, n] / AE)

    # Berechne die Momentanbeschleunigung und aktualisiere die
    # Vektorpfeile.
    u = np.concatenate([r1[:, n], r2[:, n], v1[:, n], v2[:, n]])
    a_1, a_2 = np.split(dgl(t[n], u), 4)[2:]
    pfeil_a1.set_positions(r1[:, n] / AE,
                           r1[:, n] / AE + scal_a * a_1)
    pfeil_a2.set_positions(r2[:, n] / AE,
                           r2[:, n] / AE + scal_a * a_2)

    # Stelle die Zeit in den drei anderen Diagrammen dar.
    x_pos = t[n] / tag
    linien = [linie_t_energ, linie_t_drehimp, linie_t_impuls]
    for linie in linien:
        linie.set_data([[x_pos, x_pos], linie.axes.get_ylim()])

    return linien + [plot_stern1, plot_stern2, pfeil_a1, pfeil_a2]


# Erzeuge das Animationsobjekt und starte die Animation.
ani = mpl.animation.FuncAnimation(fig, update, frames=t.size,
                                  interval=30, blit=True)
plt.show()"""



#Schwerpunktsystem
"""tag = 24 * 60 * 60
jahr = 365.25 * tag
AE = 1.495978707e11

# Skalierungsfaktor für die Darstellung der Beschleunigung
# [AE / (m/s²)].
scal_a = 20

# Simulationszeit und Zeitschrittweite [s].
t_max = 2 * jahr
dt = 1 * tag

# Massen der beiden Sterne [kg].
m1 = 2.0e30
m2 = 4.0e29

# Anfangspositionen der Sterne [m].
r0_1 = AE * np.array([0.0, 0.0])
r0_2 = AE * np.array([0.0, 1.0])

# Anfangsgeschwindigkeiten der Sterne [m/s].
v0_1 = np.array([0.0, 0.0])
v0_2 = np.array([25e3, 0.0])

# Gravitationskonstante [m³ / (kg * s²)].
G = 6.6743e-11

# Berechne die Schwerpunktsposition und -geschwindigkeit und
# ziehe diese von den Anfangsbedingungen ab.
schwerpunkt0 = (m1 * r0_1 + m2 * r0_2) / (m1 + m2)
schwerpunktsgeschwindigkeit0 = (m1 * v0_1 + m2 * v0_2) / (m1 + m2)
r0_1 -= schwerpunkt0
r0_2 -= schwerpunkt0
v0_1 -= schwerpunktsgeschwindigkeit0
v0_2 -= schwerpunktsgeschwindigkeit0


def dgl(t, u):

    r1, r2, v1, v2 = np.split(u, 4)
    a1 = G * m2 / np.linalg.norm(r2 - r1)**3 * (r2 - r1)
    a2 = G * m1 / np.linalg.norm(r1 - r2)**3 * (r1 - r2)
    return np.concatenate([v1, v2, a1, a2])


# Lege den Zustandsvektor zum Zeitpunkt t=0 fest.
u0 = np.concatenate((r0_1, r0_2, v0_1, v0_2))

# Löse die Bewegungsgleichung bis zum Zeitpunkt t_max.
result = scipy.integrate.solve_ivp(dgl, [0, t_max], u0, rtol=1e-9,
                                   t_eval=np.arange(0, t_max, dt))
t = result.t
r1, r2, v1, v2 = np.split(result.y, 4)

# Berechne die verschiedenen Energiebeiträge.
E_kin1 = 1/2 * m1 * np.sum(v1 ** 2, axis=0)
E_kin2 = 1/2 * m2 * np.sum(v2 ** 2, axis=0)
E_pot = - G * m1 * m2 / np.linalg.norm(r1 - r2, axis=0)

# Berechne den Gesamtimpuls.
impuls = m1 * v1 + m2 * v2

# Berechne die Position des Schwerpunktes.
rs = (m1 * r1 + m2 * r2) / (m1 + m2)

# Berechne den Drehimpuls.
drehimpuls = (m1 * np.cross(r1, v1, axis=0) +
              m2 * np.cross(r2, v2, axis=0))

# Erzeuge eine Figure.
fig = plt.figure(figsize=(10, 7))
fig.set_tight_layout(True)

# Erzeuge eine Axes für die Bahnkurve der Sterne.
ax_bahn = fig.add_subplot(2, 2, 1)
ax_bahn.set_xlabel('$x$ [AE]')
ax_bahn.set_ylabel('$y$ [AE]')
ax_bahn.set_aspect('equal')
ax_bahn.grid()

# Plotte die Bahnkurven der Sterne.
ax_bahn.plot(r1[0] / AE, r1[1] / AE, '-r')
ax_bahn.plot(r2[0] / AE, r2[1] / AE, '-b')

# Erzeuge Punktplots für die Positionen der Himmelskörper.
plot_stern1, = ax_bahn.plot([], [], 'o', color='red')
plot_stern2, = ax_bahn.plot([], [], 'o', color='blue')

# Erzeuge zwei Pfeile für die Beschleunigungsvektoren.
style = mpl.patches.ArrowStyle.Simple(head_length=6, head_width=3)
pfeil_a1 = mpl.patches.FancyArrowPatch((0, 0), (0, 0), color='red',
                                       arrowstyle=style)
pfeil_a2 = mpl.patches.FancyArrowPatch((0, 0), (0, 0), color='blue',
                                       arrowstyle=style)

# Füge die Pfeile zur Axes hinzu.
ax_bahn.add_patch(pfeil_a1)
ax_bahn.add_patch(pfeil_a2)

# Erzeuge eine Axes und plotte die Energie.
ax_energ = fig.add_subplot(2, 2, 2)
ax_energ.set_title('Energie')
ax_energ.set_xlabel('$t$ [d]')
ax_energ.set_ylabel('$E$ [J]')
ax_energ.grid()
ax_energ.plot(t / tag, E_kin1, '-r', label='$E_{kin,1}$')
ax_energ.plot(t / tag, E_kin2, '-b', label='$E_{kin,2}$')
ax_energ.plot(t / tag, E_pot, '-c', label='$E_{pot}$')
ax_energ.plot(t / tag, E_pot + E_kin1 + E_kin2,
              '-k', label='$E_{ges}$')
ax_energ.legend()

# Erzeuge eine Axes und plotte den Drehimpuls.
ax_drehimpuls = fig.add_subplot(2, 2, 3)
ax_drehimpuls.set_title('Drehimpuls')
ax_drehimpuls.set_xlabel('$t$ [d]')
ax_drehimpuls.set_ylabel('$L$ [kg m² / s]')
ax_drehimpuls.grid()
ax_drehimpuls.plot(t / tag, drehimpuls)

# Erzeuge eine Axes und plotte den Schwerpunkt.
ax_schwerpunkt = fig.add_subplot(2, 2, 4)
ax_schwerpunkt.set_title('Schwerpunkt')
ax_schwerpunkt.set_xlabel('$t$ [d]')
ax_schwerpunkt.set_ylabel('$r_s$ [mm]')
ax_schwerpunkt.grid()
ax_schwerpunkt.plot(t / tag, 1e3 * rs[0, :], label='$r_{s,x}$')
ax_schwerpunkt.plot(t / tag, 1e3 * rs[1, :], label='$r_{s,y}$')
ax_schwerpunkt.legend()

# Sorge dafür, dass die nachfolgenden Linien nicht mehr die
# y-Skalierung verändern.
ax_energ.set_ylim(auto=False)
ax_drehimpuls.set_ylim(auto=False)
ax_schwerpunkt.set_ylim(auto=False)

# Erzeuge drei schwarze Linien, die die aktuelle Zeit in den
# Plots für Energie, Impuls und Drehimpuls darstellen.
linie_t_energ, = ax_energ.plot([], [], '-k')
linie_t_drehimp, = ax_drehimpuls.plot([], [], '-k')
linie_t_schwerpunkt, = ax_schwerpunkt.plot([], [], '-k')


def update(n):

    # Aktualisiere die Positionen der Sterne.
    plot_stern1.set_data(r1[:, n] / AE)
    plot_stern2.set_data(r2[:, n] / AE)

    # Berechne die Momentanbeschleunigung und aktualisiere die
    # Vektorpfeile.
    u = np.concatenate([r1[:, n], r2[:, n], v1[:, n], v2[:, n]])
    a_1, a_2 = np.split(dgl(t[n], u), 4)[2:]
    pfeil_a1.set_positions(r1[:, n] / AE,
                           r1[:, n] / AE + scal_a * a_1)
    pfeil_a2.set_positions(r2[:, n] / AE,
                           r2[:, n] / AE + scal_a * a_2)

    # Stelle die Zeit in den drei anderen Diagrammen dar.
    x_pos = t[n] / tag
    linien = [linie_t_energ, linie_t_drehimp, linie_t_schwerpunkt]
    for linie in linien:
        linie.set_data([[x_pos, x_pos], linie.axes.get_ylim()])

    return linien + [plot_stern1, plot_stern2, pfeil_a1, pfeil_a2]


# Erzeuge das Animationsobjekt und starte die Animation.
ani = mpl.animation.FuncAnimation(fig, update, frames=t.size,
                                  interval=30, blit=True)
plt.show()"""