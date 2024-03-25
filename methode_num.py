import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.header("Prévision du dépôt de calcaire au sein du garnissage d’une tour aéroréfrigérante")
col1, col2, col3 = st.columns(3)
dx = col1.slider(value=5, min_value=2, max_value=10, step=1, label="Maillage dx en µm")
dy = col2.slider(value=5, min_value=2, max_value=10, step=1, label="Maillage dy en µm")
L = col3.slider(value=1, min_value=1, max_value=10, step=1, label="Longueur L en Cm")
L = L / 100

tab1, tab2, tab3, tab4 = st.tabs(["Maillage", "Ordres de grandeur", "Shah&London", "Précipitation"])

dx = dx * 1e-6
dy = dy * 1e-6

T = 308
a1, b1, c1, d1, e1 = -356.3094, -0.06091964, 21834.37, 126.8339, -1684915
a2, b2, c2, d2, e2 = -107.8871, -0.03252849, 5151.79, 38.92561, -563713.9
a3, b3, c3, d3, e3 = 1209.120, 0.31294, -34765.05, -478.782, 0
a4, b4, c4, d4, e4, f4, g4 = -4.098, -3245.2, 2.2362, -3.984, 13.957, -1262.3, 8.5641
pw = 994.0319 * 10 ** -3

k1 = 10 ** (a1 + b1 * T + c1 / T + d1 * np.log10(T) + e1 / T ** 2)
k2 = 10 ** (a2 + b2 * T + c2 / T + d2 * np.log10(T) + e2 / T ** 2)
k3 = 10 ** (a3 + b3 * T + c3 / T + d3 * np.log10(T) + e3 / T ** 2)
ke = 10 ** (a4 + b4 / T + c4 / T ** 2 + d4 / T ** 3 + (e4 + f4 / T + g4 / T ** 2) * np.log10(pw))

ksp = -171.9065 - 0.077993 * T + (2839.319 / T) + 71.595 * np.log(T)

Ca_tot = 2 * 10 ** -3
C_tot = 1.96 * 10 ** -3


def F(x, Ce):
    x1, x2, x3 = x
    eq1 = (k3 / k2) * x1 * x2 * x3 + x1 + 2 * x2 - (x1 * x3 / k2) - 2 * x3 - (ke / x1) - 2.05 * 10 ** -3
    eq2 = x2 + (k3 * x1 * x2 * x3 / k2) - Ca_tot
    eq3 = (x1 ** 2 * x3 / (k1 * k2)) + (x1 * x3 / k2) + x3 + (k3 * x1 * x2 * x3 / k2) - Ce
    return [eq1, eq2, eq3]


def J(x):
    x1, x2, x3 = x
    # Calcul des dérivées partielles de f1 par rapport à x1, x2 et x3
    df1_x1 = (k3 / k2) * x2 * x3 + 1 - (x3 / k2) + (ke / x1 ** 2)
    df1_x2 = (k3 / k2) * x1 * x3 + 2
    df1_x3 = (k3 / k2) * x1 * x2 - (x1 / k2) - 2
    # Calcul des dérivées partielles de f2 par rapport à x1, x2 et x3
    df2_x1 = (k3 * x2 * x3) / k2
    df2_x2 = 1 + (k3 * x1 * x3) / k2
    df2_x3 = (k3 * x1 * x2) / k2
    # Calcul des dérivées partielles de f3 par rapport à x1, x2 et x3
    df3_x1 = (2 * x1 * x3) / (k1 * k2) + (x3 / k2) + (k3 * x2 * x3) / k2
    df3_x2 = (k3 * x1 * x3) / k2
    df3_x3 = (x1 ** 2) / (k1 * k2) + 1 + (k3 * x1 * x2) / k2 + (x1 / k2)
    # Construction de la matrice jacobienne
    J = np.array([[df1_x1, df1_x2, df1_x3],
                  [df2_x1, df2_x2, df2_x3],
                  [df3_x1, df3_x2, df3_x3]])
    return J


precision = 1.e-20




def NewtonRaphson(Ce):
    x = np.array([10 ** -9, 10 ** -3, 10 ** -5])
    n, pas2 = 0, 4e4
    while n < 20 and pas2 > precision:
        pas = np.linalg.inv(J(x)).dot(F(x, Ce))
        x -= pas  # iteration
        pas2 = np.sqrt(pas.dot(pas))
        n += 1
    return x


C_Ca, C_H, C_CO3 = NewtonRaphson(C_tot)

print(f"Résultat finaux : \n x1: {C_Ca:.2e}  \n x2: {C_H:.2e}  \n x3: {C_CO3:.2e}")

kc1 = 10 ** (0.198 - 444 / T)
kc2 = 10 ** (2.84 - 2177 / T)
kc3 = 10 ** (-1.1 - 1737 / T)

ksp = 10 ** (-171.9065 - 0.077993 * T + (2839.319 / T) + 71.595 * np.log10(T))

R_CaCO3 = 10 * (kc1 * C_Ca + kc2 * (C_Ca ** 2 * C_CO3) / (k1 * k2) + kc3) * (1 - 10 ** ((2 / 3) * np.log10(C_H * C_CO3 / ksp))) * 10 ** -3
print(R_CaCO3)

a = 147.8
Qe = 7320
ρ = 994
q = Qe / (ρ * a * 3600)

print("La valeur du débit linéique est de ", f"{q:.2e}")

Dk = 10 ** -9
µ = 0.720 * 10 ** -3
theta = np.pi / 2
d = 150 * 10 ** -6
g = 9.81

δ = (3 * q * µ / (ρ * g)) ** (1 / 3)
print(f"La valeur de l'épaisseur de film est {δ:.2e}")

v = (ρ * g * δ ** 2) / (3 * µ)
Re = 4 * ρ * v * δ / µ
Sc = (µ / ρ) / Dk

print(f"Le nombre de Reynolds est égal à {int(Re)}")

Pe = Re * Sc
print(f"Le nombre de Péclet est égal à {Pe:.2e}")

Da = R_CaCO3 / (Ca_tot * Dk / d)
print(f"Le nombre de Damkhöler est égal à {Da:.2e}, donc l'étape limitante est la réaction de précipitation.")

X = [0]
Y = [0]

qx = 1.02
qy = 1.02

i = 1
x = dx
while x < L:
    X.append(x)
    i += 1
    if x < L / 2:
        x = X[-1] + dx * qx ** (i - 1)
    else:
        x += X[-1] - X[-2]
X.append(L)

j = 1
y = dy
while j * dy < δ:
    Y.append(y)
    j += 1
    if y < δ / 2:
        y = Y[-1] + dy * qy ** (j - 1)
    else:
        y += Y[-1] - Y[-2]
Y.append(δ)

n = len(X)
m = len(Y)

C = np.ones((m, n)) * Ca_tot


def Vx(y):
    return ρ * g * (δ * y - y ** 2 / 2) / µ


ε = 1
s1 = 0
s2 = 0
with tab1:
    for _ in range(100):
        # Points Centraux
        for i in range(1, n - 1):
            for j in range(1, m - 1):
                a_e = Dk * (Y[j + 1] - Y[j - 1]) / (2 * (X[i + 1] - X[i]))
                a_w = Dk * (Y[j + 1] - Y[j - 1]) / (2 * (X[i] - X[i - 1])) + Vx(Y[j]) * (Y[j + 1] - Y[j - 1]) / 2
                a_n = Dk * (X[i + 1] - X[i - 1]) / (2 * (Y[j + 1] - Y[j]))
                a_s = Dk * (X[i + 1] - X[i - 1]) / (2 * (Y[j] - Y[j - 1]))
                a_p = a_e + a_w + a_n + a_s

                C[j][i] = (a_e * C[j][i + 1] + a_w * C[j][i - 1] + a_n * C[j + 1][i] + a_s * C[j - 1][i]) / a_p

        # Facade Ouest
        for j in range(m):
            C[j][0] = Ca_tot

        for i in range(1, n - 1):
            # Facade Nord
            j = m - 1
            a_e = Dk * (Y[j] - Y[j - 1]) / (2 * (X[i + 1] - X[i]))
            a_w = Dk * (Y[j] - Y[j - 1]) / (2 * (X[i] - X[i - 1])) + Vx(Y[j]) * (Y[j] - Y[j - 1]) / 2
            a_s = Dk * (X[i + 1] - X[i - 1]) / (2 * (Y[j] - Y[j - 1]))
            a_p = a_e + a_w + a_s
            C[j][i] = (a_e * C[j][i + 1] + a_w * C[j][i - 1] + a_s * C[j - 1][i]) / a_p

            # Facade Sud
            j = 0
            a_e = Dk * (Y[j + 1] - Y[j]) / (2 * (X[i + 1] - X[i]))
            a_w = Dk * (Y[j + 1] - Y[j]) / (2 * (X[i] - X[i - 1])) + Vx(Y[j]) * (Y[j + 1] - Y[j]) / 2
            a_n = Dk * (X[i + 1] - X[i - 1]) / (2 * (Y[j + 1] - Y[j]))
            a_p = a_e + a_w + a_n
            b = -R_CaCO3 * ((X[i + 1] - X[i - 1]) / 2)
            C[j][i] = (a_e * C[j][i + 1] + a_w * C[j][i - 1] + a_n * C[j + 1][i] + b) / a_p

        # Facade Est
        for j in range(1, m - 1):
            i = n - 1
            a_w = Dk * (Y[j + 1] - Y[j - 1]) / (2 * (X[i] - X[i - 1])) + Vx(Y[j]) * (Y[j + 1] - Y[j - 1]) / 2
            a_n = Dk * (X[i] - X[i - 1]) / (2 * (Y[j + 1] - Y[j]))
            a_s = Dk * (X[i] - X[i - 1]) / (2 * (Y[j] - Y[j - 1]))
            a_p = a_w + a_n + a_s
            C[j][i] = (a_w * C[j][i - 1] + a_n * C[j + 1][i] + a_s * C[j - 1][i]) / a_p

        # Coin Nord Est
        i = n - 1
        j = m - 1
        a_s = Dk * (X[i] - X[i - 1]) / (2 * (Y[j] - Y[j - 1]))
        a_w = Dk * (Y[j] - Y[j - 1]) / (2 * (X[i] - X[i - 1])) + Vx(Y[j]) * (Y[j] - Y[j - 1]) / 2
        a_p = a_w + a_s
        C[j][i] = (a_w * C[j][i - 1] + a_s * C[j - 1][i]) / a_p

        # Coin Sud Est
        i = n - 1
        j = 0
        a_w = Dk * (Y[j + 1] - Y[j]) / (2 * (X[i] - X[i - 1])) + Vx(Y[j]) * (Y[j] - Y[j - 1]) / 2
        a_n = Dk * (X[i] - X[i - 1]) / (2 * (Y[j + 1] - Y[j]))
        a_p = a_w + a_n
        b = -R_CaCO3 * ((X[i] - X[i - 1]) / 2)

        C[j][i] = (a_w * C[j][i - 1] + a_n * C[j + 1][i] + b) / a_p

        if ε < 2e-5:
            break

    fig = go.Figure(data=go.Heatmap(
        z=C,
        x=X,
        y=Y,
        colorscale='Viridis'
    ))

    fig.update_layout(
        plot_bgcolor='white',
        autosize=False,
        width=1200,
        height=600,
        yaxis_range=[0, δ]
    )

    st.plotly_chart(fig)

with tab2:
    col1, col2, col3 = st.columns(3)
    col1.metric(label="Reynolds", value=int(Re))
    col2.metric(label="Péclet", value=f"{Pe:.2e}")
    col3.metric(label="Epaisseur du film", value=f"{δ:.2e} m")

def ConcentrationMelange(i):
    s1 = 0
    s2 = 0
    for j in range(m):
        if j == 0:
            s1 += C[j][i] * Vx(Y[j]) * (Y[1])
            s2 += Vx(Y[j]) * (Y[j + 1])
        elif j == m - 1:
            s1 += C[j][i] * Vx(Y[j]) * (Y[j] - Y[j - 1])
            s2 += Vx(Y[j]) * (Y[j] - Y[j - 1])
        else:
            s1 += C[j][i] * Vx(Y[j]) * ((Y[j + 1] - Y[j - 1]) / 2)
            s2 += Vx(Y[j]) * ((Y[j + 1] - Y[j - 1]) / 2)

    return (s1 / s2)


def Km(i):
    return (-R_CaCO3 / (C[0][i] - ConcentrationMelange(i)))


Dh = 4 * d


def Sh(i):
    return ((Km(i)) * Dh / Dk)


Sh_code = []
for i in range(1, n):
    Sh_code.append(Sh(i))

X_e = []
for i in range(n):
    X_e.append(X[i] / (Dh * Pe))

Sh_London = []
for i in range(1, len(X_e)):
    if X_e[i] <= 2 * 10 ** -4:
        Sh_London.append(1.49 * X_e[i] ** (-1 / 3))
    elif X_e[i] <= 10 ** -3:
        Sh_London.append(1.49 * X_e[i] ** (-1 / 3) - 0.4)
    else:
        Sh_London.append(8.325 + 8.68 * (10 ** 3 * X_e[i]) ** (-0.506) * np.exp(X_e[i]))

with tab3:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Echelle décimale")
        fig, ax = plt.subplots()
        ax.plot(X_e[1:], Sh_code)
        ax.plot(X_e[1:], Sh_London)
        ax.legend(["Code", "Shah&London"])
        ax.grid()
        st.pyplot(fig)

    with col2:
        st.subheader("Echelle logarithmique")
        fig, ax = plt.subplots()
        ax.loglog(X_e[1:], Sh_code)
        ax.loglog(X_e[1:], Sh_London)
        ax.legend(["Code", "Shah&London"])
        ax.grid()
        st.pyplot(fig)


from scipy.optimize import fsolve

φ = [0] * n
j=0
for i in range(n):
  def F2(x):
    x1, x2, x3 = x
    eq1 = (k3 / k2) * x1 * x2 * x3 + x1 + 2 * x2 - (x1 * x3 / k2) - 2 * x3 - (ke / x1) - 2.05 * 10 ** -3
    eq2 = x2 + (k3 * x1 * x2 * x3 / k2) - Ca_tot
    eq3 = (x1**2 * x3 / (k1 * k2)) + (x1 * x3 / k2) + x3 + (k3 * x1 * x2 * x3 / k2) - C[j][i]
    return [eq1, eq2, eq3]
  C_Ca_i, C_H_i, C_CO3_i = fsolve(F2, [np.array([10 ** -6, 10 ** -3, 10 ** -3])])
  φ[i] = - 10 * (kc1 * C_Ca_i + kc2 * (C_Ca_i **2 * C_CO3_i) / (k1 * k2) + kc3)*(1 - 10 ** ((2/3) * np.log10(C_H_i * C_CO3_i / ksp))) * 10 ** -3
def φ_tot(phi):
    S = 0
    for i in range (len(φ)-1):
      S += φ[i] * (X[i+1]-X[i])
    return S*100.09*24*3600*365
with tab4:
    col1, col2 = st.columns(2)
    fig, ax = plt.subplots()
    ax.plot(X,φ)
    ax.set_title('Flux de précipitation calcaire (kg/m²/an)')
    col1.pyplot(fig)
    col2.write(f"Pour une plaque de {L} m de long et 1m de large, il y a {round(φ_tot(φ), 3)} kg/an de calcaire qui se dépose dessus.")