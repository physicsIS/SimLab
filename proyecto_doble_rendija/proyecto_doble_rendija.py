"""
Simulación FDTD de propagación de ondas con doble rendija.

Se implementa una simulación bidimensional mediante el método de diferencias
finitas en el dominio del tiempo (FDTD) para estudiar el patrón de interferencia
generado por una doble rendija. La simulación incluye una capa absorbente,
una fuente armónica y la visualización del campo y de la intensidad en una
pantalla de observación.
"""

import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio

#           ********************** FUNCIONES ************************

def abrir_rend(ycentral):
    """
    Genera una apertura en la pared sólida que representa una rendija.

    Modifica la matriz global `pared` eliminando (por medio de False) el material
    en un intervalo vertical centrado en `ycentral`, con ancho `a_rendija`.

    Parámetros
    ----------
    ycentral : int
        Coordenada vertical (índice en y) del centro de la rendija.

    Retorna
    -------
    None
    """
    a = ycentral - a_rendija // 2
    b = ycentral + a_rendija // 2
    pared[x_pared:x_pared + grosor, a:b] = False

def absorbente(Nx, Ny, n_abs, sigma_max, m=3):
    """
    Construye una capa absorbente para evitar reflexiones artificiales en los
    bordes del dominio.

    Genera una matriz sigma(x,y) que implementa una absorción
:set nonumber    creciente hacia los bordes del dominio para minimizar
    dichas reflexiones.

    Parámetros
    ----------
    Nx : int
        Número de puntos en dirección x.
    Ny : int
        Número de puntos en dirección y.
    n_abs : int
        Espesor de la capa absorbente (en celdas).
    sigma_max : float
        Valor máximo del coeficiente de absorción.
    m : int
        Orden del perfil polinómico de crecimiento.

    Retorna
    -------
    sigma : ndarray (Nx, Ny)
        Matriz de coeficiente de amortiguamiento espacial.
    """
    sigma = np.zeros((Nx, Ny), dtype=np.float32)
    for i in range(Nx):
        di = min(i, Nx - 1 - i)
        for j in range(Ny):
            dj = min(j, Ny - 1 - j)
            d = min(di, dj)
            if d < n_abs:
                p = (n_abs - d) / n_abs
                sigma[i, j] = sigma_max * (p ** m)
    return sigma

def laplaciano(U):
    """
    Calcula el Laplaciano discreto en dos dimensiones mediante esquema de 5 puntos.

    Parámetros
    ---------
    U : ndarray (Nx, Ny)
        Campo escalar en el tiempo actual.

    Retorna
    -------
    L : ndarray (Nx, Ny)
        Laplaciano discreto de U.
    """
    L = np.zeros_like(U)
    L[1:-1, 1:-1] = (
        (U[2:, 1:-1] - 2.0 * U[1:-1, 1:-1] + U[:-2, 1:-1]) / (dx * dx)
      + (U[1:-1, 2:] - 2.0 * U[1:-1, 1:-1] + U[1:-1, :-2]) / (dy * dy)
    )
    return L



#                 ********************** PARÁMETROS **************************

# -malla-
Nx = 900      # extensión para malla
Ny = 600
dx = 1.0      # tamaño para celdas
dy = 1.0
c = 1.0       # velocidad

courant = 0.6                            # "courant number" necesario cuando simulamos cosas con FDTD+ondas
dt = courant * dx / (c * np.sqrt(2.0))   # paso temporal según condición de estabilidad
Nt = 4000                                # número de pasos de tiempo

# -onda-
long = 20
w = 2 * np.pi * (c / long)

# -pared y doble rendija-
x_pared = 200
grosor = 4

a_rendija = 8     # ancho de rendija
sep_rend = 80     # separación de rendijas
y0 = Ny // 2      # centro vertical
y1 = y0 - sep_rend // 2
y2 = y0 + sep_rend // 2

# -capa absorbente-
n_abs = 60
sigma_max = 0.35 / dt

# -fuente de onda-
x_fuente = 20
amplitud = 1.0

# -parámetros para el video-
paso = 4
x0_norm = x_pared + grosor + 5 # normalización para colores

color_p = np.array([1.0, 0.25, 0.65], dtype=np.float32) # máximos
color_n = np.array([0.55, 0.45, 1.0], dtype=np.float32) # mínimos
gamma = 0.8

# -pantalla de observación-
x_obs = x_pared + grosor + 400  # pantalla a la derecha
n_i = int(0.35 * Nt)            # tiempo inicial donde aún no está estabilizado


#          *********************** MAIN **************************


# -construcción de pared-
pared = np.zeros((Nx, Ny), dtype=bool)
pared[x_pared:x_pared + grosor, :] = True

abrir_rend(y1)
abrir_rend(y2)

# -capa absorbente-
sigma = absorbente(Nx, Ny, n_abs, sigma_max, m=3)

# -campos-
perfil_y = np.ones(Ny)

u_nm1 = np.zeros((Nx, Ny), dtype=np.float32)
u_n   = np.zeros((Nx, Ny), dtype=np.float32)
u_np1 = np.zeros((Nx, Ny), dtype=np.float32)

fact = (c * dt) ** 2 # pre factor

video = imageio.get_writer("doblerendija.mp4", fps=30)

I_acumulado = np.zeros(Ny, dtype=np.float64) # inicializa Intensidad para la gráfica
I_contador = 0

# - loop FDTD-
for n in range(Nt):
    t_n = n * dt

    # fuente (onda)
    fuente_n = amplitud * np.sin(w * t_n) * perfil_y
    fuente_nm1 = amplitud * np.sin(w * (t_n - dt)) * perfil_y

    u_n[x_fuente, :] = fuente_n
    u_nm1[x_fuente, :] = fuente_nm1

    # actualizamos "leapfrog"
    L = laplaciano(u_n)
    a = 2.0 - 2 * sigma * dt
    b = -1.0 + 2.0 * sigma * dt
    u_np1[:] = a * u_n + b * u_nm1 + fact * L

    # pared (condición Dirichlet u=0)
    u_np1[pared] = 0.0
    u_n[pared] = 0.0
    u_nm1[pared] = 0.0

    # bordes sin Dirichlet
    u_np1[0, :]  = u_np1[1, :]
    u_np1[-1, :] = u_np1[-2, :]
    u_np1[:, 0]  = u_np1[:, 1]
    u_np1[:, -1] = u_np1[:, -2]

    u_np1[x_fuente, :] = amplitud * np.sin(w * (t_n + dt)) * perfil_y

    # cambio
    u_nm1, u_n, u_np1 = u_n, u_np1, u_nm1

    # acumular la intensidad para la gráfica
    if n >= n_i:
        dx_obs = 40
        du_dt = (u_n[x_obs:x_obs + dx_obs, :] - u_nm1[x_obs:x_obs + dx_obs, :]) / dt
        I_acumulado += np.mean(du_dt ** 2, axis=0)
        I_contador += 1

    # guardar el frame
    if n % paso == 0:
        img = u_n.T
        m = np.max(np.abs(img[:, x0_norm:])) + 1e-6

        A = np.clip(img / m, -1.0, 1.0)
        I_p = np.clip(A, 0.0, 1.0) ** gamma
        I_n = np.clip(-A, 0.0, 1.0) ** gamma

        fondo = np.ones((Ny, Nx, 3), dtype=np.float32)
        fondo -= (1.0 - color_p) * I_p[..., None]
        fondo -= (1.0 - color_n) * I_n[..., None]
        fondo = np.clip(fondo, 0.0, 1.0)

        frame = (fondo * 255).astype(np.uint8)
        frame[pared.T] = 0
        video.append_data(frame)

video.close()

# -gráfica-

I_y = I_acumulado / max(I_contador, 1)
I_y = I_y / np.max(I_y)

y = (np.arange(Ny) - y0) * dy
theta = y / x_obs

plt.figure(figsize=(6, 4))
plt.plot(theta, I_y)
plt.xlabel(r"$\theta$")
plt.ylabel("Intensidad (normalizada)")
plt.title("Patrón de interferencia en la pantalla")
plt.tight_layout()
plt.savefig("Intensidad.png")
plt.close()
