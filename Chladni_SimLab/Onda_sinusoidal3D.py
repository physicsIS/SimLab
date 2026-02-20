import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from Onda2Dim import Onda2Dim
from Onda2Dim import cmap_nodos
from matplotlib.animation import FFMpegWriter

"""
Simulación de la ecuación de onda 2D en una membrana cuadrada
con condiciones de frontera de Dirichlet, usando el módulo 'Onda2Dim'

Se utiliza una señas sinusoidal externa colocada en el centro de la membrana
y se visualiza su evolución temporal mediante animación en DOS dimensiones.

Hecho por: Kevin Mauricio Chacón Ureña (C32060)

"""

#------------------Frecuencia de señal sinusoidal-------------

A = 1
f= 15
w = f*2*np.pi
L = 1

d=0.001
h=0.005

sim = Onda2Dim(L, 3, h, d)
t_fin=1.5

#--------------------Programación animación 3D---------------------

# Configuración de la figura 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Crear la malla de coordenadas
x = np.linspace(0, L, sim.x0)
y = np.linspace(0, L, sim.y0)
X, Y = np.meshgrid(x, y)

# Configurar la superficie inicial
Z = sim.u[0]  # Amplitud inicial
surf = ax.plot_surface(X, Y, Z, cmap=cmap_nodos, 
                        vmin=-A, vmax=A, 
                        linewidth=0, antialiased=True)

ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_zlabel('Amplitud [m]')
ax.set_zlim(-A, A)
ax.view_init(elev=30, azim=45)

cbar = plt.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
cbar.set_label("Amplitud de la onda [m]")

contador = ax.text2D(0.02, 0.95, '', transform=ax.transAxes, 
                     fontsize=12, color='black')

ax.set_title(f'Membrana estilo Chladni: Señal sinusoidal forzada a {f} Hz')


def actualizar_anim(i):
    sim.u[0, sim.x0//2, sim.y0//2] = A * np.sin(i*d* w) #Condición de señal sinosoidal en el centro
    sim.calcular_tiempo()



def animate(i):
    global surf
    
    actualizar_anim(i)
    Z_new = sim.u[0]

    surf.remove()
    surf = ax.plot_surface(
        X, Y, Z_new,
        cmap=cmap_nodos,
        vmin=-A, vmax=A,
        linewidth=0,
        antialiased=True
    )

    contador.set_text(f"t = {i*d:.3f} s")
    return surf, contador

interval, nframes = 20, int(t_fin/d)

anim = animation.FuncAnimation(fig, animate, frames=nframes,
                              repeat=False,
                              interval=interval, blit=False)

plt.show()

'''
#Si se descomenta guardará la animación como '.mp4', importante entonces remover 'plt.show()'

writer = FFMpegWriter(
    fps=int(0.1/d),   # frames por segundo coherente con d
    metadata=dict(artist='Matplotlib'),
    bitrate=1800
)

anim.save(f"Onda3Dimension (Sinusoidal {f}Hz).mp4", writer=writer)

import winsound
winsound.Beep(450, 700)
'''