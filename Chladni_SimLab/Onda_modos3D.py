import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from Onda2Dim import Onda2Dim
from Onda2Dim import cmap_nodos
from matplotlib.animation import FFMpegWriter

"""
Simulación de la ecuación de onda 2D en una membrana cuadrada
con condiciones de frontera de Dirichlet, usando el módulo 'Onda2Dim'

Se configuran modos de vibración estacionarios y se visualiza su 
evolución temporal mediante animación en DOS dimensiones.

Hecho por: Kevin Mauricio Chacón Ureña (C32060)

"""

#------------------Cond. iniciales Modo de vibración-------------

A = 10
d = 0.001
L = 1
h=0.005
sim = Onda2Dim(L, 3, h, d)
t_fin=1.5

m, n= 3,2#modos de vibración

m1,n1=1,3

for i in range(sim.x0): #posición inicial del modo
    for j in range(sim.y0):
        sim.u[0,j,i] = A * (np.sin(m*np.pi*i*h/L) * np.sin(n*np.pi*j*h/L)+np.sin(m1*np.pi*i*h/L) * np.sin(n1*np.pi*j*h/L))
        sim.u[1,j,i] = sim.u[0,j,i].copy()

#--------------------Programación animación 2D---------------------

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


ax.set_xticks([])
ax.set_yticks([])
plt.xlabel('x [m]')
plt.ylabel('y [m]')

contador = ax.text2D(0.02, 0.95, '', transform=ax.transAxes, 
                     fontsize=12, color='black')

if (m1*n1) == 0:
    ax.set_title(f'Membrana estilo Chladni: Modo de vibración {(m,n)}')
if (m1*n1)>0:
    ax.set_title(f'Membrana estilo Chladni: Modo de vibración {(m,n)}+{(m1,n1)}')
if (m1*n1)<0:
    ax.set_title(f'Membrana estilo Chladni: Modo de vibración {(m,n)}-{(abs(m1),abs(n1))}')

def animate(i):
    global surf
    
    sim.calcular_tiempo()
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

if (m1*n1) == 0:
    anim.save(f"Onda3Dimension (modo {(m,n)}).mp4", writer=writer)
if (m1*n1)>0:
    anim.save(f"Onda3Dimension (modo {(m,n)}+{(m1,n1)}).mp4", writer=writer)
if (m1*n1)<0:
    anim.save(f"Onda3Dimension (modo {(m,n)}-{(abs(m1),abs(n1))}).mp4", writer=writer)

import winsound
winsound.Beep(450, 700)
'''