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

A = 1
d = 0.001
L = 1
h=0.005
sim = Onda2Dim(L, 3, h, d)
t_fin=1.5

m, n= 3,2#modos de vibración

m1,n1=0,0

for i in range(sim.x0): #posición inicial del modo
    for j in range(sim.y0):
        sim.u[0,j,i] = A * (np.sin(m*np.pi*i*h/L) * np.sin(n*np.pi*j*h/L)+np.sin(m1*np.pi*i*h/L) * np.sin(n1*np.pi*j*h/L))
        sim.u[1,j,i] = sim.u[0,j,i].copy()


#--------------------Programación animación 2D---------------------

def actualizar_anim(i):
    sim.calcular_tiempo()

def init():
    return img,

fig, ax = plt.subplots(figsize=(8, 8),
    dpi=200)

img = ax.imshow(sim.u[0], vmin=-A, vmax=A, cmap=cmap_nodos)

contador = plt.text(
    0.0, 0.0, '',            # posición en coordenadas del eje
    fontsize=12,
    verticalalignment='top', color='white')


ax.set_xticks([])
ax.set_yticks([])
plt.xlabel('x [m]')
plt.ylabel('y [m]')

cbar = plt.colorbar(img, ax=ax)
cbar.set_label("Amplitud de la onda [m]")

if (m1*n1) == 0:
    ax.set_title(f'Membrana estilo Chladni: Modo de vibración {(m,n)}')
if (m1*n1)>0:
    ax.set_title(f'Membrana estilo Chladni: Modo de vibración {(m,n)}+{(m1,n1)}')
if (m1*n1)<0:
    ax.set_title(f'Membrana estilo Chladni: Modo de vibración {(m,n)}-{(abs(m1),abs(n1))}')

def animate(i):
    actualizar_anim(i)
    img.set_data(sim.u[0])
    contador.set_text(f"t = {i*d:.2f} s")
    return img, contador

interval, nframes = sim.d/10, int(t_fin/d)
anim = animation.FuncAnimation(fig, animate, frames=nframes,
                              repeat=False,
                              init_func=init, interval=interval, blit=True)

plt.show()

'''
#Si se descomenta guardará la animación como '.mp4', importante entonces remover 'plt.show()'

writer = FFMpegWriter(
    fps=int(0.1/d),   # frames por segundo coherente con d
    metadata=dict(artist='Matplotlib'),
    bitrate=1800
)

if (m1*n1) == 0:
    anim.save(f"Onda2Dimension (modo {(m,n)}).mp4", writer=writer)
if (m1*n1)>0:
    anim.save(f"Onda2Dimension (modo {(m,n)}+{(m1,n1)}).mp4", writer=writer)
if (m1*n1)<0:
    anim.save(f"Onda2Dimension (modo {(m,n)}-{(abs(m1),abs(n1))}).mp4", writer=writer)

import winsound
winsound.Beep(450, 700)
'''