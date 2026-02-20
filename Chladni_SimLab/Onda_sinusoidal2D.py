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

#--------------------Programación animación 2D---------------------

def actualizar_anim(i):
    sim.u[0, sim.x0//2, sim.y0//2] = A * np.sin(i*d* w) #Condición de señal sinosoidal en el centro
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

ax.set_title(f'Membrana estilo Chladni: Señal sinusoidal forzada a {f} Hz')

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

anim.save(f"Onda2Dimension (Sinusoidal {f}Hz).mp4", writer=writer)

import winsound
winsound.Beep(450, 700)
'''