import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from Onda2Dim import Onda2Dim
from Onda2Dim import cmap_nodos
from matplotlib.animation import FFMpegWriter

"""
Simulación de la ecuación de onda 2D en una membrana cuadrada
con condiciones de frontera de Dirichlet, usando el módulo 'Onda2Dim'

Se utiliza un perfil gaussiano inicial centrado en la membrana
y se visualiza su evolución temporal mediante animación en DOS dimensiones.

Hecho por: Kevin Mauricio Chacón Ureña (C32060)

"""

#------------------Cond. iniciales Perfil Gaussiano-------------

A = 1 #Amplitud
sigma = 11 #Ancho perfil Gaussiano (en unidades de grilla)

L = 1
h=0.005
d = 0.001

sim = Onda2Dim(L, 2, h, d)
t_fin=1.5

#Perfil Gaussiano inicial
xc, yc=sim.x0//2, sim.y0//2 #centro de la gota
for i in range(sim.x0):
    for j in range(sim.y0):
        sim.u[0,i,j] = A*np.exp(-((i-xc)**2+(j-yc)**2)/(2*sigma**2))
        sim.u[1,i,j] = sim.u[0,i,j]


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

ax.set_title('Membrana estilo Chladni: Perfil Gaussiano en el centro')

def animate(i):
    actualizar_anim(i) #i: frame actual
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
    fps=int(0.1/d),   # frames por segundo coherente con tu dt
    metadata=dict(artist='Matplotlib'),
    bitrate=1800
)

anim.save("Onda2Dimension (Gaussiana).mp4", writer=writer)


import winsound
winsound.Beep(1500, 700)
'''