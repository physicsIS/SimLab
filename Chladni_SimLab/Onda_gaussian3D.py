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
y se visualiza su evolución temporal mediante animación en TRES dimensiones.

Hecho por: Kevin Mauricio Chacón Ureña (C32060)

"""

#------------------Cond. iniciales Perfil Gaussiano-------------

A = 1 #Amplitud
sigma = 11 #Ancho perfil Gaussiano

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


#-----------Programación animación 3D------------

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


ax.set_title('Membrana estilo Chladni: Perfil Gaussiano en el centro')

def actualizar_anim(i):
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
    fps=int(0.1/d),   # frames por segundo coherente con tu dt
    metadata=dict(artist='Matplotlib'),
    bitrate=1800
)

anim.save("Onda3Dimension (Gaussiana).mp4", writer=writer)

import winsound
winsound.Beep(450, 700)
'''