import numpy as np
from matplotlib.colors import LinearSegmentedColormap
'''
Código hecho por estudiante Kevin Mauricio Chacón Ureña (C32060) para la sección SimLab: Simulation Challenges 
de Physics in Silico (PhiS). Agradecimientos especiales al coordinador de la sección Gabriel Álvarez Castrillo
por el apoyo.

Proyecto: Vibraciones de una membrana tipo Chladni

'''

class Onda2Dim:
    """
    Simulación numérica de la ecuación de onda bidimensional
    usando el método de diferencias finitas explícito.

    La ecuación resuelta es:

        ∂²u/∂t² = c² (∂²u/∂x² + ∂²u/∂y²)

    con condiciones de frontera de Dirichlet homogéneas (u = 0 en los bordes).

    Parameters
    ----------
    L : float
        Longitud del dominio cuadrado [m].
    c : float
        Velocidad de propagación de la onda [m/s].
    h : float
        Paso espacial (Δx = Δy) [m].
    d : float
        Paso temporal (Δt) [s].

    Attributes
    ----------
    x0 : int
        Número de puntos en la dirección x.
    y0 : int
        Número de puntos en la dirección y.
    c : float
        Velocidad de la onda.
    h : float
        Paso espacial.
    d : float
        Paso temporal.
    a2 : float
        Parámetro de estabilidad α² = (c Δt / Δx)².
    u : ndarray
        Arreglo de dimensión (3, x0, y0) que almacena la solución
        en tres tiempos consecutivos:
            u[2] → tiempo k-1
            u[1] → tiempo k
            u[0] → tiempo k+1
    """

    def __init__(self, L, c, h, d):
        """
        Inicializa la simulación de la onda 2D.
        """
        self.x0 = int(L / h)
        self.y0 = int(L / h)
        self.c = c
        self.h = h
        self.d = d
        self.a2 = (c * d / h) ** 2  # α² (condición CFL)
        self.u = np.zeros((3, self.x0, self.y0))

    def calcular_tiempo(self):
        """
        Avanza la solución un paso temporal usando el esquema explícito:

            u^{k+1}_{i,j} = α² (u vecinos - 4u_{i,j})
                            + 2u^k_{i,j}
                            - u^{k-1}_{i,j}

        Se aplican condiciones de frontera de Dirichlet homogéneas
        (u = 0 en todos los bordes).

        Notes
        -----
        El método es estable si se cumple la condición CFL:

            α² ≤ 1/2   (en 2D)

        donde α² = (c Δt / Δx)².
        """
        u, x0, y0 = self.u, self.x0, self.y0

        # Desplazamiento temporal: k → k-1, k+1 → k
        u[2] = u[1]
        u[1] = u[0]

        # Condiciones de frontera (Dirichlet homogéneas)
        u[0, 0, :] = 0
        u[0, -1, :] = 0
        u[0, :, 0] = 0
        u[0, :, -1] = 0

        # Actualización en el interior
        u[0, 1:x0-1, 1:y0-1] = (
            self.a2 * (
                u[1, 1:x0-1, 0:y0-2] +
                u[1, 1:x0-1, 2:y0] +
                u[1, 0:x0-2, 1:y0-1] +
                u[1, 2:x0, 1:y0-1] -
                4 * u[1, 1:x0-1, 1:y0-1]
            )
            + 2 * u[1, 1:x0-1, 1:y0-1]
            - u[2, 1:x0-1, 1:y0-1]
        )


# Paleta personalizada para visualización
colors = [
    (1, 1, 1),
    (1, 1, 1),
    (1, 1, 1),
    (1, 1, 1),
    (1, 0, 0.5),
    (0, 0, 0),
    (1, 0.5, 0),
    (1, 1, 1),
    (1, 1, 1),
    (1, 1, 1),
    (1, 1, 1)
]

cmap_nodos = LinearSegmentedColormap.from_list('nodos', colors, N=512)
"""
Colormap personalizado para visualización de la membrana.

Esta paleta está diseñada para resaltar los valores cercanos a cero,
haciendo que los nodos (u ≈ 0) se distingan claramente del resto
de la amplitud de la onda.

- Valores cercanos a cero → tonos oscuros/contrastantes.
- Valores grandes positivos o negativos → tonos claros.

Pensado para identificar nodos estacionarios en modos normales.
"""