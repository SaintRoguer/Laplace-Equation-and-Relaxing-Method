import numpy as numpy
import matplotlib.pyplot as matplotlib
import pandas as pandas

# Define el tamaño de la grilla
x = numpy.linspace(-50, 50, 101)
y = numpy.linspace(-80, 80, 161)
X, Y = numpy.meshgrid(x, y)

# Define matrices para el potencial y los bornes
V = numpy.zeros(X.shape)
Terminals = numpy.zeros(X.shape)

# -------------------------------------------------------------------------------------------------
# ------------------------- SECCION PARA CREAR NUEVAS CONFIGURACIONES -----------------------------
# -------------------------------------------------------------------------------------------------

# Rectangulos
for i in range(131, 141):
    for j in range (20, 81):
        Terminals[i, j] = 12

# Circulos
""" center = [50, 35]
radius = 15

for i in range(0, 161):
    for j in range (0, 101):
        distance = numpy.sqrt((i-center[1])**2 + (j-center[0])**2)
        if distance < radius:
            Terminals[i, j] = 12 """

# Triangulos
vertex1 = [50, 50]
vertex2 = [40, 20]
vertex3 = [60, 20]

def triangle_area(v1, v2, v3):
     return abs((v1[0]*(v2[1] - v3[1]) + v2[0]*(v3[1] - v1[1]) + v3[0]*(v1[1] - v2[1])) / 2.0)

def in_triangle(x, y):
    A = triangle_area(vertex1, vertex2, vertex3)
    A1 = triangle_area([x, y], vertex2, vertex3)
    A2 = triangle_area(vertex1, [x, y], vertex3)
    A3 = triangle_area(vertex1, vertex2, [x, y])
    return A == A1 + A2 + A3

for i in range(0, 161):
    for j in range (0, 101):
        if in_triangle(j, i):
            Terminals[i, j] = 12

# Guarda configuracion creada
data_frame = pandas.DataFrame(Terminals)
data_frame.to_csv('terminals_files/test.csv', index=False, header=None)

# -------------------------------------------------------------------------------------------------

# Solicita input al usuario
file_name = input("Terminals file name: ")
e = float(input("value of epsilon: "))

# Carga la configuracion de bornes desde el archivo
data_frame = pandas.read_csv("terminals_files/" + file_name + ".csv", header=None)
Terminals = data_frame.values

# Mascara para los puntos correspondientes a bornes
terminal_mask = Terminals != 0

# Inicializa los potenciales a los valores de cada borne
V = Terminals


# Funcion para aplicar relajacion
def relaxation(grid):
    result = grid.copy()

    # Desplaza la grilla de potenciales una unidad en cada direccion (creando 4 nuevas)
    left_shift = numpy.roll(grid, 1, axis=1)
    right_shift = numpy.roll(grid, -1, axis=1)
    up_shift = numpy.roll(grid, 1, axis=0)
    down_shift = numpy.roll(grid, -1, axis=0)

    # Los puntos correspondientes a los bornes mantienen su potencial (CONDICION DE DIRICHLET)
    left_shift[terminal_mask] = Terminals[terminal_mask]
    right_shift[terminal_mask] = Terminals[terminal_mask]
    up_shift[terminal_mask] = Terminals[terminal_mask]
    down_shift[terminal_mask] = Terminals[terminal_mask]

    # Se calcula el promedio de potenciales
    avg_neighbors = (left_shift + right_shift + up_shift + down_shift) / 4

    # Se considera solo los puntos que no pertenecen al borde de la region de estudio
    result[1:-1, 1:-1] = avg_neighbors[1:-1, 1:-1]

    return result

# Funcion para aplicar condiciones de Neumann
def neumann(grid):
    border_values = grid.copy()

    # Se igualan los potenciales en los bordes con los puntos inmediatamente lindantes
    border_values[0, :] = grid[1, :]        # Borde superior
    border_values[-1, :] = grid[-2, :]      # Borde inferior
    border_values[:, 0] = grid[:, 1]        # Borde izquierdo
    border_values[:, -1] = grid[:, -2]      # Borde derecho

    return border_values


# Ciclo principal
while True:
    # Salva la distribucion de potencial actual
    previousV = V

    # Calcula la nueva distribucion aplicando relajacion y condiciones de borde
    V = relaxation(V)
    V = neumann(V)

    # Comprueba si se cumple el criterio de convergencia 
    diff = numpy.abs(V - previousV).max()
    if diff < e:
        break


# Crea un figura
fig = matplotlib.figure(figsize=matplotlib.figaspect(0.4))

# Crea el primer grafico, de curvas
ax = fig.add_subplot(1, 2, 1)
levels = numpy.linspace(V.min(), V.max(), 40)
contour = ax.contourf(X, Y, V, levels=levels, cmap='coolwarm')

# Crea el segundo grafico, de superficie
ax3d = fig.add_subplot(1, 2, 2, projection='3d')
ax3d.plot_surface(X, Y, V, cmap='coolwarm', linewidth=0, antialiased=True)


# Datos extra para el grafico 1
ax.set_title('Electrostatic Potential Distribution')
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.axis('equal')
fig.colorbar(contour, ax=ax, orientation='vertical', label='electrostatic potential [V]')

# Datos extra para el grafico 2
ax3d.set_title('Electrostatic Potential Surface')
ax3d.set_xlabel('x [m]')
ax3d.set_ylabel('y [m]')
ax3d.set_zlabel('electrostatic potential [V]')

matplotlib.tight_layout()
matplotlib.show()