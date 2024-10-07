import numpy as numpy
import matplotlib.pyplot as matplotlib

x = numpy.linspace(-50, 50, 101)
y = numpy.linspace(-80, 80, 161)
X, Y = numpy.meshgrid(x, y)

V = numpy.zeros(X.shape)
Terminals = numpy.zeros(X.shape)

for i in range(40, 60):
    for j in range (40, 60):
        Terminals[i, j] = 12

for i in range(100, 120):
    for j in range (40, 60):
        Terminals[i, j] = -12


e = float(input("value of epsilon: "))


terminal_mask = Terminals != 0

V = Terminals

def relaxation(grid):
    result = grid.copy()

    left_shift = numpy.roll(grid, 1, axis=1)
    right_shift = numpy.roll(grid, -1, axis=1)
    up_shift = numpy.roll(grid, 1, axis=0)
    down_shift = numpy.roll(grid, -1, axis=0)

    left_shift[terminal_mask] = Terminals[terminal_mask]
    right_shift[terminal_mask] = Terminals[terminal_mask]
    up_shift[terminal_mask] = Terminals[terminal_mask]
    down_shift[terminal_mask] = Terminals[terminal_mask]

    avg_neighbors = (left_shift + right_shift + up_shift + down_shift) / 4

    result[1:-1, 1:-1] = avg_neighbors[1:-1, 1:-1]

    return result


def neumann(grid):
    border_values = grid.copy()

    # Borde superior
    border_values[0, :] = grid[1, :]
    # Borde inferior
    border_values[-1, :] = grid[-2, :]
    # Borde izquierdo
    border_values[:, 0] = grid[:, 1]
    # Borde derecho
    border_values[:, -1] = grid[:, -2]

    return border_values


while True:
    previousV = V
    V = relaxation(V)
    V = neumann(V)
    diff = numpy.abs(V - previousV).max()
    if diff < e:
        break


fig = matplotlib.figure(figsize=matplotlib.figaspect(0.4))

ax = fig.add_subplot(1, 2, 1)
levels = numpy.linspace(V.min(), V.max(), 40)
contour = ax.contourf(X, Y, V, levels=levels, cmap='coolwarm')


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