import numpy as numpy
import matplotlib.pyplot as matplotlib

x = numpy.linspace(-50, 50, 101)
y = numpy.linspace(-80, 80, 161)
X, Y = numpy.meshgrid(x, y)

V = numpy.zeros(X.shape)
Terminals = numpy.zeros(X.shape)

for i in range(40, 60):
    for j in range (70, 90):
        Terminals[i, j] = 12


e = 0.01


terminal_mask = Terminals > 0

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

    # Top border
    border_values[0, :] = grid[1, :]
    # Bottom border
    border_values[-1, :] = grid[-2, :]
    # Left border
    border_values[:, 0] = grid[:, 1]
    # Right border
    border_values[:, -1] = grid[:, -2]

    return border_values


while True:
    previousV = V
    V = relaxation(V)
    V = neumann(V)
    diff = numpy.abs(V - previousV).max()
    if diff < e:
        break


fig3d, ax3d = matplotlib.subplots(subplot_kw={"projection": "3d"})
surf = ax3d.plot_surface(X, Y, V, cmap='coolwarm', linewidth=0, antialiased=False)

matplotlib.style.use('_mpl-gallery-nogrid')
fig, ax = matplotlib.subplots(1, 1, figsize=(6, 6))

levels = numpy.linspace(V.min(), V.max(), 7)

contour = ax.contourf(X, Y, V, levels=levels)

# Datos extra para el grafico 2
ax.set_title('Equipotential Lines')
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.axis('equal')
fig.colorbar(contour, ax=ax, orientation='vertical')

matplotlib.tight_layout()
matplotlib.show()