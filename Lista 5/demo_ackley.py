import numpy                as np
import matplotlib.pyplot    as plt
from mpl_toolkits.mplot3d   import Axes3D
from maza_state             import ackley

x1 = np.linspace(-5, 5)
x2 = np.linspace(-5, 5)
X, Y = np.meshgrid(x1, x2)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, ackley(X, Y), cmap='ocean')

plt.show()
