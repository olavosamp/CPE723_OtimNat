import numpy                as np
import matplotlib.pyplot    as plt
from mpl_toolkits.mplot3d   import Axes3D
from maza_state             import ackley

x1 = np.linspace(-5, 5)
x2 = np.linspace(-5, 5)
X, Y = np.meshgrid(x1, x2)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

Z = np.zeros((len(X), len(Y)))
for i in range(len(X)):
    for j in range(len(Y)):
        Z[i,j] = ackley([X[i,j], Y[i,j]])

ax.plot_surface(X, Y, Z, cmap='ocean')

plt.show()
