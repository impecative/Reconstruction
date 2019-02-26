import numpy as np
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D



# generate a uniform distribution of points on a sphere
R = 1 # metre fixed radius

thetas = np.zeros(500)
phis   = np.zeros(len(thetas))
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
for i in range(len(thetas)):
    thetas[i] = 2 * np.pi * random.uniform(0,1)
    phis[i] = np.arccos(1 - 2*random.uniform(0,1))
    

    """phi = random.uniform(0, 2*np.pi)
    costheta = random.uniform(-1,1)
    # u = random.uniform(0,1)

    theta = np.arccos(costheta)
    r = R

    thetas[i] = theta
    phis[i]   = phi

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    ax.scatter(x,y,z, color="r", s=3)"""

ax.scatter(np.sin(phis)*np.cos(thetas), np.sin(phis)*np.sin(thetas), np.cos(phis), c="r", s=3)
ax.scatter(-np.sin(phis)*np.cos(thetas), -np.sin(phis)*np.sin(thetas), -np.cos(phis), c="b", s=3)

ax.set_xlabel("x ")
ax.set_ylabel("y ")
ax.set_zlabel("z ")


plt.figure()
plt.plot(thetas, phis, "bo", ms=3)

plt.show()
