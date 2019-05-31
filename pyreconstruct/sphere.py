import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib.ticker
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import FormatStrFormatter



# generate a uniform distribution of points on a sphere
R = 1 # metre fixed radius

thetas = np.zeros(500)
phis   = np.zeros(len(thetas))
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
for i in range(len(thetas)):
    thetas[i] = 2 * np.pi * random.uniform(0,1)
    phis[i] = np.arccos(2*random.uniform(0,1)-1)
    

ax.scatter(np.sin(phis)*np.cos(thetas), np.sin(phis)*np.sin(thetas), np.cos(phis), c="b", s=3)
ax.scatter(-np.sin(phis)*np.cos(thetas), -np.sin(phis)*np.sin(thetas), -np.cos(phis), c="b", s=3)

ax.set_aspect('equal')

ax.set_xlabel("x ")
ax.set_ylabel("y ")
ax.set_zlabel("z ")


print(np.where(thetas>2*np.pi))


# done properly sphere looks like this! 
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(thetas, phis, "bo", ms=3)
# ax.xaxis.set_major_formatter(FormatStrFormatter('%i $\pi$'))
# ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=np.pi))
# ax.yaxis.set_major_formatter(FormatStrFormatter('%i $\pi$'))
# ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=np.pi))

ax.set_xticks(np.linspace(0, 2*np.pi, 5))
ax.set_xticklabels(["0", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$" ])
ax.set_yticks(np.linspace(0, np.pi, 3))
ax.set_yticklabels(["0",  r"$\frac{\pi}{2}$", r"$\pi$"])

plt.xlabel(r"$\theta$")
plt.ylabel(r"$\phi$")

plt.tight_layout()
# plt.show()


# done WRONG it looks like this...

thetas2 = np.zeros(len(thetas))
phis2   = np.zeros(len(thetas))

for i in range(len(thetas2)):
    thetas2[i] = random.uniform(0, 1) * 2 * np.pi
    phis2[i]   = random.uniform(0,1) * np.pi

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(np.sin(phis2)*np.cos(thetas2), np.sin(phis2)*np.sin(thetas2), np.cos(phis2), c="b", s=3)
ax.scatter(-np.sin(phis2)*np.cos(thetas2), -np.sin(phis2)*np.sin(thetas2), -np.cos(phis2), c="b", s=3)

ax.set_aspect('equal')

ax.set_xlabel("x ")
ax.set_ylabel("y ")
ax.set_zlabel("z ")

# 2D plot of thetas vs phis
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(thetas2, phis2, "bo", ms=3)
ax.set_xticks(np.linspace(0, 2*np.pi, 5))
ax.set_xticklabels(["0", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$" ])
ax.set_yticks(np.linspace(0, np.pi, 3))
ax.set_yticklabels(["0",  r"$\frac{\pi}{2}$", r"$\pi$"])

plt.xlabel(r"$\theta$", fontsize=15)
plt.ylabel(r"$\phi$", fontsize=15)
ax.tick_params(axis = 'both', which = 'major', labelsize = 14)


plt.tight_layout()
plt.show()