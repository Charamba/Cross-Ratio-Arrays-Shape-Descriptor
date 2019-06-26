"""
============================
Circles, Wedges and Polygons
============================
"""

import numpy as np
import matplotlib
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt


fig, ax = plt.subplots()

resolution = 50  # the number of vertices
N = 3
x = np.random.rand(N)
y = np.random.rand(N)

stepRadio = 1.0/N
#radii = 0.1*np.random.rand(N)
radii = [0.1, 0.2, 0.3]
patches = []
for x1, y1, r in zip(x, y, radii):
    circle = Circle((x1, y1), r)
    patches.append(circle)

stepColor = 100.00/N
print("stepColor = ", stepColor)
colors = np.arange(0, 100, stepColor)
print("colors: ", colors)
p = PatchCollection(patches, alpha=0.4)
p.set_array(np.array(colors))
ax.add_collection(p)
fig.colorbar(p, ax=ax)

plt.show()