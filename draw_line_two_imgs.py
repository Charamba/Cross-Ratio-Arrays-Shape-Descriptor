import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import numpy as np

fig = plt.figure(figsize=(10,5))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

x,y = np.random.rand(100),np.random.rand(100)

ax1.plot(x,y,'ko')
ax2.plot(x,y,'ko')

i = 10
xyA = (x[i],y[i])
xyB = (x[5],y[5])
con = ConnectionPatch(xyA=xyA, xyB=xyB, coordsA="data", coordsB="data",
                      axesA=ax2, axesB=ax1, color="red")
ax2.add_artist(con)

ax2.plot(x[i],y[i],'ro',markersize=20)
ax1.plot(x[5],y[5],'ro',markersize=10)


plt.show()