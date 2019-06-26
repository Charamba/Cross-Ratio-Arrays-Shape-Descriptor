x_d = [18,23,25,35,65,54,34,56,72,19,23,42,18,39,37]
y_d = [202,186,187,180,156,169,174,172,153,199,193,174,198,183,178]
n=len(x_d)
import numpy.linalg
import matplotlib.pyplot as plt
B=numpy.array(y_d)
A=numpy.array(([[x_d[j], 1] for j in range(n)]))
X=numpy.linalg.lstsq(A,B)[0]
a=X[0]; b=X[1]
print("Line is: y=",a,"x+",b)

#var('x')


Xplot = []
Yplot = []

x = min(x_d)
y = a*x + b
Xplot.append(x)
Yplot.append(y)

x = max(x_d)
y = a*x + b
Xplot.append(x)
Yplot.append(y)

plt.plot(Xplot, Yplot, color='r')
#LineF=plt.plot(a*x+b,(x,min(x_d),max(x_d)))
#SP=plt.scatter_plot(zip(x_d,y_d), figsize=4, facecolor="lightgreen", edgecolor="green", markersize=30, marker='s')
plt.plot(x_d, y_d, 'bx')
plt.show()