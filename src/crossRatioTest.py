import CrossRatio
from CrossRatio import *


vector1 = [1.2832748538011696, 2.3439781818268774, 1.0041555836589726, 22.387513797774382]
vector2 = [1.2869352869352877, 2.3222390974180924, 1.004937630712958, 19.13938951974164]

tol1 = calcDistanceTolerance(vector1)
tol2 = calcDistanceTolerance(vector2)

distance = calcDistanceVector(vector1, vector2)

print("tol1 = ", tol1)
print("tol2 = ", tol2)
print("Distance = ", distance)