import math
import numpy as np
from Point import R2_Point
import copy
from Utils import *
# 0.046
#valTol = 0.76#0.0015#0.25#0.073
percentualTol = 0.2#0.081#0.05#0.012
# 0.03512794452
def simpleRatio(A, B, C):
	AC = A.euclideanDistance(C)
	BC = B.euclideanDistance(C)
	return AC/BC

def crossRatio(A, B, C, D):
	AC = A.euclideanDistance(C)
	BC = B.euclideanDistance(C)
	AD = A.euclideanDistance(D)
	BD = B.euclideanDistance(D)

	if BC == 0 or BD == 0 or AD == 0:
		# print("cross ratio value = 0")
		# print("points: ", [A, B, C, D])
		return 0

	# if (AC/BC)*(BD/AD) > 10:
	# 	print("cross ratio value = ", (AC/BC)*(BD/AD))
	# 	print("points: ", [A, B, C, D])

	return (AC/BC)*(BD/AD)

def det3(P1, P2, P3):
	(x1, y1, z1) = unpackPoint(P1)
	(x2, y2, z2) = unpackPoint(P2)
	(x3, y3, z3) = unpackPoint(P3)
	
	return x1*y2*z3 + x3*y1*z2 + x2*y3*z1 - (x3*y2*z1 + x2*y1*z3 + x1*y3*z2)

def crossRatio5(A1, B1, A2, B2, O):
	det_A1_A2_O = det3(A1, A2, O)
	det_A1_B2_B1 = det3(A1, B2, B1)


	if det_A1_A2_O == 0 or det_A1_B2_B1 == 0:
		return 0

	cp1 = det3(A1, A2, B1)*det3(A1, B2, O)/(det_A1_A2_O*det_A1_B2_B1)
	#cp2 = det3(A2, A1, B1)*det3(A2, O, B2)/(det3(A2, A1, B2)*det3(A2, O, B1))
	#O = P2_Point(0, 0, 1)

	# print(det3(A1, A2, O))
	# print(det3(A1, A2, B2))
	# print(det3(A2, B1, O))
	# print(det3(A2, B1, B2))


	#cp1 = (det3(A1, A2, O)/det3(A1, A2, B2))/(det3(A2, B1, O)/det3(A2, B1, B2))
	return (cp1)#, cp2)

def fiveCrossRatio(P1, P2, P3, P4, P5):
	det124 = det3(P1, P2, P4) 
	det135 = det3(P1, P3, P5)
	deno = det124*det135
	if deno == 0:
		return 0

	return det3(P1, P2, P3)*det3(P1, P4, P5)/deno

def suitable5CR(P1, P2, P3, P4, P5):
	q = fiveCrossRatio(P1, P2, P3, P4, P5)
	return 2*q/(q*q + 1)

def invariant5CrossRatio(P1, P2, P3, P4, P5):
	r = suitable5CR(P1, P2, P3, P4, P5)*suitable5CR(P1, P2, P3, P5, P4)*suitable5CR(P1, P2, P4, P5, P3)
	return r

def crossRatioFilter(points=[]):
	crossRatioValues = []
	calcCrossRatioTolerances = []
	nPoints = len(points)
	if nPoints >= 4:
		nSlideComb = nPoints - 2 #3
		for i in range(1, nSlideComb):
			#crossRatioValues.append(crossRatio(points[i], points[i+1], points[i+2], points[i+3]))
			crossRatioValues.append(crossRatio(points[0], points[i], points[i+1], points[-1]))
			#calcCrossRatioTolerances.append(calcCrossRatioTolerance(points[i], points[i+1], points[i+2], points[i+3]))

	else:
		return [0]
	return 	crossRatioValues#, calcCrossRatioTolerances)

def crossRatioFilter5(points=[], junctionPoint=None):
	crossRatioValues = []
	O = junctionPoint
	calcCrossRatioTolerances = []
	nPoints = len(points)
	if nPoints >= 4:
		nSlideComb = nPoints - 3
		for i in range(0, nSlideComb):
			crossRatioValues.append(crossRatio5(points[i], points[i+1], points[i+2], points[i+3], O))
			#calcCrossRatioTolerances.append(calcCrossRatioTolerance(points[i], points[i+1], points[i+2], points[i+3]))

	else:
		return ([0],[0])
	return 	crossRatioValues#, calcCrossRatioTolerances)

def invariant5CrossRatioFilter(points, vertice=None):
	crossRatioValues = []
	calcCrossRatioTolerances = []
	nPoints = len(points)

	factor = 0
	if vertice != None:
		factor = 1


	if nPoints >= 5 - factor:
		nSlideComb = nPoints - (4 - factor)
		for i in range(0, nSlideComb):

			if vertice != None:
				p0 = vertice
			else:
				p0 = points[i]

			crossRatioValues.append(crossRatio5(p0, points[i+1-factor], points[i+2-factor], points[i+3-factor], points[i+4-factor]))
	return crossRatioValues

def calcCrossRatioTolerance(p1, p2, p3, p4):
	# d12 = p1.euclideanDistance(p2)
	# d23 = p2.euclideanDistance(p3)
	# d34 = p3.euclideanDistance(p4)

	# d13 = d12 + d23
	# d24 = d23 + d34
	# d14 = d12 + d24

	# tol = (d13**2)*(d34**2)/((d14**2)*(d23**4)) + (d12**2)*(d13**2)/((d14**4)*(d23**2)) + (d12**2)*(d24**2)/((d14**2)*(d23**4)) + (d24**2)*(d34**2)/((d14**4)*(d23**2))
	
	# d14Pow2 = d14*d14
	# tol = 1/d14Pow2


	return valTol#math.sqrt(tol)

def calcDistanceTolerance(crossRatioVector):
	tol = percentualTol
	#print("crossRatioVector: ", crossRatioVector)
	seq = crossRatioVector
	return sum([r*tol for r in seq]) #math.sqrt(


def isEqualValues(val1, val2, valTol_):
	return (val2 - valTol_) <= val1 <= (val2 + valTol_)


# def isEqualVectors(crossValues1, crossValues2, tol, vectorTolerances):
# 	if len(crossValues1) == len(crossValues2):
# 		for i in range(0, len(crossValues1)):
# 			val1 = crossValues1[i]
# 			val2 = crossValues2[i]
# 			valTol_ = vectorTolerances[i]
# 			if not isEqualValues(val1, val2, valTol_):
# 				return False
# 		return True
# 	else:
# 		return False



def calcDistanceVector(vector1, vector2):
	lenVector = len(vector1)
	distance = 1000 # INF
	if lenVector == len(vector2):
		distance = 0
		for i in range(0, len(vector1)):
			val1 = vector1[i]
			val2 = vector2[i]
			diffVal = val2 - val1
			diffVal = diffVal*diffVal
			distance += diffVal
		distance = math.sqrt(distance)
	return distance


def compareCellPerCell(vector1, vector2, cell_tol):
	matchingCounter = 0
	lenVector = len(vector1)
	distance = 1000 # INF
	if lenVector == len(vector2):
		for i in range(0, len(vector1)):
			val1 = vector1[i]
			val2 = vector2[i]
			diffVal = abs(val2 - val1)
			if cell_tol*max(val1, val2) >= diffVal:
				matchingCounter += 1
	
	percentMatching = 0
	if lenVector > 0:
		percentMatching = matchingCounter/lenVector

	return percentMatching


def isEqualVectors(crossValues1, crossValues2, distanceTol):
	if len(crossValues1) == len(crossValues2):
		# for (val1, val2) in zip(crossValues1, crossValues2):
		# 	if not(abs(val2 - val1) <= val1*percentualTol):
		# 		return False
		# return True

		distance = calcDistanceVector(crossValues1, crossValues2)
		#if sumTol >= 2:
		#	print("size = ", len(crossValues1))
		#	print("distance = ", distance)
		#	print("sumTol = ", sumTol)
		#	print("sumTol*valTol = ", sumTol*valTol)
		#lenVector = len(crossValues1)
		#vectorTol = lenVector*valTol#sumTol#*valTol#lenVector*valTol
		# print("distance  = ", distance)
		# print("distanceTol = ", distanceTol)
		return (distance <= distanceTol)
	else:
		return False

'''
	lenVector = len(crossValues1)
	if lenVector == len(crossValues2):
		distance = 0
		for i in range(0, len(crossValues1)):
			val1 = crossValues1[i]
			val2 = crossValues2[i]
			diffVal = val2 - val1
			diffVal = diffVal*diffVal
			distance += diffVal
		distance = math.sqrt(distance)
		vectorTol = lenVector*valTol
		return (- vectorTol <= distance <= vectorTol)
'''

def compareCrossRatiosVectors(vector1, vector2, distanceTol):
	# print("vector1 = ", vector1)
	# print("vector2 = ", vector2)
	#sumTol = math.sqrt(sum([t*t for t in vectorTolerances])) #sum(tolerances2)
	if isEqualVectors(vector1, vector2, distanceTol):
		#print("compareCrossRatiosVectors: true,true")
		return (True, True)
	else:
		vector2_ = copy.deepcopy(vector2)
		vector2_.reverse()
		#print("compareCrossRatiosVectors: ", isEqualVectors(vector1, vector2_, sumTol))
		return (isEqualVectors(vector1, vector2_, distanceTol), False)

"""
A = R2_Point(0, 0)
B = R2_Point(0, 0.95)
C = R2_Point(0, 0.95+1.75)
D = R2_Point(0, 0.95+1.75+3.6)

E = R2_Point(0, 0.95+1.75+3.6 + 2.0)
F = R2_Point(0, 0.95+1.75+3.6 + 4.0)
G = R2_Point(0, 0.95+1.75+3.6 + 7.0)
H = R2_Point(0, 0.95+1.75+3.6 + 9.0)


edgePoints1 = [A, B, C, D, E, F, G, H]
edgePoints2 = [A, B, C, D, E, F, G, H]
edgePoints2.reverse()

crossRV1 = crossRatioFilter(edgePoints1)
crossRV2 = crossRatioFilter(edgePoints2)

print("cross ratio values1 = ", crossRV1)
print("cross ratio values2 = ", crossRV2)

print("compareCrossRatiosVectors = ", compareCrossRatiosVectors(crossRV1, crossRV2))
"""




# cr5vectorTemplate = [0.8993799740727911, 0.06552501293135299, 0.007481991191622515, 0.7269387781499284, 0.0034725190336924033, 0.06337601484562531, 0.9053433484125969]
# cr5vectorTest = [0.9539378247574579, 0.024929637360532646, 0.016669482399608818, 0.727639016927337, 0.0012661011136918786, 0.5976631136144654, 0.35658122903519274]

# cell_tol = 0.5
# print("compareCellPerCell: ", compareCellPerCell(cr5vectorTemplate, cr5vectorTest, cell_tol))