import math
from LineEstimation import *
from Point import *

def cov(X, Y):
	n = len(X)
	x_ = sum(X)/n
	y_ = sum(Y)/n

	ret = 0
	for (xi, yi) in zip(X, Y):
		ret += (xi - x_)*(yi - y_)
	return ret

def var(V):
	n = len(V)
	v_ = sum(V)/n
	
	ret = 0
	for vi in V:
		ret += (vi - v_)*(vi - v_)
	return ret

def pearsonCoefficient(X, Y):
	n = len(X)
	m = len(Y)
	if n == 0 or m == 0:
		return 0
	else:
		var_x = var(X)
		var_y = var(Y)
		if var_x == 0 or var_y == 0:
			return 0
		return cov(X, Y)/(math.sqrt(var_x*var_y))

def getVerticesIndices(verticesPairs):
	Y = [y for (x, y) in verticesPairs]
	return Y

def shifftSignalToMinValue(Y):
	minIdx = Y.index(min(Y))
	part1 = Y[minIdx+1:]
	part2 = Y[:minIdx]

	minVal = Y[minIdx]

	dif_part1 = float('inf')
	if part1:
		dif_part1 = part1[0] - minVal
	dif_part2 = float('inf')
	if part2:
		dif_part2 = part2[-1] - minVal

	if dif_part1 < dif_part2:
		part1 = [minVal] + part1 # crescente
	else:
		part2 = part2 + [minVal] # decrescente
	return part1 + part2

def removePeaksAndDepressionsValues(signal):
	S = signal
	newSignal = []

	if S[0] < S[1]:
		newSignal.append(S[0])

	for i in range(1, len(S)-1):
		if not(S[i] == min(S[i-1], S[i], S[i+1]) or S[i] == max(S[i-1], S[i], S[i+1])):
			#print(S[i])
			newSignal.append(S[i])

	if S[-1] > S[-2]:
		newSignal.append(S[-1])

	return newSignal

def filteredVerticePairs(testVertices, verticePairs, distanceValues=None):
	newPairs = []
	newDistances = []
	W = testVertices
	for w2 in W:
		for i, (v1, v2) in enumerate(verticePairs):
			if v2 == w2:
				newPairs.append((v1, v2))
				if not(distanceValues == None):
					newDistances.append(distanceValues[i])

	if not(distanceValues == None):
		return newPairs, newDistances

	return newPairs

def removeOutLiers(points):
	model = LinearLeastSquaresModel()
	minimumNumber = int(len(points)*0.3)
	iterations = 1
	Y = [p.y for p in points]
	mY = sum(Y)/len(Y)
	fitDist = mY*0.5
	numberOfcloses = int(len(points)*0.5)
	inliers = ransac(points, model, minimumNumber, 1, fitDist, numberOfcloses, debug=False,return_all=True, random=False)
	
	# #Tentando remover outliers pelo coeficiente de Pearson
	# if inliers == []:
	# 	minPercent = 0.2
	# 	X = [p.x for p in points]
	# 	minPoints = 1#int(minPercent*len(X))
		

	# 	bestPearsonValue = pearsonCoefficient(X, Y)
	# 	removeCount = 0
	# 	i = 0
	# 	outliersIndices = []
	# 	for i in range(0, len(X)):
	# 		# removing by index i
	# 		newX = X[:i] + X[i+1:] 
	# 		newY = Y[:i] + Y[i+1:]

	# 		if pearsonCoefficient(newX, newY) > bestPearsonValue:
	# 			outliersIndices.append(i)
	# 			removeCount++


	return inliers

# verticesPairs = [(0, 22), (1, 0), (2, 2), (3, 3), (5, 6), (6, 7), (9, 8), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16), (16, 18), (17, 19), (18, 20), (19, 5), (20, 21)]


# Y = getVerticesIndices(verticesPairs)
# Y = shifftSignalToMinValue(Y)
# X = range(0, len(Y))

# print("HONDA - HONDA:")
# print(pearsonCoefficient(X, Y))


# Y =[2,25,17,20,8,7,19,3,24,4,26,14,13,11]
# X = range(0, len(Y))
# print("HONDA - ADIDAS")
# print(pearsonCoefficient(X, Y))




# vertPairs = [(0, 22), (1, 0), (2, 2), (3, 3), (4, 19), (5, 6), (8, 8), (10, 12), (12, 13), (13, 14), (14, 15), (15, 16), (16, 18), (17, 4), (18, 20), (19, 21)]

# Y = getVerticesIndices(vertPairs)
# Y = shifftSignalToMinValue(Y)

# #Y = [0, 2, 3, 19, 6, 8, 12, 13, 14, 15, 16, 18, 4, 20, 21, 22]
# print("Y = ", Y)
# W = removePeaksAndDepressionsValues(Y)
# print("W = ", W)


# # for w2 in W:
# # 	for (v1, v2) in vertPairs:
# # 		if v2 == w2:
# # 			newPairs.append((v1, v2))

# newPairs = filteredVerticePairs(W, vertPairs)

# print("newPairs: ", newPairs)

# for i in range(1, len(Y) - 1):
# 	if not(Y[i] == min(Y[i-1], Y[i], Y[i+1]) or Y[i] == max(Y[i-1], Y[i], Y[i+1])): 
# 		print(Y[i])




# # dado ruim
# #[(0, 20), (1, 8), (3, 7), (5, 19), (6, 3), (7, 24), (8, 4), (9, 26), (12, 14), (14, 13), (16, 11), (17, 2), (18, 25), (19, 17)]

# #badData = [(17, 2), (18, 25), (19, 17), (0, 20), (1, 8), (3, 7), (5, 19), (6, 3), (7, 24), (8, 4), (9, 26), (12, 14), (14, 13), (16, 11)]

# badData = [(2, 1), (4, 3), (5, 4), (7, 11), (8, 14), (11, 2), (12, 9), (13, 10), (14, 8), (0, 12), (1, 16)]

# Y = [y for (x, y) in badData]

# # good Y
# #Y = [1, 2, 22, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 3, 23, 25]

# mY = sum(Y)/len(Y)
# print("len(Y): ", len(Y))
# print("average(Y): ", mY)

# points = []
# for i, y in enumerate(Y):
# 	points.append(R2_Point(i, y))


# model = LinearLeastSquaresModel()
# minimumNumber = int(len(points)*0.3)
# iterations = 1
# fitDist = mY*0.5
# numberOfcloses = int(len(points)*0.5)
# inliers = ransac(points, model, minimumNumber, 1, fitDist, numberOfcloses, debug=False,return_all=True, random=False)

# #remove outliers
# inliers = removeOutLiers(points)

# newY = []
# X = []
# if inliers:
# 	newY = [p.y for p in inliers]
# 	X = list(range(0, len(newY)))

# print("newY: ", newY)
# print("len(newY) = ", len(newY))
# print("percent size = ", len(newY)/len(Y))
# print("pearson = ", pearsonCoefficient(X, newY))




### MEDIAN FILTER TESTS

# badData = [(17, 2), (18, 25), (19, 17), (0, 20), (1, 8), (3, 7), (5, 19), (6, 3), (7, 24), (8, 4), (9, 26), (12, 14), (14, 13), (16, 11)]
# Y = [y for (x, y) in badData]

# print("Y = ", Y)

# good data
#Y  = [1, 7, 8, 14, 20, 2]


def median(val1, val2, val3):
	values = [val1, val2, val3]
	minVal = min(values)
	maxVal = max(values)

	medianVal = 0
	for v in values:
		if minVal < v < maxVal:
			medianVal = v
	return medianVal


def median1DFilter(data, growing=True):
	n = len(data)
	newData = []

	firstValue = float('inf')
	lastValue = -1

	if growing:
		firstValue = -1
		lastValue = float('inf')

	for i in range(0, n):

		val1 = firstValue
		if i != 0:
			val1 = data[i-1]

		val2 = data[i]

		val3 = lastValue
		if i != n-1:
			val3 = data[i+1]
		medianVal = median(val1, val2, val3)

		if not(medianVal in newData):
			newData.append(medianVal)
	return newData



# newY = median1DFilter(Y, growing=False)
# print("newY = ", newY)


# X = list(range(0, len(newY)))
# print("pcoef = ", pearsonCoefficient(X, newY))


# vertices:  [(0, 17), (1, 11), (2, 6), (5, 10), (6, 9), (8, 13), (13, 15), (17, 25), 
# (26, 23), (37, 26), (38, 24), (47, 2), (49, 0), (52, 18), (54, 19)]
#acessible - acessible real 2
# X = [0, 1, 2, 3]
X = [6, 8, 13, 52, 5] #<------- Pegar os índices corretos do Template para o Pearson
# X = [0, 1, 2, 3, 4]
newY = [9, 13, 15, 18, 10]
newY = [0, 1, 2, 5, 7, 8, 9, 13, 14, 15, 16, 17, 19, 20, 22, 23]#[y - min(X) for y in newY]

X = range(0, len(newY))
pcoef = pearsonCoefficient(X, newY)
# print("pcoef = ", pcoef)


correctPairsIndices =  [(9, 0), (11, 1), (14, 5), (15, 7), (16, 8), (12, 2), (17, 9), (20, 13), (21, 14), (22, 15), (23, 16), (0, 17), (2, 19), (4, 20), (5, 22), (8, 23)]

correctPairsIndices.sort(key=lambda p: int(p[1])) # ordenando pelo Y
#(x0,_) = correctPairsIndices[0]
#X = [x for (x,y) in correctPairsIndices]


# print("correctPairsIndices: ", correctPairsIndices)
# Y = [0, 1, 2, 5, 7, 8, 9, 13, 14, 15, 16, 17, 19, 20, 22, 23]#[y for (x,y) in correctPairsIndices]
# X = [x for (x, y) in correctPairsIndices if y in Y]

# X.sort()

# print("X: ", X)
# print("Y:", Y)


# tau =  [5, 7, 9, 11, 14, 16, 23, 24, 25, 35, 37]
# Q =   [15, 11, 17, 5, 3, 6, 16, 14, 13, 12, 0]

# pcoef = pearsonCoefficient(tau, Q)
# print("pcoef = ", pcoef)


#acessible - acessible real 3

# vertices = [(0, 0), (1, 1), (2, 2), (5, 3), (15, 19), (16, 8), (22, 11), (26, 21),
# 			(35, 20), (41, 23), (45, 22), (47, 9), (49, 28), (54, 13), (55, 12)]

# X = []
# Y = [0, 1, 2, 3, 8, 11, 20, 22]
# # for (x, y) in vertices:
# # 	if y in Y:
# # 		X.append(x)

# X = [x for (x, y) in vertices if y in Y]
# #X = [0, 1, 2, 5, 16, 22, 35, 45]
# print("X = ", X)
# #Y = [0, 1, 2, 3, 8, 11, 20, 22]#[0, 11, 22, 1, 2, 3, 8, 20]
# pcoef = pearsonCoefficient(X, Y)
# print("pcoef = ", pcoef)

# Y = [6,7,0,1,3,5]
# newY = shifftSignalToMinValue(Y)
# print("Y1: ", newY)

# Y = [3,2,1,0,8,7,6,5,4]
# newY = shifftSignalToMinValue(Y)
# print("Y2: ", newY)







