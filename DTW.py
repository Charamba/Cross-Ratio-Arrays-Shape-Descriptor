import numpy as np
from scipy import stats

class costFunctionBinary:
	def compute(self, val1, val2):
		if val1 != val2:
			return 1
		else:
			return 0

class DTW:
	def __init__(self, signal1, signal2, costFunctionObject=None):
		self.signal1 = signal1#list(stats.zscore(np.array(signal1)))
		self.signal2 = signal2#list(stats.zscore(np.array(signal2)))
		self.m = len(signal1)
		self.n = len(signal2)
		self.costFunctionObject = costFunctionObject
		# Init distance matrix
		self.M = []#self.init_matrix()

	def costFunctionValue(self, val1, val2):
		costValue = 0
		if self.costFunctionObject != None:
			costValue = self.costFunctionObject.compute(val1, val2)
		else:
			# Default cost function
			costValue = self.cost(val1, val2)
		return costValue

	def init_matrix(self):
		M = [[j for j in [0]*self.n] for i in [0]*self.m]
		prevValue = 0
		for i in range(0, self.m):
			costValue = self.costFunctionValue(self.signal1[i], self.signal2[0])
			M[i][0] = costValue#prevValue + costValue
			#prevValue = M[i][0]

		prevValue = 0
		for j in range(0, self.n):
			costValue = self.costFunctionValue(self.signal1[0], self.signal2[j])
			M[0][j] = costValue#prevValue + costValue
			#prevValue = M[0][j]
		return M

	def computeMatrix(self):
		self.M = self.init_matrix()
		for i in range(1, self.m):
			for j in range(1, self.n):
				minValue = min(self.M[i-1][j-1], self.M[i-1][j], self.M[i][j-1])
				costValue = self.costFunctionValue(self.signal1[i], self.signal2[j])
				self.M[i][j] = minValue + costValue

	def findOptmPath(self):
		(i, j) = (0, 0)
		m = self.m
		n = self.n
		
		P = []
		while not(i >= m and j >= n):
			P.append((i, j))
			#print("i, j = %d, %d" %(i,j))
			#print("m = ", m)
			#print("n = ", n)
			(rightVal, leftVal, diagonalVal) = (float('Inf'), float('Inf'), float('Inf'))
			if (i + 1 < m) and (j + 1 < n):
				diagonalVal = self.M[i+1][j+1]
			if j + 1 < n:
				rightVal = self.M[i][j + 1]
			if i + 1 < m:
				leftVal = self.M[i + 1][j]

			minVal = min(rightVal, leftVal, diagonalVal)
			if diagonalVal == minVal:
				i = i + 1
				j = j + 1
			elif rightVal == minVal:
				j = j + 1
			else:
				i = i + 1

		P.append((m-1, n-1))
		return P

	def findOptmPath2(self):
		
		m = self.m
		n = self.n
		(i, j) = (m-1, n-1)
		P = []
		while not(i <= 0 and j <= 0):
			P.append((i, j))
			#print("i, j = %d, %d" %(i,j))
			#print("m = ", m)
			#print("n = ", n)
			(rightVal, leftVal, diagonalVal) = (float('Inf'), float('Inf'), float('Inf'))
			if (i - 1 >= 0) and (j - 1 >= 0):
				diagonalVal = self.M[i-1][j-1]
			if j - 1 >= 0:
				print("i =", i)
				print("j =", j)
				rightVal = self.M[i][j - 1]
			if i - 1 >= 0:
				leftVal = self.M[i - 1][j]

			minVal = min(rightVal, leftVal, diagonalVal)

			if leftVal == minVal:
				i = i - 1
			elif diagonalVal == minVal:
				i = i - 1
				j = j - 1
			elif rightVal == minVal:
				j = j - 1


		P.append((0, 0))
		return P

	def distance(self):
		self.computeMatrix()

		# for row in self.M:
		# 	#for cell in row:
		# 	print(row)
			#print("")	

		return self.M[self.m-1][self.n-1]

	def cost(self, v1, v2):
		return abs(v2 - v1)

	def completeBaseValue(self, spectre, maxLength):
		baseValue = min(spectre[0], spectre[-1])
		spectreLen = len(spectre)
		complementSpectre = [baseValue]*abs(maxLength-spectreLen)
		spectre += complementSpectre
		#print("spectre: ", spectre)
		return spectre

	def equalityLengthSpectres(self):
		len1 = len(self.signal1)
		len2 = len(self.signal2)
		maxLen = max(len1, len2)
		#print("s1:")
		self.signal1 = self.completeBaseValue(self.signal1, maxLen)
		#print("s2:")
		self.signal2 = self.completeBaseValue(self.signal2, maxLen)

		self.m = len(self.signal1)
		self.n = len(self.signal2)
		#return (signal1, signal2)
		

# Example 
# templateSpectre_v2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 16, 16, 16, 16, 16, 18, 18, 18, 18, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 20, 20, 20, 20, 14, 14, 14, 18, 18, 18, 18, 18, 14, 14, 14, 14, 14, 14, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 19, 19, 16, 16, 18, 19, 19, 20, 20, 20, 20, 18, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 16, 16, 16, 16, 10, 10, 10, 10, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 12, 12, 12, 12, 12, 12, 14, 12, 12, 12, 14, 12, 12, 14, 16, 14, 12, 14, 16, 14, 12, 14, 14, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 12, 12, 12, 12, 12, 12, 12, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]

# testSpectre_v1 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 18, 16, 16, 16, 18, 18, 18, 18, 16, 16, 16, 14, 14, 14, 14, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 14, 16, 16, 16, 16, 18, 16, 20, 20, 20, 18, 16, 16, 16, 16, 16, 16, 16, 16, 16, 19, 19, 19, 20, 20, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 16, 16, 16, 16, 16, 10, 10, 10, 10, 10, 12, 12, 12, 12, 12, 12, 12, 12, 14, 12, 12, 12, 12, 12, 12, 12, 12, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 14, 14, 14, 14, 14, 14, 14, 14, 12, 12, 12, 14, 14, 14, 14, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]


# #templateSpectre_v2 = [4, 4, 4, 5, 4, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 9, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 7, 7, 9, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 8, 7, 7, 7, 7, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 7, 7, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5];
# #testSpectre_v1 = [5, 5, 7, 7, 7, 7, 7, 7, 7, 7, 7, 9, 7, 7, 7, 7, 7, 7, 7, 7, 9, 7, 7, 7, 7, 5, 5, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 9, 9, 9, 9, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 4];


# templateSpectre_v2Errado = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 18, 16, 16, 16, 16, 16, 18, 18, 18, 18, 16, 16, 16, 16, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 20, 20, 18, 18, 14, 14, 14, 18, 18, 18, 18, 18, 14, 14, 14, 20, 20, 20, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 18, 18, 18, 18, 18, 18, 18, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 18, 16, 16, 18, 19, 19, 20, 21, 21, 21, 21, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 12, 12, 12, 12, 12, 12, 14, 12, 12, 12, 14, 12, 12, 12, 16, 16, 12, 12, 16, 16, 16, 16, 16, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
# templateSpectre_v2Errado += [0]*92
# testSpectre_v2Errado = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 14, 14, 16, 14, 14, 14, 16, 16, 18, 18, 16, 16, 16, 16, 16, 16, 20, 20, 20, 20, 16, 16, 18, 16, 16, 16, 18, 16, 18, 16, 16, 20, 20, 18, 18, 18, 20, 21, 21, 21, 21, 16, 16, 16, 16, 16, 20, 20, 18, 20, 20, 19, 19, 19, 19, 19, 19, 19, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 16, 16, 14, 16, 16, 16, 16, 16, 14, 16, 16, 14, 14, 18, 18, 18, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 16, 16, 16, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 16, 14, 14, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 18, 18, 16, 14, 14, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 6, 8, 6, 8, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]



# #templateSpectre_v2Errado = [4, 4, 4, 5, 4, 7, 7, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 8, 8, 8, 8, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
# #testSpectre_v2Errado = [5, 7, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 7, 5, 5, 5, 5, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 6, 6, 8, 8, 8, 8, 8, 8, 8, 8, 6, 6, 6, 6, 6, 6, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 4, 4]



# dtw = DTW(templateSpectre_v2, testSpectre_v1)
# dtw.equalityLengthSpectres()
# print("default distance: ", dtw.distance())

# costFunctionBin = costFunctionBinary()
# dtw2 = DTW(templateSpectre_v2, testSpectre_v1, costFunctionBin)
# dtw2.equalityLengthSpectres()
# print("binary distance: ", dtw2.distance())

# print("##################")
# dtw = DTW(templateSpectre_v2Errado, testSpectre_v2Errado)
# dtw.equalityLengthSpectres()
# print("default distance: ", dtw.distance())

# costFunctionBin = costFunctionBinary()
# dtw2 = DTW(templateSpectre_v2Errado, testSpectre_v2Errado, costFunctionBin)
# dtw2.equalityLengthSpectres()
# print("binary distance: ", dtw2.distance())

# print("##################")

# templateSpectre = [1, 1, 1, 2, 2, 1, 1, 1]
# testSpectre = [0, 2, 0, 0, 0, 0]

# dtw3 = DTW(templateSpectre, testSpectre)
# print("DTW = ", dtw3.distance())

# templateSpectre = len(templateSpectre)*[0]
# testSpectre = len(testSpectre)*[3]
# dtw4 = DTW(templateSpectre, testSpectre)

# print("DTW zero percent = ", dtw4.distance())