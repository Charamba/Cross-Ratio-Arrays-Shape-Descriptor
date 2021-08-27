
class cellMatrix:
	def __init__(self, distanceValue, sortIdx1=None, sortIdx2=None):
		self.sortIdx1 = sortIdx1
		self.sortIdx2 = sortIdx2
		self.distanceValue = distanceValue

class DistanceMatchingMatrix:
	def __init__(self, distanceMatrix):
		self.cells = self.initCellsMatrix(distanceMatrix)
		self.nRows = len(cells)
		self.nCols = len(cells[0])

	def initCellsMatrix(self, distanceMatrix):
		M = distanceMatrix
		cells = []
		for i, row in enumerate(M):
			cells[i] = []
			for e in row:
				cells[i].append(e)
		return cells
	def getRow(self, idxRow):
		M = self.cells
		return M[idxRow]

	def getCol(self, idxCol):
		M = self.cells
		return [row[idxCol] for row in N]

	def distanceLevels(self, cellsList):
		s = cellsList
		sortedIndices = sorted(range(len(s)), key=lambda k: s[k])
		distanceLevels = [0]*len(sortedIndices)
		for i,idx in enumerate(sortedIndices):
			distanceLevels[idx] = i

		return distanceLevels