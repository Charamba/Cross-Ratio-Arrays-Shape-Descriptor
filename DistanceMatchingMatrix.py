
class cellMatrix:
	def __init__(self, distanceValue, lv1=None, lv2=None):
		self.lv1 = lv1
		self.testRay2 = lv2
		self.distanceValue = distanceValue
	def __eq__(self, other):
		return self.distanceValue == other.distanceValue
	def	__lt__(self, other):
		return self.distanceValue < other.distanceValue
	def __repr__(self):
		return str(self.distanceValue)
	def getLevels(self):
		return (self.lv1, self.lv2)

class DistanceMatchingMatrix:
	def __init__(self, distanceMatrix):
		self.cells = self.initCellsMatrix(distanceMatrix)
		self.nRows = len(self.cells)
		self.nCols = len(self.cells[0])

	def __repr__(self):
		returnString = ""
		for row in self.cells:
			returnString += str(row)
			returnString += "\n"
			# for e in row:
			# 	returnString.append(e)
		return returnString

	def initCellsMatrix(self, distanceMatrix):
		M = distanceMatrix
		cellsMatrix = []
		for i, row in enumerate(M):
			cellsRow = []
			for e in row:
				cellsRow.append(cellMatrix(e))
			cellsMatrix.append(cellsRow)
		return cellsMatrix
	def getRow(self, idxRow):
		M = self.cells
		return M[idxRow]
	def getCol(self, idxCol):
		M = self.cells
		return [row[idxCol] for row in M]
	def distanceLevels(self, cellsList):
		s = cellsList
		sortedIndices = sorted(range(len(s)), key=lambda k: s[k])
		distanceLevels = [0]*len(sortedIndices)
		for i,idx in enumerate(sortedIndices):
			distanceLevels[idx] = i
		return distanceLevels
	def distanceLevelsMatrix(self):
		M = self.cells
		L = []
		for i in range(0, self.nRows):
			rowi = self.getRow(i) 
			distLvls = self.distanceLevels(rowi)
			for j, dLvl in enumerate(distLvls):
				self.cells[i][j].lv1 = dLvl
		for j in range(0, self.nCols):
			colj = self.getCol(j)
			#print(self.distanceLevels(colj))
			distLvls = self.distanceLevels(colj)
			#distLvls.reverse()
			for i, dLvl in enumerate(distLvls):
				self.cells[i][j].lv2 = dLvl
		for row in self.cells:
			rowDL = [(e.lv1, e.lv2) for e in row]
			print(rowDL)
	def getMatchingVertices(self):
		M = self.cells
		values = []
		for i,row in enumerate(M):
			for j,e in enumerate(row):
				(lv1, lv2) = e.getLevels()
				if (lv1, lv2) == (0, 0):
					values.append((i, j))
		return values


# M = [[0.7, 0.1, 0.3, 0.5],[0.2, 0.5, 0.6, 7.0],[0.4, 0.8, 0.9, 0.2]]

# dmm = DistanceMatchingMatrix(M)

# print(dmm)
# dmm.distanceLevelsMatrix()

# print("matching vertices pairs:")
# print(dmm.getMatchingVertices())
