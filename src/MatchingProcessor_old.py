from ShapeDescriptor import *
from CrossFeature import *
from DTW import *
from DistanceMatchingMatrix import *
from Pearson import *
import itertools

PEARSON_THRESHOLD = 0.0
CR5_DISTANCE_PERCENTUAL_TOL = 0.1
CR5_MATCHING_TOL = 0.65

def removeDuplicityVerticesMatching(verticesPairs, distances):
	verticesPairsDict = {}

	for i, vPair in enumerate(verticesPairs):
		(v1, v2) = vPair
		new_d = distances[i]
		if v2 in verticesPairsDict:
			(old_v1, old_v2, old_d) = verticesPairsDict[v2]
			if new_d < old_d:
				verticesPairsDict[v2] = (v1, v2, new_d)
		else:
			verticesPairsDict[v2] = (v1, v2, new_d)

	verticeValues = [(v[0], v[1], v[2]) for v in verticesPairsDict.values()]
	verticeValues = sorted([(v1, v2, v3) for (v1, v2, v3) in verticeValues], key=lambda t: t[0])
	verticePairs = [(v[0], v[1]) for v in verticeValues]
	verticeDistances = [(v[2]) for v in verticeValues]

	return (verticePairs, verticeDistances)


def removeDuplicityRaysMatching(templateRays):
	raysPairsDict = {}

	for ray in templateRays:
		(s1, s2) = (ray.s, ray.bestMatchingRayPair.s)
		new_d = ray.bestMatchingDistance
		if s2 in raysPairsDict:
			old_ray = raysPairsDict[s2]
			old_d = old_ray.bestMatchingDistance
			if new_d < old_d:
				raysPairsDict[s2] = ray
		else:
			raysPairsDict[s2] = ray

	# verticeValues = [(v[0], v[1], v[2]) for r in raysPairsDict.values()]
	# verticeValues = sorted([(v1, v2, v3) for (v1, v2, v3) in verticeValues], key=lambda t: t[0])
	# verticePairs = [(v[0], v[1]) for v in verticeValues]
	# verticeDistances = [(v[2]) for v in verticeValues]

	# return (verticePairs, verticeDistances)
	return raysPairsDict.values()

def vector_metric(sigma, a, r):
	x = 1-a
	y = 1-r
	return sigma*math.sqrt(x*x + y*y)

def vector_metric_simetric(sigma, a, b):
	x = 1-a
	y = 1-b
	return sigma*math.sqrt(x*x + y*y)/math.sqrt(2)

def vector_metric_assimetric(sigma, a, b, r):
	x = 1-a
	y = 1-b
	z = 1-r
	return sigma*math.sqrt(x*x + y*y + z*z)/math.sqrt(3)

def atenuation_function(mi, rho):
	return mi*math.exp(-rho)

def gravity_distance(global_distance, rho, M, m, n):
	d = global_distance
	d2 = d*d
	return d2/((rho*(n/M)*(n/m)))/(n*n)

class MatchingProcessor:
	def __init__(self, templateDescriptor, testDescriptor):
		self.templateDescriptor = templateDescriptor
		self.testDescriptor = testDescriptor


	def compareByPencils(self, symetric_shape=False):
		#1 Percorrer todos os pencils da imagem template
		#2 Comparar pencils 
		#3 Escolher o que possui maior percentual de matches
		distanceMatrix = []
		spectrePairs = []
		distanceValues = []
		mTemplateRays = [] 
		mTestRays = []

		matching_result = False

		totalRays = 0
		distance = 0
		matchingVerticesPairs = []

		n = len(self.templateDescriptor.pencils)
		m = len(self.testDescriptor.pencils)

		nRays_template = 300 + n
		nRays_test     = 300 + m

		#print("nRays_template = ", nRays_template)
		#print("nRays_test = ", nRays_test)
		nRays_Max = max(nRays_template, nRays_test)

		for template_idx, pencil in enumerate(self.templateDescriptor.pencils):
			(test_idx, distance, templateRays, testRays) = self.findBestMatchPencil(pencil, self.testDescriptor.pencils)
			
			if distance != float('inf'):
				matchingVerticesPairs.append((template_idx, test_idx))
				distanceValues.append(distance)

			mTemplateRays += templateRays
			mTestRays += testRays


		(matchingVerticesPairs, distanceValues) = removeDuplicityVerticesMatching(matchingVerticesPairs, distanceValues)
		#print("vertices: ", matchingVerticesPairs)
		#print("distances: ", distanceValues)

		matchingVerticesPairsPoints = []

		distanceObject = sum(distanceValues) # Symetric Shape

		cr5dist = float('inf')
		cr5distance_tol = 0
		cr5MatchingPercent = 0
		pCoef = 0
		new_pCoef = 0
		Distance = float('inf') # DISTANCE MAX VALUE
		#print("distance obj: ", distanceObject)
		if len(distanceValues) != 0:
			#Distance = distanceObject/len(distanceValues)
			#print("average(dist): ", distanceObject/(len(distanceValues)))
			#print('Matching vertices Pairs: ', len(matchingVerticesPairs))9
			if len(matchingVerticesPairs) > 1:
				#print('entrou')
				if symetric_shape:
					(pCoef, inliersTestIndices) = self.calcPearsonCoefficient(matchingVerticesPairs, removeOutLiers_flag=False)
					newMatchingVerticesPairs = []
					for (ti, qi) in matchingVerticesPairs:
						if qi in inliersTestIndices:
							t_vertex = self.templateDescriptor.hullVertices[ti]
							q_vertex = self.testDescriptor.hullVertices[qi]			
							matchingVerticesPairsPoints.append((t_vertex, q_vertex))
							newMatchingVerticesPairs = matchingVerticesPairs
					
					a = len(newMatchingVerticesPairs)/n
					b = len(newMatchingVerticesPairs)/m
					r = 1.0
					sigma = distanceObject/len(distanceValues)
					print("a = ", a)
					print("b = ", b)
					print("sigma = ", sigma)
					Distance = vector_metric_simetric(sigma, a, b)
					#print("vector_metric_simetric = ", Distance)
					if a > 0:
						matching_result = True
					# if abs(pCoef) >= PEARSON_THRESHOLD or symetric_shape:
					# 	templateVertices = self.templateDescriptor.hullVertices
					# 	testVertices = self.testDescriptor.hullVertices

				if not(symetric_shape):

					templateVertices = self.templateDescriptor.hullVertices
					testVertices = self.testDescriptor.hullVertices

					(pCoef, inliersTestIndices) = self.calcPearsonCoefficient(matchingVerticesPairs, removeOutLiers_flag=False)
					matchingVerticesPairsPoints, newMatchingVerticesPairs = self.compareCrossRatioVertices_combinations(templateVertices, testVertices, matchingVerticesPairs, inliersTestIndices)
					print("old_p_coef = ", pCoef)
					if newMatchingVerticesPairs:
						(new_pCoef, new_inliersTestIndices) = self.calcPearsonCoefficient(newMatchingVerticesPairs, removeOutLiers_flag=True)

						print("NEW PEARSON COEFFICIENT: ", new_pCoef)

						#---- Calculando novos valores de distancia DTW entre espectros dos vertices que sobraram
						oldMatchingVerticesPairs_oldValues = zip(matchingVerticesPairs, distanceValues)
						new_values = []
						
						for oldPair, oldValue in zip(matchingVerticesPairs, distanceValues):
							if oldPair in newMatchingVerticesPairs:
								new_values.append(oldValue)

						new_dist_obj = sum(new_values)
						#print("NEW Distance Object:", new_dist_obj)
						new_average = new_dist_obj/len(new_values)
						vertexes_matches = len(new_values)
						#print("NEW Average Dist:", new_average)

						Distance = atenuation_function(new_average, new_pCoef)
						#gDist = gravity_distance(new_average, new_pCoef, n, m, vertexes_matches)

						#print("gravit_distance = ",gDist)
						# ------------------
						sigma = new_average
						a = len(new_values)/n
						b = len(new_values)/m
						r = new_pCoef
						print("sigma = ", sigma)
						print("a = ", a)
						print("b = ", b)
						print("r = ", r)
						Distance = vector_metric_assimetric(sigma, a, b, r)
						#print("vector_metric_assimetric = ", Distance)
						#gravit_distance_force = (new_pCoef*(float(vertexes_matches)/n)*(float(vertexes_matches)/m))/(new_average*new_average)
						#print("gravit_distance = ", 1.0/gravit_distance_force)

						# mi = new_average/nRays_Max
						# print("mi = ", mi)
						# lamb = 0.7
						# global_average_distance = lamb*(1.0-new_pCoef) + (1.0 - lamb)*mi
						# print("global_distance: ", global_average_distance)
						# Distance = global_average_distance

					if matchingVerticesPairsPoints:
						matching_result = True


		return (matching_result, new_pCoef, mTemplateRays, mTestRays, matchingVerticesPairsPoints, Distance)


	# def compareByPencils_old(self):
	# 	#1 Percorrer todos os pencils da imagem template
	# 	#2 Comparar pencils 
	# 	#3 Escolher o que possui maior percentual de matches
	# 	distanceMatrix = []
	# 	spectrePairs = []
	# 	distanceValues = []
	# 	mTemplateRays = [] 
	# 	mTestRays = []

	# 	# pencil_template = self.templateDescriptor.pencils[0]
	# 	# pencil_test = self.testDescriptor.pencils[0]

	# 	matching_result = False

	# 	totalRays = 0
	# 	distance = 0
	# 	matchingVerticesPairs = []

	# 	n = len(self.templateDescriptor.pencils)
	# 	m = len(self.testDescriptor.pencils)
	# 	for template_idx, pencil in enumerate(self.templateDescriptor.pencils):
	# 		(test_idx, distance, templateRays, testRays) = self.findBestMatchPencil(pencil, self.testDescriptor.pencils)
			
	# 		if distance != float('inf'):
	# 			matchingVerticesPairs.append((template_idx, test_idx))
	# 			distanceValues.append(distance)

	# 		mTemplateRays += templateRays
	# 		mTestRays += testRays


	# 	# print("### ANTES")
	# 	# print("vertices: ", matchingVerticesPairs)
	# 	# print("distances: ", distanceValues)

	# 	# print("### DEPOIS")
	# 	(matchingVerticesPairs, distanceValues) = removeDuplicityVerticesMatching(matchingVerticesPairs, distanceValues)
	# 	print("vertices: ", matchingVerticesPairs)
	# 	print("distances: ", distanceValues)

	# 	matchingVerticesPairsPoints = []

	# 	distanceObject = sum(distanceValues)

	# 	cr5dist = float('inf')
	# 	cr5distance_tol = 0
	# 	cr5MatchingPercent = 0
	# 	pCoef = 0

	# 	print("distance obj: ", distanceObject)
	# 	if len(distanceValues) != 0:
	# 		print("average(dist): ", distanceObject/(len(distanceValues)))
	# 		if len(matchingVerticesPairs) > 1:

	# 			(pCoef, inliersTestIndices) = self.calcPearsonCoefficient(matchingVerticesPairs)

	# 			for (ti, qi) in matchingVerticesPairs:
	# 				if qi in inliersTestIndices:
	# 					t_vertex = self.templateDescriptor.hullVertices[ti]
	# 					q_vertex = self.testDescriptor.hullVertices[qi]			
	# 					matchingVerticesPairsPoints.append((t_vertex, q_vertex))
				
	# 			if abs(pCoef) >= PEARSON_THRESHOLD:
	# 				templateVertices = self.templateDescriptor.hullVertices
	# 				testVertices = self.testDescriptor.hullVertices
	# 				cr5dist, cr5distance_tol, cr5vectorTemplate, cr5vectorTest = self.compareCrossRatioVertices(templateVertices, testVertices, matchingVerticesPairs, inliersTestIndices)
	# 				cr5dist_, cr5distance_tol_, cr5vectorTemplate_, cr5vectorTest_ = self.compareCrossRatioVertices_(templateVertices, testVertices, matchingVerticesPairs, inliersTestIndices)
	# 				cr5dist_2, cr5distance_tol_2, cr5vectorTemplate_2, cr5vectorTest_2 = self.compareCrossRatioVertices_2(templateVertices, testVertices, matchingVerticesPairs, distanceValues, inliersTestIndices)
	# 				cr5dist_c, cr5distance_tol_c, cr5vectorTemplate_c, cr5vectorTest_c, matchingVerticesPairsPoints = self.compareCrossRatioVertices_combinations(templateVertices, testVertices, matchingVerticesPairs, inliersTestIndices)

	# 				cr5MatchingPercent = compareCellPerCell(cr5vectorTemplate, cr5vectorTest, CR5_DISTANCE_PERCENTUAL_TOL)
	# 				cr5MatchingPercent_ = compareCellPerCell(cr5vectorTemplate_, cr5vectorTest_, CR5_DISTANCE_PERCENTUAL_TOL)
	# 				cr5MatchingPercent_2 = compareCellPerCell(cr5vectorTemplate_2, cr5vectorTest_2, CR5_DISTANCE_PERCENTUAL_TOL)
	# 				cr5MatchingPercent_c = compareCellPerCell(cr5vectorTemplate_c, cr5vectorTest_c, CR5_DISTANCE_PERCENTUAL_TOL)


					
	# 				print("cr5MatchingPercent = ", cr5MatchingPercent)
	# 				print("cr5MatchingPercent_ = ", cr5MatchingPercent_)
	# 				print("cr5MatchingPercent_2 = ", cr5MatchingPercent_2)
	# 				print("cr5MatchingPercent_c = ", cr5MatchingPercent_c)
	# 				if cr5MatchingPercent_ > cr5MatchingPercent:
	# 					(cr5dist, cr5distance_tol, cr5vectorTemplate, cr5vectorTest) = (cr5dist_, cr5distance_tol_, cr5vectorTemplate_, cr5vectorTest_)
	# 					cr5MatchingPercent = cr5MatchingPercent_

	# 				# print("## PEARSON COEFFICIENT = ", pCoef)
	# 				# print("## FIVE CROSS RATIO DISTANCE = ", dist)
	# 				# print("## CrossRatio5 Matching Percent = ", cr5MatchingPercent)
	# 				if cr5dist < cr5distance_tol or cr5MatchingPercent > CR5_MATCHING_TOL:
	# 					matching_result = True

	# 	return (matching_result, pCoef, cr5dist, cr5distance_tol, cr5MatchingPercent, mTemplateRays, mTestRays, matchingVerticesPairsPoints)

	def calcPearsonCoefficient(self, verticesPairs, removeOutLiers_flag=False):
		#Y = getVerticesIndices(verticesPairs)
		(x0, y0) = verticesPairs[0]
		Y = [y for (x, y) in verticesPairs]
		Y = shifftSignalToMinValue(Y)
		
		#print("Y: ", Y)
		# X = range(0, len(Y))

		# ----- remove outliers 
		points = []
		for i, y in enumerate(Y):
			points.append(R2_Point(i, y))

		inliers = points
		if removeOutLiers_flag:
			inliers = removeOutLiers(points) # retirar ou continuar com isso?

		newY = []
		X = []
		pCoef = 0
		if inliers: # by Ransac
			newY = [p.y for p in inliers]
			#newY = Y #<--- parei aqui
			X = [x for (x, y) in verticesPairs if y in newY]#list(range(0, len(newY)))
			#print("tau: ", X)
			#print("Q:  ", newY)
			minX = min(X)
			pCoef = abs(pearsonCoefficient(X, newY))
			return (pCoef, newY)
		else: # by median Filter (caso Ransac falhe!)
			X = [x for (x, y) in verticesPairs if y in Y]#list(range(0, len(Y)))
			minX = min(X)
			#print("tau: ", X)
			#print("Q:  ", Y)

			pCoef = abs(pearsonCoefficient(X, Y))

			return (pCoef, Y)
			newY1 = median1DFilter(Y, growing=True)
			X = [x for (x, y) in verticesPairs if y in newY1]#list(range(0, len(newY1)))
			minX = min(X)
			pCoef1 = abs(pearsonCoefficient(X, newY1))

			newY2 = median1DFilter(Y, growing=False)
			X = [x for (x, y) in verticesPairs if y in newY2]#list(range(0, len(newY2)))
			minX = min(X)
			pCoef2 = abs(pearsonCoefficient(X, newY2)) 

			if pCoef1 > pCoef2:
				pCoef = pCoef1
				newY = newY1
			else:
				pCoef = pCoef2
				newY = newY2
		#------------------
		

		if len(newY) == 0:
			newY = Y

		#X = list(range(0, len(newY)))

		#print("newY: ", newY)

		#print("pCoef = ", pCoef)

		return (pCoef, newY)

	def compareCrossRatioVertices_2(self, templateVertices, testVertices, verticePairs, distanceValues, inliersTestIndices=[]):
		#Y = getVerticesIndices(verticesPairs)
		Y = [y for (x, y) in verticePairs] # get test vertice indices
		if inliersTestIndices:
			Y = inliersTestIndices and Y
		Y = Y and inliersTestIndices
		Y = shifftSignalToMinValue(Y)
		W = Y
		#W = removePeaksAndDepressionsValues(Y)


		(newPairs, newDistances) = filteredVerticePairs(W, verticePairs, distanceValues)

		pairs_distances = sorted(list(zip(newPairs, newDistances)), key=lambda x:x[1])

		newPairs = [p for (p,d) in pairs_distances]
		newDistances = [d for (p,d) in pairs_distances]
		print("newDistances = ", newDistances)

		newTemplateVertices = [templateVertices[x] for (x, y) in newPairs]
		newTestVertices = [testVertices[y] for (x, y) in newPairs]

		cr5vectorTemplate = invariant5CrossRatioFilter(newTemplateVertices)
		cr5vectorTest = invariant5CrossRatioFilter(newTestVertices)


		#print("cr5vectorTemplate: ", cr5vectorTemplate)
		#print("cr5vectorTest: ", cr5vectorTest)

		if len(cr5vectorTemplate) != 0:
			distance = calcDistanceVector(cr5vectorTemplate, cr5vectorTest)
		else:
			distance = float('Inf')

		distance_tol = calcDistanceVector(cr5vectorTemplate, [0]*len(cr5vectorTemplate))*CR5_DISTANCE_PERCENTUAL_TOL

		return distance, distance_tol, cr5vectorTemplate, cr5vectorTest


	def compareCrossRatioVertices_(self, templateVertices, testVertices, verticePairs, inliersTestIndices=[]):
		#Y = getVerticesIndices(verticesPairs)
		Y = [y for (x, y) in verticePairs] # get test vertice indices
		if inliersTestIndices:
			Y = inliersTestIndices and Y
		Y = Y and inliersTestIndices
		Y = shifftSignalToMinValue(Y)
		W = Y
		#W = removePeaksAndDepressionsValues(Y)


		newPairs = filteredVerticePairs(W, verticePairs)


		newTemplateVertices = [templateVertices[x] for (x, y) in newPairs]
		newTestVertices = [testVertices[y] for (x, y) in newPairs]

		cr5vectorTemplate = invariant5CrossRatioFilter(newTemplateVertices)
		cr5vectorTest = invariant5CrossRatioFilter(newTestVertices)


		#print("cr5vectorTemplate: ", cr5vectorTemplate)
		#print("cr5vectorTest: ", cr5vectorTest)

		if len(cr5vectorTemplate) != 0:
			distance = calcDistanceVector(cr5vectorTemplate, cr5vectorTest)
		else:
			distance = float('Inf')

		distance_tol = calcDistanceVector(cr5vectorTemplate, [0]*len(cr5vectorTemplate))*CR5_DISTANCE_PERCENTUAL_TOL

		return distance, distance_tol, cr5vectorTemplate, cr5vectorTest 

	def compareCrossRatioVertices_combinations(self, templateVertices, testVertices, verticeIndicesPairs, inliersTestIndices=[]):
		Y = [y for (x, y) in verticeIndicesPairs] # get test vertice indices
		if inliersTestIndices:
			Y = inliersTestIndices and Y
		Y = Y and inliersTestIndices
		Y = shifftSignalToMinValue(Y)
		W = Y

		newIndxsPairs = filteredVerticePairs(W, verticeIndicesPairs)

		T = templateVertices
		Q = testVertices

		combinations_of_pairs = itertools.combinations(newIndxsPairs, 5)

		correctPairsIndices = []
		wrongPairsIndices = []

		# buscando a primeira combinação de pontos confiáveis
		for _5pairs in combinations_of_pairs:
			((a1, a2), (b1, b2), (c1, c2), (d1, d2), (e1, e2)) = _5pairs

			val1 = crossRatio5(T[a1], T[b1], T[c1], T[d1], T[e1])
			val2 = crossRatio5(Q[a2], Q[b2], Q[c2], Q[d2], Q[e2])

			if val1 >= 1E-4 and val2 >= 1E-4:
				if abs(val1-val2) <= val1*CR5_DISTANCE_PERCENTUAL_TOL:
					correctPairsIndices = [(a1,a2), (b1,b2), (c1,c2), (d1,d2), (e1,e2)]

					for (i, j) in correctPairsIndices:
						idx = newIndxsPairs.index((i, j))
						del newIndxsPairs[idx]
					break

		totalPairs = len(newIndxsPairs)

		if newIndxsPairs == []:
			wrongPairsIndices = newIndxsPairs

		while not(newIndxsPairs == []):#len(correctPairsIndices) + len(wrongPairsIndices) < totalPairs:
			(x1, x2) = newIndxsPairs[0]

			isWrong = True
			for comb in itertools.combinations(correctPairsIndices, 4):

				((a1, a2), (b1, b2), (c1, c2), (d1, d2)) = comb
				val1 = crossRatio5(T[a1], T[b1], T[c1], T[d1], T[x1])
				val2 = crossRatio5(Q[a2], Q[b2], Q[c2], Q[d2], Q[x2])


				#if not(val1 == 0 or val2 == 0):
				#if val1 >= 1E-4 and val2 >= 1E-4:
				if abs(val1-val2) <= val1*CR5_DISTANCE_PERCENTUAL_TOL:
					correctPairsIndices.append((x1, x2))
					isWrong = False
					break
			
			if isWrong:
				wrongPairsIndices.append((x1, x2))

			del newIndxsPairs[0]

		n_correct = len(correctPairsIndices)
		n_wrong = len(wrongPairsIndices)
		#print("len(correctPairsIndices): ", n_correct)
		#print("len(wrongPairsIndices): ", n_wrong)
		#print("# -> percentual vertices inliers: ", n_correct/(n_correct + n_wrong))

		#correctPairsIndices += wrongPairsIndices # apagar depois
		newTemplateVertices = [templateVertices[x] for (x, y) in correctPairsIndices]
		newTestVertices =     [    testVertices[y] for (x, y) in correctPairsIndices]

		matchingVerticesPairsPoints = list(zip(newTemplateVertices, newTestVertices))

		#print("correctPairsIndices = ", correctPairsIndices)

		# Corrigindo a ordem do correctPairsIndices por eventuais deslocamentos (gambiarra)
		correctPairsIndices = [(x, y) for (x, y) in verticeIndicesPairs if (x, y) in correctPairsIndices]

		return matchingVerticesPairsPoints, correctPairsIndices


	def compareCrossRatioVertices(self, templateVertices, testVertices, verticePairs, inliersTestIndices=[]):
		#Y = getVerticesIndices(verticesPairs)
		Y = [y for (x, y) in verticePairs] # get test vertice indices
		if inliersTestIndices:
			Y = inliersTestIndices and Y
		Y = Y and inliersTestIndices
		Y = shifftSignalToMinValue(Y)
		W = Y
		#W = removePeaksAndDepressionsValues(Y)


		newPairs = filteredVerticePairs(W, verticePairs)

		Xindices = sorted([x for (x, y) in newPairs])

		Yindices = [y for (x, y) in newPairs]
		minIdx = Yindices.index(min(Yindices))

		Yindices = sorted(Yindices[:minIdx]) + sorted(Yindices[minIdx:])

		newTemplateVertices = [templateVertices[x] for x in Xindices]
		newTestVertices = [testVertices[y] for y in Yindices]

		cr5vectorTemplate = invariant5CrossRatioFilter(newTemplateVertices)
		cr5vectorTest = invariant5CrossRatioFilter(newTestVertices)


		#print("cr5vectorTemplate: ", cr5vectorTemplate)
		#print("cr5vectorTest: ", cr5vectorTest)

		if len(cr5vectorTemplate) != 0:
			distance = calcDistanceVector(cr5vectorTemplate, cr5vectorTest)
		else:
			distance = float('Inf')

		distance_tol = calcDistanceVector(cr5vectorTemplate, [0]*len(cr5vectorTemplate))*CR5_DISTANCE_PERCENTUAL_TOL

		return distance, distance_tol, cr5vectorTemplate, cr5vectorTest 



	def compareFanBeamExtremePointsCR5(self, templateRays):

		testRays = [r.bestMatchingRayPair for r in templateRays]

		templateV0 = templateRays[0].edgePoints[0]
		testV0 = testRays[0].edgePoints[0]

		templateExtremePoints = [r.edgePoints[-1] for r in templateRays]
		testExtremePoints = [r.edgePoints[-1] for r in testRays]

		cr5vectorTemplate = invariant5CrossRatioFilter(templateExtremePoints)
		cr5vectorTest = invariant5CrossRatioFilter(testExtremePoints)

		if len(cr5vectorTemplate) != 0:
			distance = calcDistanceVector(cr5vectorTemplate, cr5vectorTest)
		else:
			distance = float('Inf')

		return distance

	def generateSimpleTopologySpectre(self, matchedRaysList):
		spectre = []
		if len(matchedRaysList) > 0:
			matchedRaysList = list(set(matchedRaysList))
			sortRays = sorted(matchedRaysList, key=lambda r: r.s)
			spectre_idx = [r.s for r in sortRays]

			for ray in sortRays:
				spectre.append(ray.numberOfEdgePoints)
		return spectre

	def generateKeySpectres(self, templateRays, testRays):
		templateSpectre = []
		testSpectre = []

		keyValue = 2

		for i in range(len(templateRays)):
			templateRays[i].key = keyValue
			templateRays[i].bestMatchingRayPair.key = keyValue
			templateSpectre.append(keyValue)
			keyValue += 1

		bestTestRayPairs = [templateRay.bestMatchingRayPair for templateRay in templateRays]

		for testRay in testRays:
			if testRay in bestTestRayPairs:
				i = bestTestRayPairs.index(testRay)
				testSpectre.append(bestTestRayPairs[i].key)

		return (templateSpectre, testSpectre)

	def generateTopologySpectre(self, matchedRaysList, groundValue=0, maxLengthVector=None):
		spectre = []
		if len(matchedRaysList) > 0:
			matchedRaysList = list(set(matchedRaysList))
			sortRays = sorted(matchedRaysList, key=lambda r: r.s)
			spectre_idx = [r.s for r in sortRays]

			for i in range(maxLengthVector):
				if i in spectre_idx:
					[ray] = [r for r in sortRays if r.s==i]
					spectre.append(ray.numberOfEdgePoints)
				else:
					spectre.append(groundValue)

			for ray in sortRays:
				spectre.append(ray.numberOfEdgePoints)
		return spectre		

	def generateTopologySpectre_deprecated2(self, matchedRaysList, groundValue=0, maxLengthVector=None):
		maxGapPercent = 0.2
		maxGapLen = maxLengthVector*maxGapPercent

		spectre = [groundValue]*maxLengthVector

		if len(matchedRaysList) > 0:
			matchedRaysList = list(set(matchedRaysList))
			sortRays = sorted(matchedRaysList, key=lambda r: r.s)
			spectre_idx = [r.s for r in sortRays]
			s_max = max(spectre_idx)
			if maxLengthVector == None:
				maxLengthVector = max(spectre_idx) 
			#print("spectre_idx: ", spectre_idx)

			# sortRaysDict = {}
			# for ray in sortRays:
			# 	sortRaysDict[ray.s] = ray

			#spectre = [groundValue]*maxLengthVector

			# lastValue = 0
			# for i in range(0, len(spectre)):
			# 	if i in sortRaysDict:
			# 		lastValue = sortRaysDict[i].numberOfEdgePoints
			# 	spectre[i] = lastValue

			for ray in sortRays:
				#spectre[ray.s] = ray.numberOfEdgePoints

				extremeCR = crossRatio(ray.edgePoints[0],ray.edgePoints[1], ray.edgePoints[2], ray.edgePoints[-1])				
				#spectre[ray.s] = sum(ray.crossRatioVector)#/extremeCR
				spectre.append(ray.numberOfEdgePoints)

			# retificação do sinal (preenchendo espaços com buracos)
			for i in range(0, len(sortRays)):
				if i+1 < len(sortRays):
					ray1 = sortRays[i]
					ray2 = sortRays[i+1]
					s1 = ray1.s
					s2 = ray2.s
					if abs(s2 - s1) <= maxGapLen:
						for s in range(s1+1, s2):
							spectre[s] = spectre[s1] # repetindo valor 

			# lastValue = groundValue
			# for s in range(0, s_max):
			# 	if spectre[s] == groundValue:
			# 		spectre[s] = lastValue
			# 	else:
			# 		lastValue = spectre[s]
			

			#print(spectre)

		return spectre

	def findBestMatchPencil(self, pencil, otherPencils):
		# return spectres
		(best_distance, best_matchedTemplateRays, best_matchedTestRays) = (float('inf'), [], [])
		bestIdx = 0
		for i, other in enumerate(otherPencils):
			#print("test pencil i = ", i)
			(distance, matchedTemplateRays, matchedTestRays) = self.comparePencils(pencil, other)
			#print("(i, d) = (%d, %3.2f)" %(i, distance))
			if distance < best_distance:
				#print("SWAP best distance!")
				bestIdx = i
				best_distance = distance
				best_matchedTemplateRays = matchedTemplateRays
				best_matchedTestRays = matchedTestRays

		#print("best idx = ", bestIdx)
		return (bestIdx, best_distance, best_matchedTemplateRays, best_matchedTestRays)

	def comparePencils(self, pencil1, pencil2):
		# Pegar raios do pencil 1
		# Usar a topologia desses raios para retorna lista de raios com a mesma topologia

		rays1 = pencil1.getValues()
		for i in range(0, len(rays1)):
			rays1[i].cleanMatchingVariables()

		rays2 = pencil2.getValues()
		totalRays1 = len(rays1)
		totalRays2 = len(rays2)
		totalRays = min(totalRays1, totalRays2)
		# print("len1 = ", totalRays1)
		# print("len2 = ", totalRays2)
		totalTemplateRays = len(rays1)
		templateMatchs = [0]*totalTemplateRays
		matchedTemplateRays = []
		matchedTestRays = []
		#matchPairsRays = []

		for i, templateRay in enumerate(rays1):
			for testRay in pencil2.access(templateRay.numberOfEdgePoints):
				
				#if templateRay.isMatch(testRay):
				if templateRay.isMatching(testRay):
					matchedTemplateRays.append(templateRay)
					#matchedTestRays.append(testRay)
					templateMatchs[i] = 1
					#matchPairsRays.append((templateRay.s, testRay.s))

		templateRaysIndices = []
		testRaysIndices = []
		
		matchedTemplateRays = removeDuplicityRaysMatching(matchedTemplateRays)
		matchedTemplateRays = list(set(matchedTemplateRays))
		rays = matchedTemplateRays
		matchedTemplateRays = sorted(rays, key=lambda r: r.s)


		#CR5_distance = self.compareFanBeamExtremePointsCR5(matchedTemplateRays) rempover

		raysIndicePairs = []
		distances = []

		# Pearson
		for mTemplateRay in matchedTemplateRays:
			matchedTestRays.append(mTemplateRay.bestMatchingRayPair)
		# 	(s1, s2) = (mTemplateRay.s, mTemplateRay.bestMatchingRayPair.s)
		# 	templateRaysIndices.append(s1)
		# 	testRaysIndices.append(s2)
		# 	# raysIndicePairs.append(s1,s2)
		# 	# distances.append(mTemplateRay.bestMatchingDistance)

		sumMatchedTemplateRays = sum(templateMatchs)
		# # print("templateRaysIndices = ", templateRaysIndices)
		# # print("testRaysIndices = ", testRaysIndices)
		# p = pearsonCoefficient(templateRaysIndices, testRaysIndices)
		#print(">>>> p = ", p)

		if sumMatchedTemplateRays == 0:
			return (float('inf'), matchedTemplateRays, matchedTestRays)

		percentualMatch = 0
		if totalTemplateRays != 0:
			percentualMatch = sumMatchedTemplateRays/totalTemplateRays

		mTemplateRays = list(set(matchedTemplateRays))
		mTestRays = list(set(matchedTestRays))


		# USANDO DESLOCAMENTO SIMPLIFICADO e DTW e Spectros de "rotulos"
		templateSpectre = self.generateSimpleTopologySpectre(mTemplateRays)
		testSpectre = self.generateSimpleTopologySpectre(mTestRays)

		#(templateSpectre, testSpectre) = self.generateKeySpectres(mTemplateRays, mTestRays)

		n = len(mTemplateRays)
		costBinFunct = costFunctionBinary()
		dtw = DTW(templateSpectre, testSpectre, costBinFunct)
		n = len(mTemplateRays)
		dtw_dist = dtw.distance()


		error_distance = totalRays - n
		error_distance += dtw_dist#sum([1 for a,b in zip(templateSpectre,testSpectre) if a==b])
		distance_normalized = error_distance/n

		# invertendo o espectro do query
		testSpectreReversed = testSpectre
		testSpectreReversed.reverse()

		dtw = DTW(templateSpectre, testSpectreReversed, costBinFunct)
		n = len(mTemplateRays)
		dtw_dist = dtw.distance()

		error_distance = totalRays - n
		error_distance += dtw_dist#sum([1 for a,b in zip(templateSpectre,testSpectreReversed) if a==b])
		distance_normalized_reversed = error_distance/n

		distance_normalized = min(distance_normalized, distance_normalized_reversed)

		# USANDO DTW e TOPOLOGY SPECTRUM
		# templateSpectre = self.generateTopologySpectre(mTemplateRays, maxLengthVector=totalRays1)
		# testSpectre = self.generateTopologySpectre(mTestRays, groundValue=1, maxLengthVector=totalRays2)
		# costBinFunct = costFunctionBinary()
		# dtw = DTW(templateSpectre, testSpectre, costBinFunct)
		# n = len(mTemplateRays)
		# distance_normalized = dtw.distance()

		#print("dtw normalized = ", distance_normalized)
		#print("CR5 distance = ", CR5_distance)
		#return ((1-abs(p)), matchedTemplateRays, matchedTestRays)
		#return (distance_normalized*(1-abs(p)), matchedTemplateRays, matchedTestRays)



		return (distance_normalized, matchedTemplateRays, matchedTestRays)

	def comparePencils_deprecated(self, pencil1, pencil2):
		# Pegar raios do pencil 1
		# Usar a topologia desses raios para retorna lista de raios com a mesma topologia

		rays1 = pencil1.getValues()
		rays2 = pencil2.getValues()
		totalRays1 = len(rays1)
		totalRays2 = len(rays2)
		# print("len1 = ", totalRays1)
		# print("len2 = ", totalRays2)
		totalTemplateRays = len(rays1)
		templateMatchs = [0]*totalTemplateRays
		matchedTemplateRays = []
		matchedTestRays = []
		matchPairsRays = []

		for i, templateRay in enumerate(rays1):
			for testRay in pencil2.access(templateRay.numberOfEdgePoints):
				
				if templateRay.isMatch(testRay):
					matchedTemplateRays.append(templateRay)
					matchedTestRays.append(testRay)
					templateMatchs[i] = 1
				#else:


		sumMatchedTemplateRays = sum(templateMatchs)

		if sumMatchedTemplateRays == 0:
			return (float('inf'), matchedTemplateRays, matchedTestRays)

		percentualMatch = 0
		if totalTemplateRays != 0:
			percentualMatch = sumMatchedTemplateRays/totalTemplateRays

		mTemplateRays = list(set(matchedTemplateRays))
		mTestRays = list(set(matchedTestRays))


		templateSpectre = self.generateTopologySpectre(mTemplateRays, maxLengthVector=totalRays1)
		testSpectre = self.generateTopologySpectre(mTestRays, groundValue=1, maxLengthVector=totalRays2)

		costBinFunct = costFunctionBinary()
		dtw = DTW(templateSpectre, testSpectre, costBinFunct)
		#print("dtw.distance() = ", dtw.distance())
		distance_normalized = dtw.distance()/len(mTemplateRays)
		#print("dtw normalized = ", distance_normalized)

		return (distance_normalized, matchedTemplateRays, matchedTestRays)

	def compare(self, estimateVanishPoint=False):
		matchedWhitePixels = self.templateDescriptor.whitePixelsImage

		vanishPoints = []
		matchedTemplateRays = []
		matchedTestRays = []
		badTemplRays = []
		badTestRays = []
		totalComp = 0
		distanceError = 0
		templateMatchs = [0]*len(self.templateDescriptor.raysTable.getValues())
		#testMatchs = [0]*len(self.testDescriptor.rays)

		matchByProjections = {}

		for i, templateRay in enumerate(self.templateDescriptor.raysTable.getValues()):
			mathcFlag = False
			#for testRay in enumerate(self.testDescriptor.rays):
			for testRay in self.testDescriptor.raysTable.access(templateRay.numberOfEdgePoints):
				totalComp += 1

				if templateRay.isMatch(testRay):
					matchedTemplateRays.append(templateRay)
					matchedTestRays.append(testRay)

					for wp in templateRay.whitePixels:
						matchedWhitePixels[wp] = 1  

					if estimateVanishPoint:
						testRay.estimateVanishPoints(templateRay)
						vanishPoints.append(testRay.getVanishPoint())

					templateMatchs[i] = 1
					#testMatchs[j] = 1

					distanceError += calcDistanceVector(templateRay.crossRatioVector, testRay.crossRatioVector)
					mathcFlag = True
					# if testRay.numberOfEdgePoints % 2 == 0:
					# if testRay.estimateVanishPoints(templateRay):
					# 	vanishPoints.append(testRay.getVanishPoint())
					# 	matchedTemplateRays.append(templateRay)
					# 	matchedTestRays.append(testRay)
					# else:
					# 	badTemplRays.append(templateRay)
					# 	badTestRays.append(testRay)
			if mathcFlag:
				theta = templateRay.theta
				if matchByProjections.get(theta) is None:
					matchByProjections[theta] = 1
				else:
					matchByProjections[theta] += 1

		sumMatchedTemplateRays = sum(templateMatchs)
		#sumMatchedTestRays = sum(testMatchs)
		#print("matchedTemplateRays[0].s = ", matchedTemplateRays[0].s)
		matchedTemplateRays = list(set(matchedTemplateRays))
		sortRays = sorted(matchedTemplateRays, key=lambda r: r.s)
		# SPECTRES
		print("Matched Template Rays Topology")
		for ray1 in sortRays:
			print(ray1.numberOfEdgePoints)

		print("Matched Template Rays")
		for ray1 in sortRays:
			print(sum(ray1.crossRatioVector))
		# ---------------------------------------------------------------
		matchedTestRays = list(set(matchedTestRays))
		sortRays = sorted(matchedTestRays, key=lambda r: r.s)

		print("Matched Test Rays Topology")
		for ray2 in sortRays:
			print(ray2.numberOfEdgePoints)

		print("Matched Test Rays")
		for ray2 in sortRays:
			print(sum(ray2.crossRatioVector))


		print("### MATCHING STATISTIC ###")
		print("# Template Rays: ", len(self.templateDescriptor.raysTable.getValues()))
		print("# Test Rays: ", len(self.testDescriptor.raysTable.getValues()))
		print("# Comparations: ", totalComp)
		print("# Matched Template Rays : ", sumMatchedTemplateRays)
		#print("# Matched Test Rays : ", sumMatchedTestRays)
		if len(self.templateDescriptor.raysTable.getValues()):
			print("Percentual template match: %3.2f%%" %(sumMatchedTemplateRays*100/len(self.templateDescriptor.raysTable.getValues())))
		# if len(self.testDescriptor.rays):
		# 	print("Percentual test match: %3.2f%%" %(sumMatchedTestRays*100/len(self.testDescriptor.rays)))
		print("Distance Error: ", distanceError)
		print("---------------------------------")
		matchedWhitePixelsValues = matchedWhitePixels.values()
		totalWhitePixels = len(matchedWhitePixelsValues)
		sumMatchedPixels = sum(matchedWhitePixelsValues)
		print("White pixels in template image: ", totalWhitePixels)
		print("White matched pixels in template image: ", sumMatchedPixels)
		if totalWhitePixels > 0:
			percentual = (sumMatchedPixels*100)/totalWhitePixels
			print("Percentual matched pixels: <__ %3.2f %%__>" %(percentual))

		print("Match by Projections: ")
		# matchByProjections = {}
		# for templRay in matchedTemplateRays:
		# 	theta = templRay.theta
		# 	if matchByProjections.get(theta) is None:
		# 		matchByProjections[theta] = 1
		# 	else:
		# 		matchByProjections[theta] += 1
		print("len(matchByProjections) = ", len(matchByProjections.items()))
		print("(theta, percentual match)")

		matchByProjections.items()
		
		for (theta, count) in matchByProjections.items():
			percentual = count*100/sumMatchedTemplateRays
			print("(%f, %d) ---- %3.2f" %(theta, count, percentual))

		return (matchedTemplateRays, matchedTestRays, badTemplRays, badTestRays, vanishPoints)

	def compareByCrossFeatures(self):

		matchedTemplateRays = []
		matchedTestRays = []
		# -----------------------
		vanishPoints = []
		matchedtemplateFeatures = []
		matchedtestFeatures = []
		badTemplRays = []
		badtestFeatures = []
		totalComp = 0
		distanceError = 0
		templateMatchsFeatures = [0]*len(self.templateDescriptor.crossFeatures)
		testMatchsFeatures = [0]*len(self.testDescriptor.crossFeatures)

		templateMatchsRaysCounter = {}
		testMatchRaysCounter = {}

		templateEdgePointsCounter = {}
		templateJunctionPointsCounter = {}

		for i, templateFeature in enumerate(self.templateDescriptor.crossFeatures):
			mathcFlag = False
			for j, testFeature in enumerate(self.testDescriptor.crossFeatures):
				totalComp += 1

				if templateFeature.isMatch(testFeature):
				#if templateFeature.compareRays(testFeature):
					matchedtemplateFeatures.append(templateFeature)
					matchedtestFeatures.append(testFeature)
					# ------------
					matchedTemplateRays.append(templateFeature.ray1)
					matchedTemplateRays.append(templateFeature.ray2)
					matchedTestRays.append(testFeature.ray1)
					matchedTestRays.append(testFeature.ray2)


					(s1, t1) = templateFeature.ray1.getPolarCoordinate()
					(s2, t2) = templateFeature.ray2.getPolarCoordinate()

					templateMatchsRaysCounter[(s1, t1)] = 1
					templateMatchsRaysCounter[(s2, t2)] = 1

					templateEdgePointsCounter[(s1, t1)] = templateFeature.ray1.numberOfEdgePoints - 1
					templateEdgePointsCounter[(s2, t2)] = templateFeature.ray2.numberOfEdgePoints - 1


					 
					if templateJunctionPointsCounter.get(s1, t1) is not None:
						templateJunctionPointsCounter[(s1, t1)] = templateJunctionPointsCounter.get(s1, t1) + 1
					else:
						templateJunctionPointsCounter[(s1, t1)] = 1

					if templateJunctionPointsCounter.get(s2, t2) is not None:
						templateJunctionPointsCounter[(s2, t2)] = templateJunctionPointsCounter.get(s1, t1) + 1
					else:
						templateJunctionPointsCounter[(s2, t2)] = 1

					templateMatchsFeatures[i] = 1
					testMatchsFeatures[j] = 1



		sumMatchedtemplateFeatures = sum(templateMatchsFeatures)
		sumMatchedtestFeatures = sum(testMatchsFeatures)
		print("### MATCHING STATISTIC ###")
		print("* FEATURES:")
		print("# Template Features: ", len(self.templateDescriptor.crossFeatures))
		print("# Test Features: ", len(self.testDescriptor.crossFeatures))
		print("# Comparations: ", totalComp)
		print("# Matched Template Features : ", sumMatchedtemplateFeatures)
		print("# Matched Test Features : ", sumMatchedtestFeatures)
		if len(self.templateDescriptor.crossFeatures) > 0:
			print("Percentual template match: < %3.2f%% >" %(sumMatchedtemplateFeatures*100/len(self.templateDescriptor.crossFeatures)))
		if len(self.testDescriptor.crossFeatures) > 0:
			print("Percentual test match: %3.2f%%" %(sumMatchedtestFeatures*100/len(self.testDescriptor.crossFeatures)))
		
		print("---------------------------------")
		print("* RAYS:")
		nTemplateRaysMatch = sum(templateMatchsRaysCounter.values())
		totalTemplateRays = len(self.templateDescriptor.rays)
		print(" Matched Template Rays: ", nTemplateRaysMatch)
		if totalTemplateRays > 0:
			print("Percentual template ray match: < %3.2f%% >" %(nTemplateRaysMatch*100/totalTemplateRays))

		print("---------------------------------")
		print("* EDGE POINTS + JUNCTION POINTS:")
		matchedEdgePoints = sum(templateEdgePointsCounter.values())
		matchedJunctionPoints = sum(templateJunctionPointsCounter.values())

		matchedPoints = matchedEdgePoints + matchedJunctionPoints
		totalTemplatePoints = self.templateDescriptor.numberOfEdgePoints + self.templateDescriptor.numberOfJunctionPoints
		
		print("Matched Edge points: ", matchedEdgePoints)
		print("Matched Junction points: ", matchedJunctionPoints)
		print("Matched Points: ", matchedPoints)

		if totalTemplateRays > 0:
			print("Percentual match edge points : %3.2f%%" %(matchedEdgePoints*100/self.templateDescriptor.numberOfEdgePoints))
			if self.templateDescriptor.numberOfJunctionPoints > 0:
				print("Percentual match junction points : %3.2f%%" %(matchedJunctionPoints*100/self.templateDescriptor.numberOfJunctionPoints))
			print("Percentual match points : < %3.2f%% >" %(matchedPoints*100/totalTemplatePoints))


		return (matchedTemplateRays, matchedTestRays, badTemplRays, badtestFeatures, vanishPoints)

	def compareByTripleCrossFeatures(self):
		matchedTemplateRays = []
		matchedTestRays = []
		# -----------------------
		vanishPoints = []
		matchedtemplateFeatures = []
		matchedtestFeatures = []
		badTemplRays = []
		badtestFeatures = []
		totalComp = 0
		distanceError = 0

		templateFeatures = self.templateDescriptor.tripleCrossFeaturesTable.getValues()
		testFeatures = self.testDescriptor.tripleCrossFeaturesTable.getValues()

		templateMatchsFeatures = [0]*len(templateFeatures)
		testMatchsFeatures = [0]*len(testFeatures)

		templateMatchsRaysCounter = {}
		testMatchRaysCounter = {}

		templateEdgePointsCounter = {}
		templateJunctionPointsCounter = {}

		for i, templateFeature in enumerate(templateFeatures):
			mathcFlag = False

			key = templateFeature.topoKey

			for j, testFeature in enumerate(self.testDescriptor.tripleCrossFeaturesTable.access(key)):
				totalComp += 1

				if templateFeature.isMatch(testFeature):
					# print("5-Points Cross Ratio:")
					# print(templateFeature.fivePointsCrossRatioVector)
					# print(testFeature.fivePointsCrossRatioVector)

				#if templateFeature.compareRays(testFeature):
					matchedtemplateFeatures.append(templateFeature)
					matchedtestFeatures.append(testFeature)
					# ------------
					matchedTemplateRays.append(templateFeature.ray1)
					matchedTemplateRays.append(templateFeature.ray2)
					matchedTemplateRays.append(templateFeature.ray3)
					matchedTestRays.append(testFeature.ray1)
					matchedTestRays.append(testFeature.ray2)
					matchedTestRays.append(testFeature.ray3)


					(s1, t1) = templateFeature.ray1.getPolarCoordinate()
					(s2, t2) = templateFeature.ray2.getPolarCoordinate()
					(s3, t3) = templateFeature.ray3.getPolarCoordinate()

					templateMatchsRaysCounter[(s1, t1)] = 1
					templateMatchsRaysCounter[(s2, t2)] = 1
					templateMatchsRaysCounter[(s3, t3)] = 1

					templateEdgePointsCounter[(s1, t1)] = templateFeature.ray1.numberOfEdgePoints - 2
					templateEdgePointsCounter[(s2, t2)] = templateFeature.ray2.numberOfEdgePoints - 2
					templateEdgePointsCounter[(s3, t3)] = templateFeature.ray3.numberOfEdgePoints - 2

					 
					# if templateJunctionPointsCounter.get(s1, t1) is not None:
					# 	templateJunctionPointsCounter[(s1, t1)] = templateJunctionPointsCounter.get(s1, t1) + 1
					# else:
					# 	templateJunctionPointsCounter[(s1, t1)] = 1

					# if templateJunctionPointsCounter.get(s2, t2) is not None:
					# 	templateJunctionPointsCounter[(s2, t2)] = templateJunctionPointsCounter.get(s1, t1) + 1
					# else:
					# 	templateJunctionPointsCounter[(s2, t2)] = 1

					# if templateJunctionPointsCounter.get(s3, t3) is not None:
					# 	templateJunctionPointsCounter[(s3, t3)] = templateJunctionPointsCounter.get(s1, t1) + 1
					# else:
					# 	templateJunctionPointsCounter[(s3, t3)] = 1

					templateMatchsFeatures[i] = 1
					testMatchsFeatures[j] = 1 # <==== ERRADO (corrigir)


		sumMatchedtemplateFeatures = sum(templateMatchsFeatures)
		sumMatchedtestFeatures = sum(testMatchsFeatures)
		print("### MATCHING STATISTIC ###")
		print("* TRIPLE CROSS FEATURES:")
		print("# Template Features: ", len(templateFeatures))
		print("# Test Features: ", len(testFeatures))
		print("# Comparations: ", totalComp)
		print("# Matched Template Features : ", sumMatchedtemplateFeatures)
		print("# Matched Test Features : ", sumMatchedtestFeatures)
		if len(templateFeatures) > 0:
			print("Percentual template match: < %3.2f%% >" %(sumMatchedtemplateFeatures*100/len(templateFeatures)))
		if len(testFeatures) > 0:
			print("Percentual test match: %3.2f%%" %(sumMatchedtestFeatures*100/len(testFeatures)))
		
		print("---------------------------------")
		print("* RAYS:")
		nTemplateRaysMatch = sum(templateMatchsRaysCounter.values())
		totalTemplateRays = len(self.templateDescriptor.raysTable.getValues())
		print(" Matched Template Rays: ", nTemplateRaysMatch)
		if totalTemplateRays > 0:
			print("Percentual template ray match: < %3.2f%% >" %(nTemplateRaysMatch*100/totalTemplateRays))

		print("---------------------------------")
		print("* EDGE POINTS + JUNCTION POINTS:")
		matchedEdgePoints = sum(templateEdgePointsCounter.values())
		#matchedJunctionPoints = sum(templateJunctionPointsCounter.values())

		#matchedPoints = matchedEdgePoints + matchedJunctionPoints
		#totalTemplatePoints = self.templateDescriptor.numberOfEdgePoints + self.templateDescriptor.numberOfJunctionPoints
		
		print("Matched Edge points: ", matchedEdgePoints)
		#print("Matched Junction points: ", matchedJunctionPoints)
		#print("Matched Points: ", matchedPoints)

		if totalTemplateRays > 0:
			print("Percentual match edge points : %3.2f%%" %(matchedEdgePoints*100/self.templateDescriptor.numberOfEdgePoints))
			# if self.templateDescriptor.numberOfJunctionPoints > 0:
			# 	print("Percentual match junction points : %3.2f%%" %(matchedJunctionPoints*100/self.templateDescriptor.numberOfJunctionPoints))
			#print("Percentual match points : < %3.2f%% >" %(matchedPoints*100/totalTemplatePoints))


		return (matchedTemplateRays, matchedTestRays, badTemplRays, badtestFeatures, vanishPoints)