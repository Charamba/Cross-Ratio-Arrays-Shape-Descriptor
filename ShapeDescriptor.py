import math
import numpy as np
import sys as sys
import copy
from CrossRatio import *
from Point import *
from CrossFeature import *
from TripleCrossFeature import *
from Utils import *
from FeaturesTable import *

import itertools

class RayDescriptor:
	def __init__(self, s, theta, edgePoints=[], whitePixels=[], calcCrossRatio=True):
		self.s = s
		self.theta = theta
		self.edgePoints = edgePoints
		self.numberOfEdgePoints = len(edgePoints)

		self.crossRatioVector = []
		self.distanceTolerance = []
		self.crossRatioVectorLength = 0

		self.whitePixels = whitePixels

		if calcCrossRatio:
			#print("RayDescriptor -> calcCrossRatio")
			self.calcCrossRatioVector()

		self.vanishPoint = None
		self.pencil_id = theta # initial

		self.bestMatchingDistance = float('inf')
		self.bestMatchingRayPair = None

		self.key = None

		self.P2Coordinate = None
		if len(edgePoints) >= 4:
				self.P2Coordinate = self.calcP2Coordinate(self.edgePoints[0], self.edgePoints[1])

	def cleanMatchingVariables(self):
		self.bestMatchingDistance = float('inf')
		self.bestMatchingRayPair = None

	def calcCrossRatioVector(self):
		self.crossRatioVector = crossRatioFilter(self.edgePoints) # , self.crossRatioTolerances
		#print("self.crossRatioVector = ", self.crossRatioVector)
		self.distanceTolerance = calcDistanceTolerance(self.crossRatioVector)
		self.crossRatioVectorLength = len(self.crossRatioVector)

	def CRV_length(self):
		return self.crossRatioVectorLength
	def calcDistance(self, other):
		return calcDistanceVector(self.crossRatioVector, other.crossRatioVector)

	def getPolarCoordinate(self):
		return (self.s, self.theta)


	def isMatching(self, other): # HULL SCAN
		if other.numberOfEdgePoints >= 4 and (self.numberOfEdgePoints == other.numberOfEdgePoints):
			(crossValues1, crossValues2) = (self.crossRatioVector, other.crossRatioVector)
			if len(crossValues1) == len(crossValues2):
				distance = calcDistanceVector(crossValues1, crossValues2)

				if distance <= self.distanceTolerance:
					if distance < self.bestMatchingDistance:
						self.bestMatchingDistance = distance
						self.bestMatchingRayPair = other

						self_extremeCR = crossRatio(self.edgePoints[0],self.edgePoints[1], self.edgePoints[-2], self.edgePoints[-1])
						other_extremeCR = crossRatio(other.edgePoints[0],other.edgePoints[1], other.edgePoints[-2], other.edgePoints[-1])

						#return True
						if abs(other_extremeCR - self_extremeCR) <= self_extremeCR*percentualTol: # Tol
							return True
						else:
							return False
				else:
					return False
		else:
			return False


	def isMatch(self, other): # TOMOGRAPH SCAN
		if other.numberOfEdgePoints >= 4 and (self.numberOfEdgePoints == other.numberOfEdgePoints):
			#print("MESMO NUMERO DE PONTOS!")
			#return True
			(compValue, _) = compareCrossRatiosVectors(self.crossRatioVector, other.crossRatioVector, self.distanceTolerance)
			return compValue
		else:
			#print("NUMERO DE PONTOS DIFERENTES!")
			return False
	# def equalsValue(self, other):
	# 	if other.numberOfEdgePoints >= 4 and (self.numberOfEdgePoints == other.numberOfEdgePoints):
	# 		if self.crossRatioValue != None and other.crossRatioValue != None:
	# 			return isEqualValues(self.crossRatioValue, other.crossRatioValue)
	# 		else:
	# 			return False
	# 	else:
	# 		return False
	def calcP2Coordinate(self, ep0, ep1):
		ep0 = ep0.toP2_Point()
		ep1 = ep1.toP2_Point()
		return ep0.cross(ep1)
	def getEdgePixelCoordinates(self, width, height):
		X = []
		Y = []
		halfWidth = width/2
		halfHeight = height/2
		for i in range(0, len(self.edgePoints)):
			X.append(self.edgePoints[i].x + halfWidth)
			Y.append(self.edgePoints[i].y + halfHeight)
		return (X, Y)
	def getVanishPoint(self):
		return self.vanishPoint
	def getAllPointsCoordinates(self, width, height):
		halfWidth = width/2
		halfHeight = height/2

		(X, Y) = self.getEdgePointsCoordinates(width, height)

		if self.vanishPoint != None:
			(xv, yv) = self.vanishPoint.toTuple()
			X.append(xv + halfWidth)
			Y.append(yv + halfWidth)

		return (X, Y)

	def calcVanishPointCandidate(self, t, r, b, rho, isDirect=True):
		tr_ = t.euclideanDistance(r)
		tb_ = t.euclideanDistance(b)
		rb_ = r.euclideanDistance(b)

		if tr_ == 0 or tb_ == 0 or rb_ == 0 or rho == 0:
			print("t = ", t)
			print("r = ", r)
			print("b = ", b)

		#print("tb_ = ", tb_)
		#print("rb_ = ", rb_)
		#print("t = ", t)
		#print("b = ", b)
		#print("ratio = ", rho)
		vanishPoint = None

		if rho == 0:
			print("rho = ", rho)
		if rb_ == 0:
			print("rb_ = ", rb_)

		alpha = (1.0/rho)*(tb_/rb_)
		#print("alpha = ", alpha)
		if 1-alpha:
			lambda_ = (tr_*alpha/(1-alpha))
			#print("lambda = ", lambda_)
			#if isDirect:
			#	vDir = t - r
			#else:
			#	vDir = r - t
			vDir = t - r
			vDir.r2Normalize()
			vanishPoint = t + lambda_*vDir
			#nextPoint = t + (lambda_/abs(lambda_))*vDir

			(xv, yv) = vanishPoint.toTuple()
			##### Test
			#if (-4.5368279417 <= t.x <=  -4.5368279415): #(100-150 < xv < 120-150): # or
				#vanishPoint = t - lambda_*vDir
				#print("t = ", t)
				#print("r = ", r)
				#print("b = ", b)
				#print("ratio = ", rho)
				#print("alpha = ", alpha)
				#print("lambda = ", lambda_)
				#print("vanishPoint = ", vanishPoint)
				#print("isDirect = ", isDirect)
				#print("cross ratio signature = ", self.crossRatioVector)
			# End test
		return (vanishPoint)#, nextPoint)

	def estimateVanishPoints(self, other):
		testRay = self
		templateRay = other
		testRay.pencil_id = templateRay.theta # Pencil id
		n = testRay.numberOfEdgePoints
		distTol2 = templateRay.distanceTolerance
	
		(_, isDirect) = compareCrossRatiosVectors(templateRay.crossRatioVector, testRay.crossRatioVector, distTol2)


		(idx_t, idx_r, idx_b) = ( 0, 0, 0)
		mid = int(float(n)/2.0)
		if isDirect:
			(idx_t, idx_r, idx_b) = ( 0, mid,-1)
		else:
			(idx_t, idx_r, idx_b) = ( -1, mid, 0)
		#(idx_t, idx_r, idx_b) = ( 0, mid,-1)

		(T, R, B) = (templateRay.edgePoints[0], templateRay.edgePoints[mid], templateRay.edgePoints[-1])

		# print("templateRay.edgePoints = ", templateRay.edgePoints)
		# print("idx_t, idx_r, idx_b = %d, %d, %d" %(idx_t, idx_r, idx_b))
		# print("-------------------------------------")
		rho = simpleRatio(T, R, B)

		# print("testRay.edgePoints = ", testRay.edgePoints)
		# print("idx_t, idx_r, idx_b = %d, %d, %d" %(idx_t, idx_r, idx_b))
		# print("-------------------------------------")
		(t, r, b) = (testRay.edgePoints[idx_t], testRay.edgePoints[idx_r], testRay.edgePoints[idx_b])
		vp = self.calcVanishPointCandidate(t, r, b, rho, isDirect)
		if vp:
			(xv, yv) = vp.toTuple()
			w = len(testRay.crossRatioVector)
			wVP = WeightedPoint(xv, yv, w)
			self.vanishPoint = wVP
		else:
			return False

class ShapeDescriptor:
	def __init__(self):
		#self.rays = [] 
		self.raysTable = FeaturesTable()
		self.countCrossRatioVectorLengths = [0]*20
		self.crossFeatures = []	
		self.tripleCrossFeaturesTable = FeaturesTable()
		# Points Statistic
		self.numberOfEdgePoints = 0
		self.numberOfJunctionPoints = 0

		self.whitePixelsImage = {}

		# For fanBeam scan 
		self.pencils = []
		self.hullVertices = []

	def addRay(self, s, theta, edgePoints=[], whitePixels=[], calcCrossRatio=True):
		#print("ShapeDescriptor.addRay: len(edgePoints) = ", len(edgePoints))
		if len(edgePoints) >= 4:
			for wp in whitePixels:
				self.whitePixelsImage[wp] = 0

			self.numberOfEdgePoints += len(edgePoints)
			ray = RayDescriptor(s, theta, edgePoints, whitePixels=whitePixels, calcCrossRatio=calcCrossRatio)
			#self.rays.append(ray)
			self.raysTable.update(ray, key=ray.numberOfEdgePoints)

			if calcCrossRatio:
				#print("ShapeDescriptor - addRay - calcCrossRatio = True")
				idxLen = len(ray.crossRatioVector)

				#print(sum(ray.crossRatioVector))
				#print(idxLen)
				lenMax = len(self.countCrossRatioVectorLengths)
				if lenMax > idxLen:
					self.countCrossRatioVectorLengths[idxLen] += 1
		else:
			ray = RayDescriptor(s, theta, edgePoints, whitePixels=whitePixels, calcCrossRatio=calcCrossRatio)
			self.raysTable.update(ray, key=ray.numberOfEdgePoints)

	def addPencil(self, raysList):
		raysTable = FeaturesTable()

		for ray in raysList:
			raysTable.update(ray, key=ray.numberOfEdgePoints)

		self.pencils.append(raysTable)


	def generateCrossFeatures(self, image, convexPoints=[]):
		raysPairs = pairsCombinations(self.raysTable.getValues())#(self.rays)
		#print("len(raysPairs) = ", len(raysPairs))
		for (ray1, ray2) in raysPairs:
			if True:#ray1.theta != ray2.theta:
				crossFeature = CrossFeature(ray1, ray2, image)

				if crossFeature.isValid:
					self.numberOfJunctionPoints += 1 
					self.crossFeatures.append(crossFeature)
					convexCrossValues = crossRatioFilter5(convexPoints, crossFeature.junctionPoint)
					#print("convexCrossValues: ", convexCrossValues)

	def generateTripleCrossFeatures(self, image, calcCrossRatio=True):
		raysTriples = itertools.combinations(self.raysTable.getValues(), 3)

		for (ray1, ray2, ray3) in raysTriples:
			feature = TripleCrossFeature(ray1, ray2, ray3, image, calcCrossRatio=calcCrossRatio)
			if feature.isValidFeature:
				self.tripleCrossFeaturesTable.update(feature)

		#self.tripleCrossFeaturesTable.printConfiguration()