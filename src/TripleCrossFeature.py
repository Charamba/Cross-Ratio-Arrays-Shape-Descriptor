import math
import numpy as np
import sys as sys
import copy
from CrossRatio import *
from Point import *
from Geometry import *
from Utils import *

import ShapeDescriptor as SD


class TripleCrossFeature:
	def __init__(self, ray1, ray2, ray3, image, calcCrossRatio=True):
		self.isValidFeature = False
		self.junctionPoints = []
		self.junctionColors = []

		self.ray1 = None
		self.ray2 = None
		self.ray3 = None

		self.topology1 = ()
		self.topology2 = ()
		self.topology3 = ()

		self.topoKey = None

		self.fivePointsCrossRatioVector = [] 
		self.fiveCrossRatioTol = 0

		self.initAux(ray1, ray2, ray3, image, calcCrossRatio=calcCrossRatio)
		#self.calc5CrossRatioVector()

	def __repr__(self):
		topoList = [self.topology1, self.topology2, self.topology3]
		topoList.sort()
		return str(topoList)

	def __hash__(self):
		return self.topoKey

	def initAux(self, ray1, ray2, ray3, image, calcCrossRatio=True):
		(jp12, jc12) = self.findJunctionPoint(ray1, ray2, image) # self.junctionPoints[0]
		(jp23, jc23) = self.findJunctionPoint(ray2, ray3, image) # self.junctionPoints[1]
		(jp31, jc31) = self.findJunctionPoint(ray3, ray1, image) # self.junctionPoints[2]

		self.junctionPoints = [jp12, jp23, jp31]
		self.junctionColors = [jc12, jc23, jc31]

		if jp12 == jp23 == jp31:
			#print("#### junctionPoints IS EQUALS!!!! ####")
			#print("junctionPoints: ", self.junctionPoints)
			jp23 = jp12
			jp31 = jp12

		self.isValidFeature = not(None in self.junctionPoints)

		if self.isValidFeature:
			# add junction points in edgepoints
			# print("ray1.edgePoints: ", ray1.edgePoints)
			# print("jp12: ", jp12)
			newPoints1 = self.addJunctionPoint(ray1.edgePoints, jp12)
			newPoints1 = self.addJunctionPoint(newPoints1, jp31)

			newPoints2 = self.addJunctionPoint(ray2.edgePoints, jp12)
			newPoints2 = self.addJunctionPoint(newPoints2, jp23)

			newPoints3 = self.addJunctionPoint(ray3.edgePoints, jp23)
			newPoints3 = self.addJunctionPoint(newPoints3, jp31)

			# create topology
			self.topology1 = self.createTopology(newPoints1, jp12, jp31, jc12)
			self.topology2 = self.createTopology(newPoints2, jp12, jp23, jc23)
			self.topology3 = self.createTopology(newPoints3, jp23, jp31, jc31)

			# key for features table
			topoList = [self.topology1, self.topology2, self.topology3]
			topoList.sort()
			self.topoKey = hash(str(topoList))


			# print("Points1: ", newPoints1)
			# P0 = newPoints1[0]
			# newPoints1 = removeDuplicates(newPoints1)
			# newPoints1 = sortPoints(newPoints1, P0)
			# print("NEW Points1: ", newPoints1)

			# print("Points2: ", newPoints2)
			# P0 = newPoints2[0]
			# newPoints2 = removeDuplicates(newPoints2)
			# newPoints2 = sortPoints(newPoints2, P0)
			# print("NEW Points2: ", newPoints2)

			# print("Points3: ", newPoints3)
			# P0 = newPoints3[0]
			# newPoints3 = removeDuplicates(newPoints3)
			# newPoints3 = sortPoints(newPoints3, P0)
			# print("NEW Points3: ", newPoints3)

			# create rays with new points
			self.ray1 = SD.RayDescriptor(ray1.s, ray1.theta, newPoints1, calcCrossRatio=calcCrossRatio)
			self.ray2 = SD.RayDescriptor(ray2.s, ray2.theta, newPoints2, calcCrossRatio=calcCrossRatio)
			self.ray3 = SD.RayDescriptor(ray3.s, ray3.theta, newPoints3, calcCrossRatio=calcCrossRatio)
			self.calc5CrossRatioVector()


	def addJunctionPoint(self, edgePoints, junctionPoint):
		P0 = edgePoints[0]
		new_edgePoints = edgePoints + [junctionPoint]
		return sortPoints(new_edgePoints, P0)

	def createTopology(self, points, junctionPoint1, junctionPoint2, junctionColor):
		if junctionPoint1 == junctionPoint2:
			idx1 = points.index(junctionPoint1)
			idx2 = idx1 + 1
		else:
			idx1 = points.index(junctionPoint1)
			idx2 = points.index(junctionPoint2)

		idxMin = min(idx1, idx2)
		idxMax = max(idx1, idx2)
		size = len(points)

		# -------
		firstValue = min(idxMin, size - idxMax - 1)
		midValue = idxMax - idxMin - 1
		lastValue = max(idxMin, size - idxMax - 1)

		#return (idxMin, idxMax - idxMin - 1, size - idxMax - 1)
		#print("idxMin = ", idxMin)
		#print("idxMax = ", idxMax)
		color = 'black'
		if junctionColor > 0:
			color = 'white'

		return (firstValue, midValue, lastValue, color)

	def findJunctionPoint(self, ray1, ray2, image):
		(height, width) = image.getShape()
		halfHeight = height/2
		halfWidth = width/2

		junctionPoint = ray1.P2Coordinate.cross(ray2.P2Coordinate)

		if junctionPoint.z != 0:
			junctionPoint.normalize()
		else:
			return (None, None)

		condition1 = inBoundingBox(junctionPoint, ray1.edgePoints[0], ray1.edgePoints[-1])
		condition2 = inBoundingBox(junctionPoint, ray2.edgePoints[0], ray2.edgePoints[-1])

		isValidPoint = condition1 and condition2 

		if isValidPoint:
			junctionPoint = junctionPoint.toR2_Point()
			# print("junctionPoint.y + halfHeight = ", junctionPoint.y + halfHeight)
			# print("junctionPoint.x + halfWidth = ",  junctionPoint.x + halfWidth)
			# print("halfHeight = ", halfHeight)
			# print("junctionPoint.y = ", junctionPoint.y)

			if  (0 <= int(junctionPoint.y + halfHeight) < height) and (0 <= int(junctionPoint.x + halfWidth) < width):
				#print("int(junctionPoint.y + halfHeight) = ", int(junctionPoint.y + halfHeight))
				junctionColor = image.image[int(junctionPoint.y + halfHeight)][int(junctionPoint.x + halfWidth)][0]
			else:
				junctionColor = 0
			return (junctionPoint, junctionColor)
		else:
			return (None, None)

	def isMatch(self, other):
		if self.compare5CrossRatioVector(other):
			return self.compareTopology_and_Rays(other)
		return  False#and self.compare5CrossRatioVector(other)

	def isEqualsTopology(self, other):
		return self.topoKey == other.topoKey


		# selfTopos =  [self.topology1,  self.topology2,  self.topology3]
		# otherTopos = [other.topology1, other.topology2, other.topology3]

		# condition1 = other.topology1 in selfTopos and other.topology2 in selfTopos and other.topology3 in selfTopos 
		# condition2 = self.topology1 in otherTopos and self.topology2 in otherTopos and self.topology3 in otherTopos

		# return condition1 and condition2

	def compareTopology_and_Rays(self, other):
		selfTopos =  [self.topology1,  self.topology2,  self.topology3]
		otherTopos = [other.topology1, other.topology2, other.topology3]

		isMatch = True

		#print("selfTopos: ", selfTopos)

		#print("otherTopos: ", otherTopos)

		# checking if topologies are equals
		#if other.topology1 in selfTopos and other.topology2 in selfTopos and other.topology3 in selfTopos:
		if self.isEqualsTopology(other):
			# print("Topology is ok")


			# initializing cross ratio vectors
			# print("len(self.ray1.crossRatioVector) = ", len(self.ray1.crossRatioVector))
			if len(self.ray1.crossRatioVector) == 0:
				self.ray1.calcCrossRatioVector()
			# 	print("self calc 1")
			# else:
			# 	print("self NO calc 1")

			if len(self.ray2.crossRatioVector) == 0:
				self.ray2.calcCrossRatioVector()
			# 	print("self calc 2")
			# else:
			# 	print("self NO calc 2")

			if len(self.ray3.crossRatioVector) == 0:
				self.ray3.calcCrossRatioVector()
			# 	print("self calc 3")
			# else:
			# 	print("self NO calc 3")



			if len(other.ray1.crossRatioVector) == 0:
				other.ray1.calcCrossRatioVector()
			# 	print("other calc 1")
			# else:
			# 	print("other NO calc 1")

			if len(other.ray2.crossRatioVector) == 0:
				other.ray2.calcCrossRatioVector()
			# 	print("other calc 2")
			# else:
			# 	print("other NO calc 2")

			if len(other.ray3.crossRatioVector) == 0:
				other.ray3.calcCrossRatioVector()
			# 	print("other calc 3")
			# else:
			# 	print("other NO calc 3")


			selfRays  = [self.ray1, self.ray2, self.ray3] 
			otherRays = [other.ray1, other.ray2, other.ray3]

			#return True
			# otherIndices = []
			# otherIndices.append(otherTopos.index(self.topology1))
			# otherIndices.append(otherTopos.index(self.topology2))
			# otherIndices.append(otherTopos.index(self.topology3))

			# print("indices: ", otherIndices)
			# for selfIdx, otherIdx in enumerate(otherIndices):
			# 	selfRay  = selfRays[selfIdx]
			# 	otherRay = otherRays[otherIdx]

			# 	if not(selfRay.isMatch(otherRay)):
			# 		isMatch = False
			# 		break

			indicesPairs = []

			# combine comparations by more topolgy
			for i, selfTopo in enumerate(selfTopos):
				for j, otherTopo in enumerate(otherTopos):
					if selfTopo == otherTopo:
						indicesPairs.append((i,j))

			selfRaysCheckList  = [False, False, False]
			otherRaysCheckList = [False, False, False]

			# rays comparations
			for (selfIdx, otherIdx) in indicesPairs:
			 	selfRay  = selfRays[selfIdx]
			 	otherRay = otherRays[otherIdx]

			 	# print("self CRVectors: ",  selfRay.crossRatioVector)
			 	# print("other CRVectors: ", otherRay.crossRatioVector)

			 	if selfRay.isMatch(otherRay):
			 		selfRaysCheckList[selfIdx] = True
			 		otherRaysCheckList[otherIdx] = True

			# print("selfRaysCheckList: ", selfRaysCheckList)
			# print("otherRaysCheckList: ", otherRaysCheckList)
			isMatch = all(selfRaysCheckList) and all(otherRaysCheckList)
		else:
			isMatch = False

		# if isMatch:
		# 	print("is Match!")

		return isMatch

	def compare5CrossRatioVector(self, other):
		(compValue, _) = compareCrossRatiosVectors(self.fivePointsCrossRatioVector, other.fivePointsCrossRatioVector, self.fiveCrossRatioTol)
		#print("compare5CrossRatioVector: ", compValue)
		return compValue

	def findOpositePoint(self, P0, opositeJP, edgePoints):
		majorIdx = len(edgePoints) - 1
		idx0 = edgePoints.index(P0)
		opositeJPIdx = edgePoints.index(opositeJP)
		idx = 0
		if idx0 < opositeJPIdx:
			idx = majorIdx

		opositePoint = edgePoints[idx]
		return opositePoint

	def calc5CrossRatio(self, P0, opositeJunctionP1, opositeJunctionP2, adjacentRay1, adjacentRay2, opositeRay):
		P1 = self.findOpositePoint(P0, opositeJunctionP1, adjacentRay1.edgePoints)
		P2 = self.findOpositePoint(P0, opositeJunctionP2, adjacentRay2.edgePoints)
		P3 = opositeRay.edgePoints[0]  # first point
		P4 = opositeRay.edgePoints[-1] # last point
		return invariant5CrossRatio(P0, P1, P2, P3, P4)

	def calc5CrossRatioVector(self):
		[jp12, jp23, jp31] = self.junctionPoints

		r1 = self.ray1
		r2 = self.ray2
		r3 = self.ray3

		self.fivePointsCrossRatioVector.append(self.calc5CrossRatio(jp12, jp31, jp23, r1, r2, r3))
		self.fivePointsCrossRatioVector.append(self.calc5CrossRatio(jp23, jp12, jp31, r2, r3, r1))
		self.fivePointsCrossRatioVector.append(self.calc5CrossRatio(jp31, jp23, jp12, r3, r1, r2))
		self.fivePointsCrossRatioVector.sort()

		self.fiveCrossRatioTol = calcDistanceTolerance(self.fivePointsCrossRatioVector)
		# print("5-Points Cross Ratio:")
		# print(self.fivePointsCrossRatioVector)