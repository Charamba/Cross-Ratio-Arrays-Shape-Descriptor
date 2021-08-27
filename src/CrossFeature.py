import math
import numpy as np
import sys as sys
import copy
from CrossRatio import *
from Point import *
from Geometry import *
from Utils import *
#from ShapeDescriptor import *
import ShapeDescriptor as SD

class CrossFeature:
	def __init__(self, ray1, ray2, image):
		self.isValid = False
		self.junctionPoint = self.findJunctionPoint(ray1, ray2, image)
		self.pointsTopologyIdx1 = [] # Ex.: [1,3], [2,4], [3,3] 
		self.pointsTopologyIdx2 = []
		self.junctionColor = None
		if self.isValid:
			new_edgePoints1 = self.addJunctionPoint(ray1.edgePoints, self.junctionPoint)
			junctioIdx = new_edgePoints1.index(self.junctionPoint)
			newLen1 = len(new_edgePoints1)
			self.pointsTopologyIdx1 = set([junctioIdx, newLen1 - junctioIdx - 1])

			#print(edgePoints1)
			new_edgePoints2 = self.addJunctionPoint(ray2.edgePoints, self.junctionPoint)
			junctioIdx = new_edgePoints2.index(self.junctionPoint)
			newLen2 = len(new_edgePoints2)
			self.pointsTopologyIdx2 = set([junctioIdx, newLen2 - junctioIdx - 1])
			#print(edgePoints2)
			self.ray1 = SD.RayDescriptor(ray1.s, ray1.theta, new_edgePoints1)
			self.ray2 = SD.RayDescriptor(ray2.s, ray2.theta, new_edgePoints2)
		

	def findJunctionPoint(self, ray1, ray2, image):
		# PAREI AQUI!!
		(height, width) = image.getShape()
		halfHeight = height/2
		halfWidth = width/2

		junctionPoint = ray1.P2Coordinate.cross(ray2.P2Coordinate)
		#print("junctionPoint = ", junctionPoint)

		if junctionPoint.z != 0:
			junctionPoint.normalize()
		else:
			return R2_Point(junctionPoint.x, junctionPoint.y)
		#print("junctionPoint Pos = ", junctionPoint)
		# junctionPoint.x = junctionPoint.x - width
		# junctionPoint.y = junctionPoint.y - height

		condition1 = inBoundingBox(junctionPoint, ray1.edgePoints[0], ray1.edgePoints[-1])
		condition2 = inBoundingBox(junctionPoint, ray2.edgePoints[0], ray2.edgePoints[-1])


		#print("yImg: ", int(junctionPoint.y + halfHeight))
		#print("xImg: ", )

		# junctionPoint.x = junctionPoint.x + width
		# junctionPoint.y = junctionPoint.y + height


		#print("condition1 = ", condition1)
		#print("condition2 = ", condition2)
		self.isValid = condition1 and condition2 #<---- BoundingBox VER!!!! PAREI AQUI!

		if self.isValid:
			junctionPoint = junctionPoint.toR2_Point()
			self.junctionColor = image.image[int(junctionPoint.y + halfHeight)][int(junctionPoint.x + halfWidth)][0]
			#print("junctionColor = ", self.junctionColor)

		#print("isvalid = ", self.isValid )
		#junctionPoint.normalize()
		#print("junctionPoint = ", junctionPoint)
		#(xj2, yj2, _) = junctionPoint.toTuple()
		return junctionPoint

	def addJunctionPoint(self, edgePoints, junctionPoint):
		P0 = edgePoints[0]
		#print("P0 = ", P0)
		#print("Pj = ", junctionPoint)
		new_edgePoints = edgePoints + [junctionPoint]
		#edgePoints.append(junctionPoint)
		#print(edgePoints)
		#print("sort points:")
		#print(sortPoints(edgePoints, P0))
		#print("---------------------------")
		return sortPoints(new_edgePoints, P0)

	def compareRays(self, other):
		return (self.ray1.isMatch(other.ray1) and self.ray2.isMatch(other.ray2)) or (self.ray1.isMatch(other.ray2) and self.ray2.isMatch(other.ray1))

	def isMatch(self, other):
		if self.junctionColor == other.junctionColor: 
		#if True:
			if self.pointsTopologyIdx1 == other.pointsTopologyIdx1 and self.pointsTopologyIdx2 == other.pointsTopologyIdx2:
				return self.ray1.isMatch(other.ray1) and self.ray2.isMatch(other.ray2)
			elif self.pointsTopologyIdx1 == other.pointsTopologyIdx2 and self.pointsTopologyIdx2 == other.pointsTopologyIdx1:
				return self.ray1.isMatch(other.ray2) and self.ray2.isMatch(other.ray1)

	# def calcPeripheralCrossRatioValue(self, ray1, ray2):
	# 	(A, B) = (ray1.edgePoints[0], ray1.edgePoints[-1])
	# 	(C, D) = (ray2.edgePoints[0], ray2.edgePoints[-1])
	# 	O = self.junctionPoint
	# 	ACO_area = triangleArea(A, C, O)
	# 	ADO_area = triangleArea(A, D, O)
	# 	BCO_area = triangleArea(B, C, O)
	# 	BDO_area = triangleArea(B, D, O)



	# 	print("TRIANGLE AREAS:")
	# 	print(ACO_area)
	# 	print(ADO_area)
	# 	print(BCO_area)
	# 	print(BDO_area)
	# 	print("----------------")



	# 	t = []
	# 	t0 = (ACO_area/ADO_area)*(BDO_area/BCO_area)
	# 	t.append(t0)

	# 	if t0 != 0:
	# 		t1 = 1.0/t0
	# 		t.append(t1)

	# 	t2 = 1.0 - t0
	# 	t.append(t2)

	# 	if 1.0-t0 != 0:
	# 		t3 = 1.0/(1.0-t0)
	# 		t.append(t3)

	# 	if t0 != 0:
	# 		t4 = (t0 - 1.0)/t0
	# 		t.append(t4)

	# 	if t0 -1.0 != 0:
	# 		t5 = t0/(t0-1.0)
	# 		t.append(t5)

	# 	print("FIVE POINTS CROSS RATIO:")
	# 	print("t = ", t)
	# 	print("min(t) = ", min(t))

	
	# 	(A1, B1) = (ray1.edgePoints[0], ray1.edgePoints[-1])
	# 	(A2, B2) = (ray2.edgePoints[0], ray2.edgePoints[-1])
	# 	O = self.junctionPoint
	# 	# O.y += O.y*0.25
	# 	# O.x += O.x*0.5

	# 	A1 = R2_Point(0.0,0.0)
	# 	B1 = R2_Point(10.0, 10.0)
	# 	A2 = R2_Point(0.0, 10.0)
	# 	B2 = R2_Point(10.0, 0.0)
	# 	O = R2_Point(4.0, 4.0)

	# 	cp1 = crossRatio5(A1, B1, A2, B2, O)
	# 	print("FIVE POINTS CROSS RATIO:")
	# 	print(cp1)
	# 	#print(cp2)

	# def calcCentroidCrossRatioVector(self, ray1, ray2):



