import math
import numpy as np
import sys as sys
from CrossRatio import *
from Point import *


class HorizonLine:
	def __init__(self, vanishPoints=[]):
		self.vanishPoints = vanishPoints
		self.P2_representation = self.calcP2Repr(vanishPoints)
	def calcP2Repr(self, vanishPoints):
		(x0, y0) = vanishPoints[0].toTuple()
		(x1, y1) = vanishPoints[-1].toTuple()
		P0 = P2_Point(x0, y0, 1)
		P1 = P2_Point(x1, y1, 1)
		l = P0.cross(P1)
		l.normalize()
		return l
	def getRepr(self):
		return self.P2_representation