from ShapeDescriptor import *

class VanishPencil:
	def __init__(self, vanishRay=None):
		self.vanishRays = None
		if vanishRay:
			self.vanishRays = [vanishRay] # List of vanishRays
	def updateVanishRay(self, newVanishRay):
		if self.vanishRays:
			newWeight = newVanishRay.crossRatioVectorLength
			oldWeight = self.vanishRays[0].crossRatioVectorLength

			if newWeight > oldWeight:
				self.vanishRays = [newVanishRay]
			elif newWeight == oldWeight:
				self.vanishRays.append(newVanishRay)
		else:
			self.vanishRays = [newVanishRay]
		
class VanishPencilsTable:
	def __init__(self):
		self.pencils = {} # {(pencil_id_1, vanishPencil_1), (pencil_id_2, vanishPencil_2)}
	def getPencils(self):
		return self.pencils.items()
	def getVanishPoints(self):
		vanishPoints = []
		for vanishPencil in self.pencils.values():
			rays = vanishPencil.vanishRays
			for ray in rays:
				vanishPoints.append((ray.vanishPoint, ray.pencil_id))
		return vanishPoints
	def getVirtualPoints(self):
		virtualPoints = []
		for vanishPencil in self.pencils.values():
			rays = vanishPencil.vanishRays
			(xm, ym, wm) = (0, 0, 0)
			n = len(rays)
			for ray in rays:
				(vPi, iDi) = (ray.vanishPoint, ray.pencil_id)
				(xi, yi) = vPi.toTuple()
				wm = vPi.w
				xm += xi
				ym += yi
			virtualPoints.append((WeightedPoint(xm/n, ym/n, wm), iDi))
		print("virtualPoints = ", virtualPoints)
		return virtualPoints
	def updatePencil(self, pencil_id, vanishRay):
		key = pencil_id
		if key in self.pencils:
			self.pencils[key].updateVanishRay(vanishRay)
		else:
			value = VanishPencil(vanishRay)
			self.pencils.update({key:value})