from ShapeDescriptor import *

class MatchRaysPairs:
	def __init__(self):
		self.pairs = {}
	def length(self):
		return len(self.pairs)
	def getPairs(self):
		return self.pairs.items()
	def getValues(self):
		return self.pairs.values()
	def getKeys(self):
		return self.pairs.keys()
	def updatePair(self, rayKey, rayValue):
		isUpdate = False
		if rayKey in self.pairs:
			#print("rayKey = ", rayKey)
			pack = self.pairs[rayKey]
			if type(pack) == list:
				[oldRayValue] = pack
			else:
				oldRayValue = pack 

			#print("oldRayValue = ", oldRayValue)
			
			oldDistance = rayKey.calcDistance(oldRayValue)
			newDistance = rayKey.calcDistance(rayValue)
			if oldDistance > newDistance:
				self.pairs[rayKey] = rayValue
				#self.pairs[rayKey].append(rayValue)
				isUpdate = True
		else:
			self.pairs.update({rayKey:rayValue})
			isUpdate = True
		return isUpdate
	def intersection(self, other):
		#print("len self.getKeys()", len(self.getKeys()))
		set1 = set(self.getPairs())
		set2 = set([(v,k) for k,v in list(other.getPairs())]) # reverse (k,v) |----> (v,k)
		intersectionPairs = set1 & set2
		return list(intersectionPairs)
