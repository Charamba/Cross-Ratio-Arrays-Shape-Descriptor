from TripleCrossFeature import *
from Utils import *

import itertools


class FeaturesTable:
	def __init__(self):
		self.features = {}

	def getValues(self):
		featuresList = []
		for subFeaturesList in self.features.values():
			for feature in subFeaturesList:
				featuresList.append(feature)
		return featuresList

	def printConfiguration(self):
		print("TOPOLOGY ---- Number of features:")
		for (key, val) in self.features.items():
			strConfigRow = str(val[0]) + " ---- " + str(len(val))
			print(strConfigRow)


	def access(self, topoKey):
		if topoKey in self.features:
			# print("topoKey = ", topoKey)
			# print("len = ", len(self.features[topoKey]))
			# print("self.features[topoKey] = ", self.features[topoKey])
			return self.features[topoKey]
		else:
			return []

	def update(self, feature, key=None):
		key_ = key
		if key_ is None:
			key_ = feature.topoKey

		if key_ in self.features:
			self.features[key_].append(feature)
		else:
			self.features[key_] = [feature]

	def concat(self, other):
		for otherRay in other.getValues():
			self.update(otherRay, key=otherRay.numberOfEdgePoints)