from ShapeDescriptor import *
from Plotter import *
from LineEstimation import *
from VanishPencilsTable import *
import matplotlib.pylab as pl
from matplotlib.patches import ConnectionPatch
import time

from CrossRatio import crossRatio

import pickle

from PIL import Image
from Scanner import *
from MatchingProcessor import *
#from resizeimage import resizeimage

from hull_contour import curves_from_rasterized_img
from find_homography_distance import calc_hausdorff_distance, calc_homography, calc_homography_distance, transf_points

## Close window and change progress in code
def press(event):
	#print('press', event.key)
	if event.key == 'enter':
		plt.close()
def is_even (a):
	print(a)
	if a%2 != 0:  
		return False
	return True

# def check_dimensions(image):
# 	width, height = image.width, image.height
# 	w = is_even(width)
# 	h = is_even(height)
# 	print(w, h)
# 	if not w and not h:
# 		a = image.resize((width+1, height+1), Image.NEAREST)
# 		#a = resize.resize((width+1, height+1), Image.NEAREST)
# 	elif not w and h: 
# 		a = image.resize((width+1, height+1), Image.NEAREST)
# 	elif w and not h: 
# 		a = image.thumbnail((width+1, height+1), Image.ANTIALIAS)
# 	else: 
# 		a = image
# 	return a
# ================================= FLAGS ===================================== #

FULL_POINTS = True # To CRS method

# plot: 
showScanRays = False
showMatchRays = False # MATCH RAYS
showTrajectories = False

# scan: tomographic, convex-hull
convex_hull_scan_template = True
convex_hull_scan_test = True

# convex-hull fan-beam parameters
SAMPLE = 150

template_nFanBeam = SAMPLE
test_nFanBeam = SAMPLE
emitter_points_number = SAMPLE


# compare features: rays, triple cross feature
compare = True
compareByRays = True

# HULL parameters (contour=False, convex=True)
convex_hull_flag = True

# SHAPE FLAG (symetryc=True, assimetric=False)
symetric_shape_flag = True



class costCRSFunction:
	def compute(self, val1, val2):
		logval1 = -1.0 if (val1 == -1.0) else -0.5 if (val1 == 0.0) else math.log(val1) 
		logval2 = -1.0 if (val2 == -1.0) else -0.5 if (val2 == 0.0) else math.log(val2)
		denominator = logval2 + logval1 if (logval2 + logval1) != 0 else 0.0001
		return abs((logval2 - logval1)/denominator)

# ==========================================================================
# ========================  CRS MATCHING PROCESSOR  ========================
# ==========================================================================

class CRS_MatchingProcessor(MatchingProcessor):
	def __init__(self, templateDescriptor, testDescriptor):
		self.templateDescriptor = templateDescriptor
		self.testDescriptor = testDescriptor

	def compare_spectra(self):
		distanceValues = [] 
		mTemplateRays = [] 
		mTestRays = []

		distance = 0
		matchingVerticesPairs = []

		for template_idx, pencil in enumerate(self.templateDescriptor.pencils):
			(test_idx, distance, templateRays, testRays) = self.findBestMatchPencil(pencil, self.testDescriptor.pencils)
			
			if distance != float('inf'):
				matchingVerticesPairs.append((template_idx, test_idx))
				distanceValues.append(distance)

			mTemplateRays += templateRays
			mTestRays += testRays

		(matchingVerticesPairs, distanceValues) = removeDuplicityVerticesMatching(matchingVerticesPairs, distanceValues)

		templateVertices = self.templateDescriptor.hullVertices
		testVertices = self.testDescriptor.hullVertices

		newTemplateVertices = [templateVertices[i] for (i, j) in matchingVerticesPairs]
		newTestVertices =     [    testVertices[j] for (i, j) in matchingVerticesPairs]

		matchingVerticesPairsPoints = list(zip(newTemplateVertices, newTestVertices))

		return (matchingVerticesPairsPoints, distanceValues, mTemplateRays, mTestRays)

	def compareByPencils(self):
		distanceValues = [] 
		mTemplateRays = [] 
		mTestRays = []

		distance = 0
		matchingVerticesPairs = []

		for template_idx, pencil in enumerate(self.templateDescriptor.pencils):
			(test_idx, distance, templateRays, testRays) = self.findBestMatchPencil(pencil, self.testDescriptor.pencils)
			
			if distance != float('inf'):
				matchingVerticesPairs.append((template_idx, test_idx))
				distanceValues.append(distance)

			mTemplateRays += templateRays
			mTestRays += testRays

			if template_idx >= 5:
				break

		# (matchingVerticesPairs, distanceValues) = removeDuplicityVerticesMatching(matchingVerticesPairs, distanceValues)

		# templateVertices = self.templateDescriptor.hullVertices
		# testVertices = self.testDescriptor.hullVertices

		# newTemplateVertices = [templateVertices[i] for (i, j) in matchingVerticesPairs]
		# newTestVertices =     [    testVertices[j] for (i, j) in matchingVerticesPairs]

		# matchingVerticesPairsPoints = list(zip(newTemplateVertices, newTestVertices))

		return (matchingVerticesPairs, distanceValues, mTemplateRays, mTestRays)

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

		templateSpectre = []
		testSpectre = []

		for ray in rays1:
			cr_value = -1 # if numberOfEdgePoints <= 2 
			if ray.numberOfEdgePoints >= 4:
				P0, P1, P2, P3 = ray.edgePoints[0], ray.edgePoints[1], ray.edgePoints[2], ray.edgePoints[-1]
				cr_value = crossRatio(P0, P1, P2, P3)
			elif ray.numberOfEdgePoints == 3:
				cr_value = 0
			templateSpectre.append(cr_value)

		for ray in rays2:
			cr_value = -1 # if numberOfEdgePoints <= 2 
			if ray.numberOfEdgePoints >= 4:
				P0, P1, P2, P3 = ray.edgePoints[0], ray.edgePoints[1], ray.edgePoints[2], ray.edgePoints[-1]
				cr_value = crossRatio(P0, P1, P2, P3)
			elif ray.numberOfEdgePoints == 3:
				cr_value = 0
			testSpectre.append(cr_value)

		costCRSFunct_obj = costCRSFunction()
		#(templateSpectre, testSpectre) = self.generateKeySpectres(mTemplateRays, mTestRays)

		dtw = DTW(templateSpectre, testSpectre, costCRSFunct_obj)
		dtw_dist = dtw.distance()

		#error_distance = totalRays - n
		error_distance = dtw_dist#sum([1 for a,b in zip(templateSpectre,testSpectre) if a==b])
		distance_normalized = error_distance

		# invertendo o espectro do query
		testSpectreReversed = testSpectre
		testSpectreReversed.reverse()

		dtw = DTW(templateSpectre, testSpectreReversed, costCRSFunct_obj)
		dtw_dist = dtw.distance()

		#error_distance = totalRays - n
		error_distance = dtw_dist#sum([1 for a,b in zip(templateSpectre,testSpectreReversed) if a==b])
		distance_normalized_reversed = error_distance

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
