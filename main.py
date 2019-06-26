from CrossRatio import *
from CrossRadonTransform import *
from HoughTransform import *
from ShapeDescriptor import *
from MatchRaysPairs import *
from Plotter import *
#from ransac import *
from LineEstimation import *
from HorizonLine import *
from VanishPencilsTable import *
from Image import *
from Scanner import *

import time

## Close window and change progress in code
def press(event):
	#print('press', event.key)
	if event.key == 'enter':
		plt.close()

# =============================================================================
# ============================= FLAG and Parameters ===========================
# =============================================================================

# Scan rays and Match rays
showScanRays  = True # RED Rays
showMatchRays = True # Cyan Rays

# Grid
showPixelGrid = False # show pixel grid

# SCAN CONFIGURATION
nTraj = 11#15#491#301#401#201#191#101#7 #adiddas:(491, 18)
nProj = 1#18#27#9#180#32#9

nTrajTemplate = nTraj
nProjTemplate = nProj

nTrajTest = nTraj
nProjTest = nProj

# MATCH CARDINALITY
N_By_M_Match_Cardinality = False    	# N:M
N_By_One_Match_Cardinality = False  	# N:1
One_by_N_Match_Cardinality = False  	# 1:N
One_by_One_Match_Cardinality = True 	# 1:1

# VANISH POINTS FLAGS
CRVectorlenThreshold = 1
showAllVanishPoints = True
ignoreDistance = True
limitDistance = 1000

# LINE ESTIMATION
show_Least_Square_line = True
show_Hough_line = True
show_Wighted_Hough_line = True
show_RANSAC_line = True

# PENCILS CLUSTERS
show_VanishPoint_by_angle = False
show_discarted_vanishPoints = True


# =============================================================================
# ================================= MAIN ======================================
# =============================================================================

fig = plt.figure()

# =============================================================================
# ============================== LOAD IMAGES ==================================
# =============================================================================

filename = askopenfilename(filetypes=[("all files","*"),("Bitmap Files","*.bmp; *.dib"),
										("JPEG", "*.jpg; *.jpe; *.jpeg; *.jfif"),
										("PNG", "*.png"), ("TIFF", "*.tiff; *.tif")])

templateImage = Image(misc.imread(filename, mode = 'RGB'))


filename = askopenfilename(filetypes=[("all files","*"),("Bitmap Files","*.bmp; *.dib"),
										("JPEG", "*.jpg; *.jpe; *.jpeg; *.jfif"),
										("PNG", "*.png"), ("TIFF", "*.tiff; *.tif")])

testImage = Image(misc.imread(filename, mode = 'RGB'))


# =============================================================================
# ============================== SHOW ORIGINAL IMAGE ==========================
# =============================================================================
fig.canvas.set_window_title('Original Image')
fig.canvas.mpl_connect('key_press_event', press)

#plotterTemplateImg = Plotter(templateImage)

start_time = time.time()
#(templateSinograma, templateDescriptor) = crossRadonTransform2(templateImage, nTrajTemplate, nProjTemplate) # 2,1
templateScanner = Scanner(templateImage)
templateDescriptor = templateScanner.tomographic_scan(nTrajTemplate, nProjTemplate)
print("--- %s seconds ---" % (time.time() - start_time))

print("#### TEMPLATE STATISTICS ####")
print("n. template rays: ", len(templateDescriptor.rays))
for i in range(1, len(templateDescriptor.countCrossRatioVectorLengths)):
	countLen = templateDescriptor.countCrossRatioVectorLengths[i]
	if countLen > 0:
		print("CrossRatio vector with size %d, have %d Rays" %(i, countLen))
print("---------------------------")

#plotterTestImg = Plotter(testImage)
start_time = time.time()
#(testSinograma, testDescriptor) = crossRadonTransform2(testImage, nTrajTest, nProjTest) #(67, 27)
testScanner = Scanner(testImage)
testDescriptor = testScanner.tomographic_scan(nTrajTest, nProjTest)
print("--- %s seconds ---" % (time.time() - start_time))

print("#### TEST STATISTICS ####")
print("n. test rays: ", len(testDescriptor.rays))
for i in range(1, len(testDescriptor.countCrossRatioVectorLengths)):
	countLen = testDescriptor.countCrossRatioVectorLengths[i]
	if countLen > 0:
		print("CrossRatio vector with size %d, have %d Rays" %(i, countLen))
print("---------------------------")



templateGreenRays = []
testGreenRays = []
testZeroRays = []

bestRaysPairs_1_N = MatchRaysPairs()
bestRaysPairs_N_1 = MatchRaysPairs()

templateRedRays = []
testRedRays = []

# =============================================================================
# ================================ MATCHING ===================================
# =============================================================================
countMatch = 0
totalComp  = 0
countMatchCrossRatioVectorLengths = 60*[0]

for templateRay in templateDescriptor.rays:
	testBestMatchRay = None
	minDistance = 10000

	for testRay in testDescriptor.rays:
		totalComp += 1

		#print("templateRay.crossRatioVector = ", templateRay.crossRatioVector)
		#print("testRay.crossRatioVector = ", testRay.crossRatioVector)

		if templateRay.isMatch(testRay) and testRay.CRV_length() >= CRVectorlenThreshold:# testRay.numberOfEdgePoints >= 4:

			if N_By_M_Match_Cardinality:
				testRay.estimateVanishPoints(templateRay)
				testGreenRays.append(testRay) # Não pode comentar!!!!
				
				if showMatchRays:
					templateGreenRays.append(templateRay)
				
				idxLen = len(testRay.crossRatioVector)
				countMatchCrossRatioVectorLengths[idxLen] += 1
			
			if N_By_One_Match_Cardinality or One_by_One_Match_Cardinality: # N:1 OU 1:1

				testRay.estimateVanishPoints(templateRay)
				beforeLen = bestRaysPairs_N_1.length()
				if bestRaysPairs_N_1.updatePair(testRay, templateRay):
						afterLen = bestRaysPairs_N_1.length()
						if afterLen > beforeLen:
							idxLen = len(testRay.crossRatioVector)
							if idxLen < len(countMatchCrossRatioVectorLengths):
								countMatchCrossRatioVectorLengths[idxLen] += 1
			if One_by_N_Match_Cardinality or One_by_One_Match_Cardinality: # 1:N OU 1:1
				testRay.estimateVanishPoints(templateRay)
				beforeLen = bestRaysPairs_1_N.length()
				if bestRaysPairs_1_N.updatePair(templateRay, testRay):
						afterLen = bestRaysPairs_1_N.length()
						if afterLen > beforeLen:
							idxLen = len(testRay.crossRatioVector)
							if idxLen < len(countMatchCrossRatioVectorLengths):
								countMatchCrossRatioVectorLengths[idxLen] += 1

		# Adicionar nos arrays de raios para exibir depois!
		else:
			if showScanRays:
				templateRedRays.append(templateRay)
				testRedRays.append(testRay)


if One_by_N_Match_Cardinality:
	testGreenRays = bestRaysPairs_1_N.getValues()
elif N_By_One_Match_Cardinality:
	testGreenRays = bestRaysPairs_N_1.getValues()
elif One_by_One_Match_Cardinality:
	bestPairsList = bestRaysPairs_1_N.intersection(bestRaysPairs_N_1)
	testGreenRays = [testRay for (k, testRay) in bestPairsList]
	templateGreenRays = [k for (k, testRay) in bestPairsList]

	countMatchCrossRatioVectorLengths = 60*[0]

	for testRay in testGreenRays:
		idxLen = len(testRay.crossRatioVector)
		if idxLen >= CRVectorlenThreshold and idxLen < len(countMatchCrossRatioVectorLengths):
			countMatchCrossRatioVectorLengths[idxLen] += 1
			

# # ########## PLOT ##########
# ax = fig.add_subplot(1,2,1)
# ax.set_title('Template Image')
# plt.imshow(templateImage)

#### PLOT TEMPLATE RAYS
if showPixelGrid:
	templateImg.plotPixelGrid()

if showScanRays:
	for templateRay in templateRedRays:
		templateImage.plotRay(templateRay)

if showMatchRays:
	for templateRay in templateGreenRays:
		templateImage.plotRay(templateRay, 'c', 'co')

# ax = fig.add_subplot(1,2,2)
# ax.set_title('Test Image')
# plt.imshow(testImage)

vanishPoints = []
vanishPColors = ["kx", "mx", "kx", "gx", "kx", "yx", "kx", "bx", "kx", "rx", "kx", "cx", "kx", "mx", "kx", "gx", "kx", "yx", "kx", "kx"]

validSizeCrossRatioLengths = []
print("#### MATCH STATISTICS ####")
print("Total comparation: ", totalComp)
print("Count Match: ", countMatch)
countRaysTotal = 0
countTestRaysTotal = 0
for size in range(1, len(countMatchCrossRatioVectorLengths)):
	countRays = countMatchCrossRatioVectorLengths[size]
	countTemplateRays = templateDescriptor.countCrossRatioVectorLengths[size]
	countTestRays = testDescriptor.countCrossRatioVectorLengths[size]
	if countRays > 0:
		print("CrossRatio vector with size %d, have %d Rays -- Percentual match: %3.2f %%" %(size, countRays, 100*countRays/countTestRays))

	countRaysTotal += countRays
	countTestRaysTotal += countTestRays

	if countRays <= min(countTemplateRays, countTestRays)*1 and countRays != 0:
		validSizeCrossRatioLengths.append(size)

print("validSizeCrossRatioLengths: ", validSizeCrossRatioLengths)
if countTestRaysTotal != 0:
	print("Percentual total matches: %3.2f %%" %(100*countRaysTotal/countTestRaysTotal))
print("---------------------------")


pencilsTable = VanishPencilsTable()


#### PLOT TEST RAYS
if showPixelGrid:
	testImage.plotPixelGrid()

if showScanRays:
	for testRay in testRedRays:
		testImage.plotRay(testRay)
for testRay in testGreenRays:
	if testRay.numberOfEdgePoints >= 4:
		if showMatchRays:
			testImage.plotRay(testRay, 'c', 'co')

	if (len(testRay.crossRatioVector) in validSizeCrossRatioLengths) or showAllVanishPoints:

		if showMatchRays:
			testImage.plotRay(testRay, 'c', 'co')
		vP1 = testRay.getVanishPoint()

		if vP1:
			distance = vP1.euclideanDistance(R2_Point(0,0))

			if (distance <= limitDistance) or ignoreDistance:
				vanishPoints.append(vP1)
				(x1, y1) = vP1.toTuple()

				crvLen = len(testRay.crossRatioVector)
				if crvLen <= 19:
					vpColor = vanishPColors[len(testRay.crossRatioVector)]
				else:
					vpColor = "kx"
				pencilsTable.updatePencil(testRay.pencil_id, testRay)
				if show_discarted_vanishPoints:
					testImage.plotPoint(x1, y1, color=vpColor)
					#testImage.plotCircle(x1, y1, 2+testRay.crossRatioVectorLength, 100*testRay.pencil_id/nProjTemplate)




if show_VanishPoint_by_angle:
	vanishPoints = []
	for (vPi, iDi) in pencilsTable.getVanishPoints():
		vanishPoints.append(vPi)
		(xi, yi) = vPi.toTuple()
		wi = vPi.w
		if wi <= 19:
				vpColor = vanishPColors[wi]
		else:
			vpColor = "kx"
		testImage.plotPoint(xi, yi, color=vpColor)
		testImage.plotCircle(xi, yi, 2+wi, 100*iDi/nProjTemplate)

	vanishPoints = []
	for (vPi, iDi) in pencilsTable.getVirtualPoints():
		vanishPoints.append(vPi)
		(xi, yi) = vPi.toTuple()
		wi = vPi.w

		if wi <= 19:
				vpColor = vanishPColors[wi]
		else:
			vpColor = "kx"
		testImage.plotPoint(xi, yi, color=vpColor)
		testImage.plotHexagon(xi, yi, 10*wi, 100*iDi/nProjTemplate)




#### PLOT ####
# vanishPoints_set = set(vanishPoints)#set([x for x in vanishPoints if vanishPoints.count(x) > 1])#
# vanishPoints = list(vanishPoints_set)
# print(vanishPoints)
# testImage.plotLinePoints(vanishPoints)


# =============================================================================
# ============================ HORIZON LINE ===================================
# =============================================================================

if show_Hough_line:
	## TRADITIONAL HOUGH TRANSFORM
	vanishPointsHoughSpace, houghLines = points_houghTransform(vanishPoints, weighted=True)
	vphs = vanishPointsHoughSpace.tocoo()
	try:
		idxMaxVal = vphs.data.argmax()
		maxVal = vphs.data[idxMaxVal]
		print("hough space maxVal = ", maxVal)

		#print("vanishPointsHoughSpace: ")
		#print(vanishPointsHoughSpace)

		for linePoints in houghLines.getValues():
			if len(linePoints) >= maxVal:
				testImage.plotLinePoints(linePoints, color='m')
	except ValueError:
		print("Hough space is empty, no hough line!")

if One_by_One_Match_Cardinality and show_Hough_line:
	## WEIGHTED HOUGH TRANSFORM
	vanishPointsHoughSpace, vanishHoughLines = vanishRays_houghTransform(bestPairsList, weighted = True) #points_houghTransform(vanishPoints, weighted = True)
	
	vphs = vanishPointsHoughSpace.tocoo()
	vanishRaysPairs = []
	try:
		idxMaxVal = vphs.data.argmax()
		maxVal = vphs.data[idxMaxVal]
		print("weighted hough space maxVal = ", maxVal)
		print("vanishHoughLines size = ", len(vanishHoughLines.getValues()))

		for raysTupleList in vanishHoughLines.getValues():
			bestVanishPoints = []
			testRays = [testRay for (templateRay, testRay) in raysTupleList]
			scoreWeightPoints = 0
			for testRay in testRays:
				point = testRay.getVanishPoint()
				scoreWeightPoints += point.w
				bestVanishPoints.append(point)

			if scoreWeightPoints >= maxVal:
				vanishRaysPairs = raysTupleList 
				testImage.plotLinePoints(bestVanishPoints, color='g')
	except ValueError:
		print("Weighted Hough space is empty, no hough line!")

#horizon = HorizonLine(bestVanishPoints)
#print("horizon = ", horizon.getRepr())
"""
if One_by_One_Match_Cardinality:
	## PLOT VANISH RAYS 
	for (templateVanishRay, testVanishRay) in vanishRaysPairs:
		templateImage.plotRay(templateVanishRay, 'r--', 'ro')
		testImage.plotVanishRay(testVanishRay, 'r--', 'ro')
"""

if show_Least_Square_line:
	# LeastSquare
	try:
		(Xlstsq, Ylstsq, _, _) = leastSquares(vanishPoints, weighted=True)
		P0 = R2_Point(Xlstsq[0], Ylstsq[0])
		Pf = R2_Point(Xlstsq[1], Ylstsq[1])
		testImage.plotLinePoints([P0,Pf], color="orange")
	except numpy.linalg.linalg.LinAlgError:
		print("Least Square method: not is possible!")

	model = LinearLeastSquaresModel()


if show_RANSAC_line:
	# RANSAC
	try:
		ransacReturn = ransac(vanishPoints,model, int(len(vanishPoints)*0.4), 1000, 7e3, int(len(vanishPoints)*0.2), debug=False,return_all=True, weighted=True)

		if ransacReturn:
			(XRansac, YRansac, a, b) = ransacReturn
			P0 = R2_Point(XRansac[0], YRansac[0])
			Pf = R2_Point(XRansac[1], YRansac[1])
			testImage.plotLinePoints([P0, Pf], color="cyan")
	except numpy.linalg.linalg.LinAlgError:
		print("RANSAC method: not is possible!")



# ########## PLOT ##########
ax = fig.add_subplot(1,2,1)
ax.set_title('Template Image')
(cols, rows) = templateImage.getShape()
plt.imshow(templateImage.image, interpolation='none', origin='upper', extent=[0, rows, cols, 0])
templateImage.showPatches(fig, ax)
templateImage.show()


#testImage.plotCircle(0, 0, 10, 20)
#testImage.plotCircle(50, 50, 15, 100)

ax = fig.add_subplot(1,2,2)
ax.set_title('Test Image')

(cols, rows) = testImage.getShape()
plt.imshow(testImage.image, interpolation='none', origin='upper', extent=[0, rows, cols, 0])
testImage.showPatches(fig, ax)
testImage.show()



plt.show()


#plt.imshow(vanishPointsHoughSpace)
#plt.show()







