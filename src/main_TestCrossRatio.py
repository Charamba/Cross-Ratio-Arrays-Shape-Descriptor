from CrossRatio import *
from CrossRadonTransform import *
from HoughTransform import *
from ShapeDescriptor import *
from MatchRaysPairs import *
from Plotter import *
#from ransac import *
from LineEstimation import *

import time

## Close window and change progress in code
def press(event):
	#print('press', event.key)
	if event.key == 'enter':
		plt.close()

# =============================================================================
# =============================== PLOT Parameters =============================
# =============================================================================
showScanRays  = True # RED Rays
showMatchRays = True # Cyan Rays

showPixelGrid = False # show pixel grid

# SCAN CONFIGURATION
nTraj = 1#891#301#401#201#191#101#7 #adiddas:(491, 18)
nProj = 1#18#27#27#27#9#180#32#9

# MATCH CARDINALITY
N_By_M_Match_Cardinality = False    	# N:M
N_By_One_Match_Cardinality = False  	# N:1
One_by_N_Match_Cardinality = False  	# 1:N
One_by_One_Match_Cardinality = True 	# 1:1

# VANISH POINTS FLAGS
showAllVanishPoints = True
ignoreDistance = False
lenThreshold = 1
limitDistance = 100000

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

templateImage = misc.imread(filename, mode = 'RGB')


filename = askopenfilename(filetypes=[("all files","*"),("Bitmap Files","*.bmp; *.dib"),
										("JPEG", "*.jpg; *.jpe; *.jpeg; *.jfif"),
										("PNG", "*.png"), ("TIFF", "*.tiff; *.tif")])

testImage = misc.imread(filename, mode = 'RGB')


# =============================================================================
# ============================== SHOW ORIGINAL IMAGE ==========================
# =============================================================================



fig.canvas.set_window_title('Original Image')
fig.canvas.mpl_connect('key_press_event', press)

#Theta = np.arange(0,180,1)
#Theta = np.arange(0,1,1)

plotterTemplateImg = Plotter(templateImage)

start_time = time.time()
(templateSinograma, templateDescriptor) = crossRadonTransform2(templateImage, nTraj, nProj) # 2,1
print("--- %s seconds ---" % (time.time() - start_time))

print("#### TEMPLATE STATISTICS ####")
print("n. template rays: ", len(templateDescriptor.rays))
for i in range(1, len(templateDescriptor.countCrossRatioVectorLengths)):
	countLen = templateDescriptor.countCrossRatioVectorLengths[i]
	if countLen > 0:
		print("CrossRatio vector with size %d, have %d Rays" %(i, countLen))
print("---------------------------")

plotterTestImg = Plotter(testImage)
start_time = time.time()
(testSinograma, testDescriptor) = crossRadonTransform2(testImage, nTraj, nProj) #(67, 27)
print("--- %s seconds ---" % (time.time() - start_time))

print("#### TEST STATISTICS ####")
print("n. test rays: ", len(testDescriptor.rays))
for i in range(1, len(testDescriptor.countCrossRatioVectorLengths)):
	countLen = testDescriptor.countCrossRatioVectorLengths[i]
	if countLen > 0:
		print("CrossRatio vector with size %d, have %d Rays" %(i, countLen))
print("---------------------------")

#templateRay = RayDescriptor(0, 0)
#templateRay.crossRatioVector = [1.3714285714285714, 1.149212233549583, 1.4290204295442641, 1.1684981684981686, 2.3125]
#templateRay.crossRatioVector =[1.421591804570528, 1.1000322476620445, 3.676767676767677, 1.0241541964866623, 2.5906071019473083, 1.101679389312977, 1.552466896426628, 1.412754485451334, 1.190162037037037, 1.1326650943396228, 1.7395348837209303, 1.5148005148005148, 1.2096681415929202, 1.0353811184136095, 1.4502258658189726, 1.9347826086956519, 1.3089247062461347, 1.00704720894817, 4.320862845115, 1.0916076249090187, 1.2074395924089176, 1.57446265853602, 1.2128607809847198, 1.255924978687127, 2.173862586232834]

#print("len(descriptor.rays) = ", len(descriptor.rays))

templateGreenRays = []
testGreenRays = []
testZeroRays = []

bestRaysPairs_1_N = MatchRaysPairs()
bestRaysPairs_N_1 = MatchRaysPairs()

templateRedRays = []
testRedRays = []

# ########## MATCHING ##########
countMatch = 0
totalComp  = 0
countMatchCrossRatioVectorLengths = 60*[0]

for templateRay in templateDescriptor.rays:
	#print("TEMPLATE RAY")
	#print("cross ratio signature: ", templateRay.crossRatioVector)
	#print("edge points: ")
	#for edgeP in templateRay.edgePoints:
		#print(edgeP)
	
	testBestMatchRay = None
	minDistance = 1000
	for testRay in testDescriptor.rays:
		totalComp += 1

		if templateRay.isMatch(testRay) and testRay.numberOfEdgePoints >= 4:
			print(templateRay.crossRatioVector)
			print(testRay.crossRatioVector)
			if N_By_M_Match_Cardinality:
				#if testRay.numberOfEdgePoints >= 4:
				testRay.estimateVanishPoints(templateRay)
				testGreenRays.append(testRay) # Não pode comentar!!!!
				
				if showMatchRays:
					templateGreenRays.append(templateRay)
				
				idxLen = len(testRay.crossRatioVector)
				countMatchCrossRatioVectorLengths[idxLen] += 1
			
			if N_By_One_Match_Cardinality or One_by_One_Match_Cardinality: # N:1 OU 1:1
				# matchDistance = templateRay.calcDistance(testRay)
				# if minDistance > matchDistance:
				# 	testBestMatchRay = testRay
				# 	minDistance = matchDistance
				testRay.estimateVanishPoints(templateRay)
				beforeLen = bestRaysPairs_N_1.length()
				if bestRaysPairs_N_1.updatePair(testRay, templateRay):
						afterLen = bestRaysPairs_N_1.length()
						if afterLen > beforeLen:
							idxLen = len(testRay.crossRatioVector)
							countMatchCrossRatioVectorLengths[idxLen] += 1
			if One_by_N_Match_Cardinality or One_by_One_Match_Cardinality: # 1:N OU 1:1
				testRay.estimateVanishPoints(templateRay)
				beforeLen = bestRaysPairs_1_N.length()
				if bestRaysPairs_1_N.updatePair(templateRay, testRay):
						afterLen = bestRaysPairs_1_N.length()
						if afterLen > beforeLen:
							idxLen = len(testRay.crossRatioVector)
							countMatchCrossRatioVectorLengths[idxLen] += 1

		# Adicionar nos arrays de raios para exibir depois!
		else:
			if showScanRays:
				templateRedRays.append(templateRay)
				testRedRays.append(testRay)
	# if N_By_One_Match_Cardinality: #Unicidade parcial OU Unicidade total
	# 	if testBestMatchRay:
	# 		testBestMatchRay.estimateVanishPoints(templateRay)
	# 		testGreenRays.append(testBestMatchRay)
	# 		idxLen = len(testBestMatchRay.crossRatioVector)
	# 		countMatchCrossRatioVectorLengths[idxLen] += 1
	# 		if showMatchRays:
	# 			templateGreenRays.append(templateRay)

	# elif One_by_N_Match_Cardinality:
	# 	if testBestMatchRay:
	# 		testBestMatchRay.estimateVanishPoints(templateRay)

	# 		if bestRaysPairs_1:N.updatePair(testBestMatchRay, templateRay):
	# 			testGreenRays.append(testBestMatchRay)
	# 			idxLen = len(testBestMatchRay.crossRatioVector)
	# 			countMatchCrossRatioVectorLengths[idxLen] += 1
	# 			if showMatchRays:
	# 				templateGreenRays.append(templateRay)

if One_by_N_Match_Cardinality:
	testGreenRays = bestRaysPairs_1_N.getValues()#getKeys()
	#print("testGreenRays = ", testGreenRays)
elif N_By_One_Match_Cardinality:
	testGreenRays = bestRaysPairs_N_1.getValues()#getKeys()
elif One_by_One_Match_Cardinality:

	print("len N:1 = ", len(bestRaysPairs_N_1.getKeys()))
	print("len 1:N = ", len(bestRaysPairs_1_N.getKeys()))

	bestPairsList = bestRaysPairs_1_N.intersection(bestRaysPairs_N_1)
	testGreenRays = [testRay for (k, testRay) in bestPairsList]
	#print("testGreenRays = ", testGreenRays)
	print("len 1:1 = ", len(testGreenRays))
	countMatchCrossRatioVectorLengths = 60*[0]

	for testRay in testGreenRays:
		idxLen = len(testRay.crossRatioVector)
		if idxLen >= lenThreshold:
			countMatchCrossRatioVectorLengths[idxLen] += 1
			

# ########## PLOT ##########
ax = fig.add_subplot(1,2,1)
ax.set_title('Template Image')
plt.imshow(templateImage)

#### PLOT TEMPLATE RAYS
if showPixelGrid:
	plotterTemplateImg.plotPixelGrid()

if showScanRays:
	for templateRay in templateRedRays:
		plotterTemplateImg.plotRay(templateRay)

if showMatchRays:
	for templateRay in templateGreenRays:
		plotterTemplateImg.plotRay(templateRay, 'c', 'co')
#	eP1 = templateRay.edgePoints[0]
	#(x1, y1) = eP1.toTuple()
	#plotterTemplateImg.plotPoint(x1, y1, color='yo')


ax = fig.add_subplot(1,2,2)
ax.set_title('Test Image')
plt.imshow(testImage)

vanishPoints = []
vanishPColors = ["kx", "mx", "kx", "gx", "kx", "yx", "kx", "bx", "kx", "rx", "kx", "cx", "kx", "mx", "kx", "gx", "kx", "yx", "kx", "kx"]

validSizeCrossRatioLengths = []
print("#### MATCH STATISTICS ####")
print("Total comparation: ", totalComp)
print("Count Match: ", countMatch)
for size in range(1, len(countMatchCrossRatioVectorLengths)):
	countRays = countMatchCrossRatioVectorLengths[size]
	if countRays > 0:
		print("CrossRatio vector with size %d, have %d Rays" %(size, countRays))

	countTemplateRays = templateDescriptor.countCrossRatioVectorLengths[size]
	countTestRays = testDescriptor.countCrossRatioVectorLengths[size]
	if countRays <= min(countTemplateRays, countTestRays)*1 and countRays != 0:
	#if size == 13:
		validSizeCrossRatioLengths.append(size)

print("validSizeCrossRatioLengths: ", validSizeCrossRatioLengths)
print("---------------------------")


#### PLOT TEST RAYS
if showPixelGrid:
	plotterTestImg.plotPixelGrid()

if showScanRays:
	for testRay in testRedRays:
		plotterTestImg.plotRay(testRay)
for testRay in testGreenRays:
	#print("testRay = ", testRay)
	if testRay.numberOfEdgePoints >= 4:
		if showMatchRays:
			plotterTestImg.plotRay(testRay, 'c', 'co')

	if (len(testRay.crossRatioVector) in validSizeCrossRatioLengths) or showAllVanishPoints:

		if showMatchRays:
			plotterTestImg.plotRay(testRay, 'c', 'co')
		vP1 = testRay.vanishPointCandidate1

		##vP2 = testRay.vanishPointCandidate2
		#eP1 = testRay.edgePoints[0]
		#nextP2 = testRay.nextPoint2

		#plotterTestImg.plotPoint(x2, y2, color='g')

		#(xe1, ye1) = eP1.toTuple()
		#plotterTestImg.plotPoint(xe1, ye1, color='yo')
		if vP1:
			distance = vP1.euclideanDistance(R2_Point(0,0))

			if (distance <= limitDistance) or ignoreDistance:
				vanishPoints.append(vP1)
				(x1, y1) = vP1.toTuple()
				##(x2, y2) = vP2.toTuple()
				#print("xe1 = ", xe1)
				#if xe1 > 50:
				#	color = 'go'
				#else:
				#color = 'ro'
				crvLen = len(testRay.crossRatioVector)
				if crvLen <= 19:
					vpColor = vanishPColors[len(testRay.crossRatioVector)]
				else:
					vpColor = "kx"
				plotterTestImg.plotPoint(x1, y1, color=vpColor)
				##plotterTestImg.plotPoint(x2, y2, color='go')



		#(x2, y2) = nextP2.toTuple()
		#plotterTestImg.plotPoint(x2, y2, color='g', marker='+')

'''
for i in range(0, len(testGreenRays)):
	testRay = testGreenRays[i]
	if len(testRay.crossRatioVector) >= 5:
		for j in range(i+1, len(testGreenRays)):
			testRay2 = testGreenRays[j]
			if len(testRay2.crossRatioVector) >= 5:
					vP2 = testRay.P2Coordinate.cross(testRay2.P2Coordinate)
					vP2.normalize()
					(x2, y2, _) = vP2.toTuple()
					plotterTestImg.plotPoint(x2, y2, color="mx")
'''


#print("vanishPoints = ", vanishPoints)
vanishPoints_set = set(vanishPoints)
vanishPoints = list(vanishPoints_set)


## TRADITIONAL HOUGH TRANSFORM
vanishPointsHoughSpace, houghLines = points_houghTransform(vanishPoints)
vphs = vanishPointsHoughSpace.tocoo()
try:
	idxMaxVal = vphs.data.argmax()
	maxVal = vphs.data[idxMaxVal]
	print("hough space maxVal = ", maxVal)

	#print("vanishPointsHoughSpace: ")
	#print(vanishPointsHoughSpace)

	for linePoints in houghLines.getValues():
		if len(linePoints) >= maxVal:
			plotterTestImg.plotLinePoints(linePoints, color='b')
except ValueError:
	print("Hough space is empty, no hough line!")

## WEIGHTED HOUGH TRANSFORM
vanishPointsHoughSpace, houghLines = points_houghTransform(vanishPoints, weighted = True)
vphs = vanishPointsHoughSpace.tocoo()
try:
	idxMaxVal = vphs.data.argmax()
	maxVal = vphs.data[idxMaxVal]
	print("weighted hough space maxVal = ", maxVal)

	#print("vanishPointsHoughSpace: ")
	#print(vanishPointsHoughSpace)
	print("houghLines size = ", len(houghLines.getValues()))

	for linePoints in houghLines.getValues():
		#print("type linePoints: ", type(linePoints))
		#print("linePoints size = ", len(linePoints))
		scoreWeightPoints = 0
		for point in linePoints:
			scoreWeightPoints += point.w

		if scoreWeightPoints >= maxVal:
			#print("line with %d points" %(len(linePoints)))
			plotterTestImg.plotLinePoints(linePoints, color='g')
except ValueError:
	print("Weighted Hough space is empty, no hough line!")

# LeastSquare
try:
	(Xlstsq, Ylstsq, _, _) = leastSquares(vanishPoints)

	P0 = R2_Point(Xlstsq[0], Ylstsq[0])
	Pf = R2_Point(Xlstsq[1], Ylstsq[1])
	plotterTestImg.plotLinePoints([P0,Pf], color="orange")
except numpy.linalg.linalg.LinAlgError:
	print("Least Square method: not is possible!")

# RANSAC
# n_samples = 500
# n_inputs = 1
# n_outputs = 1

# A_noisy = []
# B_noisy = []

# input_columns = range(n_inputs) # the first columns of the array
# output_columns = [n_inputs+i for i in range(n_outputs)] # the last columns of the array
# debug = False
model = LinearLeastSquaresModel()

# #print("vanishPoints = ", vanishPoints)
# for vp in list(vanishPoints):
# 	(x, y) = vp.toTuple()
# 	A_noisy.append([x])
# 	B_noisy.append([y])

# A_noisy = np.transpose(A_noisy)
# B_noisy = np.transpose(B_noisy)

#print("A_noisy = ", A_noisy)
#sort_idxs = numpy.argsort(A_noisy[:, 0])
#A_col0_sorted = A_noisy[sort_idxs] # maintain as rank-2 array

#all_data = numpy.hstack((A_noisy,B_noisy)) #50, 1000, 7e3, 300

try:
	ransacReturn = ransac(vanishPoints,model, int(len(vanishPoints)*0.1), 1000, 7e3, int(len(vanishPoints)*0.2), debug=False,return_all=True)

	if ransacReturn:
		(XRansac, YRansac, a, b) = ransacReturn

		P0 = R2_Point(XRansac[0], YRansac[0])
		Pf = R2_Point(XRansac[1], YRansac[1])
		plotterTestImg.plotLinePoints([P0, Pf], color="cyan")
except numpy.linalg.linalg.LinAlgError:
	print("RANSAC method: not is possible!")


# linear_fit,resids,rank,s = scipy.linalg.lstsq(all_data[:,input_columns], all_data[:,output_columns])

# print("linear_fit = ", linear_fit)
# linear_fit = np.transpose(linear_fit)

# if 1:
# 	import pylab
# 	pylab.plot( A_col0_sorted[:,0], numpy.dot(A_col0_sorted,linear_fit)[:,0], label='RANSAC fit' )



#angle = 0.6467
#(x0, y0) = calcPoint(19, math.cos(angle), math.sin(angle), -1000)
#(xf, yf) = calcPoint(19, math.cos(angle), math.sin(angle),  1000)

#(x, y) = (502.662027, -199.745271)
#(x, y) = (449.898912, -171.711916)
#(x, y) = (329.752412, -109.633635)
#(x, y) = (531.171458, -213.775755)
#(x, y) = (452.782545, -174.144583)



#P0 = R2_Point(329.752412, 109.633635)#(99.563625, 99.842987)
#Pf = R2_Point(531.171458, 213.775755)#(180.909506, 207.502487)
#plotterTestImg.plotLine(P0, Pf, color="k")


plt.show()


#plt.imshow(vanishPointsHoughSpace)
#plt.show()







