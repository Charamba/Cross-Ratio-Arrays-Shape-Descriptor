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
from MatchingProcessor import *

## Close window and change progress in code
def press(event):
	#print('press', event.key)
	if event.key == 'enter':
		plt.close()

# ================================= FLAGS =====================================

# plot:
showScanRays = False
showMatchRays = True
showJunctionPoints = False
showTrajectories = True

# scan: tomographic, convex-hull
tomographic_scan_template = True
convex_hull_scan_template = False

tomographic_scan_test = True
convex_hull_scan_test = False

# tomographic parameters
(template_nTraj, template_nProj) = (91, 6)
(test_nTraj, test_nProj) = (121, 18)

# convex-hull fan-beam parameters
template_nFanBeam = 18
test_nFanBeam = 18

# compare features: rays, triple cross feature
compare = True
compareByRays = True
compareByTripleCrossFeatures = False


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
# ============================== LOAD IMAGES ==================================
# =============================================================================
fig.canvas.set_window_title('Original Image')
fig.canvas.mpl_connect('key_press_event', press)


# =============================================================================
# ============================== SHOW IMAGES ==================================
# =============================================================================

templateRays = []
testRays = []

# Scanners
templateScanner = Scanner(templateImage)    
testScanner = Scanner(testImage)

# Convex-Hull
convexHullVertices = templateImage.convex_hull()
print("**** template len(convexHullVertices): ", len(convexHullVertices))
templateImage.plotLinePoints(convexHullVertices, color="yo", ciclic=True, writeOrder=True)
templateImage.plotLinePoints(convexHullVertices, color="y", ciclic=True)


print("Scanning template image...")
if convex_hull_scan_template:
	templateDescriptor_1 = templateScanner.hull_scan(convexHullVertices, fanBeamRays=template_nFanBeam, showTrajectories=showTrajectories)

if tomographic_scan_template:
	templateDescriptor_2 = templateScanner.tomographic_scan(template_nTraj, template_nProj, calcCrossRatio=True, showTrajectories=showTrajectories)

templateDescriptor = None
if convex_hull_scan_template and tomographic_scan_template:
	templateDescriptor_1.raysTable.concat(templateDescriptor_2.raysTable)
	templateDescriptor = templateDescriptor_1
elif convex_hull_scan_template:
	templateDescriptor = templateDescriptor_1
elif tomographic_scan_template:
	templateDescriptor = templateDescriptor_2


print("Scan complete!")
#templateDescriptor.generateCrossFeatures(templateImage, convexHullVertices)


if compareByTripleCrossFeatures:
	print("Generating Triple Cross Features...")
	templateDescriptor.generateTripleCrossFeatures(templateImage, calcCrossRatio=True)

convexHullVertices = testImage.convex_hull()
print("**** test len(convexHullVertices): ", len(convexHullVertices))
testImage.plotLinePoints(convexHullVertices, color="yo", ciclic=True, writeOrder=True)
testImage.plotLinePoints(convexHullVertices, color="y", ciclic=True)

#convexHullPixels = testImage.convexHullPixels(convexHullVertices)
#testImage.plotVisitedPixels(convexHullPixels, 80, True)
print("")
print("Scanning test image...")
if convex_hull_scan_test:
	testDescriptor_1 = testScanner.hull_scan(convexHullVertices, fanBeamRays=test_nFanBeam, showTrajectories=showTrajectories)

if tomographic_scan_test:
	#(nTraj, nProj) = (3, 3)#(21,12)#(51,6)#(33, 18)#(21,12)#(33,9)#(89,9)#(18,3)#(55, 9)#(55,6)#(99,9)#(55,6)#(99,9)#(99,6)#(66,2)#(21,2)#(15,9)
	testDescriptor_2 = testScanner.tomographic_scan(test_nTraj, test_nProj, calcCrossRatio=True, showTrajectories=showTrajectories)

testDescriptor = None
if convex_hull_scan_test and tomographic_scan_test:
	testDescriptor_1.raysTable.concat(testDescriptor_2.raysTable)
	testDescriptor = testDescriptor_1
elif convex_hull_scan_test:
	testDescriptor = testDescriptor_1
elif tomographic_scan_test:
	testDescriptor = testDescriptor_2


print("Scan complete!")


#testDescriptor.generateCrossFeatures(testImage, convexHullVertices)
if compareByTripleCrossFeatures:
	print("Generating Triple Cross Features...")
	testDescriptor.generateTripleCrossFeatures(testImage, calcCrossRatio=False)

#testImage.plotPixelGrid()

if showScanRays:
	for templateRay in templateDescriptor.rays:
		#templateImage.plotRay(templateRay)
		templateImage.plotLinePoints(templateRay.edgePoints, color="r", correction=True)
		templateImage.plotLinePoints(templateRay.edgePoints, color="ro", correction=True, writeOrder=False)

	for testRay in testDescriptor.rays:
		#testRay.edgePoints
		#testImage.plotRay(testRay)
		testImage.plotLinePoints(testRay.edgePoints, color="r", correction=True)
		testImage.plotLinePoints(testRay.edgePoints, color="ro", correction=True, writeOrder=False)


print("Comparing features...")
matchingProcessor = MatchingProcessor(templateDescriptor, testDescriptor)
print("Finish!")
print("Ploting...")


if compare:
	if compareByRays:
		(mTemplRays, mTestRays, bTemplRays, bTestRays, vanishPoints) = matchingProcessor.compare()
	#(mTemplRays, mTestRays, bTemplRays, bTestRays, vanishPoints) = matchingProcessor.compareByCrossFeatures()

	if compareByTripleCrossFeatures:
		(mTemplRays, mTestRays, bTemplRays, bTestRays, vanishPoints) = matchingProcessor.compareByTripleCrossFeatures()
	if showMatchRays:
		for templRay in mTemplRays:
			templateImage.plotLinePoints(templRay.edgePoints, color="c", correction=True)
			templateImage.plotLinePoints(templRay.edgePoints, color="co", correction=True, writeOrder=False)
			#templateImage.plotVisitedPixels(templRay.edgePoints, 10, correction=True)
		for testRay in mTestRays:
			testImage.plotLinePoints(testRay.edgePoints, color="c", correction=True)
			testImage.plotLinePoints(testRay.edgePoints, color="co", correction=True, writeOrder=False)
			#testImage.plotVisitedPixels(testRay.edgePoints, 10, correction=True)
	# for templRay in bTemplRays:
	# 	templateImage.plotLinePoints(templRay.edgePoints, color="m", correction=True)
	# 	templateImage.plotLinePoints(templRay.edgePoints, color="mo", correction=True)

	# for testRay in bTestRays:
	# 	testImage.plotLinePoints(testRay.edgePoints, color="m", correction=True)
	# 	testImage.plotLinePoints(testRay.edgePoints, color="mo", correction=True)

	for vp in vanishPoints:
		if vp is not None:
			(xv, yv) = vp.toTuple()
			testImage.plotPoint(xv, yv, color="bx")

	if showJunctionPoints:
		# Show junction points
		for cf in templateDescriptor.crossFeatures:
			(xcf, ycf) = cf.junctionPoint.toTuple()
			templateImage.plotPoint(xcf, ycf, color='bo')

		for cf in testDescriptor.crossFeatures:
			(xcf, ycf) = cf.junctionPoint.toTuple()
			testImage.plotPoint(xcf, ycf, color='bo')



# Show Template Image
ax = fig.add_subplot(1,2,1)
ax.set_title('Template Image')
(cols, rows) = templateImage.getShape()
plt.imshow(templateImage.image, interpolation='none', origin='upper', extent=[0, rows, cols, 0])
templateImage.showPatches(fig, ax)
templateImage.show()

# Show Test Image
ax = fig.add_subplot(1,2,2)
ax.set_title('Test Image')
(cols, rows) = testImage.getShape()
plt.imshow(testImage.image, interpolation='none', origin='upper', extent=[0, rows, cols, 0])
testImage.showPatches(fig, ax)
testImage.show()


plt.show()