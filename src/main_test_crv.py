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

## Close window and change progress in code
def press(event):
	#print('press', event.key)
	if event.key == 'enter':
		plt.close()

# ================================= FLAGS =====================================




# ================================= Transform =================================

#transfMatrix = [0.66612, -0.45833, 110; -0.00775, 0.55148, 31.0; -0.00004, -0.00132, 1.0]

def transfProj(P):
	(x, y) = P.toTuple() # R2_Point
	z = 1
	P_ = P2_Point(0.66612*x -0.45833*y + 110.0*z, -0.00775*x + 0.55148*y + 31.0*z, -0.00004*x -0.00132*y + 1.0*z )#P2_Point(400*x + 720*y, 1440*y, -x + 6*y + 600*z)
	return P_.toR2_Point()


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

# Points and lines
shift_x = 50
#A = R2_Point(375,120)#(4,153)#(77,30) #(230, 5)
#L = R2_Point(82,7)#(358,13)#(350,30) #(171, 206)

templateInitPoints = [R2_Point(375.5,120.5), R2_Point(4,153), R2_Point(350,30), R2_Point(230.5, 5.5), R2_Point(200.5,6.5), R2_Point(324.5,6.5)]
templateFinalPoints = [R2_Point(82.5,7.5), R2_Point(358,13), R2_Point(77,30), R2_Point(171.5, 206.5), R2_Point(200.5,223.5), R2_Point(324.5,223.5)]


convexHullVertices = templateImage.convex_hull()
templateImage.plotLinePoints(convexHullVertices, color="yo", ciclic=True)
templateImage.plotLinePoints(convexHullVertices, color="y", ciclic=True)

convexHullVertices = testImage.convex_hull()
testImage.plotLinePoints(convexHullVertices, color="yo", ciclic=True)
testImage.plotLinePoints(convexHullVertices, color="y", ciclic=True)


for P0, Pf in zip(templateInitPoints, templateFinalPoints):
	(visitedTemplatePixels, edgePoints, s) = templateImage.calc_edgePoints(P0, Pf)
	print("len template edgePoints = ", len(edgePoints))
	templateImage.plotLinePoints(edgePoints, color="co", writeOrder=True)
	templateImage.plotLinePoints([P0, Pf], color="yo", correction=False)
	templateImage.plotLinePoints([P0, Pf], color="c--", correction=False)
	templateRays.append(RayDescriptor(s, 1, edgePoints))#edgePoints))
	##templateImage.plotRay(RayDescriptor(s, 1, edgePoints)) # <======
	# Transform
	P0_ = transfProj(P0)
	Pf_ = transfProj(Pf)


	(visitedTestPixels, edgePoints, s) = testImage.calc_edgePoints(P0_, Pf_)
	print("len test edgePoints = ", len(edgePoints))
	testImage.plotLinePoints(edgePoints, color="co", writeOrder=True)
	testImage.plotLinePoints([P0_, Pf_], color="co", correction=False)
	testImage.plotLinePoints([P0_, Pf_], color="c--", correction=False)
	testRays.append(RayDescriptor(s, 1, edgePoints))#edgePoints))
	##testImage.plotRay(RayDescriptor(s, 1, edgePoints)) # <======
	# Plot pixels
	templateImage.plotVisitedPixels(visitedTemplatePixels, 10)
	testImage.plotVisitedPixels(visitedTestPixels, 10)



for (templateRay, testRay) in zip(templateRays, testRays):
	if testRay.CRV_length() == templateRay.CRV_length():
		testRay.estimateVanishPoints(templateRay)
		vp = testRay.getVanishPoint()
		(xv, yv) = vp.toTuple()
		#(xv, yv) = testImage.toDefaultCoordinate(xv, yv)
		testImage.plotPoint(xv, yv, color='bx')



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