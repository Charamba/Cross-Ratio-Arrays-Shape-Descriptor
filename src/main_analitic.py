from CrossRatio import *
from CrossRadonTransform import *
from ShapeDescriptor import *
from Plotter import *
from Point import *


## Close window and change progress in code
def press(event):
	#print('press', event.key)
	if event.key == 'enter':
		plt.close()

# ================================= Transform =================================
def transfProj(P):
	(x, y) = P.toTuple() # R2_Point
	z = 1
	P_ = P2_Point(400*x + 720*y, 1440*y, -x + 6*y + 600*z)
	return P_.toR2_Point()
	

# ================================= LINE ======================================
class Line:
	def __init__(self, P0, Pf, finiteLine=True):
		self.P0 = P0
		self.Pf = Pf
		self.isFiniteLine = finiteLine
		(xf, yf) = Pf.toTuple()
		(x0, y0) = P0.toTuple()
		if (xf - x0) != 0:
			self.a = (yf - y0)/(xf - x0)
			self.b = yf - self.a*xf
			self.isVertical = False
		else:
			self.a = 0
			self.b = 0
			self.isVertical = True
	def getCoeficients(self):
		return (self.a, self.b)
	def calcIntersection(self, other):
		(a1, b1) = self.getCoeficients()
		(a2, b2) = other.getCoeficients()
		(xi, yi) = (None, None)

		if (a1 == a2) and not self.isVertical:
			print("a1 = ", a1)
			print("a2 = ", a2)
			return (None, None)

		if other.isVertical:
			if not self.isVertical:
				xi = other.P0.x
				yi = a2*xi + b2
				#return (xi, yi)
			#else:
			#	if b1 == b2:
			#		(xi) = other.P0.toTuple()
		else:
			(a1, b1) = self.getCoeficients()
			(a2, b2) = other.getCoeficients()

			if not self.isVertical:
				xi = (b2 - b1)/(a1 - a2)
			else:
				xi = self.P0.x
			yi = a2*xi + b2

		if (xi, yi) != (None, None):
			(x0, y0) = other.P0.toTuple()
			(xf, yf) = other.Pf.toTuple()
			xMin = min(x0, xf)
			xMax = max(x0, xf)
			yMin = min(y0, yf)
			yMax = max(y0, yf)
			if (xMin <= xi <= xMax) and (yMin <= yi <= yMax):
				return (xi, yi)
			else:
				return (None, None)
		return (xi, yi)

		#elif b1 == b2:
def crossRadonTransform_analitic(image, nTraj, nProj, ObjLines=[]):
	#print("radonTransform")
	# Compute the number of rows and columns in the image
	(row_num, col_num, _) = image.shape
	# Initialize the transformed image
	diagonal = int(math.sqrt(row_num*row_num + col_num*col_num))
	tMax = nProj
	sMax = int(diagonal/2)
	ds = diagonal/(nTraj+1)
	
	shapeResult = (nTraj, nProj, 3)#(diagonal, nProj, 3)
	sinograma = np.zeros(shapeResult)#, dtype=np.uint8)
	
	width = col_num
	height = row_num
	dTheta = np.pi/nProj #3.14159265/nProj
	halfWidth = width/2
	halfHeight = height/2
	halfD = diagonal/2 # meia diagonal da imagem

	#Images indices: s, t, y, x
	du = u_step

	descriptor = ShapeDescriptor()

	# Compute transformed image
	for t in range(0, tMax):
		theta = t*dTheta
		cosT = math.cos(theta)
		sinT = math.sin(theta)
		for sIdx in range(0, nTraj): #np.arange(0, sMax + ds, ds):
			s = sIdx*ds - sMax
			#s = int(s)
			"""
			countR = sinograma[sIdx][t][0]
			countG = sinograma[sIdx][t][1]
			countB = sinograma[sIdx][t][2]
			"""

			X = []
			Y = []
			firstPoint = True
			#print('s, t = %f, %f' %(s, t))
			
			edgePoints = []
			#for u in np.arange(-halfD, halfD+du, du):
			# Compute the point p in the original image which is mapped to the
			# (row, col) pixel in the transformed image
			
			(x, y) = calcPoint(s, cosT, sinT, -halfD)
			P0 = R2_Point(x, y)
			(x, y) = calcPoint(s, cosT, sinT,  halfD)
			Pf = R2_Point(x, y)
			rayLine = Line(P0, Pf)

			for objLine in ObjLines:
				(xi, yi) = rayLine.calcIntersection(objLine)

				if (xi, yi) != (None, None):
					print("xi, yi = (%f, %f)" %(xi, yi))
					xIdx =  xi + halfWidth
					yIdx = -yi + halfHeight

					if 0 <= xIdx < width and 0 <= yIdx < height:
						#X.append(xIdx)
						#Y.append(yIdx)
						X.append(xIdx)
						Y.append(yIdx)
						"""
						countR = countR + int(image[yIdx][xIdx][0])
						countG = countG + int(image[yIdx][xIdx][1])
						countB = countB + int(image[yIdx][xIdx][2])
						"""
						#if isEdgePoint(s, cosT, sinT, u, du, halfWidth, halfHeight, image, firstPoint):
						#if (xi, yi) != (None, None):
							#plt.plot([xIdx], [yIdx], 'ro')
						edgePoints.append(R2_Point(xi, yi))
					#firstPoint = False
				descriptor.addRay(s, theta, edgePoints)
			#plt.plot(X, Y, 'r')
			"""
			sinograma[sIdx][t][0] = countR
			sinograma[sIdx][t][1] = countG
			sinograma[sIdx][t][2] = countB
			"""

	#row_sums = sinograma.sum(axis=1)
	#new_matrix = sinograma / row_sums[:, np.newaxis]
	return (sinograma, descriptor)


# =============================================================================
# ================================= MAIN ======================================
# =============================================================================

fig = plt.figure()

# =============================================================================
# ============================== LOAD OBJECTS =================================
# =============================================================================

shapeImage = (600, 600, 3)
templateImage = np.zeros(shapeImage)
testImage = np.zeros(shapeImage)

A = R2_Point(0, 0)
B = R2_Point(200, 0)
L = R2_Point(0,200)
K = R2_Point(200, 200) #(200,200)
M = R2_Point(0, 150)
N = R2_Point(200, 150)
P = R2_Point(0, 100)
Q = R2_Point(200, 100)

S = R2_Point(50, -10)
T = R2_Point(150, 10)
ST_ = Line(S, T)

AB_ = Line(A, B)
LK_ = Line(L, K)
MN_ = Line(M, N)
PQ_ = Line(P, Q)
shift = 5
Ls = R2_Point(L.x - shift, L.y)
As = R2_Point(A.x - shift, A.y)
LA_ = Line(Ls, As)
Ks = R2_Point(K.x + shift, K.y)
Bs = R2_Point(B.x + shift, B.y)
KB_ = Line(K, B)

templateLines = [AB_, PQ_, MN_, LK_, LA_, KB_] #[LK_, MN_, PQ_, AB_, LA_, KB_]


(xi, yi) = ST_.calcIntersection(AB_)
print("xi, yi = (%f, %f)" %(xi, yi))


A1 = transfProj(A)
B1 = transfProj(B)
L1 = transfProj(L)
K1 = transfProj(K)
M1 = transfProj(M)
N1 = transfProj(N)
P1 = transfProj(P)
Q1 = transfProj(Q)

Ls1 = transfProj(Ls)
As1 = transfProj(As)
Ks1 = transfProj(Ks)
Bs1 = transfProj(Bs)

AB1_ = Line(A1, B1)
LK1_ = Line(L1, K1)
MN1_ = Line(M1, N1)
PQ1_ = Line(P1, Q1)
LA1_ = Line(Ls1, As1)
KB1_ = Line(Ks1, Bs1)
testLines = [AB1_, PQ1_, MN1_, LK1_, LA1_, KB1_] #[LK1_, MN1_, PQ1_, AB1_, LA1_, KB1_]


# =============================================================================
# ============================== SHOW ORIGINAL IMAGE ==========================
# =============================================================================

fig.canvas.set_window_title('Original Image')
fig.canvas.mpl_connect('key_press_event', press)

#Theta = np.arange(0,180,1)
#Theta = np.arange(0,1,1)
nTraj = 41#61#7
nProj = 9#18#9
plotterTemplateImg = Plotter(templateImage)
(templateSinograma, templateDescriptor) = crossRadonTransform_analitic(templateImage, nTraj, nProj, templateLines) # 2,1
 
plotterTestImg = Plotter(testImage)
(testSinograma, testDescriptor) = crossRadonTransform_analitic(testImage, nTraj, nProj, testLines)


#templateRay = RayDescriptor(0, 0)
#templateRay.crossRatioVector = [1.3714285714285714, 1.149212233549583, 1.4290204295442641, 1.1684981684981686, 2.3125]
#templateRay.crossRatioVector =[1.421591804570528, 1.1000322476620445, 3.676767676767677, 1.0241541964866623, 2.5906071019473083, 1.101679389312977, 1.552466896426628, 1.412754485451334, 1.190162037037037, 1.1326650943396228, 1.7395348837209303, 1.5148005148005148, 1.2096681415929202, 1.0353811184136095, 1.4502258658189726, 1.9347826086956519, 1.3089247062461347, 1.00704720894817, 4.320862845115, 1.0916076249090187, 1.2074395924089176, 1.57446265853602, 1.2128607809847198, 1.255924978687127, 2.173862586232834]

#print("len(descriptor.rays) = ", len(descriptor.rays))

templateGreenRays = []
testGreenRays = []

templateRedRays = []
testRedRays = []

# ########## MATCHING ##########
countMatch = 0
totalComp  = 0
countCrossRatioVectorLengths = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
for templateRay in templateDescriptor.rays:
	print("TEMPLATE RAY")
	print("cross ratio signature: ", templateRay.crossRatioVector)
	print("edge points: ")
	for edgeP in templateRay.edgePoints:
		print(edgeP)
	for testRay in testDescriptor.rays:
		totalComp += 1
		#print("crossRatioVector = ", testDescriptor.rays[i].crossRatioVector)
		if templateRay.equals(testRay):
			#plotterTestImg.plotRay(testRay, 'g', 'go')
			if testRay.numberOfEdgePoints >= 4:
				testRay.estimateVanishPoints(templateRay)
			templateGreenRays.append(templateRay)
			testGreenRays.append(testRay)

			countMatch += 1
			#print("#### crossRatioVectors ####")
			#print("test CRV = ", testRay.crossRatioVector)
			#print("template CRV = ", templateRay.crossRatioVector)
			idxLen = len(testRay.crossRatioVector)
			countCrossRatioVectorLengths[idxLen] += 1

		else:
			templateRedRays.append(templateRay)
			testRedRays.append(testRay)
			#plotterTestImg.plotRay(testRay)

# PLOT
ax = fig.add_subplot(1,2,1)
ax.set_title('Template Image')
plt.imshow(templateImage)


plotterTemplateImg.plotLine(As, Ls, color="y")
plotterTemplateImg.plotLine(A, B, color="y")
plotterTemplateImg.plotLine(Ks, Bs, color="y")
plotterTemplateImg.plotLine(L, K, color="y")
plotterTemplateImg.plotLine(M, N, color="y")
plotterTemplateImg.plotLine(P, Q, color="y")
#plotterTemplateImg.plotLine(S, T, color="r")
#plotterTemplateImg.plotPoint(xi, yi, color='ro')

#for templateRay in templateRedRays:
#	plotterTemplateImg.plotRay(templateRay)
for templateRay in templateGreenRays:
	plotterTemplateImg.plotRay(templateRay, 'c', 'co')
	#if templateRay.edgePoints:
		#eP1 = templateRay.edgePoints[0]
		#(x1, y1) = eP1.toTuple()
		#plotterTemplateImg.plotPoint(x1, y1, color='yo')


ax = fig.add_subplot(1,2,2)
ax.set_title('Test Image')
plt.imshow(testImage)

plotterTestImg.plotLine(As1, Ls1, color="y")
plotterTestImg.plotLine(A1, B1, color="y")
plotterTestImg.plotLine(Ks1, Bs1, color="y")
plotterTestImg.plotLine(L1, K1, color="y")
plotterTestImg.plotLine(M1, N1, color="y")
plotterTestImg.plotLine(P1, Q1, color="y")

#for testRay in testRedRays:
#	plotterTestImg.plotRay(testRay)
for testRay in testGreenRays:
	#plotterTestImg.plotRay(testRay, 'c', 'co')
	if testRay.numberOfEdgePoints >= 4:
		plotterTestImg.plotRay(testRay, 'c', 'co')
		vP1 = testRay.vanishPointCandidate1
		#print("Vp1 = (%f, %f)" %testRay.vanishPointCandidate1.toTuple())
		##vP2 = testRay.vanishPointCandidate2
		#eP1 = testRay.edgePoints[0]
		#nextP2 = testRay.nextPoint2


		#plotterTestImg.plotPoint(x2, y2, color='g')

		#(xe1, ye1) = eP1.toTuple()
		#plotterTestImg.plotPoint(xe1, ye1, color='yo')

		(x1, y1) = vP1.toTuple()
		##(x2, y2) = vP2.toTuple()
		#print("xe1 = ", xe1)
		#if xe1 > 50:
		#	color = 'go'
		#else:
		#color = 'ro'
		plotterTestImg.plotPoint(x1, -y1, color='ro')
		##plotterTestImg.plotPoint(x2, y2, color='go')


		#(x2, y2) = nextP2.toTuple()
		#plotterTestImg.plotPoint(x2, y2, color='g', marker='+')

plt.show()

print("n. template rays: ", len(templateDescriptor.rays))
print("n. test rays: ", len(testDescriptor.rays))

print("#### STATISTICS ####")
print("Total comparation: ", totalComp)
print("Count Match: ", countMatch)
print("---------------------------")
for i in range(1, len(countCrossRatioVectorLengths)):
	countLen = countCrossRatioVectorLengths[i]
	if countLen > 0:
		print("CrossRatio vector with size %d, have %d Rays" %(i, countLen))
