from Geometry import *
from Point import *
from Plotter import *
from Utils import *
import math
from hull_contour import *

from scipy.spatial import ConvexHull

class Image(Plotter):
	def __init__(self, matplotlib_image):
		(row_num, col_num, _) = matplotlib_image.shape
		Plotter.__init__(self, row_num, col_num)
		self.image = matplotlib_image

	def getShape(self):
		(height, width, _) = self.image.shape
		return (height, width)

	# def calcImageBorderPoints(s, cosT, sinT, xLimit, yLimit):
	# 	return calcBorderPoints(s, cosT, sinT, -xLimit, xLimit-1, -yLimit, yLimit-1)

	def calcBandPixels(self, P0, Pf):
		lamb = P0.euclideanDistance(Pf)

		(x0_px, y0_px) = P0.toTuple()
		(xf_px, yf_px) = Pf.toTuple()
		(x0, y0) = self.toDefaultCoordinate(x0_px, y0_px)
		(xf, yf) = self.toDefaultCoordinate(xf_px, yf_px)

		sinT = calc_sinT(x0_px, xf_px, lamb)
		cosT = calc_cosT(y0_px, yf_px, lamb)
		s = calc_s(xf, yf, cosT, sinT)

		(visitedPixels, borderPoints) = self.calcPixelBorderPoints(P0, Pf, s, cosT, sinT, bandPixels=True)

		return visitedPixels


	def calcPixelBorderPoints(self, P0, Pf, s=None, cosT=None, sinT=None, bandPixels=False):
		(height, width, _) = self.image.shape
		halfHeight = height/2#(math.floor(height/2))
		halfWidth = width/2#(math.floor(width/2))

		(x0, y0) = P0.toTuple()
		(xf, yf) = Pf.toTuple()

		# l = None
		# if s==None and cosT==None and sinT==None:
		# 	P0h = P0.toP2_Point()
		# 	Pfh = Pf.toP2_Point()
		# 	l = P0h.cross(Pfh)
		# 	l.normalize()

		(x0Pixel, y0Pixel) = (math.floor(x0-halfWidth), math.floor(y0-halfHeight))
		(xfPixel, yfPixel) = (math.floor(xf-halfWidth), math.floor(yf-halfHeight))

		l = None
		flag =  False
		if s==None and cosT==None and sinT==None:
			P0h = P2_Point(x0-halfWidth, y0-halfHeight, 1)
			Pfh = P2_Point(xf-halfWidth, yf-halfHeight, 1)
			l = P0h.cross(Pfh)
			if l.z == 0:
				return ([], [])
			l.normalize()
			flag = True



		borderPoints = []
		visitedPixels = []
		if cosT != 0 or flag:
			# horizontal
			yMin = min(y0Pixel, yfPixel)
			yMax = max(y0Pixel, yfPixel)
			for y in range(yMin, yMax+1):
				x = None
				if l == None:
					x = calcXbyY(s, cosT, sinT, y)
				else:
					ph1 = P2_Point(-halfWidth, y, 1)
					ph2 = P2_Point(halfWidth, y, 1)

					rHorizontal = ph1.cross(ph2)
					p = l.cross(rHorizontal)
					if p.z != 0:
						p = p.toR2_Point()
						(x, y) = p.toTuple()
					else:
						x = halfWidth + 1 # gambi
				if -halfWidth <= x <= halfWidth:
					borderPoints.append(R2_Point(x, y))
					visitedPixels.append(R2_Point(math.floor(x+halfWidth), math.floor(y+halfHeight)))
					# if bandPixels:
					# 	visitedPixels.append(R2_Point(math.floor(x+halfWidth+1), math.floor(y+halfHeight)))
					# 	visitedPixels.append(R2_Point(math.floor(x+halfWidth-1), math.floor(y+halfHeight)))
					
		if sinT != 0 or flag:
			# vertical
			xMin = min(x0Pixel, xfPixel)
			xMax = max(x0Pixel, xfPixel)
			for x in range(xMin, xMax+1):
				y = None
				if l == None:
					y = calcYbyX(s, cosT, sinT, x)
				else:
					pv1 = P2_Point(x, -halfHeight, 1)
					pv2 = P2_Point(x, halfHeight, 1)

					rVertical = pv1.cross(pv2)
					p = l.cross(rVertical)

					if p.z != 0:
						p = p.toR2_Point()
						(x, y) = p.toTuple() # (_, y)
					else:
						y = halfHeight + 1 # gambi
				if -halfHeight <= y <= halfHeight:
					borderPoints.append(R2_Point(x, y))
					visitedPixels.append(R2_Point(math.floor(x+halfWidth), math.floor(y+halfHeight)))
					# if bandPixels:
					# 	visitedPixels.append(R2_Point(math.floor(x+halfWidth), math.floor(y+halfHeight+1)))
					# 	visitedPixels.append(R2_Point(math.floor(x+halfWidth), math.floor(y+halfHeight-1)))

					

		# if sinT == 0:
		# 	borderPoints = sorted(borderPoints, key=lambda point: point.y)
		# else:
		# 	borderPoints = sorted(borderPoints, key=lambda point: point.x)
		
		return (visitedPixels, borderPoints)

	def getHightFrequencePoint(self, borderPoints, s, cosT, sinT):
		(height, width, _) = self.image.shape
		halfHeight = height/2
		halfWidth = width/2

		visitedWhitePixels = []

		edgePoints = []

		if len(borderPoints) >= 2:
			P0 = borderPoints[0]
			Pf = borderPoints[-1]

			# ADD FIRST POINT
			# (x0, y0) = P0.toTuple()
			# addP0_flag = False
			# if (-halfHeight < y0 < halfHeight) and (-halfWidth < x0 < halfWidth):
			# 	if self.image[int(y0+halfHeight)][int(x0+halfWidth)][0] > 0:
			# 		addP0_flag = True
			# elif (-halfHeight < y0-1 < halfHeight) and (-halfWidth < x0-1 < halfWidth):
			# 	if self.image[int(y0-1+halfHeight)][int(x0-1+halfWidth)][0] > 0:
			# 		addP0_flag = True
			# elif (-halfHeight < y0-1 < halfHeight) and (-halfWidth < x0 < halfWidth):
			# 	if self.image[int(y0-1+halfHeight)][int(x0+halfWidth)][0] > 0:
			# 		addP0_flag = True
			# elif (-halfHeight < y0 < halfHeight) and (-halfWidth < x0-1 < halfWidth):
			# 	if self.image[int(y0+halfHeight)][int(x0-1+halfWidth)][0] > 0:
			# 		addP0_flag = True

			# if addP0_flag:
			# 	edgePoints.append(P0) #(R2_Point(xf-halfWidth, yf-halfHeight))

			lastPointIndx = len(borderPoints)
			for i, point in enumerate(borderPoints):
				if i+1 < len(borderPoints):
					P = borderPoints[i]
					Pnext = borderPoints[i+1]
					vDir = Pnext - P
					(x, y) = point.toTuple()
					#u = calcUbyXY(s, cosT, sinT, x, y)
					#(nextX, nextY)= calcPoint(s, cosT, sinT, u + 0.01)
					#(prevX, prevY) = calcPoint(s, cosT, sinT, u - 0.01)

					nextPoint = point + 0.001*vDir
					prevPoint = point - 0.001*vDir
					(nextX, nextY) = nextPoint.toTuple()
					(prevX, prevY) = prevPoint.toTuple()

					if int(nextX+halfWidth) < width and int(prevX+halfWidth) < width and int(nextY+halfHeight) < height and int(prevY+halfHeight) < height:
						
						nextPixelValue = int(self.image[int(nextY+halfHeight)][int(nextX+halfWidth)][0])
						prevPixelValue = int(self.image[int(prevY+halfHeight)][int(prevX+halfWidth)][0])
						
						pixelValue = int(self.image[int(y+halfHeight)][int(x+halfWidth)][0])

						if pixelValue > 0:
							visitedWhitePixels.append((int(x+halfHeight), int(y+halfWidth)))

						if nextPixelValue != prevPixelValue:# or (pixelValue > 0 and (i == 0 or i == lastPointIndx-1)):
							edgePoints.append(point)
					# elif self.image[int(y+halfHeight)][int(x+halfWidth)][0] > 0: # extreme white points
					# 	edgePoints.append(point)

			# ADD LAST POINT
			# (xf, yf) = Pf.toTuple()
			# addPf_flag = False
			# if (-halfHeight < yf < halfHeight) and (-halfWidth < xf < halfWidth):
			# 	if self.image[int(yf+halfHeight)][int(xf+halfWidth)][0] > 0:
			# 		addPf_flag = True
			# elif (-halfHeight < yf-1 < halfHeight) and (-halfWidth < xf-1 < halfWidth):
			# 	if self.image[int(yf-1+halfHeight)][int(xf-1+halfWidth)][0] > 0:
			# 		addPf_flag = True
			# elif (-halfHeight < yf-1 < halfHeight) and (-halfWidth < xf < halfWidth):
			# 	if self.image[int(yf-1+halfHeight)][int(xf+halfWidth)][0] > 0:
			# 		addPf_flag = True
			# elif (-halfHeight < yf < halfHeight) and (-halfWidth < xf-1 < halfWidth):
			# 	if self.image[int(yf+halfHeight)][int(xf-1+halfWidth)][0] > 0:
			# 		addPf_flag = True

			# if addPf_flag:
			# 	edgePoints.append(Pf) #(R2_Point(xf-halfWidth, yf-halfHeight))


		return edgePoints, list(set(visitedWhitePixels))

	def calc_edgePoints(self, P0, Pf, s=None, theta=None, showTrajectories=False, FULL_POINTS=False):
		#print("P0 = ", P0)
		visitedWhitePixels = []
		lamb = P0.euclideanDistance(Pf)

		(x0_px, y0_px) = P0.toTuple()
		(xf_px, yf_px) = Pf.toTuple()
		(x0, y0) = self.toDefaultCoordinate(x0_px, y0_px)
		(xf, yf) = self.toDefaultCoordinate(xf_px, yf_px)

		cosT = None
		sinT = None
		# if s is None:
		# 	sinT = calc_sinT(x0_px, xf_px, lamb)  #(L_x - A_x)/lamb
		# 	cosT = calc_cosT(y0_px, yf_px, lamb)  # - (L_y - A_y)/lamb
		# 	s = calc_s(xf, yf, cosT, sinT)
		# else:
		if theta != None:
			cosT = math.cos(theta)
			sinT = math.sin(theta)
		#edgePoints = newBresenham(x0, y0, xf, yf, image)  #bresenham_line_original(x0, y0, xf, yf, image) #bresenham_line(s, cosT, sinT, x0, y0, xf, yf, image)
		
		eCorrectPoints = []

		(visitedPixels, borderPoints) = self.calcPixelBorderPoints(P0, Pf, s, cosT, sinT)

		borderPoints = removeDuplicates(borderPoints)
		visitedPixels = removeDuplicates(visitedPixels)

		#=============== DEBUG VISITED PIXELS ======================#
		# (x0_, y0_) = P0.toTuple()
		# self.plotPoint(x0_, y0_, color='yo', correction=False)
		# (xf_, yf_) = Pf.toTuple()
		# self.plotPoint(xf_, yf_, color='yo', correction=False)
		#self.plotLinePoints([borderPoints[0], borderPoints[-1]], color='go', correction=True)
		#Pf = P0 + 100*(Pf - P0)
		if showTrajectories:
			self.plotLinePoints([P0, Pf], color='g--', correction=False) #<--- Trajectories
		# self.plotLinePoints(borderPoints, color='co')
		#self.plotVisitedPixels(visitedPixels, 10)

		#edgePoints = borderPoints

		#borderPoints = [Pi for (Pi, d) in sorted([(Pi, P0_px.euclideanDistance(Pi)) for Pi in borderPoints], key=lambda t: t[1])]

		#edgePoints = self.getHightFrequencePoint(borderPoints, s, cosT, sinT)
		#edgePoints = borderPoints

		# SORT by Distances
		#points_distances = [(Pi, P0.euclideanDistance(Pi)) for Pi in edgePoints]
		#print("points_distances = ", points_distances)

		#print("P0 = ", P0)
		#print("Pf = ", Pf)
		P0_px = R2_Point(x0, y0)
		Pf_px = R2_Point(xf, yf)
		#print("edgePoints = ", edgePoints)

		# SORT borderPoints by distance to P0
		#borderPoints = [Pi for (Pi, d) in sorted([(Pi, P0_px.euclideanDistance(Pi)) for Pi in borderPoints], key=lambda t: t[1])]
		borderPoints = sortPoints(borderPoints, P0_px)
		#distances = [d for (Pi, d) in sorted([(Pi, P0_px.euclideanDistance(Pi)) for Pi in edgePoints], key=lambda t: t[1])]
		
		(edgePoints, visitedWhitePixels) = self.getHightFrequencePoint(borderPoints, s, cosT, sinT)
		#edgePoints = sortPoints(edgePoints, P0_px)


		#print("distances = ", distances)
		(height, width, _) = self.image.shape
		halfHeight = height/2.0
		halfWidth = width/2.0

		# pxVal0 = self.image[math.floor(x0+halfHeight)][math.floor(y0+halfWidth)][0]
		# print("math.floor(xf+halfHeight) = ", math.floor(xf+halfHeight))
		# print("math.floor(yf+halfHeight) = ", math.floor(yf+halfHeight))
		# pxValf = int(self.image[math.floor(xf+halfHeight)][math.floor(yf+halfWidth)][0])


		pxVal0 = 0
		pxValf = 0
		if 0 <= x0_px <= height and 0 <= y0_px <= width:
			if int(x0_px) == width:
				x0_px = x0_px - 1
			if int(y0_px) == height :
				y0_px = y0_px - 1
			pxVal0 = self.image[math.floor(y0_px)][math.floor(x0_px)][0]

		if 0 <= xf_px <= width and 0 <= yf_px <= height:
			# print("ANTES")
			# print("xf_px = ", xf_px)
			# print("yf_px = ", yf_px)
			# print("width = ", width)
			# print("height = ", height)
			if int(xf_px) == width:
				xf_px = xf_px - 1.0
			if int(yf_px) == height:
				yf_px = yf_px - 1.0
			# print("DEPOIS")
			# print("xf_px = ", xf_px)
			# print("yf_px = ", yf_px)
			pxValf = self.image[math.floor(yf_px)][math.floor(xf_px)][0]
			if pxValf > 0:
				countWhitePixels = 0

		# Extreme points
		if pxVal0 > 0 or FULL_POINTS:
			edgePoints = [P0_px] + edgePoints
		if pxValf > 0 or FULL_POINTS:
			edgePoints = edgePoints + [Pf_px]

		return (visitedWhitePixels, edgePoints, s)

	def getWhitePoints(self):
		whitePoints = []

		(height, width) = self.getShape()
		halfHeight = int(math.floor(height/2))
		halfWidth = int(math.floor(width/2))

		for x in range(width):
			for y in range(height):
				pxVal = self.image[y][x][0]
				if pxVal > 0:
					xCoord = int(x - halfWidth)
					yCoord = int(y - halfHeight)
					whitePoints.append(R2_Point(xCoord,yCoord))
					whitePoints.append(R2_Point(xCoord+1,yCoord))
					whitePoints.append(R2_Point(xCoord,yCoord+1))
					whitePoints.append(R2_Point(xCoord+1,yCoord+1))
		return whitePoints

	def points_bouding_box_area(self, curve_points):
		(x_min, y_min) = float('Inf'), float('Inf')
		(x_max, y_max) = float('-Inf'), float('-Inf')

		for (x, y) in curve_points:
			if x <= x_min:
				x_min = x
			
			if x >= x_max:
				x_max = x

			if y <= y_min:
				y_min = y
			
			if y >= y_max:
				y_max = y

		dx = abs(x_max - x_min)
		dy = abs(y_max - y_min)
		return dx*dy

	def contour_hull(self, nPoints=10):
		curves = curves_from_rasterized_img(self.image)
		vertices = []

		(height, width) = self.getShape()
		halfHeight = int(math.floor(height/2))
		halfWidth = int(math.floor(width/2))

		external_curve = []
		area_max = 0
		for curve in curves:
			area = self.points_bouding_box_area(curve)
			if area > area_max:
				area_max = area
				external_curve = curve

		#for external_curve in curves:
		N = len(external_curve)
		step = int(N/nPoints)
		if step == 0:
			step = 1
		for i,(x, y) in enumerate(external_curve):
			if i % step == 0:
				xCoord = int(x - halfWidth)
				yCoord = int(y - halfHeight)
				vertices.append(R2_Point(xCoord, yCoord))
		vertices = [vertices[-1]] + vertices
		return vertices

	def convex_hull(self, interpoints_sample=0):
		whitePoints = self.getWhitePoints()
		vertices = convex_hull_vertices(whitePoints, interpoints_sample=interpoints_sample)
		# wPListofList = []
		# for wP in whitePoints:
		# 	(x, y) = wP.toTuple()
		# 	wPListofList.append([x, y])

		# hull = ConvexHull(wPListofList)
		# vertices = []

		# for vIndex in hull.vertices:
		# 	[x, y] = wPListofList[vIndex]
		# 	vertices.append(R2_Point(x, y))
		return vertices

	def convex_hull_add_points(self, additive_points_number=0):
		whitePoints = self.getWhitePoints()
		vertices = convex_hull_vertices_add_points(whitePoints, additive_points_number=additive_points_number)

		return vertices

	def convexHullPixels(self, vertices):
		visitedPixels = []
		for i in range(0, len(vertices)):
			if i+1 != len(vertices):
				P0 = vertices[i]
				Pf = vertices[i+1]
				visitedPixels = visitedPixels + self.calcBandPixels(P0, Pf)
		P0 = vertices[-1]
		Pf = vertices[0]
		visitedPixels = visitedPixels + self.calcBandPixels(P0, Pf)
		return visitedPixels
