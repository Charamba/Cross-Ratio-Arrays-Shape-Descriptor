# For image aquisition
import matplotlib.pyplot as plt
from tkinter import *
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk
import matplotlib.image as mpimg
import matplotlib.patches
from matplotlib.patches import Circle, Wedge, Polygon, Rectangle
from matplotlib.collections import PatchCollection

from ShapeDescriptor import *

class Plotter:
	def __init__(self, row_num, col_num):
		self.width = col_num
		self.height = row_num
		self.buffer = []
		# --------------
		self.patches = []
		self.patchColors = []
		# --------------
		self.words = []


	def toDefaultCoordinate(self, x, y):
		(halfW, halfH) = (self.width/2.0, self.height/2.0)
		return (x - halfW, y - halfH)

	def toImageCoordinate(self, x, y):
		(halfW, halfH) = (self.width/2.0, self.height/2.0)
		return (x + halfW, y + halfH)

	def plotText(self, x, y, text, fontsize=12):
		self.words.append((x, y, text, fontsize))

	def plotRay(self, ray, lineColor='r', edgePointColor='ro', writeOrder=False):
		(X, Y) = ray.getEdgePixelCoordinates(self.width, self.height)
		#plt.plot(X, Y, lineColor)
		self.buffer.append((X, Y, lineColor))
		if edgePointColor != None:
			self.buffer.append((X, Y, edgePointColor))

		if writeOrder:
			for i, (x, y) in enumerate(zip(X, Y)):
				self.plotText(x, y, str(i))
			#plt.plot(X, Y, edgePointColor)
	def plotVanishRay(self, ray, lineColor='r--', edgePointColor=None):
		halfW = self.width/2.0
		halfH = self.height/2.0
		(X, Y) = ray.getEdgePixelCoordinates(self.width, self.height)
		#plt.plot(X, Y, lineColor)
		(xv, yv) = ray.getVanishPoint().toTuple()
		X.append(xv + halfW)
		Y.append(yv + halfH)
		self.buffer.append((X, Y, lineColor))
		if edgePointColor != None:
			self.buffer.append((X, Y, edgePointColor))
	def plotPoint(self, x, y, color='bo', correction=True):
		if correction:
			halfW = self.width/2.0
			halfH = self.height/2.0
		else:
			halfW = 0
			halfH = 0
		#plt.plot([x + halfW], [y + halfH], color)
		self.buffer.append(([x + halfW], [y + halfH], color))
		#plt.scatter(x + halfW, y + halfH, c=color, marker="o")
	def plotLine(self, P0, Pf, color="w", correction=True):
		halfW = self.width/2.0
		halfH = self.height/2.0
		(x0, y0) = P0.toTuple()
		(xf, yf) = Pf.toTuple()
		if correction:
			yf = -yf
			y0 = -y0
			x0 += halfW
			xf += halfW
			y0 += halfH
			yf += halfH
		self.buffer.append(([x0, xf], [y0, yf], color))
		#plt.plot([x0, xf], [y0, yf], color)
	def plotLinePoints(self, points, color='b', correction=True, ciclic=False, writeOrder=False):
		X = []
		Y = []
		halfWidth = 0
		halfHeight = 0
		if correction:
			halfWidth = self.width/2.0
			halfHeight = self.height/2.0
		for i, point in enumerate(points):
			X.append(point.x + halfWidth)
			Y.append(point.y + halfHeight)
			if writeOrder:
				self.plotText(point.x + halfWidth, point.y + halfHeight, str(i))
		#plt.plot(X, Y, color)

		if ciclic:
			X.append(points[0].x + halfWidth)
			Y.append(points[0].y + halfHeight)

		self.buffer.append((X, Y, color))
	def plotPixelGrid(self, color='m'):
		wMax = self.width
		hMax = self.height

		heights = range(0, hMax+1)
		widths  = range(0, wMax+1)

		for hi in heights:
			#plt.plot([0, wMax+1], [hi, hi], color)
			self.buffer.append(([0, wMax+1], [hi, hi], color))

		for wi in widths:
			#plt.plot([wi, wi], [0, hMax+1], color)
			self.buffer.append(([wi, wi], [0, hMax+1], color))

	def plotCircle(self, x, y, r, color):
		halfW = self.width/2
		halfH = self.height/2
		circle = Circle((x+halfW, y+halfH), r)
		self.patches.append(circle)
		self.patchColors.append(color)
	def plotHexagon(self, x, y, r, color):
		halfW = self.width/2
		halfH = self.height/2
		hexagon = matplotlib.patches.RegularPolygon((x+halfW, y+halfH), 6, r)
		self.patches.append(hexagon)
		self.patchColors.append(color)
	def plotRectangle(self, x, y, color, correction=True):
		halfW = 0
		halfH = 0
		if correction:
			halfW = self.width/2
			halfH = self.height/2
		rect = Rectangle((x+halfW, y+halfH), 1, 1)
		self.patches.append(rect)
		self.patchColors.append(color)
	def plotVisitedPixels(self, visitedPixels, color, correction=False):
		#(X, Y) = ray.getEdgePixelCoordinates(self.width, self.height)
		#for (x,y) in zip(X, Y):
		for vPixel in visitedPixels:
			(x,y) = vPixel.toTuple()
			self.plotRectangle(math.floor(x), math.floor(y), color, correction=correction)
	def showPatches(self, fig, ax):
		p = PatchCollection(self.patches, alpha=0.4)
		p.set_array(np.array(self.patchColors))
		ax.add_collection(p)
		#fig.colorbar(p, ax=ax)
	def showText(self):
		for (x, y, text, fontsize_) in self.words:
			plt.text(x, y, text, fontsize=fontsize_, color='green')
	def show(self):
		self.showText()
		for (X, Y, color) in self.buffer:
			plt.plot(X, Y, color)

