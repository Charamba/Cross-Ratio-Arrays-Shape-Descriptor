import matplotlib.pyplot as plt
from tkinter import *
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk
import matplotlib.image as mpimg
from scipy import misc
import math
import numpy as np
import sys as sys
from point import P2_Point
from point import R2_Point
import copy


def normalizeImg(image):
	(row_num, col_num, _) = image.shape
	maxVal = 0
	minValNonZero = sys.maxsize
	for j in range(0, col_num):
		for i in range(0, row_num):
			pxValue = image[i][j][0]
			if maxVal < pxValue:
				maxVal = pxValue
			if minValNonZero > pxValue > 0:
				minValNonZero = pxValue

	minValNonZero = 0
	for j in range(0, col_num):
		for i in range(0, row_num):
			pxValue = image[i][j][0]
			if pxValue > 0:
				image[i][j][0] = int((pxValue-minValNonZero)*255/(maxVal-minValNonZero))
				image[i][j][1] = int((pxValue-minValNonZero)*255/(maxVal-minValNonZero))
				image[i][j][2] = int((pxValue-minValNonZero)*255/(maxVal-minValNonZero))
			#print("pxValnorm = ", image[i][j][0])

	return image

def rotation(vs, theta):
	xs = vs.x
	ys = vs.y

	cosT = math.cos(theta)
	sinT = math.sin(theta)

	xu = xs*cosT - ys*sinT
	yu = xs*sinT + ys*cosT
	vu = R2_Point(xu, yu)
	return vu

def makeP2Line(P0, v, angle):
	v.r2Normalize
	v_angle = rotation(v, angle)
	(xv, yv) = v_angle.toTuple()
	(x0, y0, _) = P0.toTuple()
	p2Line = P2_Point(yv, -xv, -yv*x0 + xv*y0)
	return p2Line

filename = askopenfilename(filetypes=[("all files","*"),("Bitmap Files","*.bmp; *.dib"),
										("JPEG", "*.jpg; *.jpe; *.jpeg; *.jfif"),
										("PNG", "*.png"), ("TIFF", "*.tiff; *.tif")])

image = misc.imread(filename, mode = 'RGB')
#plt.imshow(image)
(ySize, xSize, _) = image.shape

# Linha do horizonte
(x0, y0) = (0, 38)
(xf, yf) = (785, 38)
#plt.plot([x0, xf], [y0, yf])

# origem (0,0)
p00 = R2_Point(0,0)
p00.to_img_coord(xSize, ySize)
#plt.plot([p00.x], [p00.y], 'x')
p00 = p00.toP2_Point()

# Pontos de fuga
pfb = R2_Point(785, 38)#(665,38) #(785, 38)
pfb.to_cartesian_coord(xSize, ySize)
PFn = copy.deepcopy(pfb)
pfb = pfb.toP2_Point()
rfb = p00.cross(pfb)

pfa = R2_Point(0,38)#(194,38) #(0,38)
pfa.to_cartesian_coord(xSize, ySize)
PF0 = copy.deepcopy(pfa)
pfa = pfa.toP2_Point()
rfa = p00.cross(pfa)

vh = PFn - PF0
vh.r2Normalize()

p = rfa.cross(rfb)
p.normalize()
#plt.plot([p.x], [p.y], 'ro')

tMax = 180
sMax = 160

dTheta = (PFn.euclideanDistance(PF0))/(tMax)
ds = np.pi/sMax #3.14159265/sMax

(xb0, yb0) = (300,104)
(xbf, ybf) = (527,241)

shapeResult = (sMax, tMax, 3)
sinograma = np.zeros(shapeResult)

for t in range(0, tMax):
	theta = t*dTheta
	PFt = PF0 + theta*vh
	#PFt.to_img_coord(xSize, ySize)
	#plt.plot([PFt.x], [PFt.y], 'x')

	for s in range(1, sMax):
		countR = sinograma[s][t][0]
		countG = sinograma[s][t][1]
		countB = sinograma[s][t][2]

		angle_s = s*ds

		# calculando reta de fuga
		Rts = makeP2Line(PFt.toP2_Point(), vh, angle_s)
		Rts.normalize()

		bottomSideLine = P2_Point(0, 1, ySize/2) # linha inferior da imagem
		bottomSideLine.normalize()
		pbound = Rts.cross(bottomSideLine) 
		pboundImg = pbound.toR2_Point()
		pboundImg.to_img_coord(xSize, ySize)
		PFImg = copy.deepcopy(PFt)
		PFImg.to_img_coord(xSize, ySize)
		#plt.plot([PFImg.x, pboundImg.x], [PFImg.y, pboundImg.y], 'r--')

		# 
		raySize = PFt.euclideanDistance(pbound)

		nu = 100
		du = raySize/nu

		vu = pboundImg - PFImg
		vu.r2Normalize()
		for u in range(0, nu):
			Pxy = PFImg + du*u*vu

			#Pxy.to_img_coord(width, height)
			xIdx = Pxy.x
			yIdx = Pxy.y
			#print("img: x,y = %f,%f" %(Pxy.x, Pxy.y))
			if xb0 <= xIdx < xbf and yb0 <= yIdx < ybf:
			#if 0 <= xIdx < xSize and 0 <= yIdx < ySize:
				#plt.plot([xIdx], [yIdx], 'r.')
				countR = countR + image[yIdx][xIdx][0]
				countG = countG + image[yIdx][xIdx][1]
				countB = countB + image[yIdx][xIdx][2]

		sinograma[s][t][0] = countR
		sinograma[s][t][1] = countG
		sinograma[s][t][2] = countB


sinograma = normalizeImg(sinograma)
plt.imshow(image)
#plt.imshow(sinograma)

		#for x in range(xb0, xbf):
		#	for y in range(yb0, ybf):
		#		Pxy = R2_Point(x, y)
		#		Pxy.to_cartesian_coord(xSize, ySize)



plt.show()



