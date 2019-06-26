# For image aquisition
import matplotlib.pyplot as plt
from tkinter import *
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk
import matplotlib.image as mpimg
from scipy import misc
import math
import numpy as np
import sys as sys
import copy

## Close window and change progress in code
def press(event):
	#print('press', event.key)
	if event.key == 'enter':
		plt.close()

def isWhite(pxVal):
	return pxVal > 0 

def isBorderPoint(s, cosT, sinT, u, du, halfWidth, halfHeight, image, firstPoint=False):
	(x0, y0) = calcPoint(s, cosT, sinT, u)
	(x1, y1) = calcPoint(s, cosT, sinT, u + du)
	xIdx0 = x0 + halfWidth
	yIdx0 = y0 + halfHeight

	xIdx1 = x1 + halfWidth
	yIdx1 = y1 + halfHeight
	pxVal0 = int(image[yIdx0][xIdx0][0])
	if 0 <= xIdx1 < 2*halfWidth and 0 <= yIdx1 < 2*halfHeight:
		if isWhite(pxVal0) and firstPoint:  #Se Primeiro ponto for branco
			return True
		else:
			pxVal1 = int(image[yIdx1][xIdx1][0]) # Pontos intermediarios havendo mudança de intesidade
			return (pxVal1 - pxVal0) != 0
	else:
		return isWhite(pxVal0) # Ultimo Ponto sendo branco

def calcPoint(s, cosT, sinT, u):	
	x = s*cosT - u*sinT
	y = s*sinT + u*cosT
	return(x, y)

def crossRadonTransform(image, nTraj, nProj):
	print("radonTransform")
	# Compute the number of rows and columns in the image
	(row_num, col_num, _) = image.shape
	# Initialize the transformed image
	diagonal = int(math.sqrt(row_num*row_num + col_num*col_num))
	tMax = nProj
	sMax = int(diagonal/2)
	ds = diagonal/(nTraj+1)
	
	shapeResult = (diagonal, nProj, 3)
	sinograma = np.zeros(shapeResult)#, dtype=np.uint8)
	
	(dims, dimt, _) = sinograma.shape
	print("size img = (%d, %d)" %(dims, dimt))

	width = col_num
	height = row_num
	dTheta = np.pi/nProj #3.14159265/nProj
	halfWidth = width/2
	halfHeight = height/2
	halfD = diagonal/2 # meia diagonal da imagem

	#Images indices: s, t, y, x
	du = 1.0

	# Compute transformed image
	for t in range(0, tMax):
		theta = t*dTheta
		cosT = math.cos(theta)
		sinT = math.sin(theta)
		for s in np.arange(-sMax, sMax + ds, ds):
			s = int(s)
			countR = sinograma[s][t][0]
			countG = sinograma[s][t][1]
			countB = sinograma[s][t][2]

			X = []
			Y = []
			firstPoint = True
			#print('s, t = %f, %f' %(s, t))
			for u in np.arange(-halfD, halfD+du, du):
				# Compute the point p in the original image which is mapped to the
				# (row, col) pixel in the transformed image
				
				(x, y) = calcPoint(s, cosT, sinT, u)
				
				xIdx = x + halfWidth
				yIdx = y + halfHeight

				if 0 <= xIdx < width and 0 <= yIdx < height:
					countR = countR + int(image[yIdx][xIdx][0])
					countG = countG + int(image[yIdx][xIdx][1])
					countB = countB + int(image[yIdx][xIdx][2])
					if isBorderPoint(s, cosT, sinT, u, du, halfWidth, halfHeight, image, firstPoint):
						X.append(xIdx)
						Y.append(yIdx)
						plt.plot([xIdx], [yIdx], 'ro')
					firstPoint = False

			plt.plot(X, Y, 'g')
			sinograma[s][t][0] = countR
			sinograma[s][t][1] = countG
			sinograma[s][t][2] = countB

	#row_sums = sinograma.sum(axis=1)
	#new_matrix = sinograma / row_sums[:, np.newaxis]

	return sinograma

def makeDescriptor(image, nTraj, nProj):
	# Compute the number of rows and columns in the image
	(row_num, col_num, _) = image.shape
	# Initialize the transformed image
	diagonal = int(math.sqrt(row_num*row_num + col_num*col_num))
	tMax = nProj
	sMax = int(diagonal/2)
	ds = diagonal/(nTraj+1)
	
	shapeResult = (diagonal, nProj, 3)
	sinograma = np.zeros(shapeResult)#, dtype=np.uint8)

	width = col_num
	height = row_num
	dTheta = np.pi/nProj #3.14159265/nProj
	halfWidth = width/2
	halfHeight = height/2
	halfD = diagonal/2 # meia diagonal da imagem

	#Images indices: s, t, y, x
	du = 1.0

	# Compute transformed image
	for t in range(0, tMax):
		theta = t*dTheta
		cosT = math.cos(theta)
		sinT = math.sin(theta)
		for s in np.arange(-sMax, sMax + ds, ds):
			s = int(s)
			countR = sinograma[s][t][0]
			countG = sinograma[s][t][1]
			countB = sinograma[s][t][2]

			X = []
			Y = []
			firstPoint = True
			#print('s, t = %f, %f' %(s, t))
			for u in np.arange(-halfD, halfD+du, du):
				# Compute the point p in the original image which is mapped to the
				# (row, col) pixel in the transformed image
				
				(x, y) = calcPoint(s, cosT, sinT, u)
				
				xIdx = x + halfWidth
				yIdx = y + halfHeight

				if 0 <= xIdx < width and 0 <= yIdx < height:
					countR = countR + int(image[yIdx][xIdx][0])
					countG = countG + int(image[yIdx][xIdx][1])
					countB = countB + int(image[yIdx][xIdx][2])
					if isBorderPoint(s, cosT, sinT, u, du, halfWidth, halfHeight, image, firstPoint):
						X.append(xIdx)
						Y.append(yIdx)
						plt.plot([xIdx], [yIdx], 'ro')
					firstPoint = False

			plt.plot(X, Y, 'g')
			sinograma[s][t][0] = countR
			sinograma[s][t][1] = countG
			sinograma[s][t][2] = countB

	#row_sums = sinograma.sum(axis=1)
	#new_matrix = sinograma / row_sums[:, np.newaxis]

	return sinograma


# =============================================================================
# ============================== LOAD IMAGE ===================================
# =============================================================================

filename = askopenfilename(filetypes=[("all files","*"),("Bitmap Files","*.bmp; *.dib"),
										("JPEG", "*.jpg; *.jpe; *.jpeg; *.jfif"),
										("PNG", "*.png"), ("TIFF", "*.tiff; *.tif")])

image = misc.imread(filename, mode = 'RGB')

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('Radon Scanner')

(row_num, col_num, _) = image.shape

# =============================================================================
# ============================== SHOW ORIGINAL IMAGE ==========================
# =============================================================================

fig.canvas.set_window_title('Original Image')
fig.canvas.mpl_connect('key_press_event', press)

#Theta = np.arange(0,180,1)
#Theta = np.arange(0,1,1)
nTraj = 61
nProj = 1
sinograma = crossRadonTransform(image, nTraj, nProj)

plt.imshow(image)
plt.show()
#sinograma = normalizeImg(sinograma)
