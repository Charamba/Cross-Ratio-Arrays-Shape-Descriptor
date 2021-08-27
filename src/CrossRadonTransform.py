#!-*- conding: utf8 -*-
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

from Point import *
from ShapeDescriptor import *

from bresenham import bresenham


u_step = 0.5#0.05


# def isEdgePoint_(oldX, oldY, currentX, currentY, image, firstPoint=False):
# 	xIdx0 = oldX
# 	yIdx0 = oldY
# 	xIdx1 = currentX
# 	yIdx1 = currentY

# 	(row_num, col_num, _) = image.shape
# 	width = col_num
# 	height = row_num
# 	halfWidth = int(width/2)
# 	halfHeight = int(height/2)

# 	pxVal0 = int(image[yIdx0][xIdx0][0])
# 	if 0 <= xIdx1 < 2*halfWidth and 0 <= yIdx1 < 2*halfHeight:
# 		if isWhite(pxVal0) and firstPoint:  #Se Primeiro ponto for branco
# 			return True
# 		else:
# 			# pxVal1 = int(image[yIdx1][xIdx1][0])
# 			pxVal1 = int(image[yIdx1][xIdx1][0]) # Pontos intermediarios havendo mudanca de intesidade
# 			return (pxVal1 - pxVal0) != 0
# 	else:
# 		return isWhite(pxVal0) # Ultimo Ponto sendo branco

def hasChangedPixelValue(i, longest, oldVal, pxVal):
	#return True
	return ((i == 0 or i == longest) and isWhite(pxVal)) or (pxVal != oldVal)

def isDiagonal(xOld, yOld, x, y):
	return not (xOld == x or yOld == y)

def choiceEdgePoint(borderPoints, xOld, yOld):
	if borderPoints:
		oldPixelPoint = R2_Point(xOld, yOld)
		if len(borderPoints) == 2:
			bp1 = borderPoints[0]
			bp2 = borderPoints[1]
			d1 = oldPixelPoint.euclideanDistance(bp1)
			d2 = oldPixelPoint.euclideanDistance(bp2)
			if d1 <= d2:
				return bp2
			else:
				return bp1
		elif len(borderPoints) == 1:
			return borderPoints[0]
		else:
			return oldPixelPoint


def bresenham_line_deprecated(s, cosT, sinT, x, y, x2, y2, image):
	(row_num, col_num, _) = image.shape
	width = col_num
	height = row_num
	halfWidth = width/2
	halfHeight = height/2	

	x =  int(x + halfWidth)
	y =  int(y + halfHeight)
	x2 = int(x2 + halfWidth)
	y2 = int(y2 + halfHeight)
	xOld = x
	yOld = y


	w = x2 - x
	h = y2 - y
	dx1 = 0
	dx2 = 0
	dy1 = 0
	if w<0: dx1 = -1 
	elif w>2: dx1 = 1
	if h<0: dy1 = -1 
	elif h>0: dy1 = 1
	if w<0: dx2 = -1 
	elif w>0: dx2 = 1
	dy2 = 0
	longest  = abs(w)
	shortest = abs(h)

	if longest<=shortest:
		longest = abs(h)
		shortest = abs(w)
		if h<0:
			dy2 = -1
		elif h>0:
			dy2 = 1
		dx2 = 0

	numerator = longest >> 1
	edgePoints = []
	oldVal = 0
	#return [R2_Point(x - halfWidth, y - halfHeight), R2_Point(x2 - halfWidth, y2 - halfHeight)]
	for i in range (0, longest+1): 
		pxVal = int(image[y][x][0])
		
		#if isWhite(pxVal):

		if hasChangedPixelValue(i, longest, oldVal, pxVal):
			if x != xOld and y != yOld:
				xFloat = float(x - halfWidth)
				yFloat = calcYbyX(s, cosT, sinT, xFloat)

				xInf = min(x, xOld)
				xSup = max(x, xOld)
				yInf = min(y, yOld)
				ySup = max(y, yOld)
				if isValidBorderPoints(xInf, yFloat, xSup, ySup):
					point =  R2_Point(xFloat, yFloat)
					if not point in edgePoints:
						edgePoints.append(point)
				else:
					yFloat = float(y - halfHeight)
					xFloat = calcXbyY(s, cosT, sinT, yFloat)

					xInf = min(x, xOld)
					xSup = max(x, xOld)
					yInf = min(y, yOld)
					ySup = max(y, yOld)
					if isValidBorderPoints(xFloat, yInf, xSup, ySup):
						point =  R2_Point(xFloat, yFloat)
						if not point in edgePoints:
							edgePoints.append(point)

			elif x == xOld:
				
				if sinT != 0:
					xFloat = float(x - halfWidth)
					yFloat = calcYbyX(s, cosT, sinT, xFloat)
				else:
					(xFloat, yFloat) = (float(x - halfWidth), float(y - halfHeight))

				#(xFloat, yFloat) = (float(x - halfWidth), float(y - halfHeight))
				point =  R2_Point(xFloat, yFloat)
				if not point in edgePoints:
					edgePoints.append(point)
			elif y == yOld:
				if cosT != 0:
					yFloat = float(y - halfHeight)
					xFloat = calcXbyY(s, cosT, sinT, yFloat)
				else:
					(xFloat, yFloat) = (float(x - halfWidth), float(y - halfHeight))
				
				#(xFloat, yFloat) = (float(x - halfWidth), float(y - halfHeight))
				point =  R2_Point(xFloat, yFloat)
				if not point in edgePoints:
					edgePoints.append(point)

		oldVal = pxVal
		(xOld, yOld) = (x, y)
		#putpixel(x,y,color)
		numerator += shortest
		if numerator>=longest:
			numerator -= longest
			x += dx1
			y += dy1
		else:
			x += dx2
			y += dy2
	return edgePoints

def bresenham_line_original(x, y, x2, y2, image):
	(row_num, col_num, _) = image.shape
	width = col_num
	height = row_num
	halfWidth = width/2
	halfHeight = height/2	

	x =  int(x + halfWidth)
	y =  int(y + halfHeight)
	x2 = int(x2 + halfWidth)
	y2 = int(y2 + halfHeight)
	xOld = x
	yOld = y

	w = x2 - x
	h = y2 - y
	dx1 = 0
	dx2 = 0
	dy1 = 0
	if w<0: dx1 = -1 
	elif w>2: dx1 = 1
	if h<0: dy1 = -1 
	elif h>0: dy1 = 1
	if w<0: dx2 = -1 
	elif w>0: dx2 = 1
	dy2 = 0
	longest  = abs(w)
	shortest = abs(h)

	if longest<=shortest:
		longest = abs(h)
		shortest = abs(w)
		if h<0:
			dy2 = -1
		elif h>0:
			dy2 = 1
		dx2 = 0

	numerator = longest >> 1
	edgePoints = []
	oldVal = 0
	#return [R2_Point(x - halfWidth, y - halfHeight), R2_Point(x2 - halfWidth, y2 - halfHeight)]
	for i in range (0, longest+1): 
		pxVal = int(image[y][x][0])
		
		xCorrect = x
		yCorrect = y

		pxVal = int(image[y][x][0])
		if isWhite(pxVal):
			ePoint = R2_Point(xCorrect - halfWidth, yCorrect - halfHeight)
			edgePoints.append(ePoint)

		numerator += shortest
		if numerator>=longest:
			numerator -= longest
			x += dx1
			y += dy1
		else:
			x += dx2
			y += dy2
	return edgePoints


def bresenham_line(s, cosT, sinT, x, y, x2, y2, image):
	(row_num, col_num, _) = image.shape
	width = col_num
	height = row_num
	halfWidth = width/2
	halfHeight = height/2	

	x =  int(x + halfWidth)
	y =  int(y + halfHeight)
	x2 = int(x2 + halfWidth)
	y2 = int(y2 + halfHeight)
	xOld = x
	yOld = y


	w = x2 - x
	h = y2 - y
	dx1 = 0
	dx2 = 0
	dy1 = 0
	if w<0: dx1 = -1 
	elif w>2: dx1 = 1
	if h<0: dy1 = -1 
	elif h>0: dy1 = 1
	if w<0: dx2 = -1 
	elif w>0: dx2 = 1
	dy2 = 0
	longest  = abs(w)
	shortest = abs(h)

	if longest<=shortest:
		longest = abs(h)
		shortest = abs(w)
		if h<0:
			dy2 = -1
		elif h>0:
			dy2 = 1
		dx2 = 0

	numerator = longest >> 1
	edgePoints = []
	oldVal = 0
	#return [R2_Point(x - halfWidth, y - halfHeight), R2_Point(x2 - halfWidth, y2 - halfHeight)]
	for i in range (0, longest+1): 
		pxVal = int(image[y][x][0])
		
		xCorrect = x
		yCorrect = y

		# =====
		#xInf = xCorrect#min(x, xOld)
		#xSup = xCorrect+1#max(x, xOld)
		#yInf = yCorrect#min(y, yOld)
		#ySup = yCorrect+1#max(y, yOld)	
		#borderPoints = calcBorderPoints(s, cosT, sinT, xInf - halfWidth, xSup - halfWidth, yInf - halfHeight, ySup - halfHeight)
		ePoint = R2_Point(xCorrect - halfWidth, yCorrect - halfHeight) #choiceEdgePoint(borderPoints, xOld, yOld)
		edgePoints.append(ePoint)
		# =====
		
		if isDiagonal(xOld, yOld, x, y):
			xInf = xOld#min(x, xOld)
			xSup = xOld+1#max(x, xOld)
			yInf = yOld#min(y, yOld)
			ySup = yOld+1#max(y, yOld)

			#crossPixel 1
			(xCross1, yCross1) = (x, yOld)
			(xInf, xSup, yInf, ySup) = (xCross1, xCross1+1, yCross1, yCross1+1)
			borderPoints = calcBorderPoints(s, cosT, sinT, xInf - halfWidth, xSup - halfWidth, yInf - halfHeight, ySup - halfHeight)
			if len(borderPoints):
				#if len(borderPoints) == 2:
				xCorrect = xCross1
				yCorrect = yCross1
			else:
				#crossPixel 2
				(xCross2, yCross2) = (xOld, y)
				(xInf, xSup, yInf, ySup) = (xCross2, xCross2+1, yCross2, yCross2+1)
				borderPoints = calcBorderPoints(s, cosT, sinT, xInf - halfWidth, xSup - halfWidth, yInf - halfHeight, ySup - halfHeight)
				if borderPoints:
					#if len(borderPoints) == 2:
					xCorrect = xCross2
					yCorrect = yCross2
		

		if hasChangedPixelValue(i, longest, oldVal, pxVal):
		#if True:
			xInf = xCorrect#min(x, xOld)
			xSup = xCorrect+1#max(x, xOld)
			yInf = yCorrect#min(y, yOld)
			ySup = yCorrect+1#max(y, yOld)		
			
			#print("xInf, xSup, yInf, ySup = (%f, %f, %f, %f)" %(xInf, xSup, yInf, ySup))
			borderPoints = calcBorderPoints(s, cosT, sinT, xInf - halfWidth, xSup - halfWidth, yInf - halfHeight, ySup - halfHeight)
			#print("borderPoints = ", borderPoints)

			#(xFloat, yFloat) = (float(x - halfWidth), float(y - halfHeight))
			#ePoint = R2_Point(xFloat, yFloat)
			#edgePoints.append(R2_Point(float(x - halfWidth), float(y - halfHeight))) # Impreciso

			
			ePoint = choiceEdgePoint(borderPoints, xOld, yOld)
			#if ePoint:
			#	edgePoints.append(ePoint)

			if ePoint and not ePoint in edgePoints:
				edgePoints.append(ePoint) # high precision
			else: # Pixel visitado pelo bresenham, mas a reta não passa
				(xPixel, yPixel) = (xCorrect, yCorrect)
				#edgePoints.append(R2_Point(float(xPixel - halfWidth), float(yPixel - halfWidth))) # Pegando os índices do pixel: low precision ERRADO!
				# -- corrigindo pixel, encontrando ponto em que a reta passa
				yCorrect = calcYbyX(s, cosT, sinT, xCorrect - halfWidth)
				#print("xCorrect = ", xCorrect)
				#print("yCorrect + halfHeight = ", yCorrect + halfHeight)

				if 0 <= xCorrect < width and 0 <= yCorrect + halfHeight < height:
					pxVal = (image[yCorrect + halfHeight][xCorrect][0])
					if hasChangedPixelValue(i, longest, oldVal, pxVal):
					#if True:
						ePoint = R2_Point(float(xCorrect - halfWidth), float(yCorrect))

						if not ePoint in edgePoints:
							edgePoints.append(ePoint)

			

			#for bPoint in borderPoints:
			#	if bPoint and not bPoint in edgePoints:
			#		edgePoints.append(bPoint)



			# Verificando se os borderPoints são edgePoints
			'''
			for bPoint in borderPoints:
				#print("bPoint = ", bPoint)
				#print("xOld, yOld = %f, %f" %(xOld, yOld))
			#if borderPoints:
				#bPoint = borderPoints[0]
				(xb, yb) = bPoint.toTuple()
				(xb, yb) = (xb + halfWidth, yb + halfHeight)
				edgePoints.append(bPoint)
				#break
				if int(xb) == xOld or int(yb) == yOld:
					#print("xb = ", xb)
					#print("yb = ", yb)
					if not bPoint in edgePoints:
						#(xFloat, yFloat) = (float(xb - halfWidth), float(yb - halfHeight))
						#ePoint = R2_Point(xb, yb)#(xFloat, yFloat)
						print(bPoint)
						print("pxval = %d, oldVal = %d" % (pxVal, oldVal))
						#edgePoints.append(bPoint) # Preciso
						#break
			'''
		oldVal = pxVal
		(xOld, yOld) = (x, y)
		#putpixel(x,y,color)
		numerator += shortest
		if numerator>=longest:
			numerator -= longest
			x += dx1
			y += dy1
		else:
			x += dx2
			y += dy2
	return edgePoints

def isWhite(pxVal):
	return pxVal > 0.0 

def isEdgePoint_deprecated(s, cosT, sinT, u, du, halfWidth, halfHeight, image, firstPoint=False):
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
			# pxVal1 = int(image[yIdx1][xIdx1][0])
			pxVal1 = int(image[yIdx1][xIdx1][0]) # Pontos intermediarios havendo mudanca de intesidade
			return (pxVal1 - pxVal0) != 0
	else:
		return isWhite(pxVal0) # Ultimo Ponto sendo branco

def isValidBorderPoints(x, y, xLimitINF, xLimitSUP, yLimitINF, yLimitSUP):
	#return True
	tol = 0.0
	return (xLimitINF - tol <= x <= xLimitSUP + tol) and (yLimitINF - tol <= y <= yLimitSUP + tol)
	#return (-halfWidth <= x <= halfWidth) and (-halfHeight <= y <= halfHeight)

def calcPoint(s, cosT, sinT, u):	
	x = s*cosT - u*sinT
	y = s*sinT + u*cosT
	return(x, y)

def calcUbyX(s, cosT, sinT, xLimit):
	u = (s*cosT - xLimit)/sinT
	return u

def calcUbyY(s, cosT, sinT, yLimit):
	u = (yLimit - s*sinT)/cosT
	return u

def calcUbyXY(s, cosT, sinT, x, y):
	if cosT != 0:
		return calcUbyY(s, cosT, sinT, y)
	else:
		return calcUbyX(s, cosT, sinT, x)

def calcYbyX(s, cosT, sinT, x):
	y = (s - x*cosT)/sinT
	return y

def calcXbyY(s, cosT, sinT, y):
	x = (s - y*sinT)/cosT
	return x

def calcBorderPoints(s, cosT, sinT, xLimitINF, xLimitSUP, yLimitINF, yLimitSUP):
	# continuar aqui!
	borderPoints = []

	if sinT != 0:
		# borda lateral esquerda
		x = xLimitINF
		y = calcYbyX(s, cosT, sinT, x)
		#print("borda lateral esquerda (%f, %f)" %(x,y))
		#print("xLimitINF, xLimitSUP, yLimitINF, yLimitSUP = %f,%f,%f,%f" %(xLimitINF, xLimitSUP, yLimitINF, yLimitSUP))
		if isValidBorderPoints(x, y, xLimitINF, xLimitSUP, yLimitINF, yLimitSUP):
			borderPoints.append(R2_Point(x, y))

		# borda lateral direita
		x = xLimitSUP
		y = calcYbyX(s, cosT, sinT, x)
		#print("borda lateral direita (%f, %f)" %(x,y))
		#print("xLimitINF, xLimitSUP, yLimitINF, yLimitSUP = %f,%f,%f,%f" %(xLimitINF, xLimitSUP, yLimitINF, yLimitSUP))
		if isValidBorderPoints(x, y, xLimitINF, xLimitSUP, yLimitINF, yLimitSUP):
			borderPoints.append(R2_Point(x, y))

		if cosT != 0:
			# borda superior
			y = yLimitSUP
			x = calcXbyY(s, cosT, sinT, y)
			#print("borda superior (%f, %f)" %(x,y))
			#print("xLimitINF, xLimitSUP, yLimitINF, yLimitSUP = %f,%f,%f,%f" %(xLimitINF, xLimitSUP, yLimitINF, yLimitSUP))
			if isValidBorderPoints(x, y, xLimitINF, xLimitSUP, yLimitINF, yLimitSUP):
				borderPoints.append(R2_Point(x, y))

			# borda inferior
			y = yLimitINF
			x = calcXbyY(s, cosT, sinT, y)
			#print("borda inferior (%f, %f)" %(x,y))
			#print("xLimitINF, xLimitSUP, yLimitINF, yLimitSUP = %f,%f,%f,%f" %(xLimitINF, xLimitSUP, yLimitINF, yLimitSUP))
			if isValidBorderPoints(x, y, xLimitINF, xLimitSUP, yLimitINF, yLimitSUP):
				borderPoints.append(R2_Point(x, y))
	else:
		# borda superior
		y = yLimitSUP
		x = calcXbyY(s, cosT, sinT, y)
		#print("*borda superior (%f, %f)" %(x,y))
		#print("xLimitINF, xLimitSUP, yLimitINF, yLimitSUP = %f,%f,%f,%f" %(xLimitINF, xLimitSUP, yLimitINF, yLimitSUP))
		if isValidBorderPoints(x, y, xLimitINF, xLimitSUP, yLimitINF, yLimitSUP):
			borderPoints.append(R2_Point(x, y))

		# borda inferior
		y = yLimitINF
		x = calcXbyY(s, cosT, sinT, y)
		#print("*borda inferior (%f, %f)" %(x,y))
		#print("xLimitINF, xLimitSUP, yLimitINF, yLimitSUP = %f,%f,%f,%f" %(xLimitINF, xLimitSUP, yLimitINF, yLimitSUP))
		if isValidBorderPoints(x, y, xLimitINF, xLimitSUP, yLimitINF, yLimitSUP):
			borderPoints.append(R2_Point(x, y))

	#if len(borderPoints) == 1:
	#	print("borderPoints: ", borderPoints)

	return borderPoints

# def calcImageBorderPoints(s, cosT, sinT, xLimit, yLimit):
# 	return calcBorderPoints(s, cosT, sinT, -xLimit, xLimit-1, -yLimit, yLimit-1)

# REFAZER (Já refeito) calcImageBorderPoints sem parametro u 
def calcImageBorderPoints_u(s, cosT, sinT, halfWidth, halfHeight):
	borderPoints = []
	if sinT != 0:
		u = calcUbyX(s, cosT, sinT, -halfWidth)
		#u = 1
		(x, y) = calcPoint(s, cosT, sinT, u)
		if isValidBorderPoints(x, y, halfWidth, halfHeight):
			borderPoints.append((x, y))

		u = calcUbyX(s, cosT, sinT, halfWidth-1)
		#u = 1
		(x, y) = calcPoint(s, cosT, sinT, u)
		if isValidBorderPoints(x, y, halfWidth, halfHeight):
			borderPoints.append((x, y))

		if cosT != 0:
			u = calcUbyY(s, cosT, sinT, -halfHeight)
			#u = 1
			(x, y) = calcPoint(s, cosT, sinT, u)
			if isValidBorderPoints(x, y, halfWidth, halfHeight):
				borderPoints.append((x, y))

			u = calcUbyY(s, cosT, sinT, halfHeight-1)
			#u = 1
			(x, y) = calcPoint(s, cosT, sinT, u)
			if isValidBorderPoints(x, y, halfWidth, halfHeight):
				borderPoints.append((x, y))
	else:
		u = calcUbyY(s, cosT, sinT, -halfHeight)
		#u = 1
		(x, y) = calcPoint(s, cosT, sinT, u)
		if isValidBorderPoints(x, y, halfWidth, halfHeight):
			borderPoints.append((x, y))

		u = calcUbyY(s, cosT, sinT, halfHeight-1)
		#u = 1
		(x, y) = calcPoint(s, cosT, sinT, u)
		if isValidBorderPoints(x, y, halfWidth, halfHeight):
			borderPoints.append((x, y))
	return borderPoints

# Deprecated
def crossRadonTransform(image, nTraj, nProj):
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
	temporaryDescriptor = ShapeDescriptor()
	#temporaryFirstEdgePoint = R2_Point(0, 0)
	# Compute transformed image
	for t in range(0, tMax):
		theta = t*dTheta
		cosT = math.cos(theta)
		sinT = math.sin(theta)
		for sIdx in range(0, nTraj+1): #np.arange(0, sMax + ds, ds):
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
			for u in np.arange(-halfD, halfD+du, du):
				# Compute the point p in the original image which is mapped to the
				# (row, col) pixel in the transformed image
				
				(x, y) = calcPoint(s, cosT, sinT, u)
				
				xIdx = x + halfWidth
				yIdx = y + halfHeight

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
					if isEdgePoint_deprecated(s, cosT, sinT, u, du, halfWidth, halfHeight, image, firstPoint):
						#plt.plot([xIdx], [yIdx], 'ro')
						edgePoints.append(R2_Point(x, y))
						#if firstPoint:
						#	(xt, yt) = calcPoint(s, cosT, sinT, u + 0.9)
						#	temporaryFirstEdgePoint = R2_Point(xt, yt)

					firstPoint = False

			descriptor.addRay(s, theta, edgePoints)
			#edgePoints[0] = temporaryFirstEdgePoint
			#temporaryDescriptor.addRay(s, theta, edgePoints)
			#plt.plot(X, Y, 'r')
			"""
			sinograma[sIdx][t][0] = countR
			sinograma[sIdx][t][1] = countG
			sinograma[sIdx][t][2] = countB
			"""

	#row_sums = sinograma.sum(axis=1)
	#new_matrix = sinograma / row_sums[:, np.newaxis]
	return (sinograma, descriptor)

def crossRadonTransform2(image, nTraj, nProj):
	#print("radonTransform")
	# Compute the number of rows and columns in the image
	(row_num, col_num, _) = image.shape
	# Initialize the transformed image
	diagonal = math.sqrt(row_num*row_num + col_num*col_num)
	tMax = nProj
	sMax = diagonal/2
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
	temporaryDescriptor = ShapeDescriptor()
	#temporaryFirstEdgePoint = R2_Point(0, 0)
	# Compute transformed image
	for t in range(0, tMax):
		theta = t*dTheta
		cosT = math.cos(theta)
		sinT = math.sin(theta)
		for sIdx in range(0, nTraj+1): #np.arange(0, sMax + ds, ds):
			s = sIdx*ds - sMax

			X = []
			Y = []
			firstPoint = True
			#print('s, t = %f, %f' %(s, t))
			
			edgePoints = []
			#for u in np.arange(-halfD, halfD+du, du):


			borderPoints = calcImageBorderPoints(s, cosT, sinT, halfWidth, halfHeight)
			if len(borderPoints) == 2:
				(x1, y1) = borderPoints[0].toTuple()
				(x2, y2) = borderPoints[1].toTuple()

				edgePoints = bresenham_line(s, cosT, sinT, x1, y1, x2, y2, image)
				
				
				##edgePoints = bresenham(x1, y1, x2, y2, image) # PAREI AQUI!
				descriptor.addRay(s, theta, edgePoints)
			#edgePoints[0] = temporaryFirstEdgePoint
			#temporaryDescriptor.addRay(s, theta, edgePoints)
			#plt.plot(X, Y, 'r')
			"""
			sinograma[sIdx][t][0] = countR
			sinograma[sIdx][t][1] = countG
			sinograma[sIdx][t][2] = countB
			"""

	#row_sums = sinograma.sum(axis=1)
	#new_matrix = sinograma / row_sums[:, np.newaxis]
	return (sinograma, descriptor)


def newBresenham(x, y, x2, y2, image):
	(row_num, col_num, _) = image.shape
	width = col_num
	height = row_num
	halfWidth = width/2
	halfHeight = height/2

	#halfHeight = 0
	#halfWidth = 0
	x =  int(x + halfWidth)
	y =  int(y + halfHeight)
	x2 = int(x2 + halfWidth)
	y2 = int(y2 + halfHeight)

	points = []
	coordinates_list = list(bresenham(x, y, x2, y2))
	longest = len(coordinates_list)
	i = 0
	oldPxVal = 0
	for (xi, yi) in coordinates_list:
		i += 1
		pxVal = int(image[yi][xi][0])
		if hasChangedPixelValue(i, longest, oldPxVal, pxVal):
			points.append(R2_Point(xi-halfWidth, yi-halfHeight))
		oldPxVal = pxVal
	return points

