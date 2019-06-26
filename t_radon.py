
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
from point import P2_Point
from point import R2_Point

## Close window and change progress in code
def press(event):
	#print('press', event.key)
	if event.key == 'enter':
		plt.close()

def rotation(vs, theta):
	xs = vs.x
	ys = vs.y

	cosT = math.cos(theta)
	sinT = math.sin(theta)

	xu = xs*cosT - ys*sinT
	yu = xs*sinT + ys*cosT
	vu = R2_Point(xu, yu)
	return vu


def radonTransform(image, nProj):
	print("radonTransform")
	# Compute the number of rows and columns in the image
	(row_num, col_num, _) = image.shape
	# Initialize the transformed image
	diagonal = int(math.sqrt(row_num*row_num + col_num*col_num))
	tMax = nProj
	sMax = diagonal
	
	shapeResult = (diagonal, nProj, 3)
	sinograma = np.zeros(shapeResult)#, dtype=np.uint8)
	
	(dims, dimt, _) = sinograma.shape
	print("size img = (%d, %d)" %(dims, dimt))

	width = col_num
	height = row_num
	dTheta = np.pi/nProj #3.14159265/nProj
	halfWidth = width/2
	halfD = diagonal/2 # meia diagonal da imagem

	#Images indices: s, t, y, x
	du = 1.0

	# Compute transformed image
	for t in range(0, tMax):
		theta = t*dTheta
		cosT = math.cos(theta)
		sinT = math.sin(theta)
		for s in range(0, sMax):
			countR = sinograma[s][t][0]
			countG = sinograma[s][t][1]
			countB = sinograma[s][t][2]
			#print('s, t = %f, %f' %(s, t))
			for u in np.arange(-halfD, halfD+du, du):
				# Compute the point p in the original image which is mapped to the
				# (row, col) pixel in the transformed image
				
				s_ = s - halfWidth
				x = s_*cosT - u*sinT
				y = s_*sinT + u*cosT
				
				xIdx = x + halfWidth
				yIdx = y + halfWidth

				if 0 <= xIdx < width and 0 <= yIdx < width:
					countR = countR + int(image[yIdx][xIdx][0])
					countG = countG + int(image[yIdx][xIdx][1])
					countB = countB + int(image[yIdx][xIdx][2])

			sinograma[s][t][0] = countR
			sinograma[s][t][1] = countG
			sinograma[s][t][2] = countB

	#row_sums = sinograma.sum(axis=1)
	#new_matrix = sinograma / row_sums[:, np.newaxis]

	return sinograma

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

def projRadonTransform(image, PF0, PFn, n):

	print("proj radonTransform")
	# Compute the number of rows and columns in the image
	(row_num, col_num, _) = image.shape
	diagonal = int(math.sqrt(row_num*row_num + col_num*col_num))
	halfD = diagonal/2 # meia diagonal da imagem
	# Initialize the transformed image
	width = col_num
	height = row_num
	PF0.to_cartesian_coord(width, height)
	PFn.to_cartesian_coord(width, height)

	ds = (PFn - PF0)/n

	tMax = 180
	sMax = n
	
	vs = PFn - PF0
	vs.r2Normalize()

	shapeResult = (sMax, tMax, 3)
	sinograma = np.zeros(shapeResult)#, dtype=np.uint8)
	
	(dims, dimt, _) = sinograma.shape
	print("size img = (%d, %d)" %(dims, dimt))


	dTheta = np.pi/tMax #3.14159265/tMax

	#Images indices: s, t, y, x
	du = 3.0

	# Compute transformed image

	
	for s in range(-sMax, sMax):

		PFs = PF0 + s*vs

		for t in range(0, tMax):
			theta = t*dTheta

			vu = rotation(vs, theta)

			countR = sinograma[s][t][0]
			countG = sinograma[s][t][1]
			countB = sinograma[s][t][2]
			#print('s, t = %f, %f' %(s, t))
			for u in np.arange(-diagonal, diagonal+du, du):
				# Compute the point p in the original image which is mapped to the
				# (row, col) pixel in the transformed image
				
				Pxy = PFs + u*vu
				#print("cart: x,y = %f,%f" %(Pxy.x, Pxy.y))
				Pxy.to_img_coord(width, height)
				xIdx = Pxy.x
				yIdx = Pxy.y
				#print("img: x,y = %f,%f" %(Pxy.x, Pxy.y))
				if 0 <= xIdx < width and 0 <= yIdx < height:
					#print("RGB = (%f, %f, %f)" %(countR, countG, countB))
					countR = countR + image[yIdx][xIdx][0]
					countG = countG + image[yIdx][xIdx][1]
					countB = countB + image[yIdx][xIdx][2]

			sinograma[s][t][0] = countR
			sinograma[s][t][1] = countG
			sinograma[s][t][2] = countB

	#row_sums = sinograma.sum(axis=1)
	#new_matrix = sinograma / row_sums[:, np.newaxis]

	return sinograma

def makeP2Line(P0, v, angle):
	v.r2Normalize
	v_angle = rotation(v, angle)
	(xv, yv) = v_angle.toTuple()
	(x0, y0, _) = P0.toTuple()
	p2Line = P2_Point(yv, -xv, -yv*x0 + xv*y0)
	return p2Line

def boudingBoxIntersection(r, P0, Pf):
	(x0, y0) = P0.toTuple()
	(xf, yf) = Pf.toTuple()
	(a, b, c) = r.toTuple()

	x1 = x0
	y1 = y0
	x2 = xf
	y2 = yf	

	if b != 0:
		yp = (-x0*a - 1)/b
		if y0 <= yp <= yf:
			(x1, y1) = (x0, yp)
		elif a != 0:
			xp = (-y0*b - 1)/a
			(x1, y1) = (xp, y0)
	elif a != 0:
		xp = (-y0*b - 1)/a
		(x1, y1) = (xp, y0)

	if b != 0:
		yp = (-xf*a - 1)/b
		if y0 <= yp <= yf:
			(x2, y2) = (xf, yf)
		elif a != 0:
			xp = (-yf*b - 1)/a
			(x2, y2) = (xp, yf)
	elif a != 0:
		xp = (-yf*b - 1)/a
		(x2, y2) = (xp, yf)

	P1 = R2_Point(x1, y1)
	P2 = R2_Point(x2, y2)
	return P1, P2

def projRadonTransform2(image, PFa, PFb, x0, y0, xf, yf, n=100):
	print("proj radonTransform")
	# Compute the number of rows and columns in the image
	(row_num, col_num, _) = image.shape
	diagonal = int(math.sqrt(row_num*row_num + col_num*col_num))
	halfD = diagonal/2 # meia diagonal da imagem
	# Initialize the transformed image
	width = col_num
	height = row_num

	tMax = 18
	sMax = n

	#(x0, y0, xf, yf) = boudingBox

	# Passando para coordenadas cartesianas
	PFa.to_cartesian_coord(width, height)
	PFb.to_cartesian_coord(width, height)

	PFa = PFa.toP2_Point()
	PFb = PFb.toP2_Point()

	Lh = PFa.cross(PFb) # horizon line

	# Vetor diretor da linha do horizonte
	vh = PFb - PFa
	vh = R2_Point(vh.x, vh.y)
	vh.r2Normalize()

	pv1 = R2_Point(x0, yf)
	pv2 = R2_Point(xf, yf)
	pv1.to_cartesian_coord(width, height)
	pv2.to_cartesian_coord(width, height)
	pv1 = pv1.toP2_Point()
	pv2 = pv2.toP2_Point()

	LeftLine  = makeP2Line(P2_Point(x0, yf, 1), vh, math.pi*179/180)
	RightLine = makeP2Line(P2_Point(xf, yf, 1), vh, math.pi*1/180)

	PF0 = Lh.cross(LeftLine)
	PFn = Lh.cross(RightLine)
	PF0.normalize()
	PFn.normalize()
	print("PF0 = ", PF0)
	print("PFn = ", PFn)

	dTheta = (PFn.euclideanDistance(PF0))/tMax

	shapeResult = (sMax, tMax, 3)
	sinograma = np.zeros(shapeResult)
	
	(dims, dimt, _) = sinograma.shape
	print("size sinograma = (%d, %d)" %(dims, dimt))


	ds = np.pi/sMax #3.14159265/sMax

	PF0 = PFn.toR2_Point()
	for t in range(0, tMax):
		theta = t*dTheta
		PFt = PF0 + theta*vh
		PFtPrint = copy.deepcopy(PFt)
		PFtPrint.to_img_coord(width, height)
		print("PFt = ", PFtPrint)
		for s in range(0, sMax):
			angle_s = s*ds

			# calculando reta de fuga
			Rts = makeP2Line(PFt.toP2_Point(), vh, angle_s)
			Rts.normalize()
			countR = sinograma[s][t][0]
			countG = sinograma[s][t][1]
			countB = sinograma[s][t][2]

			# Descobrindo intersecção com a bouding box
			P0vertex = R2_Point(x0, y0)
			Pfvertex = R2_Point(xf, yf)
			P0vertex.to_cartesian_coord(width, height)
			Pfvertex.to_cartesian_coord(width, height)

			(P1, P2) = boudingBoxIntersection(Rts, P0vertex, Pfvertex)

			distance = P1.euclideanDistance(P2)
			if distance != 0:
		
				du = 2/(P0vertex.euclideanDistance(Pfvertex))
				#du = 1.0
				#print()
				vu = P2 - P1
				vu.r2Normalize()
				for u in np.arange(-distance, distance+du, du):
					# Compute the point p in the original image which is mapped to the
					# (row, col) pixel in the transformed image
					
					Pxy = P1 + u*vu
					#print("cart: x,y = %f,%f" %(Pxy.x, Pxy.y))
					Pxy.to_img_coord(width, height)
					xIdx = Pxy.x
					yIdx = Pxy.y
					#print("img: x,y = %f,%f" %(Pxy.x, Pxy.y))
					if 0 <= xIdx < width and 0 <= yIdx < height:
						print("x,y = %d, %d " %(xIdx, yIdx))
						countR = countR + image[yIdx][xIdx][0]
						countG = countG + image[yIdx][xIdx][1]
						countB = countB + image[yIdx][xIdx][2]
						print("RGB = (%f, %f, %f)" %(countR, countG, countB))

			sinograma[s][t][0] = countR
			sinograma[s][t][1] = countG
			sinograma[s][t][2] = countB

	#row_sums = sinograma.sum(axis=1)
	#new_matrix = sinograma / row_sums[:, np.newaxis]

	return sinograma

# =============================================================================
# ============================== LOAD IMAGE ===================================
# =============================================================================

print("Oi")
filename = askopenfilename(filetypes=[("all files","*"),("Bitmap Files","*.bmp; *.dib"),
										("JPEG", "*.jpg; *.jpe; *.jpeg; *.jfif"),
										("PNG", "*.png"), ("TIFF", "*.tiff; *.tif")])

image = misc.imread(filename, mode = 'RGB')

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('Click to build line segments')
print("Click to build line segments")
(row_num, col_num, _) = image.shape

# =============================================================================
# ============================== SHOW ORIGINAL IMAGE ==========================
# =============================================================================

fig.canvas.set_window_title('Original Image')
fig.canvas.mpl_connect('key_press_event', press)
plt.imshow(image)
plt.show()

print("Pre Start..")
#sinograma = radonTransform(image, 180)
PFa = R2_Point(0, 38)
PFb = R2_Point(785, 38) #281,38 
n = 5
(x0, y0) = (300,104)
(xf, yf) = (527,241)
#sinograma = projRadonTransform(image, PF0, PFn, n)
print("Starting..")
sinograma = projRadonTransform2(image, PFa, PFb, x0, y0, xf, yf, n)
sinograma = normalizeImg(sinograma)

print("")

#image_[0] *= 255.0/image_[0].max()
#image_[1] *= 255.0/image_[1].max() 
#image_[2] *= 255.0/image_[2].max() 

#image_ /= image_.max()/255.0    # Normalize image, Uses 1+image.size divisions
# =============================================================================

fig = plt.figure()
fig.canvas.set_window_title('Rectified Image (Stratified Metric Rectification)')
fig.canvas.mpl_connect('key_press_event', press)
plt.imshow(sinograma)
plt.show()

