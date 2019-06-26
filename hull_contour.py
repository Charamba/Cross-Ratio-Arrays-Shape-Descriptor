import matplotlib.pyplot as plt
from raster import * 

from scipy import misc
import numpy as np
import math

def getCornerPoints(image, width, height):
	cornerPoints = []

	#(height, width) = image.shape()
	halfHeight = int(math.floor(height/2))
	halfWidth = int(math.floor(width/2))

	for x in range(width):
		for y in range(height):
			pxVal = pixel_value(image, x, y)
			if pxVal > 0:
				xCoord = x
				yCoord = y

				pxLeftUPVal = pixel_value(image, x-1, y-1)
				pxUPVal     = pixel_value(image, x, y-1)
				pxRightUPVal = pixel_value(image, x+1, y-1)

				pxLeftVal = pixel_value(image, x-1, y)
				pxRightVal = pixel_value(image, x+1, y)

				pxLeftDownVal = pixel_value(image, x-1, y+1)
				pxDownVal = pixel_value(image, x, y+1)
				pxRightDownVal = pixel_value(image, x+1, y+1)


				if (pxLeftVal==0 or pxLeftUPVal==0 or pxUPVal==0):  
					cornerPoints.append((x,y))
				if (pxRightVal==0 or pxRightUPVal==0 or pxUPVal==0):
					cornerPoints.append((x+1,y))
				if (pxLeftVal==0 or pxLeftDownVal==0 or pxDownVal==0):  
					cornerPoints.append((x,y+1))
				if (pxRightVal==0 or pxRightDownVal==0 or pxDownVal==0):
					cornerPoints.append((x+1,y+1))
				
	return cornerPoints

def top_side_condition(point, image):
	(x,y) = point
	return (pixel_value(image,x,y) == 255 and  pixel_value(image,x,y-1) == 0)

def bottom_side_condition(point, image):
	(x,y) = point
	return (pixel_value(image,x,y) == 0 and  pixel_value(image,x,y-1) == 255)

def left_side_condition(point, image):
	(x,y) = point
	return (pixel_value(image,x,y) == 255 and  pixel_value(image, x-1,y) == 0)

def right_side_condition(point, image):
	(x,y) = point
	return (pixel_value(image,x,y) == 0 and  pixel_value(image,x-1,y) == 255)

def horizontal_condition(point, image):
	is_up_side =  top_side_condition(point, image)
	is_down_side = bottom_side_condition(point, image)
	return is_up_side or is_down_side

def vertical_condition(point, image):
	is_left_side = left_side_condition(point, image)
	is_right_side = right_side_condition(point, image)
	return is_left_side or is_right_side

def findNextPoint(point, points, image):
	(x, y) = point
	P = point
	next_point = None

	right_possible_point = (x+1,y)
	down_possible_point  = (x,y+1)
	left_possible_point  = (x-1,y)
	up_possible_point    = (x,y-1)

	#horizontal_condition = ((pixel_value(x,y,image) == 255 and  pixel_value(x,y-1,image) == 0) or (pixel_value(x,y,image) == 0 and  pixel_value(x,y-1,image) == 255))
	#vertical_condition   = ((pixel_value(x,y,image) == 255 and  pixel_value(x-1,y,image) == 0) or (pixel_value(x,y,image) == 0 and  pixel_value(x-1,y,image) == 255))

	if right_possible_point in points and horizontal_condition(point, image):#and is_segmented_contour(P, right_possible_point, image):
		next_point = right_possible_point
	elif down_possible_point in points and vertical_condition(point, image): #and is_segmented_contour(P, down_possible_point, image):
		next_point = down_possible_point
	elif left_possible_point in points and horizontal_condition(left_possible_point, image): #and is_segmented_contour(P, left_possible_point, image):
		next_point = left_possible_point
	elif up_possible_point in points and vertical_condition(up_possible_point, image): #and is_segmented_contour(P, up_possible_point, image):
		next_point = up_possible_point
	# else:
	# 	# encontrando o mais próximo por força bruta
	# 	next_
	# 	for next_point in points:



	return next_point

def is_segmented_contour(P, nextP, image):
	(xp, yp) = P
	(xnp, ynp) = nextP

	(x,y) = (None, None)

	up_down = False
	left_right = False 

	flag = False

	if yp == ynp:#abs(xp - xnp) > 0:
		left_right = True
		x = min(xp, xnp)
		y = yp

		# if ( (pixel_value(image,x,y) == 0 and (pixel_value(image,x,y-1) == 255 or pixel_value(image,x,y+1))) or (pixel_value(image,x,y) == 255 and (pixel_value(image,x,y+1) == 0 or pixel_value(image,x,y-1) == 0) ):
		# 	flag = True
		#------------------
		# if (pixel_value(image,x,y) == 0 and pixel_value(image,x-1,y) == 255) or (pixel_value(image,x,y) == 255 and pixel_value(image,x+1,y) == 0):
		# 	flag = True


	if xp == xnp:#abs(yp - ynp) > 0:
		up_down = True
		y = min(yp, ynp)
		x = xp

		# if ( (pixel_value(image,x,y) == 0 and (pixel_value(image,x-1,y) == 255 or pixel_value(image,x+1,y) == 255))) or (pixel_value(image,x,y) == 255 and (pixel_value(image,x+1,y) == 0 or pixel_value(image,x-1,y) == 0) ):
		# 	flag = True
		#-------------------------
		# if (pixel_value(image,x,y) == 0 and pixel_value(image,x,y-1) == 255) or (pixel_value(image,x,y) == 255 and pixel_value(image,x,y+1) == 0):
		# 	flag = True

	return flag

def is_oriented(points, image):
	P0 = points[0]
	P1 = points[1]

	orientation = False

	(x,y) = P0
	right_possible_point = (x+1,y)
	down_possible_point  = (x,y+1)
	left_possible_point  = (x-1,y)
	up_possible_point    = (x,y-1)

	if right_possible_point == P1:
		if top_side_condition(P0, image):
			orientation = True
	elif down_possible_point == P1:
		if right_side_condition(P0, image):
			orientation = True
	elif left_possible_point == P1:
		if bottom_side_condition(P1, image):
			orientation = True
	elif up_possible_point == P1:
		if left_side_condition(P1, image):
			orientation = True

	return orientation



def sortContourPoints(cornerPoints, image):
	orig_corner_points = cornerPoints
	ContourPointsSorted = []

	list_curves = []

	(xp,yp) = cornerPoints[0]
	P = (xp, yp)
	del cornerPoints[0]

	#print("P0 = ", P)
	flag = False

	while cornerPoints:
		next_point = findNextPoint(P, cornerPoints, image)

		if not(next_point==None):
			ContourPointsSorted.append(next_point)
			idx_next = cornerPoints.index(next_point)

			(xp,yp) = next_point
			P = (xp, yp)
			del cornerPoints[idx_next]
		else:
			# if len(ContourPointsSorted) > 1:
			# 	break
			#print("x,y = (%d, %d)"%(xp, yp))
			if not(is_oriented(ContourPointsSorted, image)):
				ContourPointsSorted.reverse()
			#print()

			list_curves.append(ContourPointsSorted)
			ContourPointsSorted = []
			flag = True 
			(xp,yp) = cornerPoints[0]
			P = (xp, yp)
			del cornerPoints[0]

	#if not(flag):

	if not(is_oriented(ContourPointsSorted, image)):
		ContourPointsSorted.reverse()
	list_curves.append(ContourPointsSorted)

	

	return list_curves


def plotLinePoints(points, color='b', ciclic=False, writeOrder=False):
	X = []
	Y = []

	linePoints = []

	for i, point in enumerate(points):
		(x, y) = point
		X.append(x)
		Y.append(y)
		# if writeOrder:
		# 	self.plotText(point.x + halfWidth, point.y + halfHeight, str(i))
	#plt.plot(X, Y, color)

	if ciclic:
		(x, y) = points[0]
		X.append(x)
		Y.append(y)

	return (X, Y, color)

def two_pixels_sequence(pixels_sequence, pixel_orig):
	new_sequence = []
	(x, y) = pixel_orig 

	if (x,y) in pixels_sequence and (x,y-1) in pixels_sequence: # up
		new_sequence = [(x,y), (x,y-1)]
	elif (x,y) in pixels_sequence and (x-1,y) in pixels_sequence: # right
		new_sequence = [(x-1,y), (x,y)]
	elif (x-1,y-1) in pixels_sequence and (x-1,y) in pixels_sequence: # down
		new_sequence = [(x-1,y-1), (x-1,y)]
	elif (x-1,y-1) in pixels_sequence and (x,y-1) in pixels_sequence: # left
		new_sequence = [(x,y-1), (x-1,y-1)]

	return new_sequence

def three_pixels_sequence(pixels_sequence, pixel_orig):
	new_sequence = []
	(x, y) = pixel_orig

	if (x-1,y-1) in pixels_sequence and (x-1,y) in pixels_sequence and (x,y) in pixels_sequence: # up
		new_sequence = [(x-1,y-1), (x-1,y), (x,y)]
	elif (x-1,y) in pixels_sequence and (x,y) in pixels_sequence and (x,y-1) in pixels_sequence: # right
		new_sequence = [(x-1,y), (x,y), (x,y-1)]
	elif (x,y) in pixels_sequence and (x,y-1) in pixels_sequence and (x-1, y-1) in pixels_sequence: # down
		new_sequence = [(x,y), (x,y-1), (x-1, y-1)]
	elif (x,y-1) in pixels_sequence and (x-1,y-1) in pixels_sequence and (x-1,y) in pixels_sequence: # left
		new_sequence = [(x,y-1), (x-1,y-1), (x-1,y)]

	return new_sequence


def mid_point_pixels_sequence(image, pixel_orig):
	sequence_aux = []
	final_seq = []

	(x,y) = pixel_orig
	pixels_seq = [(x-1, y-1), (x-1, y), (x, y), (x, y-1)]

	for pixel in pixels_seq:
		(xp, yp) = pixel

		if pixel_value(image, xp, yp) == 255:  
			sequence_aux.append((xp, yp))

	n_sequence = len(sequence_aux)

	if n_sequence == 1:
		final_sequence = sequence_aux
	elif n_sequence == 2:
		final_sequence = two_pixels_sequence(sequence_aux, pixel_orig)
	elif n_sequence == 3:
		final_sequence = three_pixels_sequence(sequence_aux, pixel_orig)


	final_sequence = [(x+0.5, y+0.5) for (x,y) in final_sequence]

	return final_sequence

# def priority_pixel_sort(point, image):
# 	(x,y) = point
# 	pixel_values = []
	
# 	pixel_values.append(pixel_value(image, x-1, y-1))
# 	pixel_values.append(pixel_value(image, x-1, y))
# 	pixel_values.append(pixel_value(image, x, y))
# 	pixel_values.append(pixel_value(image, x, y-1))
	
# 	pixels_sequence = [(x-1, y-1), (x-1, y), (x, y), (x, y-1)] 


# 	f = lambda x: x/255
# 	values = [f(x) for x in pixel_values]

# 	for pixel in pixels_sequence:

def interior_curve(curve_points, image):
	#print("curve_points: ", curve_points)
	new_curve = []

	for point in curve_points:
		#print("point: ", point)
		new_mid_points_pixels_seq = mid_point_pixels_sequence(image, point)

		for new_pixel in new_mid_points_pixels_seq:
			if not(new_pixel in new_curve):
				new_curve.append(new_pixel)

	return new_curve


def curves_from_rasterized_img(img):
	#img, rows, cols = add_border(img)
	(rows, cols, _) = img.shape
	corner_pts = getCornerPoints(img, cols, rows)
	corner_pts = list(set(corner_pts))
	#curves = sortContourPoints(corner_pts, img)
	curves = []
	for i, curve in enumerate(sortContourPoints(corner_pts, img)):
		# in_curve = interior_curve(curve, img)
		# curves.append(in_curve)
		curves.append(curve)
	return curves

def curves_from_rasterized_img_with_slice_points(img, slice_points_dict):
	interior_curves = curves_from_rasterized_img(img)
	new_curves = []
	for curve in interior_curves:
		(X, Y, _) = plotLinePoints(curve, color='r', ciclic=False, writeOrder=False)
		new_curve = []

		for x, y in zip(X, Y):
			x_idx, y_idx = int(x), int(y)
			if (x_idx, y_idx) in slice_points_dict.keys():
				p = slice_points_dict[(x_idx, y_idx)]
				new_curve.append(p)
		new_curves.append(new_curve)
	return new_curves


