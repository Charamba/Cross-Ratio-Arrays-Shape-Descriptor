from Point import *

def calc_s(x, y, cosT, sinT):
	s = x*cosT + y*sinT
	return s

def calc_sinT(x0, xf, lamb):
	return (xf - x0)/lamb

def calc_cosT(y0, yf, lamb):
	return - (yf - y0)/lamb

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

def triangleArea(A, B, C):
	return abs((A.x*(B.y - C.y) + B.x*(C.y - A.y) + C.x*(A.y - B.y))/2.0)
