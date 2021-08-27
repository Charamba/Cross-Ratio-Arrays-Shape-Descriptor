import numpy.linalg
from Point import *
#from bresenham import bresenham

def random_partition(n,n_data):
	"""return n random rows of data (and also the other len(data)-n rows)"""
	all_idxs = numpy.arange( n_data )
	numpy.random.shuffle(all_idxs)
	idxs1 = all_idxs[:n]
	idxs2 = all_idxs[n:]
	return idxs1.tolist(), idxs2.tolist()

def leastSquares(points,  weighted=False):
	x_d = []
	y_d = []
	for point in points:
		(x, y) = point.toTuple()
		if weighted:
			w = point.w
			x_d += [x]*w
			y_d += [y]*w
		else:
			x_d.append(x)
			y_d.append(y)

	n=len(x_d)

	B=numpy.array(y_d)
	A=numpy.array(([[x_d[j], 1] for j in range(n)]))
	X=numpy.linalg.lstsq(A,B)[0]
	a=X[0]; b=X[1]
	#print("Line is: y=",a,"x+",b)

	Xplot = []
	Yplot = []

	x = min(x_d)
	y = a*x + b
	Xplot.append(x)
	Yplot.append(y)

	x = max(x_d)
	y = a*x + b
	Xplot.append(x)
	Yplot.append(y)
	return (Xplot, Yplot, a, b)

def calcResidue(a, b, xi, yi):
	yi_ = a*xi + b
	return (yi - yi_)

def calcTotalError(points, a, b):
	totalError = 0
	
	n = len(points)
	for point in points:
		(xi, yi) = point.toTuple()
		ei = calcResidue(a, b, xi, yi)
		totalError += ei*ei
		
	return totalError

class LinearLeastSquaresModel:
	"""linear system solved using linear least squares

	This class serves as an example that fulfills the model interface
	needed by the ransac() function.
	
	"""
	#def __init__(self):
	def fit(self, data, weighted=False):
		return leastSquares(data, weighted=weighted)
	def get_error( self, data, model):
		(_, _, a, b) = model
		totalError = calcTotalError(data, a, b)
		n = len(data)
		err_per_point = totalError/n
		return err_per_point

def getDataByIndex(data, idxs):
	returnData = []

	if type(idxs) != list:
		print("idxs = ", idxs)
		print("type(idxs) = ", type(idxs))
		returnData = [data[idxs]]
	else:
		for i in idxs:
			returnData.append(data[i])
		#if type(returnData) != type(list):
		#   returnData = [returnData]
	return returnData

def ransac(data,model,n,k,t,d,debug=False,return_all=False, weighted=False, random=True):
	"""fit model parameters to data using the RANSAC algorithm
	
This implementation written from pseudocode found at
http://en.wikipedia.org/w/index.php?title=RANSAC&oldid=116358182

{{{
Given:
	data - a set of observed data points
	model - a model that can be fitted to data points
	n - the minimum number of data values required to fit the model
	k - the maximum number of iterations allowed in the algorithm
	t - a threshold value for determining when a data point fits a model
	d - the number of close data values required to assert that a model fits well to data
Return:
	bestfit - model parameters which best fit the data (or nil if no good model is found)
iterations = 0
bestfit = nil
besterr = something really large
while iterations < k {
	maybeinliers = n randomly selected values from data
	maybemodel = model parameters fitted to maybeinliers
	alsoinliers = empty set
	for every point in data not in maybeinliers {
		if point fits maybemodel with an error smaller than t
			 add point to alsoinliers
	}
	if the number of elements in alsoinliers is > d {
		% this implies that we may have found a good model
		% now test how good it is
		bettermodel = model parameters fitted to all points in maybeinliers and alsoinliers
		thiserr = a measure of how well model fits these points
		if thiserr < besterr {
			bestfit = bettermodel
			besterr = thiserr
		}
	}
	increment iterations
}
return bestfit
}}}
"""
	alsoinliers = []
	iterations = 0
	bestfit = None
	besterr = numpy.inf
	best_inlier_idxs = None
	while iterations < k:
		maybeinliers = data
		test_points = data
		maybe_idxs, test_idxs = random_partition(n,len(data))
		if random:
			# print("maybe_idxs = ", maybe_idxs)
			# print("test_idxs = ", test_idxs)
			maybeinliers = getDataByIndex(data, maybe_idxs)
			test_points = getDataByIndex(data, test_idxs)
		#print("maybeinliers = ", maybeinliers)

		maybemodel = model.fit(maybeinliers, weighted=weighted)
		test_err = model.get_error( test_points, maybemodel)
		#print("test_err = ", test_err)
		#print("len test_idxs = ", len(test_idxs))
		also_idxs = test_idxs[test_err < t] # select indices of rows with accepted points
		#print("also_idxs = ", also_idxs)

		#print("also_idxs = ", also_idxs)
		#alsoinliers = getDataByIndex(data, also_idxs)
		alsoinliers = []
		test_err = 0
		for point in test_points:
			(_, _, a, b) = maybemodel
			(xi, yi) = point.toTuple()
			ei = abs(calcResidue(a, b, xi, yi))
			#ei = ei*ei
			test_err += ei
			if ei < t:
				alsoinliers.append(point)

		#if alsoinliers != list:
		#	alsoinliers = [alsoinliers]

		if debug:
			print ('test_err.min()',test_err.min())
			print ('test_err.max()',test_err.max())
			print ('numpy.mean(test_err)',numpy.mean(test_err))
			#print("alsoinliers = ", alsoinliers)
			print ('iteration %d:len(alsoinliers) = %d'%(iterations,len(alsoinliers)))
		if len(alsoinliers) > d:
			betterdata = maybeinliers + alsoinliers  #numpy.concatenate( (maybeinliers, alsoinliers) )
			bettermodel = model.fit(betterdata)
			better_errs = model.get_error( betterdata, bettermodel)

			thiserr = 0
			for point in betterdata:
				(_, _, a, b) = bettermodel
				(xi, yi) = point.toTuple()
				ei = abs(calcResidue(a, b, xi, yi))
				#ei = ei*ei
				thiserr += ei
				#if ei < t:
				#	alsoinliers.append(point)

			thiserr = numpy.mean(thiserr)#( better_errs )
			if thiserr < besterr:
				bestfit = bettermodel
				besterr = thiserr
				#best_inlier_idxs = numpy.concatenate( (maybe_idxs, also_idxs) )
				#print("best_inlier_idxs = ", best_inlier_idxs)
		iterations+=1
	if bestfit is None:
		print("did not meet fit acceptance criteria")
		return None
		#raise ValueError("did not meet fit acceptance criteria")

	alsoinliers = sorted(alsoinliers, key=lambda p: p.x)
	# print("alsoinliers: ", alsoinliers)

	return alsoinliers

	if return_all:
		return bestfit#, {'inliers':best_inlier_idxs}
	else:
		return bestfit