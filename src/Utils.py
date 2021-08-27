from Point import *
from scipy.spatial import ConvexHull

def unpackPoint(P):
	if type(P) == P2_Point:
		#P.normalize()
		(x, y, z) = P.toTuple()
	else:
		(x, y) = P.toTuple()
		z = 1
	return (x, y, z)

def sortPoints(points, P0):
	# SORT Points by distance to P0
	return [Pi for (Pi, d) in sorted([(Pi, P0.euclideanDistance(Pi)) for Pi in points], key=lambda t: t[1])]

def removeDuplicates(seq):
	seen = set()
	seen_add = seen.add
	return [x for x in seq if not (x in seen or seen_add(x))]

def pairsCombinations(elements):
	pairs = []
	for i in range(0, len(elements)):
		pairs += [(elements[i],e2) for e2 in elements[i+1:]]
	return pairs

def inBoundingBox(P, P1, P2):
	# print("P1 = ", P1)
	# print("P2 = ", P2)

	xMin = min(P1.x, P2.x)
	xMax = max(P1.x, P2.x)

	yMin = min(P1.y, P2.y)
	yMax = max(P1.y, P2.y)
	(x, y, _) = unpackPoint(P)
	# print("xMin = ", xMin)
	# print("x = ", x)
	# print("xMax = ", xMax)
	# print("yMin = ", yMin)
	# print("y = ", y)
	# print("yMax = ", yMax)
	tol = 1E-9
	#(math.floor(xMin) <= math.floor(x) <= math.floor(xMax)) and (math.floor(yMin) <= math.floor(y) <= math.floor(yMax))
	return (yMin - tol <= y <= yMax + tol) and (xMin - tol <= x <= xMax + tol)
	#return (xMin - tol <= x <= xMax + tol) or (yMin - tol <= y <= yMax + tol)

def convex_hull_vertices(points, interpoints_sample=0):
	wPListofList = []
	for point in points:
		(x, y) = point.toTuple()
		wPListofList.append([x, y])

	hull = ConvexHull(wPListofList, incremental=True)
	vertices = []

	for vIndex in hull.vertices:
		[x, y] = wPListofList[vIndex]
		vertices.append(R2_Point(x, y))

	hull_points = []
	if interpoints_sample > 0:
		vertices += [vertices[0]]
		for i in range(0, len(vertices) - 1):
			p0 = vertices[i]
			pf = vertices[i+1]
			vd = pf - p0
			#vd.r2Normalize()

			for l in np.arange(0, 1, 1/interpoints_sample):
				pl = p0 + l*vd
				hull_points.append(pl)

	else:
		hull_points = vertices

	return hull_points


def convex_hull_vertices_add_points(points, additive_points_number=0):
	wPListofList = []
	for point in points:
		(x, y) = point.toTuple()
		wPListofList.append([x, y])

	hull = ConvexHull(wPListofList, incremental=True)
	vertices = []
	length_Hull = 0
	for vIndex in hull.vertices:
		[x, y] = wPListofList[vIndex]
		vertices.append(R2_Point(x, y))

	for i in range(0, len(vertices) - 1):
		p0 = vertices[i]
		pf = vertices[i+1]
		vd = pf - p0
		length_Hull += vd.length()

	p0 = vertices[-1]
	pf = vertices[0]
	vd = pf - p0
	length_Hull += vd.length()

	hull_points = []
	residue_length = 0
	partial_points_number = 0
	if additive_points_number > 0:
		step_size = length_Hull/additive_points_number
		vertices += [vertices[0]]
		for i in range(0, len(vertices) - 1):
			p0 = vertices[i]
			pf = vertices[i+1]
			vd = pf - p0
			seg_len = vd.length()
			#vd.r2Normalize()

			n_inter_points = seg_len/step_size
			partial_points_number += n_inter_points
			if n_inter_points <= 1.0:
				residue_length += seg_len

			# for l in np.arange(0, 1, 1/n_inter_points):
			# 	pl = p0 + l*vd
			# 	hull_points.append(pl)

		residue_points_number = residue_length/step_size
		step_size = length_Hull/(additive_points_number + residue_points_number)
		for i in range(0, len(vertices) - 1):
			p0 = vertices[i]
			pf = vertices[i+1]
			vd = pf - p0
			seg_len = vd.length()
			#vd.r2Normalize()

			n_inter_points = seg_len/step_size

			for l in np.arange(0, 1, 1/n_inter_points):
				pl = p0 + l*vd
				hull_points.append(pl)

	else:
		hull_points = vertices
	#print("residue_length = ", residue_length)
	return hull_points

def convex_hull_vertices_add_points_new(points, additive_points_number=0):
	wPListofList = []
	for point in points:
		(x, y) = point.toTuple()
		wPListofList.append([x, y])

	hull = ConvexHull(wPListofList, incremental=True)
	vertices = []
	length_Hull = 0
	for vIndex in hull.vertices:
		[x, y] = wPListofList[vIndex]
		vertices.append(R2_Point(x, y))

	for i in range(0, len(vertices) - 1):
		p0 = vertices[i]
		pf = vertices[i+1]
		vd = pf - p0
		length_Hull += vd.length()

	p0 = vertices[-1]
	pf = vertices[0]
	vd = pf - p0
	length_Hull += vd.length()

	hull_points = []
	if additive_points_number > 0:
		step_size = length_Hull/additive_points_number
		vertices += [vertices[0]]
		for i in range(0, len(vertices) - 1):
			p0 = vertices[i]
			pf = vertices[i+1]
			vd = pf - p0
			seg_len = vd.length()
			#vd.r2Normalize()

			n_inter_points = seg_len/step_size
			for l in np.arange(0, 1, 1/n_inter_points):
				pl = p0 + l*vd
				hull_points.append(pl)

		for i in range(0, len(vertices) - 1):
			p0 = vertices[i]
			pf = vertices[i+1]
			vd = pf - p0
			seg_len = vd.length()
			#vd.r2Normalize()

			n_inter_points = seg_len/step_size
			for l in np.arange(0, 1, 1/n_inter_points):
				pl = p0 + l*vd
				hull_points.append(pl)



	else:
		hull_points = vertices

	return hull_points
	
