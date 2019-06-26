from ShapeDescriptor import *
from Plotter import *
from LineEstimation import *
from VanishPencilsTable import *
import matplotlib.pylab as pl
from matplotlib.patches import ConnectionPatch
import time

import pickle

from PIL import Image
from Scanner import *
from MatchingProcessor import *
#from resizeimage import resizeimage

from hull_contour import curves_from_rasterized_img
from find_homography_distance import calc_hausdorff_distance, calc_homography, calc_homography_distance, transf_points


## Close window and change progress in code
def press(event):
	#print('press', event.key)
	if event.key == 'enter':
		plt.close()
def is_even (a):
	print(a)
	if a%2 != 0:  
		return False
	return True

# def check_dimensions(image):
# 	width, height = image.width, image.height
# 	w = is_even(width)
# 	h = is_even(height)
# 	print(w, h)
# 	if not w and not h:
# 		a = image.resize((width+1, height+1), Image.NEAREST)
# 		#a = resize.resize((width+1, height+1), Image.NEAREST)
# 	elif not w and h: 
# 		a = image.resize((width+1, height+1), Image.NEAREST)
# 	elif w and not h: 
# 		a = image.thumbnail((width+1, height+1), Image.ANTIALIAS)
# 	else: 
# 		a = image
# 	return a
# ================================= FLAGS ===================================== #

# plot: 
showScanRays = False
showMatchRays = False # MATCH RAYS
showTrajectories = False

# scan: tomographic, convex-hull
convex_hull_scan_template = True
convex_hull_scan_test = True

# convex-hull fan-beam parameters
template_nFanBeam = 5
test_nFanBeam = 5
emitter_points_number = 5


# compare features: rays, triple cross feature
compare = True
compareByRays = True

# HULL parameters (contour=False, convex=True)
convex_hull_flag = True

# SHAPE FLAG (symetryc=True, assimetric=False)
symetric_shape_flag = True


# =============================================================================
# ================================= MAIN ======================================
# =============================================================================

fig = plt.figure()

# =============================================================================
# ============================== LOAD IMAGES ==================================
# =============================================================================

filename = askopenfilename(filetypes=[("all files","*"),("Bitmap Files","*.bmp; *.dib"),
										("JPEG", "*.jpg; *.jpe; *.jpeg; *.jfif"),
										("PNG", "*.png"), ("TIFF", "*.tiff; *.tif")])

templateImage = Image(misc.imread(filename, mode = 'RGB'))
name1 = filename
#templateImage.plotPixelGrid()

filename = askopenfilename(filetypes=[("all files","*"),("Bitmap Files","*.bmp; *.dib"),
										("JPEG", "*.jpg; *.jpe; *.jpeg; *.jfif"),
										("PNG", "*.png"), ("TIFF", "*.tiff; *.tif")])
name2 = filename
testImage = Image(misc.imread(filename, mode = 'RGB'))
#testImage.plotPixelGrid()
#fixed = check_dimensions(testImage)
#print(fixed.width,fixed.height)


# =============================================================================
# ============================== LOAD IMAGES ==================================
# =============================================================================
fig.canvas.set_window_title('Original Image')
fig.canvas.mpl_connect('key_press_event', press)


# =============================================================================
# ============================== SCAN IMAGES ==================================
# =============================================================================

templateRays = []
testRays = []

# Scanners
templateScanner = Scanner(templateImage)    
testScanner = Scanner(testImage)

# Convex-Hull
convexHullVertices_orig = templateImage.convex_hull()
original_vertex_number = len(convexHullVertices_orig)

#emitter_points_number = 46


convexHullVertices_orig = testImage.convex_hull()
original_vertex_number = len(convexHullVertices_orig)


convexHullVertices = testImage.convex_hull_add_points(additive_points_number=add_points_number)

if convex_hull_flag:
	print("Hull type: CONVEX")
	print("test:")
	add_points_number = emitter_points_number - original_vertex_number
	if add_points_number <= 0:
		add_points_number = 1

	print("emitter_points_number = ", emitter_points_number)
	print("original_vertex_number = ", original_vertex_number)
	print("add_points_number = ", add_points_number)

	convexHullVertices = testImage.convex_hull_add_points(additive_points_number=add_points_number)#convex_hull(interpoints_sample=0)
	print("(*) Scanning template image...")
	print("- convex hull vertices: ", len(convexHullVertices))
else:
	print("Hull type: CONTOUR")
	print("test:")
	add_points_number = template_nFanBeam - original_vertex_number
	if add_points_number <= 0:
		add_points_number = 1
	print("nFanBeam = ", template_nFanBeam)
	print("original_vertex_number = ", original_vertex_number)
	print("add_points_number = ", add_points_number)

	convexHullVertices = testImage.convex_hull_add_points(additive_points_number=add_points_number)#convex_hull(interpoints_sample=0)
	print("(*) Scanning template image...")
	print("- convex hull vertices: ", len(convexHullVertices))


print("(*) Scanning test image...")
print("- convex hull vertices: ", len(convexHullVertices))


start_time = time.time()

if convex_hull_flag:
	emitter_points = convexHullVertices
	emitter_points_orig = convexHullVertices_orig
	testDescriptor = testScanner.hull_scan(emitter_points, convexHullVertices_orig, fanBeamRays=template_nFanBeam, showTrajectories=showTrajectories, verticeIndexList=[])#0 #45
else:
	emitter_points = testImage.contour_hull(nPoints=emitter_points_number)
	print("n emitter_points = ", len(emitter_points))
	emitter_points_orig = emitter_points
	targetPoints = convexHullVertices
	print("target points = ", len(targetPoints))
	testDescriptor = testScanner.hull_scan_for_contour(emitter_points, emitter_points_orig, targetPoints, fanBeamRays=template_nFanBeam, showTrajectories=showTrajectories, verticeIndexList=[])

testImage.plotLinePoints(emitter_points, color="k", ciclic=True)
testImage.plotLinePoints(emitter_points, color="y", ciclic=True)
testImage.plotLinePoints(emitter_points, color="ko", ciclic=True, writeOrder=False)
testImage.plotLinePoints(emitter_points, color="yo", ciclic=True, writeOrder=False)

end_time = time.time()
time_elapsed = end_time - start_time
print("Scan complete! time: ", time_elapsed)

print("(**) Comparing features...")

matchingProcessor = MatchingProcessor(templateDescriptor, testDescriptor)
test_vertices_rectif = []
templ_edge_pts = []
test_edge_pts = []

if compare:
	start_time = time.time()
	(matchedVerticesPairs, distance_values, mTemplRays, mTestRays) = matchingProcessor.compareByPencils()

	matchedVerticesPairs_json_data = {}
	for (i, (pair, distance)) in enumerate(zip(matchedVerticesPairs, distance_values)):
		matchedVerticesPairs_json_data[i] = (pair, distance)

	name1 = name1.split("/")[-1].split(".")[0]
	name2 = name2.split("/")[-1].split(".")[0]
	str_filename = name1 + "-" + name2 + "-" + str(template_nFanBeam) + ".txt"
	with open(str_filename, 'wb') as outfile:  
		pickle.dump(matchedVerticesPairs_json_data, outfile)

	(matchedVerticesPairs, distanceValues) = fanbeams_threshold_distance(matchedVerticesPairs, distance_values)

	end_time = time.time()
	time_elapsed = end_time - start_time
	print("Comparation complete! time: ", time_elapsed)
	print("--------------------------------")
	print("N. Rays per fan-beam: ", template_nFanBeam)
	print("Template image dimensions:", templateImage.getShape())
	print("Query image dimensions:", testImage.getShape())

	template_hull_points = templateDescriptor.hullVertices
	query_hull_points = testDescriptor.hullVertices

	template_pts = np.array([[templ_pt.x, templ_pt.y] for (templ_pt, _) in matchedVerticesPairs])
	test_pts     = np.array([[test_pt.x, test_pt.y] for (_, test_pt) in matchedVerticesPairs])

	## Estimando homografia e erro de transferência pelos vértices associados
	
	(dist_hull, new_template_pts, new_test_pts, hinv) = calc_homography_distance(template_pts, test_pts)
	print("percentual transfer error distance (hull vertices): ", dist_hull)
	# test_pts = [R2_Point(point[0],point[1]) for point in test_pts]
	# test_vertices_rectif = transf_points(hinv, new_test_pts)
	# test_vertices_rectif = [R2_Point(point[0],point[1]) for point in test_vertices_rectif]
	
	matchedVerticesPairs = []

	# new_template_pts = template_pts
	# new_test_pts = test_pts
	for templ_pt, test_pt in zip(new_template_pts, new_test_pts):
		templ_pt = R2_Point(templ_pt[0], templ_pt[1])
		test_pt  = R2_Point(test_pt[0], test_pt[1])
		matchedVerticesPairs.append((templ_pt, test_pt))

	for templRay, testRay in zip(mTemplRays, mTestRays):
		templ_edge_pts = templ_edge_pts + templRay.edgePoints
		test_edge_pts  = test_edge_pts + testRay.edgePoints

	templ_edge_pts = np.array([[templ_pt.x, templ_pt.y] for templ_pt in templ_edge_pts])
	test_edge_pts  = np.array([[test_pt.x, test_pt.y] for test_pt in test_edge_pts])

	(dist_edge, new_edge_templ_pts, new_edge_test_pts, hinv) = calc_homography_distance(templ_edge_pts, test_edge_pts)

	print("percentual transfer error distance (edge points): ", dist_edge)
	if showMatchRays:
		for templRay in mTemplRays:
			#templateImage.plotLinePoints(templRay.edgePoints, color="k", correction=True)
			templateImage.plotLinePoints(templRay.edgePoints, color="c", correction=True)
			#templateImage.plotLinePoints(templRay.edgePoints, color="ko", correction=True, writeOrder=False)
			templateImage.plotLinePoints(templRay.edgePoints, color="co", correction=True, writeOrder=False)

		for testRay in mTestRays:
			#testImage.plotLinePoints(testRay.edgePoints, color="k", correction=True)
			testImage.plotLinePoints(testRay.edgePoints, color="c", correction=True)
			#testImage.plotLinePoints(testRay.edgePoints, color="ko", correction=True, writeOrder=False)
			testImage.plotLinePoints(testRay.edgePoints, color="co", correction=True, writeOrder=False)




#print("Finish!")

print("Ploting...")


n = len(matchedVerticesPairs)
colors = plt.cm.rainbow(np.linspace(0,1,n))
print("matchedVerticesPairs = ", matchedVerticesPairs)

if showScanRays:
	for templateRay in templateDescriptor.raysTable.getValues():
		templateImage.plotLinePoints(templateRay.edgePoints, color="r", correction=True)
		templateImage.plotLinePoints(templateRay.edgePoints, color="ro", correction=True, writeOrder=False)

	for testRay in testDescriptor.raysTable.getValues():
		testImage.plotLinePoints(testRay.edgePoints, color="r", correction=True)
		testImage.plotLinePoints(testRay.edgePoints, color="ro", correction=True, writeOrder=False)

# Show Template Image
ax1 = fig.add_subplot(1,2,1)
#ax1.set_title('Template Image')
(cols, rows) = templateImage.getShape()
plt.imshow(templateImage.image, interpolation='none', origin='upper', extent=[0, rows, cols, 0])
#templateImage.showPatches(fig, ax1)
ax1.axis('off')
templateImage.show()


dist_hull_str = '%.4f' % dist_hull
dist_edge_str = '%.4f' % dist_edge

(xT, yT) = templateImage.getShape()
(xQ, yQ) = testImage.getShape()

#x_label, y_label = -xT*0.15, max(yT, yQ)

x_label, y_label = -5, .5
plt.title("Hull points distance = " + dist_hull_str, fontsize=15, color='red')



# Show Test Image
ax2 = fig.add_subplot(1,2,2)
#ax2.set_title('Query Image')
(cols, rows) = testImage.getShape()

plt.imshow(testImage.image, interpolation='none', origin='upper', extent=[0, rows, cols, 0])
#plt.imshow(realImage, interpolation='none', origin='upper', extent=[0, rows, cols, 0])
#testImage.showPatches(fig, ax2)
ax2.axis('off')
testImage.show()

plt.title("Edge points distance = " + dist_edge_str, fontsize=15, color='blue')

for i,(t_vertex, q_vertex) in enumerate(matchedVerticesPairs):

	(xt, yt) = t_vertex.toTuple()
	(xq, yq) = q_vertex.toTuple()

	xyT = templateImage.toImageCoordinate(xt, yt)
	xyQ = testImage.toImageCoordinate(xq, yq)

	con = ConnectionPatch(xyA=xyQ, xyB=xyT, coordsA="data", coordsB="data",
                		  axesA=ax2, axesB=ax1, color=colors[i], linewidth=2.0)
	ax2.add_artist(con)


plt.show()