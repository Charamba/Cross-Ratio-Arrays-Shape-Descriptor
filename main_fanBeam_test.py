from Image import *
from Scanner import *
from MatchingProcessor import *
from scipy import misc
from Utils import *


# generate segments:
def genLines(points):
	lines = []
	for i in range(-1, len(points)):
		if i+1 < len(points):
			p1 = points[i].toP2_Point()
			p2 = points[i+1].toP2_Point()
			line = p1.cross(p2)
			lines.append((line, p1, p2))

	return lines

def genNewVertices(lines):
	newVertices = []
	linesPairs = itertools.combinations(lines, 2)

	for (line1, line2) in linesPairs:
		(l1, p1A, p1B) = line1
		(l2, p2A, p2B) = line2
		if l1.z != 0 and l2.z != 0:
			newVertex = l1.cross(l2)
			if newVertex.z != 0:
				newVertices = newVertices + [p1A, newVertex.toR2_Point(), p2A]
				#newVertices.append(newVertex.toR2_Point())

	return newVertices#convex_hull_vertices(newVertices)


## Close window and change progress in code
def press(event):
	#print('press', event.key)
	if event.key == 'enter':
		plt.close()

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


filename = askopenfilename(filetypes=[("all files","*"),("Bitmap Files","*.bmp; *.dib"),
										("JPEG", "*.jpg; *.jpe; *.jpeg; *.jfif"),
										("PNG", "*.png"), ("TIFF", "*.tiff; *.tif")])

testImage = Image(misc.imread(filename, mode = 'RGB'))


# =============================================================================
# ============================== LOAD IMAGES ==================================
# =============================================================================
fig.canvas.set_window_title('Original Image')
fig.canvas.mpl_connect('key_press_event', press)


# CONVEX HULL

templateVerticesLv1 = templateImage.convex_hull()
print("**** test len(templateVerticesLv1): ", len(templateVerticesLv1))
templateImage.plotLinePoints(templateVerticesLv1, color="yo", ciclic=True, writeOrder=False)
templateImage.plotLinePoints(templateVerticesLv1, color="y", ciclic=True)


testVerticesLv1  = testImage.convex_hull()
print("**** test len(testVerticesLv1): ", len(testVerticesLv1))
testImage.plotLinePoints(testVerticesLv1, color="yo", ciclic=True, writeOrder=False)
testImage.plotLinePoints(testVerticesLv1, color="y", ciclic=True)



templateVerticesLv2 = genNewVertices(genLines(templateVerticesLv1))
templateImage.plotLinePoints(templateVerticesLv2, color="ro", ciclic=True, writeOrder=False)
templateImage.plotLinePoints(templateVerticesLv2, color="r", ciclic=True)


hullTemplateV2 = convex_hull_vertices(templateVerticesLv2)
templateImage.plotLinePoints(hullTemplateV2, color="mo", ciclic=True)

templateImage.plotLinePoints(templateVerticesLv2, color="r", ciclic=True)

testVerticesLv2 = genNewVertices(genLines(testVerticesLv1))
testImage.plotLinePoints(testVerticesLv2, color="ro", ciclic=True, writeOrder=False)
testImage.plotLinePoints(testVerticesLv2, color="r", ciclic=True)

hullTestV2 = convex_hull_vertices(testVerticesLv2)
testImage.plotLinePoints(hullTestV2, color="mo", ciclic=True)



# Show Template Image
ax = fig.add_subplot(1,2,1)
ax.set_title('Template Image')
(cols, rows) = templateImage.getShape()
plt.imshow(templateImage.image, interpolation='none', origin='upper', extent=[0, rows, cols, 0])
templateImage.showPatches(fig, ax)
templateImage.show()

# Show Test Image
ax = fig.add_subplot(1,2,2)
ax.set_title('Test Image')
(cols, rows) = testImage.getShape()
plt.imshow(testImage.image, interpolation='none', origin='upper', extent=[0, rows, cols, 0])
testImage.showPatches(fig, ax)
testImage.show()

plt.show()