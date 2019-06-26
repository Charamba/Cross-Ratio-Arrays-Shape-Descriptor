from CrossRatio import *
from CrossRadonTransform import *
from HoughTransform import *
from ShapeDescriptor import *
from MatchRaysPairs import *
from Plotter import *
#from ransac import *
from LineEstimation import *
from HorizonLine import *
from VanishPencilsTable import *

from Image import *
from Scanner import *
from MatchingProcessor import *

# ================================= Transform =================================
def transfProj(P):
	(x, y) = P.toTuple() # R2_Point
	z = 1
	P_ = P2_Point(400*x + 720*y, 1440*y, -x + 6*y + 600*z)
	return P_.toR2_Point()


A = R2_Point(2, 5)
B = R2_Point(5, 1)
C = R2_Point(3, 5)
D = R2_Point(1, 1)
E = R2_Point(0.5, 2)
F = R2_Point(6, 1)


r1 = RayDescriptor(0, 0.0, [A, B], calcCrossRatio=False)
r2 = RayDescriptor(0, 0.1, [C, D], calcCrossRatio=False)
r3 = RayDescriptor(0, 0.2, [E, F], calcCrossRatio=False)





print("F(A, B, C, D, E) = ", invariant5CrossRatio(A, B, C, D, E))
print("F(A, B, D, C, E) = ", invariant5CrossRatio(A, B, D, C, E))

A_ = transfProj(A)
B_ = transfProj(B)
C_ = transfProj(C)
D_ = transfProj(D)
E_ = transfProj(E)





print("F(A_, B_, C_, D_, E_) = ", invariant5CrossRatio(A_, B_, C_, D_, E_))
print("F(A_, D_, E_, B_, C_) = ", invariant5CrossRatio(A_, D_, E_, B_, C_))



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

