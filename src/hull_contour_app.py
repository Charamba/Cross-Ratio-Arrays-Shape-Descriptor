from tkinter.filedialog import askopenfilename
from hull_contour import *

fig = plt.figure()
# =============================================================================
# ============================== LOAD IMAGES ==================================
# =============================================================================
if __name__ == '__main__':
	filename = ""
	filename = askopenfilename(filetypes=[("all files","*"),("Bitmap Files","*.bmp; *.dib"),
											("JPEG", "*.jpg; *.jpe; *.jpeg; *.jfif"),
											("PNG", "*.png"), ("TIFF", "*.tiff; *.tif")])

	original_image = misc.imread(filename, mode = 'RGB')

	original_image, rows, cols = add_border(original_image)
	#(rows, cols, _) = original_image.shape

	# ax1 = fig.add_subplot(1,2,1)
	# plt.imshow(original_image, interpolation='none', origin='upper', extent=[0, cols, rows, 0])

	cornerPoints = getCornerPoints(original_image, cols, rows)

	cornerPoints_orig = cornerPoints

	cornerPoints = list(set(cornerPoints))
	curves_list = sortContourPoints(cornerPoints, original_image)

	ax2 = fig.add_subplot(1,1,1)
	#original_image = paint_pixels(original_image, rows, cols, cornerPoints_orig, color_array=[100,100,100])
	plt.imshow(original_image, interpolation='none', origin='upper', extent=[0, cols, rows, 0])	


	color_array = ['orange', 'c', 'm', 'g', 'y', 'r', 'b']
	n_curves = float(len(curves_list))

	N = int((n_curves)/(len(color_array)))
	color_array = color_array*(N+1)

	for i, curve in enumerate(curves_list):
		(X, Y, _) = plotLinePoints(curve, color='r', ciclic=False, writeOrder=False)
		interiorCurve_ = interior_curve(curve, original_image)
		(Xint, Yint, _) = plotLinePoints(interiorCurve_, color='r', ciclic=False, writeOrder=False)

		#plt.plot(X, Y, color=color_line, linewidth=3.0)
		color = color_array[i]
		plt.plot(X, Y, color=color, linewidth=3.0)
		plt.plot(Xint, Yint, color=color, linestyle='--', linewidth=1.0)
		#plt.plot([X[0]], [Y[0]], color='r', marker='o')

	plt.show()