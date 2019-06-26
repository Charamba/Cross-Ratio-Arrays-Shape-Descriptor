import matplotlib.pylab as pl
from matplotlib.patches import ConnectionPatch
import time

import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from PIL import Image

from tkinter import *
from tkinter.filedialog import askopenfilename

from PIL import Image, ImageTk
import matplotlib.image as mpimg
import scipy
from scipy import misc


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

templateImage = misc.imread(filename, mode = 'RGB')


filename = askopenfilename(filetypes=[("all files","*"),("Bitmap Files","*.bmp; *.dib"),
										("JPEG", "*.jpg; *.jpe; *.jpeg; *.jfif"),
										("PNG", "*.png"), ("TIFF", "*.tiff; *.tif")])

testImage = misc.imread(filename, mode = 'RGB')


# =============================================================================
# ============================== LOAD IMAGES ==================================
# =============================================================================
fig.canvas.set_window_title('Original Image')
fig.canvas.mpl_connect('key_press_event', press)


# =============================================================================
# ============================== SCAN IMAGES ==================================
# =============================================================================


# Show Template Image
ax1 = fig.add_subplot(1,1,1)

(cols, rows, _) = templateImage.shape

#templateImage.showPatches(fig, ax1)

template_counter = 0
query_counter = 0

c = [255.0, 75.0, 75.0]
for x in range(cols):
	for y in range(rows):
		if templateImage[x][y][0] > 0:
			template_counter += 1
		if testImage[x][y][0] > 0:
			query_counter += 1
		if templateImage[x][y][0] > 0 and testImage[x][y][0] == 0:
		 	templateImage[x][y][0] = c[0]
		 	templateImage[x][y][1] = c[1]
		 	templateImage[x][y][2] = c[2]

plt.imshow(templateImage, interpolation='none', origin='upper', extent=[0, rows, cols, 0])
oclusion_percent = 100*(1.0 - float(query_counter)/template_counter)


# templateImage.show()


# Show Test Image
#ax2 = fig.add_subplot(1,2,2)
print('Oclusion Percent: %.2f', oclusion_percent)
oclusion_percent_str = str("%.2f" % oclusion_percent) + "%"
#plt.suptitle("Percentual de oclusão: " + oclusion_percent_str, fontsize=24, bbox={'facecolor':[c[0]/255,c[1]/255,c[2]/255], 'alpha':1.0, 'pad':10})
(cols, rows, _) = testImage.shape

#ax1.set_title(r'Percentual de oclusão $\approx$' + oclusion_percent_str, fontsize=16, color='r')
ax1.axis('off')


plt.imshow(templateImage, interpolation='none', origin='upper', extent=[0, rows, cols, 0])

#plt.quiver([100], [100], [10], [0], [10], alpha=.5)
#plt.imshow(realImage, interpolation='none', origin='upper', extent=[0, rows, cols, 0])
#testImage.showPatches(fig, ax2)
#ax2.axis('off')
#testImage.show()

plt.savefig("oclusion_percent" + ".png")
#plt.show()