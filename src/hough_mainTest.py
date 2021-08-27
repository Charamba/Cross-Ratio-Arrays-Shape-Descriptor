from HoughTransform import *
from Plotter import *

## Close window and change progress in code
def press(event):
	#print('press', event.key)
	if event.key == 'enter':
		plt.close()

# =============================================================================
# ================================= MAIN ======================================
# =============================================================================

fig = plt.figure()


# ########## PLOT ##########
ax = fig.add_subplot(1,2,1)
ax.set_title('Template Image')

plt.imshow(templateImage)