import numpy as np
import cv2
import math
import sys
import copy

image_name = sys.argv[1]
path_result = sys.argv[2]
alpha = float(sys.argv[3]) #oclusion area percentage

base_name = image_name.split('.')[0] + '_p' + str(alpha)

l = 102.0 #image side length
pi = math.pi
r = 2.0*l*math.sqrt(alpha*pi)/pi

print("ratio = ", r)

# Create a black image
img_original = cv2.imread(image_name)#np.zeros((512,512,3), np.uint8)

img_00 = copy.deepcopy(img_original)
img_0l = copy.deepcopy(img_original)
img_l0 = copy.deepcopy(img_original)
img_ll = copy.deepcopy(img_original)

l = int(l)
 
# Draw a diagonal blue line with thickness of 5 px
#cv2.line(img,(0,0),(511,511),(255,0,0),5)

cv2.circle(img_00,(0,0), int(r), (0,0,0), -1)
cv2.circle(img_0l,(0,l), int(r), (0,0,0), -1)
cv2.circle(img_l0,(l,0), int(r), (0,0,0), -1)
cv2.circle(img_ll,(l,l), int(r), (0,0,0), -1)

font = cv2.FONT_HERSHEY_SIMPLEX
#cv2.putText(img,'OpenCV',(10,500), font, 4,(255,255,255),2,cv2.LINE_AA)

cv2.imwrite(path_result + base_name + 'c00' +'.bmp', img_00)
cv2.imwrite(path_result + base_name + 'c0l' +'.bmp', img_0l)
cv2.imwrite(path_result + base_name + 'cl0' +'.bmp', img_l0)
cv2.imwrite(path_result + base_name + 'cll' +'.bmp', img_ll)





