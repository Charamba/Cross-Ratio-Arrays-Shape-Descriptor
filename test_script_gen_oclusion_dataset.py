import numpy as np
import cv2
import math
import sys
import copy
import os


def generate_oclusion_dataset(template_folder, image_name, path_result, alpha):
    base_name = image_name.split('.')[0] + '_p' + str(alpha)

    l = 102.0 #image side length
    pi = math.pi
    r = 2.0*l*math.sqrt(alpha*pi)/pi

    #print("ratio = ", r)

    # Create a black image
    img_original = cv2.imread(template_folder + image_name)#np.zeros((512,512,3), np.uint8)
    img_00 = copy.deepcopy(img_original)
    img_0l = copy.deepcopy(img_original)
    img_l0 = copy.deepcopy(img_original)
    img_ll = copy.deepcopy(img_original)

    l = int(l)

    cv2.circle(img_00,(0,0), int(r), (0,0,0), -1)
    cv2.circle(img_0l,(0,l), int(r), (0,0,0), -1)
    cv2.circle(img_l0,(l,0), int(r), (0,0,0), -1)
    cv2.circle(img_ll,(l,l), int(r), (0,0,0), -1)

    cv2.imwrite(path_result + base_name + 'c00' +'.bmp', img_00)
    cv2.imwrite(path_result + base_name + 'c0l' +'.bmp', img_0l)
    cv2.imwrite(path_result + base_name + 'cl0' +'.bmp', img_l0)
    cv2.imwrite(path_result + base_name + 'cll' +'.bmp', img_ll)

# MAIN
template_folder = sys.argv[1]
path_result = sys.argv[2]
alpha = float(sys.argv[3]) #oclusion area percentage

img_names = os.listdir(template_folder)

for img_name in img_names:
    generate_oclusion_dataset(template_folder, img_name, path_result, alpha)





