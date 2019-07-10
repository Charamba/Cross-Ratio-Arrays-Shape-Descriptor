import numpy as np
import cv2
import math
import sys
import copy
import os

def get_white_pixel_corners(img):
    dimensions = img.shape
    whitePoints = []
    # height, width, number of channels in image
    height = img.shape[0]
    width = img.shape[1]
    channels = img.shape[2]

    for i in range(0,height):
        for j in range(0,width):
            pxVal = img[i][j][0]
            if pxVal > 0:
                xCoord = j 
                yCoord = i 
                whitePoints.append((xCoord,yCoord))
                whitePoints.append((xCoord+1,yCoord))
                whitePoints.append((xCoord,yCoord+1))
                whitePoints.append((xCoord+1,yCoord+1))
    return whitePoints

def get_white_pixels(img):
    dimensions = img.shape
    whitePoints = []
    # height, width, number of channels in image
    height = img.shape[0]
    width = img.shape[1]
    channels = img.shape[2]

    for i in range(0,height):
        for j in range(0,width):
            pxVal = img[i][j][0]
            if pxVal > 0:
                xCoord = j 
                yCoord = i 
                whitePoints.append((xCoord,yCoord))

    return whitePoints

def paint_pixel(img, i, j, color):
    (h, w, channels) = img.shape
    for i in range(0, h):
        for j in range(0, w):
            img[i][j] = color

def gen_oclusion(img, nPixels, type="UP"):
    (h, w, channels) = img.shape
    I_range = list(range(0, h))
    J_range = list(range(0, w))
    
    if type == "DOWN":
        I_range.reverse()
    elif type == "RIGHT":
        J_range.reverse()

    count = 0
    if type == "UP" or type == "DOWN":
        for i in I_range:
            for j in J_range:
                pxVal = img[i][j][0] 
                if pxVal > 0 and count < nPixels:
                    count+=1
                    img[i][j] = (0,0,0)
    elif type == "LEFT" or type == "RIGHT":
        for j in J_range:
            for i in I_range:
                pxVal = img[i][j][0] 
                if pxVal > 0 and count < nPixels:
                    count+=1
                    img[i][j] = (0,0,0)
                    

def bbox_object(img):

    corners = get_white_pixel_corners(img)

    xMax = float('-inf')
    yMax = float('-inf')
    xMin = float('inf')
    yMin = float('inf')

    dimensions = img.shape
    # height, width, number of channels in image
    height = img.shape[0]
    width = img.shape[1]
    channels = img.shape[2]

    for c in corners:
        (xc, yc) = c
        if xMax < xc:
            xMax = xc
        if xMin > xc:
            xMin = xc
        
        if yMax < yc:
            yMax = yc
        if yMin > yc:
            yMin = yc

    return xMin, yMin, xMax, yMax



def generate_oclusion_dataset(template_folder, image_name, path_result, alpha):

    str_alpha = str(int(alpha*100))
    base_name = image_name.split('.')[0] + '_' + str_alpha + 'pc'

    #print("ratio = ", r)

    # Create a black image
    img_original = cv2.imread(template_folder + image_name)#np.zeros((512,512,3), np.uint8)
    img_UP = copy.deepcopy(img_original)
    img_DOWN = copy.deepcopy(img_original)
    img_LEFT = copy.deepcopy(img_original)
    img_RIGHT = copy.deepcopy(img_original)

    (xMin, yMin, xMax, yMax) = bbox_object(img_original)

    # l1 = xMax - xMin
    # l2 = yMax - yMin
    # A = l1*l2

    # lmin = min(l1,l2)

    # #l = 102.0 #image side length
    # pi = math.pi
    # r = 2.0*math.sqrt(A*alpha*pi)/pi

    # if r > lmin:
    #     print("image_name: " + image_name + " r = " + str(r) + ", lmin = " + str(lmin))

    # cv2.circle(img_original,(xMin,yMin), int(r), (255,0,0), -1)
    # cv2.circle(img_original,(xMin,yMax), int(r), (255,0,0), -1)
    # cv2.circle(img_original,(xMax,yMin), int(r), (255,0,0), -1)
    # cv2.circle(img_original,(xMax,yMax), int(r), (255,0,0), -1)

    # cv2.imwrite(path_result + base_name + 'c00' +'.bmp', img_original)
    # cv2.imwrite(path_result + base_name + 'c0l' +'.bmp', img_original)
    # cv2.imwrite(path_result + base_name + 'cl0' +'.bmp', img_original)
    # cv2.imwrite(path_result + base_name + 'cll' +'.bmp', img_original)

    totalWhitePìxels = len(get_white_pixels(img_original))
    nMissedPixels = alpha*totalWhitePìxels

    # print("totalWhitePìxels = ", totalWhitePìxels)
    # print("nPixels = ", nMissedPixels)

    gen_oclusion(img_UP,   nMissedPixels)
    gen_oclusion(img_DOWN, nMissedPixels, "DOWN")
    gen_oclusion(img_LEFT, nMissedPixels, "LEFT")
    gen_oclusion(img_RIGHT, nMissedPixels,"RIGHT")

    #cv2.circle(img_00,(xMin,yMin), int(r), (0,0,0), -1)
    # cv2.circle(img_0l,(xMin,yMax), int(r), (0,0,0), -1)
    # cv2.circle(img_l0,(xMax,yMin), int(r), (0,0,0), -1)
    # cv2.circle(img_ll,(xMax,yMax), int(r), (0,0,0), -1)

    cv2.imwrite(path_result + base_name + 'UP' +'.bmp', img_UP)
    cv2.imwrite(path_result + base_name + 'DOWN' +'.bmp', img_DOWN)
    cv2.imwrite(path_result + base_name + 'LEFT' +'.bmp', img_LEFT)
    cv2.imwrite(path_result + base_name + 'RIGHT' +'.bmp', img_RIGHT)

# MAIN
argc = len(sys.argv)

if argc < 4:
    print("Correct use:")
    print(">> python3 test_script_gen_oclusion_dataset.py <template_folder> <result_folder> <alpha>")
else:
    template_folder = sys.argv[1]
    path_result = sys.argv[2]
    alpha = float(sys.argv[3]) #oclusion area percentage
    

    img_names = os.listdir(template_folder)

    for img_name in img_names:
        generate_oclusion_dataset(template_folder, img_name, path_result, alpha)





