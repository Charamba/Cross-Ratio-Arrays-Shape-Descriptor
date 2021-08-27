# For image aquisition
import matplotlib.pyplot as plt
from tkinter import *
from Point import *
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk
import matplotlib.image as mpimg
from scipy import misc
from scipy import sparse
import math
import numpy as np

class HoughLines:
    def __init__(self):
        self.lines = {}
    def getLines(self):
        return self.lines.items()
    def getValues(self):
        return self.lines.values()
    def updateLine(self, key, value):
        if key in self.lines:
            self.lines[key].append(value)
        else:
            self.lines.update({key:[value]})

def image_houghTransform(image):
    # Compute the number of rows and columns in the image
    (row_num, col_num, _) = image.shape
    # Initialize the transformed image
    d = int(math.sqrt(row_num*row_num + col_num*col_num))
    tMax = d
    shapeResult = (d, d, 3)
    image_ = np.zeros(shapeResult, dtype=np.uint8)
    #print("d = ", d)
    (dimx, dimy, _) = image_.shape
    print("size img = (%d, %d)" %(dimx, dimy))

    # Compute transformed image
    for t in range(0, tMax):
        for x in range(0, col_num):
            for y in range(0, row_num):
                # Compute the point p in the original image which is mapped to the
                # (row, col) pixel in the transformed image
                #p = Point(col, row, 1)
                #p.to_img_coord(col_num, row_num)
                #p.transform(H)
                #p.normalize()
                # Get the (row_px, col_px) pixel coordinates of p
                #(row_px, col_px)  = p.get_pixel_coord(col_num, row_num)
                t_rad = (math.pi*t)/tMax
                r = int(x*math.cos(t_rad) + y*math.sin(t_rad))
                # If the pixel is part of the image, get it's color
                pxValue = image[x][y][0]
                if ((0 < r < d) and (pxValue > 0)):
                    # Original image
                    #pixelValue = image[x][y]
                    # Rectified image
                    #print("(r, t) = (%d, %d)"%(r, t))
                    image_[r][t][0] += 50 #R
                    image_[r][t][1] += 50 #G
                    image_[r][t][2] += 50 #B

    return image_

def points_houghTransform(points,  x0=0, y0=0, weighted=False):
    # Compute the number of rows and columns in the image
    minX = sys.float_info.max
    maxX = sys.float_info.min
    minY = sys.float_info.max
    maxY = sys.float_info.min

    for point in points:
        (x, y) = point.toTuple()
        if x < minX:
            minX = x
        if x > maxX:
            maxX = x
        if y < minY:
            minY = y
        if y > maxY:
            maxY = y
    #pMax = R2_Point(maxX, maxY)
    #pMin = R2_Point(minX, minY)

    #d = pMin.euclideanDistance(pMax)
    #print("pMax = ", pMax)
    #print("pMin = ", pMin)
    #d += 1.0
    #print("d = ", d)
    d = 1000000#int(d)
    #d = int(math.sqrt(row_num*row_num + col_num*col_num))
    tMax = 360

    houghSpace = sparse.lil_matrix((d,d))
    houghLines = HoughLines()

    #print("d = ", d)
    # Compute transformed image
    for t in range(0, tMax):
        for point in points:
            (x, y) = point.toTuple()


            #x = int(x)
            #y = int(y)

            # Compute the point p in the original image which is mapped to the
            # (row, col) pixel in the transformed image
            #p = Point(col, row, 1)
            #p.to_img_coord(col_num, row_num)
            #p.transform(H)
            #p.normalize()
            # Get the (row_px, col_px) pixel coordinates of p
            #(row_px, col_px)  = p.get_pixel_coord(col_num, row_num)
            t_rad = (2*math.pi*t)/tMax
            r = int(x*math.cos(t_rad) + y*math.sin(t_rad))
            # If the pixel is part of the image, get it's color
            if (0 < r < d):
                # Original image
                # pixelValue = image[x][y]
                # Rectified image
                # print("(r, t) = (%d, %d)"%(r, t))
                w = 1.0
                if weighted:
                    w = point.w

                houghSpace[r, t] += w
                if weighted:
                    houghLines.updateLine((r,t), WeightedPoint(x, y, w))
                else:
                    houghLines.updateLine((r,t), R2_Point(x, y))
                # 54, 34760   r == 19 and t == 2103
                #if r == 54 and t == 34760:
                #    print("(x, y) = (%f, %f)" %(x, y))
                #    if houghSpace[r, t] == 5.0:
                #        print("(r, t) = (%d, %d)" %(r, t))
                #        print("t_rad = ", t_rad)


                #try:
                #    houghSpace[r, t]  += 1.0
                #except Exception as e:
                #    print("(r, t) = %f, %f" %(r, t))
                #    print("houghSpace[r, t] = ", houghSpace[r, t])
    return houghSpace, houghLines

def vanishRays_houghTransform(raysPairs,  x0=0, y0=0, weighted=False):
# Compute the number of rows and columns in the image
    minX = sys.float_info.max
    maxX = sys.float_info.min
    minY = sys.float_info.max
    maxY = sys.float_info.min

    d = 100000#int(d)
    #d = int(math.sqrt(row_num*row_num + col_num*col_num))
    tMax = 360

    houghSpace = sparse.lil_matrix((d,d))
    houghLines = HoughLines()

    for (templateRay, testRay) in raysPairs:
        point = testRay.getVanishPoint()
        if point is not None:
            (x, y) = point.toTuple()
            if x < minX:
                minX = x
            if x > maxX:
                maxX = x
            if y < minY:
                minY = y
            if y > maxY:
                maxY = y
        #pMax = R2_Point(maxX, maxY)
        #pMin = R2_Point(minX, minY)

        #d = pMin.euclideanDistance(pMax)
        #print("pMax = ", pMax)
        #print("pMin = ", pMin)
        #d += 1.0
        #print("d = ", d)


        #print("d = ", d)
        # Compute transformed image
        for t in range(0, tMax):
            for (templateRay, testRay) in raysPairs:
                point = testRay.getVanishPoint()
                if point is not None:
                    (x, y) = point.toTuple()
                    #point = WeightedPoint(x, y, len(ray.crossRatioVector))
                    #x = int(x)
                    #y = int(y)

                    # Compute the point p in the original image which is mapped to the
                    # (row, col) pixel in the transformed image
                    #p = Point(col, row, 1)
                    #p.to_img_coord(col_num, row_num)
                    #p.transform(H)
                    #p.normalize()
                    # Get the (row_px, col_px) pixel coordinates of p
                    #(row_px, col_px)  = p.get_pixel_coord(col_num, row_num)
                    t_rad = (2*math.pi*t)/tMax
                    r = int(x*math.cos(t_rad) + y*math.sin(t_rad))
                    # If the pixel is part of the image, get it's color
                    if (0 < r < d):
                        # Original image
                        # pixelValue = image[x][y]
                        # Rectified image
                        # print("(r, t) = (%d, %d)"%(r, t))
                        w = 1.0
                        if weighted:
                            w = point.w

                        houghSpace[r, t] += w
                        if weighted:
                            houghLines.updateLine((r,t), (templateRay, testRay))
                        else:
                            houghLines.updateLine((r,t), R2_Point(x, y))
                        # 54, 34760   r == 19 and t == 2103
                        #if r == 54 and t == 34760:
                        #    print("(x, y) = (%f, %f)" %(x, y))
                        #    if houghSpace[r, t] == 5.0:
                        #        print("(r, t) = (%d, %d)" %(r, t))
                        #        print("t_rad = ", t_rad)


                        #try:
                        #    houghSpace[r, t]  += 1.0
                        #except Exception as e:
                        #    print("(r, t) = %f, %f" %(r, t))
                        #    print("houghSpace[r, t] = ", houghSpace[r, t])
    return houghSpace, houghLines


'''
def inverseHoughTransform(euclidianPoints, houghPoints, peakValue=1):
    d = 100000#int(d)
    #d = int(math.sqrt(row_num*row_num + col_num*col_num))
    tMax = d
    for t in range(0, tMax):
        for point in euclidianPoints:
            (x, y) = point.toTuple()
'''


#-----------------------

"""
A = R2_Point(1.2, 3.0)
B = R2_Point(-1.5, 2.0)
C = R2_Point(0.2, 1.0)
D = R2_Point(1.0, 2.0)
E = R2_Point(0.0, 6.0)

points = [A, B, C, D, E]

houghSpace = points_houghTransform(points)
"""


