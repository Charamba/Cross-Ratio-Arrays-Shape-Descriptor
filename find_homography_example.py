#!/usr/bin/env python
 
import cv2
import numpy as np

from numpy.linalg import inv

 
if __name__ == '__main__' :
 
    # Read source image.
    #im_src = cv2.imread('book2.jpg')
    # Four corners of the book in source image
    pts_src = np.array([[141, 131], [480, 159], [493, 630],[64, 601], [1,0]])
 
 
    # Read destination image.
    #im_dst = cv2.imread('book1.jpg')
    # Four corners of the book in destination image.
    pts_dst = np.array([[318, 256],[534, 372],[316, 670],[73, 473], [400, 203]])
 
    # Calculate Homography
    h, status = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC,5.0)
    print("status = ", status)
    #a = np.array([[1., 2.], [3., 4.]])
    
    
    print("Homography: ", h)

    p1_ = np.matmul(h, [141, 131, 1])
    print("p1_: ", p1_)
    p1_normaliz = [p1_[0]/p1_[2], p1_[1]/p1_[2], p1_[2]/p1_[2]]
    print("p1_normaliz. : ", p1_normaliz)

    # INVERSA
    h_inv = inv(h)
    p1_rectif = np.matmul(h_inv, [318, 256, 1])
    print("p1_rectif: ", p1_rectif)
    p1_rect_normaliz = [p1_rectif[0]/p1_rectif[2], p1_rectif[1]/p1_rectif[2], p1_rectif[2]/p1_rectif[2]]
    print("p1_rect_normaliz. : ", p1_rect_normaliz)


    # Warp source image to destination based on homography
    #im_out = cv2.warpPerspective(im_src, h, (im_dst.shape[1],im_dst.shape[0]))
     
    # Display images
    #cv2.imshow("Source Image", im_src)
    #cv2.imshow("Destination Image", im_dst)
    #cv2.imshow("Warped Source Image", im_out)
 
    cv2.waitKey(0)