import cv2
import numpy as np

from numpy.linalg import inv
from scipy.spatial.distance import directed_hausdorff

import math

def bbox_dimensions(points):
    X = [p[0] for p in points]
    Y = [p[1] for p in points]

    min_x = min(X)
    min_y = min(Y)
    max_x = max(X)
    max_y = max(Y)
    dbbx = max_x - min_x
    dbby = max_y - min_y
    return (dbbx, dbby)

def calc_homography(pairs_points):
    if len(pairs_points) == 0:
        return float('Inf')

    pts_src = np.array([[src_pt[0], src_pt[1]]  for (src_pt, _) in pairs_points])
    pts_dst = np.array([[dst_pt[0], dst_pt[1]]  for (_, dst_pt) in pairs_points])
    if (len(pts_src) == 0 or len(pts_dst) == 0):
        return float('Inf')
    #print("pts_dst = ", len(pts_dst))

    h_inv, status = cv2.findHomography(pts_dst, pts_src, cv2.LMEDS)

    if h_inv is None:
        h_inv, status = cv2.findHomography(pts_dst, pts_src, cv2.RANSAC, 3.0)

    return h_inv, status

def transf_points(h, points):
    rectif_points = []
    for pt_dst in points:
        #pt_dst = pts_dst[i]
        #print("pt_dst: ", pt_dst)
        #print("h_inv: ", h_inv)
        
        p_rectif = np.matmul(h, [pt_dst[0], pt_dst[1], 1])
        #p_rectif = np.matmul(np.array(h), np.array([pt_dst.x, pt_dst.y, 1]))

        p_rectif_norm = [p_rectif[0]/p_rectif[2], p_rectif[1]/p_rectif[2], p_rectif[2]/p_rectif[2]]

        (xrn, yrn) = (p_rectif_norm[0], p_rectif_norm[1])
        rectif_points.append((xrn, yrn))
    return rectif_points

def calc_hausdorff_distance(template_hull_points, query_hull_points, pairs_points):
    h_inv = calc_homography(pairs_points)
    (dbbx, dbby) = bbox_dimensions(template_hull_points)
    max_bb = max(dbbx, dbby)
    (dbbx, dbby) = (max_bb, max_bb)

    #print("h_inv = ", h_inv)
    #print("query_hull_points = ", query_hull_points)
    query_hull_points_rect = transf_points(h_inv, query_hull_points)

    template_hull_points = [(p.x/dbbx,p.y/dbby) for p in template_hull_points]
    query_hull_points_rect = [(p.x/dbbx,p.y/dbby) for p in query_hull_points]

    return directed_hausdorff(template_hull_points, query_hull_points_rect)[0], directed_hausdorff(query_hull_points_rect, template_hull_points)[0]

def calc_homography_distance(template_pts, test_pts):#(pairs_points):
    if len(test_pts) == 0:
        return float('Inf'), None, None, None

    h_inv, status = cv2.findHomography(test_pts, template_pts, cv2.LMEDS)#,9.0)
    n = 0
    if h_inv is None:
        return float('Inf'), None, None, None
    dist = 0 if len(test_pts) > 0 else float('Inf')
    dist_array = []
    (dbbx, dbby) = bbox_dimensions(template_pts)
    #print("dbbx = ", dbbx)
    #print("dbby = ", dbby)

    new_template_pts = []
    new_test_pts = []

    for i, s in enumerate(status):
        if s == 1:
            orig_pt = template_pts[i]
            dst_pt = test_pts[i]

            new_template_pts.append(orig_pt)
            new_test_pts.append(dst_pt)

            #print("dst_pt: ", dst_pt)
            #print("h_inv: ", h_inv)
            p_rectif = np.matmul(h_inv, [dst_pt[0], dst_pt[1], 1])
            p_rectif_norm = [p_rectif[0]/p_rectif[2], p_rectif[1]/p_rectif[2], p_rectif[2]/p_rectif[2]]

            (xo, yo)   = (orig_pt[0], orig_pt[1]) 
            (xrn, yrn) = (p_rectif_norm[0], p_rectif_norm[1])

            dx = (xo-xrn)/dbbx
            dy = (yo-yrn)/dbby

            error_geom = math.sqrt(dx*dx + dy*dy)
            dist += error_geom
            dist_array.append(error_geom)
            #if s == 1:
            n += 1
        #else s==0:



    matchings_percent = float(sum(status))/len(status)
    #print("matchings_percent = ", matchings_percent)
    #print("Total = ", len(status))
    #print("n = ", n)
    dist_array.sort()
    errors = np.array(dist_array)#dist_array[0:4]
    #median_error = np.median(dist_array)

    #return (1.0-matchings_percent), median_error/n
    # math.sqrt((dist/n)**2 + (1.0 - n/150)**2)
    return 100*(dist/n), new_template_pts, new_test_pts, h_inv#*(1.0 - n/150) if n/150 >= 0.1 else float('Inf')#(1.0-matchings_percent)#(1 - matchings_percent)#np.median(errors)/n#1.0/n#dist/n


if __name__ == '__main__' :
    # acessible vs acessible test 0
    pairs_points = [((10.630961, -151.628088), (4.908575, -200.799395)), ((61.261922, -75.256176), (70.817150, -124.598790)), ((86.577403, -37.070221), (136.725725, -48.398186)), ((137.208364, 39.301691), (202.634300, 27.802419)), ((187.839325, 115.673603), (268.542875, 104.003024)), ((198.000000, 131.000000), (272.000000, 108.000000)), ((201.000000, 138.000000), (274.000000, 111.000000)), ((210.000000, 161.000000), (289.000000, 140.000000)), ((210.000000, 164.000000), (289.000000, 142.000000)), ((193.000000, 171.000000), (259.000000, 152.000000)), ((128.461571, 197.015371), (174.000000, 178.000000)), ((82.635361, 212.783400), (76.547010, 203.561440)), ((-6.000000, 243.000000), (-13.000000, 227.000000)), ((-16.000000, 246.000000), (-22.000000, 229.000000)), ((-29.000000, 249.000000), (-37.000000, 232.000000)), ((-37.000000, 250.000000), (-52.000000, 234.000000)), ((-51.000000, 251.000000), (-63.000000, 235.000000)), ((-66.000000, 251.000000), (-100.000000, 235.000000)), ((-75.000000, 250.000000), (-118.000000, 233.000000)), ((-88.000000, 248.000000), (-140.000000, 229.000000)), ((-109.000000, 242.000000), (-153.000000, 226.000000)), ((-114.000000, 240.000000), (-160.000000, 224.000000)), ((-133.000000, 231.000000), (-192.000000, 212.000000)), ((-143.000000, 225.000000), (-210.000000, 203.000000)), ((-155.000000, 216.000000), (-220.000000, 197.000000)), ((-174.000000, 198.000000), (-241.000000, 182.000000)), ((-180.000000, 191.000000), (-249.000000, 175.000000)), ((-186.000000, 183.000000), (-263.000000, 161.000000)), ((-192.000000, 174.000000), (-278.000000, 142.000000)), ((-201.000000, 157.000000), (-283.000000, 134.000000)), ((-207.000000, 142.000000), (-293.000000, 114.000000)), ((-210.000000, 133.000000), (-296.000000, 106.000000)), ((-213.000000, 121.000000), (-301.000000, 87.000000)), ((-215.000000, 108.000000), (-302.000000, 81.000000)), ((-216.000000, 99.000000), (-303.000000, 73.000000)), ((-216.000000, 75.000000), (-303.000000, 48.000000)), ((-213.000000, 54.000000), (-301.000000, 34.000000)), ((-212.000000, 49.000000), (-298.000000, 22.000000)), ((-210.000000, 41.000000), (-291.000000, 3.000000)), ((-176.485566, -54.400419), (-248.845287, -88.506571)), ((-145.971132, -140.800839), (-206.690575, -180.013143)), ((-118.000000, -220.000000), (-168.000000, -264.000000)), ((-114.000000, -228.000000), (-166.000000, -268.000000)), ((-112.000000, -231.000000), (-153.000000, -283.000000)), ((-103.000000, -240.000000), (-147.000000, -287.000000)), ((-92.000000, -246.000000), (-131.000000, -294.000000)), ((-89.000000, -247.000000), (-127.000000, -295.000000)), ((-85.000000, -248.000000), (-122.000000, -296.000000)), ((-65.000000, -247.000000), (-94.000000, -295.000000)), ((-62.000000, -246.000000), (-90.000000, -294.000000)), ((-54.000000, -242.000000), (-81.000000, -291.000000)), ((-51.000000, -240.000000), (-77.000000, -289.000000)), ((-43.000000, -232.000000), (-61.000000, -277.000000))]

    # honda vs acessible test 30
    #pairs_points = [((918.000000, -767.000000), (-217.000000, 100.000000)), ((925.573130, -528.688114), (88.558067, -77.347163)), ((929.359694, -409.532171), (-207.000000, -78.000000)), ((933.146259, -290.376228), (-67.137316, 147.129403)), ((936.932824, -171.220285), (103.000000, 163.000000)), ((940.719389, -52.064342), (94.930112, 2.421396)), ((944.505954, 67.091601), (-106.705974, 141.194104)), ((948.292518, 186.247544), (29.000000, 161.000000)), ((959.652213, 543.715372), (106.000000, 145.000000)), ((965.000000, 712.000000), (-130.069293, -128.862356)), ((845.786661, 757.810363), (-205.000000, 109.000000)), ((726.573323, 758.620725), (104.488178, 122.074233)), ((488.146645, 760.241450), (-197.000000, -85.000000)), ((368.933306, 761.051813), (-244.000000, 68.000000)), ((249.719968, 761.862175), (-163.534647, -106.931178)), ((130.506629, 762.672538), (-211.000000, -75.000000)), ((-227.133387, 765.103626), (-29.673233, -194.655891)), ((-359.000000, 766.000000), (44.000000, -226.000000)), ((-417.000000, 766.000000), (106.000000, 141.000000)), ((-536.199721, 764.024314), (35.222959, 161.676409)), ((-960.914207, 637.787412), (-27.568658, 153.064701)), ((-965.000000, 88.000000), (-258.000000, 5.000000)), ((-964.000000, 79.000000), (-257.000000, -2.000000)), ((-963.000000, 74.000000), (-255.000000, -12.000000)), ((-961.000000, 66.000000), (-216.000000, -71.000000)), ((-933.000000, 16.000000), (-253.000000, 49.000000)), ((-927.000000, 9.000000), (-139.000000, 136.000000)), ((-907.000000, -10.000000), (-144.000000, 135.000000)), ((-894.000000, -20.000000), (-200.000000, -83.000000)), ((655.692461, -661.145138), (102.000000, 168.000000)), ((905.000000, -763.000000), (-249.000000, 58.000000)), ((915.000000, -767.000000), (-180.000000, 123.000000))]

    # dividno os pontos em duas listas
    pts_src = np.array([[src_pt.x, src_pt.y]  for (src_pt, _) in pairs_points])
    pts_dst = np.array([[dst_pt.x, dst_pt.y]  for (_, dst_pt) in pairs_points])

    dist = calc_homography_distance(pts_src, pts_dst)
    #print("dist = ", dist)


    #cv2.waitKey(0)