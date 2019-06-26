import pickle
import sys
import numpy as np
from MatchingProcessor import fanbeams_threshold_distance

from find_homography_distance import calc_hausdorff_distance, calc_homography, calc_homography_distance, transf_points


def calc_espectre_distance(matching_filename, threshold_factors, use_fanbeam_threshold=False):

    with open(matching_filename, 'rb') as pickle_in:
        unpickled_dict = pickle.load(pickle_in)

    #print(unpickled_dict)

    (matchedVerticesPairs, distanceValues) = zip(*unpickled_dict.values())

    matchedVerticesPairs = list(matchedVerticesPairs)
    distanceValues = list(distanceValues)

    if use_fanbeam_threshold:
        (matchedVerticesPairs, distanceValues) = fanbeams_threshold_distance(matchedVerticesPairs, distanceValues, threshold_factors)

    template_pts = np.array([[templ_pt.x, templ_pt.y] for (templ_pt, _) in matchedVerticesPairs])
    test_pts     = np.array([[test_pt.x,  test_pt.y]   for (_, test_pt) in matchedVerticesPairs])

    ## Estimando homografia e erro de transferência pelos vértices associados

    (dist, new_template_pts, new_test_pts, hinv) = calc_homography_distance(template_pts, test_pts)
    
    n_pts = None
    if new_template_pts == [] or new_template_pts == None:
        dist = float('Inf')
        n_pts = 0
    else:
        n_pts = len(new_template_pts)
        if n_pts == 4:
            dist = float('Inf')
            

    return dist, n_pts 


# filename = sys.argv[1]
# threshold_factor = float(sys.argv[2])
# print("filename: ", filename)


# dist = calc_espectre_distance(filename, threshold_factor)
# print("dist = ", dist)




# print(matchedVerticesPairs)
# print(distanceValues)


