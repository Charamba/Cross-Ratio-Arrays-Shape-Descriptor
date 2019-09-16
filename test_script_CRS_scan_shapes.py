
from multiprocessing import Process


from ShapeDescriptor import *
from Plotter import *
from LineEstimation import *
from VanishPencilsTable import *
import matplotlib.pylab as pl
from matplotlib.patches import ConnectionPatch
import time
import os
import pickle

from PIL import Image
from Scanner import *
from MatchingProcessor import MatchingProcessor
#from resizeimage import resizeimage

from hull_contour import curves_from_rasterized_img
from find_homography_distance import calc_hausdorff_distance, calc_homography, calc_homography_distance, transf_points

FULL_POINTS=True

# def check_dimensions(image):
# 	width, height = image.width, image.height1
# 	w = is_even(width)
# 	h = is_even(height)
# 	print(w, h)
# 	if not w and not h:
# 		a = image.resize((width+1, height+1), Image.NEAREST)
# 		#a = resize.resize((width+1, height+1), Image.NEAREST)
# 	elif not w and h: 
# 		a = image.resize((width+1, height+1), Image.NEAREST)
# 	elif w and not h: 
# 		a = image.thumbnail((width+1, height+1), Image.ANTIALIAS)
# 	else: 
# 		a = image
# 	return a


# =============================================================================
# ============================== SCAN IMAGES ==================================
# =============================================================================


def thread_function(shapes_folder, shape_filenames, misc_images, output_folder, sample):
    for shape_filename, misc_img in zip(shape_filenames, misc_images):
        shape_filename = shapes_folder + shape_filename
        shape_image = Image(misc_img)

        # Scanners
        shape_scanner = Scanner(shape_image)    

        # Convex-Hull
        convexHullVertices_orig = shape_image.convex_hull()
        original_vertex_number = len(convexHullVertices_orig)

        convexHullVertices = []
        emitter_points = []

        nFanBeam = sample
        emitter_points_number = sample

        add_points_number = emitter_points_number - original_vertex_number
        if add_points_number <= 0:
            add_points_number = 1

        convexHullVertices = shape_image.convex_hull_add_points(additive_points_number=add_points_number)

        emitter_points = convexHullVertices
        shape_descriptor = shape_scanner.hull_scan(emitter_points, convexHullVertices_orig, fanBeamRays=nFanBeam, FULL_POINTS=FULL_POINTS)

        if not(output_folder[-1] == "/"):
            output_folder += "/"

        output_filename = output_folder + shape_filename.split("/")[-1].split(".")[0] + ".desc"

        print("Saving ", output_filename)
        with open(output_filename, 'wb') as myFile:
            pickle.dump(shape_descriptor, myFile)
        

# MAIN
if __name__ == '__main__':

    start_time = time.time()

    argc = len(sys.argv)
    if 4 <= argc <= 5:
        shapes_folder = sys.argv[1]
        descriptors_folder = sys.argv[2]
        sample = int(sys.argv[3])
        n_processes = int(sys.argv[4])

        img_extensions = ["bmp", "gif", "jpeg", "jpg", "png"]

        img_names = os.listdir(shapes_folder)
        img_names = [n for n in img_names if n.split(".")[-1].lower() in img_extensions]

        n_img = len(img_names)
        if n_processes > n_img:
            n_processes = n_img

        full_names_lists = []
        full_img_lists = []
        it = 0
        pid = 0
        for name in img_names:
            if it == 0:
                full_names_lists.append([name])
                # print("shapes_folder: ", shapes_folder)
                # print(name)
                misc_img = misc.imread(shapes_folder + name, mode = 'RGB')
                full_img_lists.append([misc_img])
            else:
                full_names_lists[pid].append(name)
                misc_img = misc.imread(shapes_folder + name, mode = 'RGB')
                full_img_lists[pid].append(misc_img)
            
            pid += 1
            if pid == n_processes:
                it += 1
                pid = 0

        processes = []
        idx = 0
        print("Running in " + str(len(full_names_lists)) + " processes ...")

        for names_list, img_list in zip(full_names_lists, full_img_lists):
            p = Process(target=thread_function, args=(shapes_folder, names_list, img_list, descriptors_folder, sample))
            processes.append(p)
            idx += 1

        for p in processes:
            p.start()

        for p in processes:
            p.join()

    else:
        print("Usage: ")
        print(" ")
        print(">> python test_script_scan_shapes.py <shapes_directory> <descriptors_directory> <sample> <processes>")

    end_time = time.time()
    time_elapsed = end_time - start_time

    print("\nTime elapsed: ", time_elapsed)
