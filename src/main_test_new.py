import pickle
import os
import hull_contour
from hull_contour import *
#from CrossRatio import *
from scipy import misc
import ast


#from CrossRadonTransform import *
#from HoughTransform import *
#from ShapeDescriptor import *
#from MatchRaysPairs import *
#from Plotter import *
#from ransac import *
#from LineEstimation import *
#from HorizonLine import *
from VanishPencilsTable import *
import matplotlib.pylab as pl
from matplotlib.patches import ConnectionPatch
from find_homography_distance import calc_hausdorff_distance


from Image import *
from Scanner import *
from MatchingProcessor import *



# ================================= FLAGS ===================================== # 

# plot: 
showScanRays = False
showMatchRays = False
showTrajectories = False

# scan: tomographic, convex-hull
convex_hull_scan_template = True

tomographic_scan_test = False
convex_hull_scan_test = True
# tomographic parameters
(template_nTraj, template_nProj) = (91, 6)
(test_nTraj, test_nProj) = (121, 18)

# convex-hull fan-beam parameters
template_nFanBeam = 30
test_nFanBeam = 30
emitter_points_number = 30
#add_points_number = 45

# compare features: rays, triple cross feature
compare = True
compareByRays = True
compareByTripleCrossFeatures = False

# booleano para controlar se os dados do template estao armazenados 
generated_template = True 
generated_test = True 

# HULL parameters (contour=False, convex=True)
convex_hull_flag = False

# SHAPE FLAG (symetryc=True,assimetric=False)
symetric_shape_flag=True

diff_shape = False
symetric_temp = ('A', 'B', 'C', 'D', 'E', 'H', 'M', 'N', 'Z', 'S', 'O', 'T', 'U', 'V', 'Y', 'X')
def get_name(directory):
	directory = directory.split('/')
	size = len(directory)
	name = directory[size-1]
	name = name.split('.')
	name = name[0]
	return name

## Close window and change progress in code
def press(event):
	#print('press', event.key)
	if event.key == 'enter':
		plt.close()

fig = plt.figure()

def convert2dict(inlist):
        i = 0
        homo = {}
        #digit = ('1', '2', '3', '4', '5', '6', '7', '8', '9')
        while(i < len(inlist)-2): 
            test_name = inlist[i]
            temp_name = inlist[i+1]
            pairs = inlist[i+2]
            #if((temp in digit) or (test in digit)):
            #    i+=3
            #    continue
            pairs = ast.literal_eval(pairs)
            if(test_name not in homo.keys()): 
                homo.update({test_name: {}})
            homo[test_name].update({temp_name:pairs})
            i+=3 
        return homo

def main_test(template_nFanBeam_input, test_nFanBeam_input, emitter_points_number_input):
    template_nFanBeam = template_nFanBeam_input
    test_nFanBeam = test_nFanBeam_input
    emitter_points_number = emitter_points_number_input  
    file_type = 2#int(input('1. Escolher imagens individualmente \n' + '2. Ler arquivo com todas as imagens\n'))
    tem_filename = []
    tst_filename = []
    if(file_type == 1):
        temp = askopenfilename(filetypes=[("all files","*"),("Bitmap Files","*.bmp; *.dib"),("JPEG", "*.jpg; *.jpe; *.jpeg; *.jfif"),("PNG", "*.png"), ("TIFF", "*.tiff; *.tif")])
        tst = askopenfilename(filetypes=[("all files","*"),("Bitmap Files","*.bmp; *.dib"),
											("JPEG", "*.jpg; *.jpe; *.jpeg; *.jfif"),
											("PNG", "*.png"), ("TIFF", "*.tiff; *.tif")])
        tem_filename.append(temp)
        tst_filename.append(tst)
    else:
        with open('arq_temp.txt', 'r') as arq_temp:
            tem_filename = arq_temp.read()
            tem_filename = tem_filename.split()
        with open('arq_test.txt', 'r') as arq_test:
            tst_filename = arq_test.read()
            tst_filename = tst_filename.split()
    with open('resultados_elevation50_caracteres/elevation50/r' +' ' + str(emitter_points_number) + ' ' + str(test_nFanBeam) + '_pairs.txt', 'r') as r_file:
        pair_match = r_file.read()
        pair_match = pair_match.split('\n')
    pairs = {}
    pairs = convert2dict(pair_match)
    fig.canvas.set_window_title('Original Image')
    fig.canvas.mpl_connect('key_press_event', press)


    for t in tst_filename:
        testImage = Image(misc.imread(t, mode = 'RGB'))
        test_name = get_name(t)
        
        ncurves_test = hull_contour.curves_from_rasterized_img(testImage.image)

        file_name = 'resultados_elevation50_caracteres/elevation50/testes' + str(emitter_points_number_input) +'/' + str(test_nFanBeam) + test_name + '.txt'
        try:
            with open(file_name, 'rb') as myFile:
                testDescriptor = pickle.load(myFile)
        except FileNotFoundError: 
            testScanner = Scanner(testImage)
            convexHullVertices_orig = testImage.convex_hull()
            
            original_vertex_number = len(convexHullVertices_orig)
            convexHullVertices = testImage.convex_hull(interpoints_sample=0)
            
            if convex_hull_flag: 
                add_points_number = emitter_points_number - original_vertex_number
                if add_points_number <= 0:
                    add_points_number = 1
                convexHullVertices = testImage.convex_hull_add_points(additive_points_number=add_points_number)#convex_hull(interpoints_sample=0)
            else: 
                add_points_number = test_nFanBeam - original_vertex_number
                if add_points_number <= 0:
                    add_points_number = 1
                convexHullVertices = testImage.convex_hull_add_points(additive_points_number=add_points_number)#  
            
            if convex_hull_flag:
                emitter_points = convexHullVertices
                emitter_points_orig = convexHullVertices_orig
                testDescriptor = testScanner.hull_scan(emitter_points, convexHullVertices_orig, fanBeamRays=test_nFanBeam, showTrajectories=showTrajectories, verticeIndexList=[])#0 #45
            else:
                emitter_points = testImage.contour_hull(nPoints=emitter_points_number)
                emitter_points_orig = emitter_points
                targetPoints = convexHullVertices
                testDescriptor = testScanner.hull_scan_for_contour(emitter_points, emitter_points_orig, targetPoints, fanBeamRays=test_nFanBeam, showTrajectories=showTrajectories, verticeIndexList=[])
            with open(file_name, 'wb') as myFile:
                pickle.dump(testDescriptor, myFile)
                myFile.close()

        for x in tem_filename:
            templateImage = Image(misc.imread(x, mode = 'RGB'))
            ncurves_temp = hull_contour.curves_from_rasterized_img(templateImage.image)
            temp_name = get_name(x)
            
            if(len(ncurves_test) == len(ncurves_temp)): 
                 
                file_name = 'resultados_elevation50_caracteres/elevation50/templates'+ str(emitter_points_number_input) +'/' + str(template_nFanBeam) + temp_name + '.txt'
                try:
                    with open(file_name, 'rb') as myFile:
                        templateDescriptor = pickle.load(myFile)
                except FileNotFoundError: 
                    templateScanner = Scanner(templateImage)
                    convexHullVertices_orig = templateImage.convex_hull()
                    original_vertex_number = len(convexHullVertices_orig)

                    if convex_hull_flag: 
                        add_points_number = emitter_points_number - original_vertex_number
                        if add_points_number <= 0:
                            add_points_number = 1
                        convexHullVertices = templateImage.convex_hull_add_points(additive_points_number=add_points_number)#convex_hull(interpoints_sample=0)
                    else: 
                        add_points_number = test_nFanBeam - original_vertex_number
                        if add_points_number <= 0:
                            add_points_number = 1
                        convexHullVertices = templateImage.convex_hull_add_points(additive_points_number=add_points_number)#  
                    
                    if convex_hull_flag:
                        emitter_points = convexHullVertices
                        emitter_points_orig = convexHullVertices_orig
                        templateDescriptor = templateScanner.hull_scan(emitter_points, convexHullVertices_orig, fanBeamRays=test_nFanBeam, showTrajectories=showTrajectories, verticeIndexList=[])#0 #45
                    else:
                        emitter_points = templateImage.contour_hull(nPoints=emitter_points_number)
                        emitter_points_orig = emitter_points
                        targetPoints = convexHullVertices
                        templateDescriptor = templateScanner.hull_scan_for_contour(emitter_points, emitter_points_orig, targetPoints, fanBeamRays=test_nFanBeam, showTrajectories=showTrajectories, verticeIndexList=[])
                    
                    with open(file_name, 'wb') as myFile:
                        pickle.dump(templateDescriptor, myFile)
                
                if test_name in pairs.keys() and temp_name in pairs[test_name].keys(): 
                    matchingVerticesPairsPoints = pairs[test_name][temp_name]
                    #dist1, dist2 = calc_hausdorff_distance(templateDescriptor.hullVertices, testDescriptor.hullVertices, matchingVerticesPairsPoints)
                    #rnew = test_name + '\n' + temp_name +'\n' + str(dist1) + '\n' + str(dist2)+'\n'
                    print(test_name, temp_name)
                    if matchingVerticesPairsPoints != []:
                        dist1, dist2 = calc_hausdorff_distance(templateDescriptor.hullVertices, testDescriptor.hullVertices, matchingVerticesPairsPoints)
                        rnew = test_name + '\n' + temp_name +'\n' + str(dist1) + '\n' + str(dist2)+'\n'
                    else: 
                        rnew = test_name + '\n' + temp_name +'\n' + 'Inf' + '\n' + 'Inf'+'\n'
                    with open('resultados_elevation50_caracteres/elevation50/r' +' ' + str(emitter_points_number) + ' ' + str(test_nFanBeam) + '_pairs new.txt' , 'a') as arq:
                        arq.write(rnew)

                else: 
                    matchingProcessor = MatchingProcessor(templateDescriptor, testDescriptor)
                    if compare:
                        if compareByRays:
                            symetric_shape = True 
                            sigma = 0
                            a = b = 0
                            if(temp_name in symetric_temp): 
                                symetric_shape = True
                            (matching_result, new_pCoef, _, _, matchingVerticesPairsPoints, distance, sigma, a, b) = matchingProcessor.compareByPencils(symetric_shape)
                            result = test_name + '\n' + temp_name +'\n' + str(symetric_shape)+'\n'+ str(new_pCoef)+'\n'+ str(matching_result)+ '\n' + str(distance) + '\n' + str(sigma) + '\n'+ str(a) + '\n' + str(b) + '\n'
                            r = test_name + '\n' + temp_name +'\n' + str(matchingVerticesPairsPoints) + '\n'
                            print(test_name, temp_name, distance)
                            '''
                            with open('resultados/elevation10/r' +' ' + str(emitter_points_number) + ' ' + str(test_nFanBeam) + '.txt' , 'a') as arq:
                                arq.write(result)
                            with open('resultados/elevation10/r' +' ' + str(emitter_points_number) + ' ' + str(test_nFanBeam) + '_pairs.txt' , 'a') as arq:
                                arq.write(r)
                            '''
                    if matchingVerticesPairsPoints != []:
                        dist1, dist2 = calc_hausdorff_distance(templateDescriptor.hullVertices, testDescriptor.hullVertices, matchingVerticesPairsPoints)
                        rnew = test_name + '\n' + temp_name +'\n' + str(dist1) + '\n' + str(dist2)+'\n'
                    else: 
                        rnew = test_name + '\n' + temp_name +'\n' + 'Inf' + '\n' + 'Inf'+'\n'
                    with open('resultados_elevation50_caracteres/elevation50/r' +' ' + str(emitter_points_number) + ' ' + str(test_nFanBeam) + '_pairs new.txt' , 'a') as arq:
                        arq.write(rnew)
            
            else: 
                #dist1, dist2 = calc_hausdorff_distance(templateDescriptor.hullVertices, testDescriptor.hullVertices, [])
                symetric_shape = True
                if(temp_name in symetric_temp): 
                    symetric_shape = True
                distance = float('Inf') 
                new_pCoef = 0 
                a = b = 0
                matching_result = False 
                sigma = float('Inf') 
                result = test_name + '\n' + temp_name +'\n' + str(symetric_shape)+'\n'+str(new_pCoef)+'\n'+ str(matching_result)+ '\n' + str(distance) + '\n' + str(sigma) + '\n' + str(a) + '\n' + str(b) + '\n'
                print(test_name, temp_name, distance)
                r = test_name + '\n' + temp_name +'\n' + str([]) + '\n'
                rnew = test_name + '\n' + temp_name +'\n' + 'Inf' + '\n' + 'Inf'+'\n'
                '''
                with open('resultados/elevation10/r' +' ' + str(emitter_points_number) + ' ' + str(test_nFanBeam) + '.txt' , 'a') as arq:
                    arq.write(result)
                with open('resultados/elevation10/r' +' ' + str(emitter_points_number) + ' ' + str(test_nFanBeam) + '_pairs.txt' , 'a') as arq:
                    arq.write(r)
                '''
                with open('resultados_elevation50_caracteres/elevation50/r' +' ' + str(emitter_points_number) + ' ' + str(test_nFanBeam) + '_pairs new.txt' , 'a') as arq:
                    arq.write(rnew)
                

def main (): 
    raios = [150]
    pontos = [150]
    for p in pontos: 
        for r in raios: 
            if(r == 100 and p == 100):
                break
            else: 
                print("pontos = ", p, "raios = ", r)
                main_test(r, r, p)       


if __name__ == "__main__": 
    main()