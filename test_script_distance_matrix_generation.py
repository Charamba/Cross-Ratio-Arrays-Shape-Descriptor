import pickle
import os
import sys
from calc_distance_by_matching_vertices import calc_espectre_distance


# ARGS
matching_folder = sys.argv[1]
# a = float(sys.argv[2])
# b = float(sys.argv[3])
# c = float(sys.argv[4])
# d = float(sys.argv[5])
threshold_factors = []#[a, b, c, d]

# MAIN
matching_results_list = os.listdir(matching_folder)

templ_query_aux_list = [] #tuples

def get_minor_value(comp_dict):
    minor_val = float('Inf')
    minor_templ_name = ""
    minor_pts = None

    for (templ_name, (dist, n_pts)) in comp_dict.items():
        if dist < minor_val and n_pts >= 15:
            minor_val = dist
            minor_templ_name = templ_name
            minor_pts = n_pts
    
    return (minor_templ_name, minor_val, minor_pts)


for matching_filename in matching_results_list:

    #print(matching_filename.split("-"))
    [template_name, query_name, _] = matching_filename.split("-")
    #template_name = template_name[3:]
    #query_name = query_name[3:]
    full_file_name = matching_folder + "/" + matching_filename
    dist, n_pts = calc_espectre_distance(full_file_name, threshold_factors)

    templ_query_aux_list.append((template_name, query_name, dist, n_pts))

# Assembly dictionary
matrix_distance_dict = {}

for (template_name, query_name, dist, n_pts) in templ_query_aux_list:

    if not(query_name in matrix_distance_dict):
        matrix_distance_dict[query_name] = {}
    
    matrix_distance_dict[query_name][template_name] = dist, n_pts

#print("dict: ", matrix_distance_dict)
acertos = 0

for (query_name, comp_dict) in matrix_distance_dict.items():
    (minor_templ_name, minor_val, n_pts) = get_minor_value(comp_dict)
    matching_status = "ERROR"
    
    if minor_templ_name == query_name.split('_')[0]:
        matching_status = "OK!"
        acertos = acertos + 1
    else:
        matching_status = matching_status + "  (" + query_name.split('_')[0] + ", " +  str(comp_dict[query_name.split('_')[0]]) + ")"
    
    print(query_name + " x " + minor_templ_name + ": " + str(minor_val) + "  " + matching_status + " pts = " + str(n_pts))

n_items = len(matrix_distance_dict.items())

acuracia = float(acertos)/n_items

print("Acertos = ", acertos)
print("Accuracy = ", acuracia)