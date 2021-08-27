import scipy.io
import sys

number_of_items = 360 #180 (SL) # 200 (200 CE2-occlusion)# 
classes_var_number = 9 #9 # 5 # 9

def get_minor_value(comp_dict):
    minor_val = float('Inf')
    minor_associated_idx = None

    for i, dist in comp_dict.items():
        #print(item)
        if dist < minor_val:
            minor_val = dist
            minor_associated_idx = i
    
    return (minor_associated_idx, minor_val)

argc = len(sys.argv)

if argc >= 2:
    score_file = sys.argv[1] # Ex.: 'data_el45_score.mat'
    mat = scipy.io.loadmat(score_file)

    score_matrix = list(mat['Score'])

    accuracy_matrix = {}

    template_indices = list(range(0, number_of_items, classes_var_number))

    # Assembly dictionary
    # matrix_distance_dict = {}

    # for (template_name, query_name, dist) in templ_query_aux_list:

    #     if not(query_name in matrix_distance_dict):
    #         matrix_distance_dict[query_name] = {}
        
    #     matrix_distance_dict[query_name][template_name] = dist

    # print("dict: ", matrix_distance_dict)

    

    for j in range(0, number_of_items):
        if j not in template_indices:
            for i in template_indices:
                dist = score_matrix[i][j]

                if not j in accuracy_matrix:
                    accuracy_matrix[j] = {}

                accuracy_matrix[j][int(i/classes_var_number)] = dist
            #print(len(accuracy_matrix[j]))

    #print(len(accuracy_matrix))

    acertos = 0
    

    for (j, comp_dict) in accuracy_matrix.items():
        (minor_associated_idx, minor_val) = get_minor_value(comp_dict)
        matching_status = "ERROR"
        
        if minor_associated_idx == int(j/classes_var_number):
            matching_status = "OK!"
            acertos = acertos + 1
        else:
            matching_status = matching_status + "  (" + str(int(j/classes_var_number)) + ", " +  str(comp_dict[int(j/classes_var_number)]) + ")"
        
        print('query_' + str(int(j/classes_var_number)) + " <--> template_" + str(minor_associated_idx) + ": " + str(minor_val) + "  " + matching_status)


    n_items = len(accuracy_matrix.items())
    acuracia = float(acertos)/n_items
    print("Acertos = ", acertos)
    print("Accuracy = ", acuracia)
else:
    print("Correct use:")
    print(">> test_script_HCNC_acuracy.py <score_file.mat>")