import scipy.io


def get_minor_value(comp_dict):
    minor_val = float('Inf')
    minor_associated_idx = None

    for i, dist in comp_dict.items():
        #print(item)
        if dist < minor_val:
            minor_val = dist
            minor_associated_idx = i
    
    return (minor_associated_idx, minor_val)

mat = scipy.io.loadmat('data_el45_score.mat')

score_matrix = list(mat['Score'])

accuracy_matrix = {}

template_indices = list(range(0, 360, 9))

# Assembly dictionary
# matrix_distance_dict = {}

# for (template_name, query_name, dist) in templ_query_aux_list:

#     if not(query_name in matrix_distance_dict):
#         matrix_distance_dict[query_name] = {}
    
#     matrix_distance_dict[query_name][template_name] = dist

# print("dict: ", matrix_distance_dict)


for j in range(0, 360):
    if j not in template_indices:
        for i in template_indices:
            dist = score_matrix[i][j]

            if not j in accuracy_matrix:
                accuracy_matrix[j] = {}

            accuracy_matrix[j][int(i/9)] = dist
        #print(len(accuracy_matrix[j]))

#print(len(accuracy_matrix))

acertos = 0

for (j, comp_dict) in accuracy_matrix.items():
    (minor_associated_idx, minor_val) = get_minor_value(comp_dict)
    matching_status = "ERROR"
    
    if minor_associated_idx == int(j/9):
        matching_status = "OK!"
        acertos = acertos + 1
    else:
        matching_status = matching_status + "  (" + str(int(j/9)) + ", " +  str(comp_dict[int(j/9)]) + ")"
    
    print('query_' + str(int(j/9)) + " <--> template_" + str(minor_associated_idx) + ": " + str(minor_val) + "  " + matching_status)


n_items = len(accuracy_matrix.items())
acuracia = float(acertos)/n_items
print("Acertos = ", acertos)
print("Accuracy = ", acuracia)