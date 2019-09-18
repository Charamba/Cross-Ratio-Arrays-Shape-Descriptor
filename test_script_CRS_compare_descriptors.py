import pickle
import sys
import os
from MatchingProcessor import MatchingProcessor
from multiprocessing import Process

from main_CRS import CRS_MatchingProcessor

def compare_descriptors(name_descriptor1, name_descriptor2):
    with open(name_descriptor1, 'rb') as pickle_in:  
        templateDescriptor = pickle.load(pickle_in)

    with open(name_descriptor2, 'rb') as pickle_in:  
        testDescriptor = pickle.load(pickle_in)

    matchingProcessor = CRS_MatchingProcessor(templateDescriptor, testDescriptor)

    (matchedVerticesPairs, distance_values, mTemplRays, mTestRays) = matchingProcessor.compareByPencils()

    matchedVerticesPairs_data = {}
    for (i, (pair, distance)) in enumerate(zip(matchedVerticesPairs, distance_values)):
        matchedVerticesPairs_data[i] = (pair, distance)

    return (matchedVerticesPairs_data, distance_values)

def check_file(name1, name2, folder_path):
    str_filename = folder_path + name1 + "-" + name2 + "-" + str(150) + ".txt"
    return os.path.isfile(str_filename) 

def save_matching_results(name1, name2, folder_path, data):
    #open pickle
    str_filename = folder_path + name1 + "-" + name2 + "-" + str(150) + ".txt"
    with open(str_filename, 'wb') as outfile:  
        pickle.dump(data, outfile)

def matching_template_with_query_pairs(template_folder, template_descriptors, query_folder, result_folder):
    query_descriptors = os.listdir(query_folder)

    names_in_pairs = []
    for template_descriptor in template_descriptors:
        template_descriptor = template_folder + "/" + template_descriptor
        
        for test_descriptor in query_descriptors:

            test_descriptor = query_folder + "/" + test_descriptor
            
            name1 = template_descriptor.split("/")[-1].split(".")[0]
            name2 = test_descriptor.split("/")[-1].split(".")[0]

            name1_vs_name2 = name1 + " vs " + name2
            if not check_file(name1, name2, result_folder):
                
                names_in_pairs.append((template_descriptor, test_descriptor))
                # matchedVerticesPairs_data = compare_descriptors(template_descriptor, test_descriptor)
                # save_matching_results(name1, name2, result_folder, matchedVerticesPairs_data)
                # print(name1_vs_name2 + ": Compared!")
            else:
                print(name1_vs_name2 + ": Already compared!")
    
    return names_in_pairs

def thread_function(names_in_pairs, result_folder):
    for (template_descriptor, test_descriptor) in names_in_pairs:
        name1 = template_descriptor.split("/")[-1].split(".")[0]
        name2 = test_descriptor.split("/")[-1].split(".")[0]

        name1_vs_name2 = name1 + " vs " + name2

        print(name1_vs_name2 + ": Comparing...")
        (matchedVerticesPairs_data, distance_values) = compare_descriptors(template_descriptor, test_descriptor)
        save_matching_results(name1, name2, result_folder, (matchedVerticesPairs_data, distance_values))
        print(name1_vs_name2 + ": Compared!")


# MAIN
argc = len(sys.argv)
#max_processes = 10
if argc >= 4:
    template_folder = sys.argv[1]
    query_folder  = sys.argv[2]
    result_folder = sys.argv[3]
    n_processes = int(sys.argv[4])

    template_names = os.listdir(template_folder)
    query_names = os.listdir(query_folder)

    n_templates = len(template_names)
    n_query = len(query_names)


    full_names_in_pairs = matching_template_with_query_pairs(template_folder, template_names, query_folder, result_folder)

    full_names_pairs_lists = []
    it = 0
    pid = 0
    for template_name, query_name in full_names_in_pairs:
        if it == 0:
            full_names_pairs_lists.append([(template_name, query_name)])
        else:
            full_names_pairs_lists[pid].append((template_name, query_name))

        pid += 1
        if pid == n_processes:
            it += 1
            pid = 0

    processes = []
    for names_in_pairs in full_names_pairs_lists:
        p = Process(target=thread_function, args=(names_in_pairs, result_folder))
        processes.append(p)


    # ------------------------------------
    # full_names_lists = []
    # it = 0
    # pid = 0
    # for template_name in template_names:
    #     if it == 0:
    #         full_names_lists.append([template_name])
    #     else:
    #         full_names_lists[pid].append(template_name)

    #     pid += 1
    #     if pid == n_processes:
    #         it += 1
    #         pid = 0

    # processes = []
    # for template_names in full_names_lists:
    #     print("template_folder: ", template_folder)
    #     print("query_folder: ", query_folder)
    #     p = Process(target=thread_function, args=(template_folder, template_names, query_folder, result_folder))
    #     processes.append(p)





        
    for p in processes:
        p.start()

    for p in processes:
        p.join()


else:
    print("Usage: ")
    print(" ")
    print(">> python test_script_CRS_compare_descriptors.py template_descriptors_directory query_descriptors_directory matching_results_directory")
