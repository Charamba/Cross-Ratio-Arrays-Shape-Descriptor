import math
from tabulate import tabulate
import scipy.io
import os
import operator 

results = {}
template = set()
test = set()

##################
# MATRIX DO CONCORRENTE 
###################

def get_name(directory):
	directory = directory.split('/')
	size = len(directory)
	name = directory[size-1]
	name = name.split('.')
	name = name[0]
	return name 

def is_temp(name): 
    name = name.split('_')
    if(len(name) > 1):
        return False
    return True

def get_conc_matrix(): 
	conc = scipy.io.loadmat('matlab.mat', mdict=None, appendmat=True) 
	matrix = {}
	a = os.listdir('data/data') 
	b = conc['Score']
	for i in range(len(conc['Score'])):
		aux1 = a[i]
		if(is_temp(aux1)): 
			continue
		matrix.update({get_name(aux1): {}})
		for j in range(len(conc['Score'])):

			if(i == j): continue
			cur_im = a[j]
			if not is_temp(cur_im): 
				continue
			cur_dist = b[i][j]
			matrix[get_name(aux1)].update({get_name(cur_im): [cur_dist]})
	return matrix 

def set_matrix (tp, fp, fn, tn, belongs, result):
	if((belongs) & (result)):
		tp += 1
	elif((belongs) & (not result)):
		fp += 1
	elif((not belongs) & (result)): 
		fn += 1
	else:
		tn += 1
	return tp, fp, fn, tn

def get_accuracy (tp, tn, fp, fn): 
	return ((tp + tn)/(tp + tn + fp + fn))

def get_sensibility (tp, tn, fp, fn):
	if(tp+fn == 0):
		return 0
	else:
		return (tp/(tp+fn))

def get_efficiency(tp, tn, fp, fn):
	if(tn+fp == 0):
		return 0
	else:
		return (tn/(tn+fp))

def get_ppv(tp, tn, fp, fn):
	if(tp+fp == 0):
		return 0
	else:
		return (tp/(tp+fp))

def get_PHI(tp, tn, fp, fn):
	dom = math.sqrt((tp + fp)*(tp + fn)*(tn + fp)*(tn + fn))
	if (dom == 0):
		return 0
	else: 
		return (tp*tn - fp*fn) / dom

def str2bool(a):
	if (a == 'True'):
		return True
	else: 
		return False

def checkClass(temp_name, test_name): 
	temp_name = temp_name.split('.')
	temp_name = temp_name[0]

	test_name = test_name.split('_')
	test_name = test_name[0]
	
	if(test_name.lower() == temp_name.lower()): 
		return True
	else: 
		return False

def get_matrix_parameters (tp, tn, fp, fn):

	print('\nAcuracia: ' + str(get_accuracy(tp, tn, fp, fn)))
	print('Sensibilidade: ' + str(get_sensibility (tp, tn, fp, fn)))
	print('Eficiencia: ' + str(get_efficiency(tp, tn, fp, fn)))
	print('Valor Predetivo Positivo: ' + str(get_ppv(tp, tn, fp, fn)))
	print('Valor Predetivo Negativo: ' + str(get_ppv(tp, tn, fp, fn)))
	print('Coeficiente de Mathews: ' + str(get_PHI(tp, tn, fp, fn)) + '\n')

def make_matrix(matrix_type):
	temp_list = list(template)
	test_list = list(test)

	matrix = [[' ' for x in range(len(test))] for y in range(len(template)+1)]
	
	for x in range (len(test)): 
		matrix[0][x] = test_list[x]
	for y in range(len(template)):
		matrix[y+1][0] = temp_list[y]
	
	test_cases = matrix[0]
	for x in range(len(test_cases)):
		aux = test_cases[x]
		aux1 = results[aux]
		j = 1
		while(j<len(matrix)): 
			l = matrix[j]
			aux3 = l[0]
			a = aux1[aux3][matrix_type]
			matrix[j].append(a)
			j+=1
	
	for i in range(len(matrix)):
		aux = matrix[i]
		matrix[i] = filter(lambda x: x != ' ', aux)
	
	del matrix[0]
	return (test_cases, matrix)

def get_best_match(matrix):
	test_list = list(test)
	temp_list = list(template)

	for x in test_list:
		best_match = ''
		best_dist = float('inf')
		for y in temp_list:
			aux = matrix[x][y]
			cur_dist = aux[3]
			if(cur_dist <= best_dist):
				best_match = y
				best_dist = cur_dist

		for y in temp_list:
			matrix[x][y].append(best_match)

	return matrix

def set_paramets(c,matrix_type):
	
	test_list = list(test)
	temp_list = list(template)
	belongs = False 
	tp = tn = fp = fn = 0.0 
	c = get_best_match(c)
	for x in test_list:
		for y in temp_list: 
			aux = c[x][y]
			if(matrix_type == 1 or matrix_type == 2):
				if(matrix_type == 1): 
					belongs = checkClass( aux[4], x)
					print(belongs, x, aux[4])
				 
			(tp, fp, fn, tn) = set_matrix(tp, fp, fn, tn, belongs, True)
	return (tp, fp, fn, tn)

def get_near_neighbor(matrix):
	min_dist = 0 
	min_temp = ''
	test_list = list(test)
	temp_list = list(template)
	for x in test_list:

		for y in temp_list: 
			aux = matrix[x][y]
			if aux[0] < min_dist: 
				min_dist = aux[0]
				min_temp = y
		
		for y in temp_list: 
			matrix[x][y].append(min_temp)
	return matrix

def show_matrix(head, matrix, flag):
	if (flag == 1): 
		title = '\n Matriz de Confusão do Resultado do descritor\n\n' 
	elif(flag == 2): 
		title = '\n Matriz de Confusão do Coeficiente de Pearson\n\n'
	else: 
		title = '\n Matriz de Confusão do Best Match\n\n'
	table = tabulate(matrix, tablefmt = "pipe") 
	return title.upper() + table 

def main():
	matrix_type = int(input('1. Best Mach\n2. Distance\n3. Sair\n'))
	while(matrix_type < 3):
		tp = tn = fp = fn = 0.0 
		i = 0
		pCoef = 0 
		matching_result = False 
		with open('r.txt', 'r') as file: 
			data = file.read() 
			data = data.split()
			size = len(data)
		
		while i<(size - 5):
			 
			test_name = data[i]
			template_name = data[i+1]
			comp_type = str2bool(data[i+2])
			#print(comp_type)
			if(not comp_type): 
				pCoef = float(data[i+3])
				matching_result = str2bool(data[i+4])
				distance = float(data[i+5])#*math.sqrt(3)
				#distance = distance/
			else: 
				distance = math.floor(float(data[i+5]))#*math.sqrt(2)
				#distance = distance/
			test.add(test_name)
			template.add(template_name)
			test_result = [comp_type, matching_result, pCoef, distance]

			if test_name not in results.keys():
				results[test_name] = {}
				if template_name not in results[test_name].keys():
					results[test_name][template_name] = []

			results[test_name][template_name] = test_result
			i += 6

		
		#conc = get_conc_matrix()
		#print(conc)
		#(tp, fp, fn, tn) = set_paramets(conc, 1)
		#print((tp, fp, fn, tn))
		#get_matrix_parameters(tp, tn, fp, fn)
	
		(tp, fp, fn, tn) = set_paramets(results, matrix_type)
		(a,b) = make_matrix(matrix_type-1)
		print(show_matrix(a, b, matrix_type))
		get_matrix_parameters(tp, tn, fp, fn)
		matrix_type = int(input('1. Best Mach\n2. Distance\n3. Sair\n'))

if __name__ == '__main__':
	main()

