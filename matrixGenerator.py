import math
from tabulate import tabulate
import scipy.io
import os
import operator 
import numpy as np

import sys

results = {}
template = set()
test = set()

##################
# MATRIX DO CONCORRENTE 
###################
#test = ('adidas_test90', 'adidas_test210', 'Cocacola_test30', 'Cocacola_test90', 'Cocacola_test210', 'nike_test30', 'nike_test90','nike_test210', 'peugeot_test30', 'peugeot_test90', 'peugeot_test210', 'polo_test30','polo_test90', 'polo_test210', 'puma_test90', 'puma_test210', 'ufpe_test30', 'ufpe_test90' , 'ufpe_test210')
#template = ('ufpe', 'nike', 'adidas', 'Coca-Cola', 'peugeot-logo', 'POLO')



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
	a = os.listdir('data') 
	b = conc['Score']
	for i in range(len(conc['Score'])):
		aux1 = a[i]
		print(aux1)
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

	print('\nAcuracia              : ' + str(get_accuracy(tp, tn, fp, fn)))
	print('Sensibilidade           : ' + str(get_sensibility (tp, tn, fp, fn)))
	print('Eficiencia              : ' + str(get_efficiency(tp, tn, fp, fn)))
	print('Valor Predetivo Positivo: ' + str(get_ppv(tp, tn, fp, fn)))
	print('Valor Predetivo Negativo: ' + str(get_ppv(tp, tn, fp, fn)))
	print('Coeficiente de Mathews  : ' + str(get_PHI(tp, tn, fp, fn)) + '\n')

def unpack_parameters_values(b):
	sigma = max(b[3], b[4])
	Dist = min(b[3], b[4])
	(alpha, beta, pearson) = (b[5], b[6], b[2])
	return (sigma, alpha, beta, pearson, Dist)

def mat(results): 
	matrix = []
	new_mat = {}
	# b[2] = Pearson
	# b[3] = Distancia 
	# b[3] = Sigma
	# b[5] = Alpha 
	# b[6] = Beta
	for x in test: 
		g = []
		for y in template: 
			b = results[x][y]
			a = [b[2]]               #Mudar o parametro de vizualiacao aqui
			g.append(a) 
			if(type(g[0][0]) is not str): 
				g[0].insert(0, x)
		matrix.append(g)
	# Formula 
	for x in test:
		if x not in new_mat.keys(): 
			new_mat.update({x: {}})
		for y in template:
			b = results[x][y]
			if(not b[0]): 
				(sigma, alpha, beta, pearson, Dist) = unpack_parameters_values(b)
				print("sigma, alpha, beta, pearson = %f, %f, %f, %f" %(sigma, alpha, beta, pearson))
				#Dist = Dist*math.sqrt((1-alpha)**2 + (1-beta)**2+ (1-pearson)**2)/math.sqrt(3)
				#Dist = Dist*math.sqrt((1-alpha)**2 + (1-beta)**2+ (1-pearson)**2)/math.sqrt(3)
				#Dist = Dist*math.sqrt((1-alpha)**2 + (1-beta)**2+ (1-pearson)**2)/math.sqrt(3)
				aux = sigma*math.sqrt((1-alpha)**2 + (1-beta)**2) 
				#aux = sigma#*math.sqrt((1-alpha)**2 + (1-beta)**2 + (1-pearson)**2)/math.sqrt(3) #if pearson >= 0.2 else float('Inf')#b[2]*b[4]*b[5]*b[6] # Formula nao simetrico 
			else:
				(sigma, alpha, beta, pearson, Dist) = unpack_parameters_values(b)
				maxPercent = max(alpha, beta)
				aux = sigma*math.sqrt((1-alpha)**2 + (1-beta)**2) 
				#aux = sigma#*math.sqrt((1-alpha)**2 + (1-beta)**2 + (1-pearson)**2)/math.sqrt(3)#sigma*(1-maxPercent)#((0.1*alpha)**2 + (0.9*beta)**2)                          # Formula Simetrico 
			new_mat[x].update({y: [aux]}) 

	table = tabulate(matrix, headers= list(template), tablefmt = "fancy") 
	a, b, c, d = set_paramets(new_mat,1, concorrente=True)
	get_matrix_parameters(a, b, c, d )
	return table 

def make_matrix(results, matrix_type):
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

def get_best_match(matrix, concorrente=False, new_formula=False):
	test_list = list(test)
	temp_list = list(template)

	for x in test_list:
		best_match = ''
		best_pearson = 0
		best_dist = float('inf')
		for y in temp_list:
			aux = matrix[x][y]
			
			if( not concorrente): 
				if(not new_formula): 
					cur_dist = aux[3]
				else: 
					cur_dist = aux[8]
				if(cur_dist <= best_dist):
					if(not aux[0]): 
						if(aux[2] >= best_pearson): 
							best_pearson = aux[2]
							best_match = y
							best_dist = cur_dist
					else: 
						best_match = y
						best_dist = cur_dist
			else: 
				cur_dist = aux[0]
				if cur_dist <= best_dist: 
					best_dist = cur_dist
					best_match = y 

		for y in temp_list:
			matrix[x][y].append(best_match)

	return matrix

def set_paramets(c, matrix_type, concorrente=False, new_formula=False):
	
	test_list = list(test)
	temp_list = list(template)
	belongs = False 
	tp = tn = fp = fn = 0.0 
	c = get_best_match(c,concorrente, new_formula)
	for x in test_list:
		for y in temp_list: 
			aux = c[x][y]
			if(matrix_type == 1 or matrix_type == 2):
				if(matrix_type == 1): 
					if(concorrente):
						belongs = checkClass( aux[1], x)
						#if(not belongs): 
						#	print(aux[1], x)
					elif(new_formula): 
						belongs = checkClass(aux[9], x)
					else: 
						belongs = checkClass( aux[7], x)
						#c[x][y].append(belongs)
					#print(belongs, x, aux[5])
				 
			(tp, fp, fn, tn) = set_matrix(tp, fp, fn, tn, belongs, True)
	return (tp, fp, fn, tn)

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
	result_file_path = sys.argv[1]
	matrix_type = int(input('1. Best Mach\n2.Sair\n'))
	
	while(matrix_type < 2):
		
		tp = tn = fp = fn = 0.0 
		i = 0
		pCoef = 0 
		matching_result = False 
		with open(result_file_path, 'r') as file: 
			data = file.read() 
			data = data.split()
			size = len(data)
		
		while i<(size - 8):
			 
			test_name = data[i]
			template_name = data[i+1]
			comp_type = str2bool(data[i+2])

			if(not comp_type): 
				pCoef = float(data[i+3])
				matching_result = str2bool(data[i+4])
				distance = float(data[i+5])#*math.sqrt(3)
				
			else: 
				distance = float(data[i+5])#*math.sqrt(2)

			sigma = float(data[i+6])
			alpha = float(data[i+7])
			beta = float(data[i+8])
			test.add(test_name)
			template.add(template_name)
			test_result = [comp_type, matching_result, pCoef, distance, sigma, alpha, beta]

			if test_name not in results.keys():
				results[test_name] = {}
				if template_name not in results[test_name].keys():
					results[test_name][template_name] = []

			results[test_name][template_name] = test_result
			i += 9
		
		(tp, fp, fn, tn) = set_paramets(results, matrix_type, concorrente=False)
		(a,b) = make_matrix(results, matrix_type-1)
		#print(test)
		print(show_matrix(a, b, matrix_type))
		get_matrix_parameters(tp, tn, fp, fn)
		print('\n\n Nova Formula' )
		r = mat(results)
		print(r)
		matrix_type = int(input('1. Best Mach\n2. Sair\n'))

if __name__ == '__main__':
	main()

