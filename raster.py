import matplotlib.pyplot as plt
from scipy import misc
import numpy as np


def pixel_value(image, x, y):
	value = 0

	# print("x, y: ", (x, y))
	# print("type(image): ", type(image))
	# print("image.shape: ", image.shape)
	try:
		value = image[y][x]
		# print("value = ", value)
		#print("type(value) = ", type(value))

		if type(value) == np.ndarray:
			#print("value = ", value)
			#print("is list = ", value[0])
			return int(value[0])

		if not(type(value) == int) or not(type(value) == float):
			#print("value[0] = ", value[0])
			if not(type(value[0]) == int):
				#print("value[0][0] = ", value[0][0])
				return int(value[0][0])
			#print("return * int(value[0]):", int(value[0]))
			return int(value[0])
		else:
			#print("return 2, value: ", value)
			return value
	except:
		#print("except val:", value)
		return value
		#pass
		#print("Fora dos limites da imagem")

	#print("value in px val: ", value)
	return value

def has_border(image, row, col):
	row_image = image[row][col:]

	for cel in row_image:
		if cel[0] == 255:
			return True
	return False

def is_not_tangent(image, x, y):
	if pixel_value(image, x-2, y-2) or pixel_value(image, x-1, y-2) or pixel_value(image, x, y-2) or pixel_value(image, x+1, y-2) or pixel_value(image, x+2, y-2):
		if pixel_value(image, x-2, y+2) or pixel_value(image, x-1, y+2) or pixel_value(image, x, y+2) or pixel_value(image, x+1, y+2) or pixel_value(image, x+2, y+2):
			return True
		else:
			return False
	else:
		return False

def check_pixels_bottom_top_in_line(image, tg_border_pixels, root_pixels=[]):
	have_pixel_bottom = False
	have_pixel_top = False

	tg_border_pixels.sort(key=lambda p: int(p[0])) # ordenando pelo x

	(x0, y0) = tg_border_pixels[0]
	(xf, yf) = tg_border_pixels[-1]

	tangent_pixels = []
	for xi in range(x0, xf+1):
		tangent_pixels.append((xi, y0))

	for (x,y) in tangent_pixels:
		top_pixel11_is_blank = pixel_value(image,x-1,y-1)==255 and not((x-1,y-1) in root_pixels)
		top_pixel12_is_blank = pixel_value(image,  x,y-1)==255 and not((x  ,y-1) in root_pixels)
		top_pixel13_is_blank = pixel_value(image,x+1,y-1)==255 and not((x+1,y-1) in root_pixels)

		bop_pixel11_is_blank = pixel_value(image,x-1,y+1)==255 and not((x-1,y+1) in root_pixels)
		bop_pixel12_is_blank = pixel_value(image,  x,y+1)==255 and not((x  ,y+1) in root_pixels)
		bop_pixel13_is_blank = pixel_value(image,x+1,y+1)==255 and not((x+1,y+1) in root_pixels)

		top_white_pixel    = top_pixel11_is_blank or top_pixel12_is_blank or top_pixel13_is_blank
		bottom_white_pixel = bop_pixel11_is_blank or bop_pixel12_is_blank or bop_pixel13_is_blank

		if top_white_pixel:
			have_pixel_top = True

		if bottom_white_pixel:
			have_pixel_bottom = True

	return have_pixel_bottom^have_pixel_top


def find_tangent_borders_in_row(image, rows, cols, y_row):
	tangents_border_pixels_pairs = []

	sentido_inicial = 0
	sentido_final = 0
	for y in [y_row]:
		tg_border_pixels_temp = []
		for x in range(cols):
			if pixel_value(image, x, y) == 255: # o atual pixel eh branco
				top_white_pixel    = (pixel_value(image,x-1,y-1)==255) or (pixel_value(image,x,y-1)==255) or (pixel_value(image,x+1,y-1)==255)
				bottom_white_pixel = (pixel_value(image,x-1,y+1)==255) or (pixel_value(image,x,y+1)==255) or (pixel_value(image,x+1,y+1)==255)

				if (top_white_pixel or bottom_white_pixel): #tem branco em cima ou embaixo
					if pixel_value(image, x-1, y) == 0: #and pixel_value(image, x+1, y) == 255: # o anterior eh preto (entrando)
						if top_white_pixel:
							sentido_inicial = 1 
						if bottom_white_pixel:
							sentido_inicial = -1

						if top_white_pixel and bottom_white_pixel:
							sentido_inicial = 0

						if sentido_inicial != 0:
							tg_border_pixels_temp.append((x,y))

					if pixel_value(image, x+1, y) == 0: #o proximo eh preto (saindo)
						if top_white_pixel:
							sentido_final = 1
						if bottom_white_pixel:
							sentido_final =-1
			
						if top_white_pixel and bottom_white_pixel:
							sentido_final = 0

						if sentido_inicial == sentido_final and sentido_inicial != 0:
							# print("IS TANGENT!")
							# print("y = ", y)
							tg_border_pixels_temp.append((x,y))

							if check_pixels_bottom_top_in_line(image, tg_border_pixels_temp):
								tangents_border_pixels_pairs += [tg_border_pixels_temp]
						
						tg_border_pixels_temp = []
						sentido_inicial = 0
						sentido_final = 0


	return tangents_border_pixels_pairs


def find_tangent_borders(image, rows, cols):
	tangents_border_pixels_pairs = []

	sentido_inicial = 0
	sentido_final = 0
	for y in range(rows):
		tg_border_pixels_temp = []
		for x in range(cols):
			if pixel_value(image, x, y) == 255: # o atual pixel eh branco
				top_white_pixel    = (pixel_value(image,x-1,y-1)==255) or (pixel_value(image,x,y-1)==255) or (pixel_value(image,x+1,y-1)==255)
				bottom_white_pixel = (pixel_value(image,x-1,y+1)==255) or (pixel_value(image,x,y+1)==255) or (pixel_value(image,x+1,y+1)==255)

				# entrando
				if (top_white_pixel or bottom_white_pixel): #tem branco em cima ou embaixo
					if pixel_value(image, x-1, y) == 0: #and pixel_value(image, x+1, y) == 255: # o anterior eh preto (entrando)
						if top_white_pixel:
							sentido_inicial = 1 
						if bottom_white_pixel:
							sentido_inicial = -1

						if top_white_pixel and bottom_white_pixel:
							sentido_inicial = 0

						if sentido_inicial != 0:
							tg_border_pixels_temp.append((x,y))

					if pixel_value(image, x+1, y) == 0: #o proximo eh preto (saindo)
						if top_white_pixel:
							sentido_final = 1
						if bottom_white_pixel:
							sentido_final =-1
			
						if top_white_pixel and bottom_white_pixel:
							sentido_final = 0

						if sentido_inicial == sentido_final and sentido_inicial != 0:
							# print("IS TANGENT!")
							# print("y = ", y)
							tg_border_pixels_temp.append((x,y))

							if check_pixels_bottom_top_in_line(image, tg_border_pixels_temp):
								tangents_border_pixels_pairs += [tg_border_pixels_temp]
						
						tg_border_pixels_temp = []
						sentido_inicial = 0
						sentido_final = 0


	return tangents_border_pixels_pairs

def expand_row(image, row_blank_pixels, rows, cols, y_inc):
	first_blank = row_blank_pixels[0]
	last_blank  = row_blank_pixels[-1]

	# print("first_blank: ", first_blank)
	# print("last_blank: ", last_blank)

	(xl, yl) = last_blank
	xi = xl
	last_part = []
	y_dec = -y_inc
	is_valid = True
	while xi < cols:
		
		if pixel_value(image, xi, yl) == 255:
			last_part.append((xi, yl))
		else: # Se preto, pare
			break
		xi += 1

	if last_part:
		(x0, y0) = last_part[0]
		(x1, y1) = last_part[-1]

	# ---------------------
	(xf, yf) = first_blank
	xi = xf
	first_part = []
	y_dec = -y_inc
	is_valid = True
	while xi >= 0:
		if pixel_value(image, xi, yf) == 255:
			first_part.append((xi, yf))
		else: # Se preto, pare
			break
		xi -= 1

	if first_part:
		(x0, y0) = first_part[0]
		(x1, y1) = first_part[-1]


	if is_valid == False:
		row_blank_pixels = []

	complete_line = first_part + row_blank_pixels + last_part
	complete_line.sort(key=lambda p: int(p[0])) # ordenando pelo x
	

	return first_part + row_blank_pixels + last_part

def expand_inactive_pixels(image, rows, cols, inactive_pixels):
	new_inactive_pixels = []

	for (x, y) in inactive_pixels:
		nextPixelVal = pixel_value(image, x+1, y)
		prevPixelVal = pixel_value(image, x-1, y)

		x_inc = 0
		xi_values = []
		if nextPixelVal == 255: # o proximo eh branco (entrando)
			x_inc = 1
			xi_values = list(np.arange(x, cols, 1))
		else:
			x_inc = -1
			xi_values = list(np.arange(x, -1, -1))

		for xi in xi_values:
			top_pixel11_is_blank = pixel_value(image,xi-1,y-1)==255 #and not((xi-1,y-1) in root_pixels)
			top_pixel12_is_blank = pixel_value(image,  xi,y-1)==255 #and not((xi  ,y-1) in root_pixels)
			top_pixel13_is_blank = pixel_value(image,xi+1,y-1)==255 #and not((xi+1,y-1) in root_pixels)

			bop_pixel11_is_blank = pixel_value(image,xi-1,y+1)==255 #and not((xi-1,y+1) in root_pixels)
			bop_pixel12_is_blank = pixel_value(image,  xi,y+1)==255 #and not((xi  ,y+1) in root_pixels)
			bop_pixel13_is_blank = pixel_value(image,xi+1,y+1)==255 #and not((xi+1,y+1) in root_pixels)

			top_white_pixel    = top_pixel11_is_blank or top_pixel12_is_blank or top_pixel13_is_blank
			bottom_white_pixel = bop_pixel11_is_blank or bop_pixel12_is_blank or bop_pixel13_is_blank

			if bottom_white_pixel or top_white_pixel:
				break
			else:
				new_inactive_pixels.append((xi,y))

	return inactive_pixels + new_inactive_pixels


def find_extreme_relevant_root_pixels(image, pixel_line, root_pixels):
	relevant_root_pixels = []
	#root_pixels = []

	for (x,y) in pixel_line:
		top_pixel11_is_blank = pixel_value(image,x-1,y-1)==255 and not((x-1,y-1) in root_pixels)
		top_pixel12_is_blank = pixel_value(image,  x,y-1)==255 and not((x  ,y-1) in root_pixels)
		top_pixel13_is_blank = pixel_value(image,x+1,y-1)==255 and not((x+1,y-1) in root_pixels)

		bop_pixel11_is_blank = pixel_value(image,x-1,y+1)==255 and not((x-1,y+1) in root_pixels)
		bop_pixel12_is_blank = pixel_value(image,  x,y+1)==255 and not((x  ,y+1) in root_pixels)
		bop_pixel13_is_blank = pixel_value(image,x+1,y+1)==255 and not((x+1,y+1) in root_pixels)

		top_white_pixel    = top_pixel11_is_blank or top_pixel12_is_blank or top_pixel13_is_blank
		bottom_white_pixel = bop_pixel11_is_blank or bop_pixel12_is_blank or bop_pixel13_is_blank

		if (top_white_pixel or bottom_white_pixel): #tem branco em cima ou embaixo
			relevant_root_pixels.append((x,y))


	relevant_root_pixels.sort(key=lambda p: int(p[0]))

	#print("len(relevant_root_pixels) = ", len(relevant_root_pixels))

	(first_pixel, last_pixel) = (None, None)
	if relevant_root_pixels: 
		(first_pixel, last_pixel) = relevant_root_pixels[0], relevant_root_pixels[-1]
		(xf, yf) = first_pixel
		(xl, yl) = last_pixel

		dx = xf-xl

		if dx <= 1 and len(relevant_root_pixels) == 2 and  len(pixel_line) > 2:
			(first_pixel, last_pixel) = (None, None)

	return (first_pixel, last_pixel)


def is_root_tangent_line(image, pixel_line, root_pixels, all_tangent_pixels):
	#first_pixel = pixel_line[0]
	#last_pixel  = pixel_line[-1]
	# print("-------------------------------")
	# print("pixel_line[0] = ", pixel_line[0])
	# print("pixel_line[-1] = ", pixel_line[-1])

	root_pixels_original = root_pixels

	root_pixels += all_tangent_pixels

	pixel_line = list(set(pixel_line)) #eliminando duplicatas
	pixel_line.sort(key=lambda p: int(p[0]))
	print("len(pixel_line) = ", len(pixel_line))
	print("pixel_line = ", pixel_line)
	(first_pixel, last_pixel) = find_extreme_relevant_root_pixels(image, pixel_line, root_pixels)
	is_tangent = False
	print("first_pixel: ", first_pixel)
	print("last_pixel: ", last_pixel)
	print("-------------------------")

	if (first_pixel == None or last_pixel == None):

		return is_tangent


	root_pixels += all_tangent_pixels

	pixel_borders = [first_pixel, last_pixel]
	
	tg_border_pixels_temp = []

	sentido_inicial = 0
	sentido_final = 0


	for i,(x,y) in enumerate(pixel_borders):
		top_pixel11_is_blank = pixel_value(image,x-1,y-1)==255 and not((x-1,y-1) in root_pixels)
		top_pixel12_is_blank = pixel_value(image,  x,y-1)==255 and not((x  ,y-1) in root_pixels)
		top_pixel13_is_blank = pixel_value(image,x+1,y-1)==255 and not((x+1,y-1) in root_pixels)

		bop_pixel11_is_blank = pixel_value(image,x-1,y+1)==255 and not((x-1,y+1) in root_pixels)
		bop_pixel12_is_blank = pixel_value(image,  x,y+1)==255 and not((x  ,y+1) in root_pixels)
		bop_pixel13_is_blank = pixel_value(image,x+1,y+1)==255 and not((x+1,y+1) in root_pixels)

		top_white_pixel    = top_pixel11_is_blank or top_pixel12_is_blank or top_pixel13_is_blank
		bottom_white_pixel = bop_pixel11_is_blank or bop_pixel12_is_blank or bop_pixel13_is_blank

		if (top_white_pixel or bottom_white_pixel): #tem branco em cima ou embaixo

			if i == 0:#
				#print("i=0, (x,y) = ", (x,y))
				#if pixel_value(image, x-1, y) == 0: #and pixel_value(image, x+1, y) == 255: # o anterior eh preto (entrando)
				if top_white_pixel:
					sentido_inicial = 1 
				if bottom_white_pixel:
					sentido_inicial = -1

				if top_white_pixel and bottom_white_pixel:
					sentido_inicial = 0

				if sentido_inicial != 0:
					tg_border_pixels_temp.append((x,y))

			if i == 1:
				#print("i=1, (x,y) = ", (x,y))
				#if pixel_value(image, x+1, y) == 0:# or (x+1, y) in root_pixels: #o proximo eh preto (saindo)
				if top_white_pixel:
					sentido_final = 1
				if bottom_white_pixel:
					sentido_final =-1

				if top_white_pixel and bottom_white_pixel:
					sentido_final = 0


				# print("sentido_inicial: ", sentido_inicial)
				# print("sentido_final: ", sentido_final)

				if tg_border_pixels_temp and sentido_inicial == sentido_final and (sentido_inicial != 0):
					is_tangent = True
					# print("IS ROOT TANGENT!")
					# print("sentido_inicial: ", sentido_inicial)
					# print("sentido_final: ", sentido_final)
					# print("y = ", y)
					tg_border_pixels_temp.append((x,y))

				
				tg_border_pixels_temp = []
				sentido_inicial = 0
				sentido_final = 0

	return is_tangent



def rooting_tangents(image, tangent_pixels=[], all_tangent_pixels_pairs=[], rows=0, cols=0):
	blank_inf_pixels = []
	blank_sup_pixels = []
	root_pixels = []

	tangent_pixels.sort(key=lambda p: int(p[0]))
	(x0,y0) = tangent_pixels[0] # corner left
	(x1,y1) = tangent_pixels[1] # corner right

	root_pixels = [(x,y0) for x in range(x0, x1+1)] # expandindo linha tangente inicial e armazenando na lista de root_pixels

	tangent_pixels = root_pixels#[(x,y0) for x in range(x0, x1+1)]#root_pixels
	#root_pixels = []

	#image = paint_pixels(image, rows, cols, tangent_pixels, color_array=[0,0,0]) # pintando de preto os pixels tangentes (removendo)


	all_tangent_pixels = []
	for pair in all_tangent_pixels_pairs:

		pair.sort(key=lambda p: int(p[0]))

		(x0,y0) = pair[0] # corner left
		(x1,y1) = pair[1] # corner right

		tangent_pixels_temp = [(x,y0) for x in range(x0, x1+1)]

		all_tangent_pixels += tangent_pixels_temp

	# analisando se eh para cima ou para baixo
	for (x,y) in tangent_pixels:
		for i in [-1,0,1]:
			if pixel_value(image, x-i, y+1)==255 and not((x-i, y+1) in blank_inf_pixels):
				blank_inf_pixels.append((x-i, y+1))
				
			if pixel_value(image, x-i, y-1)==255 and not((x-i, y-1) in blank_sup_pixels):
				blank_sup_pixels.append((x-i, y-1))
				

	row_level = []
	if len(blank_inf_pixels): # rooting top-down
		y_inc_signal = 1
		row_level = blank_inf_pixels
	else:                     # rooting bottom-up
		y_inc_signal = -1
		row_level = blank_sup_pixels

	is_continuos = True


	y_inc = y_inc_signal

	row_level.sort(key=lambda p: int(p[0]))
	if len(row_level):
		while is_continuos:
			next_row_level = []

			row_level = expand_row(image, row_level, rows, cols, y_inc)
			row_level.sort(key=lambda p: int(p[0])) # ordenando pelo x

			# print("row_level = ", row_level)
			is_continuos = is_continuos_row(row_level)
			# print("is_continuos = ", is_continuos)
			is_tangent = False

			#if is_root_tangent_line(image, row_level, root_pixels):
			#if is_root_tangent_line(image, row_level, root_pixels):

			if is_root_tangent_line(image, row_level, root_pixels, all_tangent_pixels):# is_root_tangent_line(image, row_level, root_pixels): # <----

				#if check_pixels_bottom_top_in_line(image, row_level, all_tangent_pixels + root_pixels):
				#if check_pixels_bottom_top_in_line(image, row_level, root_pixels):
				is_tangent = True

			if is_continuos and is_tangent:
				# print("is_continuos and is_tangent")
				# print("row_level")
				root_pixels += row_level#[row_level[0], row_level[-1]] 
				for (x,y) in row_level:
					for i in [-1,0,1]:
						if pixel_value(image, x+i, y+y_inc)==255 and not((x+i, y+y_inc) in next_row_level):
							next_row_level.append((x+i, y+y_inc))
				next_row_level.sort(key=lambda p: int(p[0])) # ordenando pelo x
			row_level = next_row_level

			#print("is_continuos: ", is_continuos)
			
			if not(len(row_level)):
				#print("row_level: ", row_level)
				break

	return root_pixels + tangent_pixels

def find_tangents(image, rows, cols):
	corner_pixels = []
	for y in range(rows):
		for x in range(cols):
			if is_corner(image, x, y):
				corner_pixels.append((x,y))
	return corner_pixels

def is_tangent(image, x, y):
	value = False
	if pixel_value(image, x, y) == 255:

		if top_blank ^ bot_blank:  # either top row or bottom row must be blank (but not both)
			im_corner = True

		prev2_blank = pixel_value(image, x-2, y) == 0
		prev_blank =  pixel_value(image, x-1, y) == 0
		next_blank =  pixel_value(image, x+1, y) == 0
		next2_blank = pixel_value(image, x+2, y) == 0
		
		case1pixel = prev_blank and next_blank  # single pixel corner
		case2pixel = prev_blank and not next_blank and next2_blank  # 2-pixel corner (left edge)
		case3pixel = prev2_blank and not prev_blank and next_blank  # 2-pixel corner (right edge)

		def check_row(range_vals, y_shift):
			row_sum = 0
			for i in range_vals:
				row_sum = row_sum + pixel_value(image, x+i, y+y_shift)
			return row_sum == 0

		my_range = []
		if case1pixel:
			my_range = [-1,0,1]

		if case2pixel or case3pixel:
			my_range = [-2,-1,0,1,2]

		top_blank = check_row(my_range,-1)
		bot_blank = check_row(my_range, 1)


	return im_corner

def is_corner(image, x, y):
	value = False
	if pixel_value(image, x, y) == 255:

		im_corner = False

		prev2_blank = pixel_value(image, x-2, y) == 0
		prev_blank = pixel_value(image, x-1, y) == 0
		next_blank = pixel_value(image, x+1, y) == 0
		next2_blank = pixel_value(image, x+2, y) == 0
		
		case1pixel = prev_blank and next_blank  # single pixel corner
		case2pixel = prev_blank and not next_blank and next2_blank  # 2-pixel corner (left edge)
		case3pixel = prev2_blank and not prev_blank and next_blank  # 2-pixel corner (right edge)

		def check_row(range_vals, y_shift):
			row_sum = 0
			for i in range_vals:
				row_sum = row_sum + pixel_value(image, x+i, y+y_shift)
			return row_sum == 0

		my_range = []
		if case1pixel:
			my_range = [-1,0,1]

		if case2pixel or case3pixel:
			my_range = [-2,-1,0,1,2]

		top_blank = check_row(my_range, -1)
		bot_blank = check_row(my_range, 1)
		if top_blank ^ bot_blank:  # either top row or bottom row must be blank (but not both)
			im_corner = True

		#if case2pixel:
		return im_corner

def is_continuos_row_image(image, row_pixels):
	n = len(row_pixels)
	is_continuos = True
	

	for ib,(x, y) in enumerate(row_pixels): # encontrando o prmeiro branco
		if pixel_value(image, x, y) == 255:
			break

	relevant_pixels = []
	for i in range(ib,len(row_pixels)): # pegando os pixels relevantes
		(xi, yi) = row_pixels[i]

		val = 0
		if pixel_value(image, xi, yi)==255:
			relevant_pixels.append((xi, yi))

	for i in range(len(relevant_pixels)-1): # verificando se ha descontinuidade
		(xi, yi) = row_pixels[i]
		(xj, yj) = row_pixels[i+1]

		if xj - xi > 1:
			is_continuos = False

	return is_continuos

def is_continuos_row(row_pixels):
	# print("row_pixels[0] = ", row_pixels[0])
	# print("row_pixels[-1] = ", row_pixels[-1])

	n = len(row_pixels)
	is_continuos = True
	if n == 1:
		is_continuos = False
	else:
		for i in range(0, n-1):
			(xi,     _) = row_pixels[i]
			(x_next, _) = row_pixels[i+1]

			if abs(x_next - xi) > 1:
				is_continuos = False
				break
	#print("is_continuos: ", is_continuos)
	return is_continuos

def rooting_horn(image, corner_pixels=[]):
	blank_inf_pixels = []
	blank_sup_pixels = []
	horn_pixels = corner_pixels
	# analisando se eh para cima ou para baixo
	for (x,y) in corner_pixels:
		for i in range(-1,2):
			if pixel_value(image, x-i, y+1)==255 and not((x-i, y+1) in blank_inf_pixels):
				blank_inf_pixels.append((x-i, y+1))
			if pixel_value(image, x-i, y-1)==255 and not((x-i, y-1) in blank_sup_pixels):
				blank_sup_pixels.append((x-i, y-1))

	row_level = []
	if len(blank_inf_pixels): # rooting top-down
		y_inc_signal = 1
		row_level = blank_inf_pixels
	else:                     # rooting bottom-up
		y_inc_signal = -1
		row_level = blank_sup_pixels

	is_continuos = True

	y_inc = y_inc_signal

	row_level.sort(key=lambda p: int(p[0]))
	if len(row_level):
		while is_continuos:
			next_row_level = []

			is_continuos = is_continuos_row(row_level)

			if is_continuos:
				horn_pixels += row_level
				for (x,y) in row_level:
					for i in [-1, 0, 1]:
						if pixel_value(image, x+i, y+y_inc)==255 and not((x+i, y+y_inc) in next_row_level):
							next_row_level.append((x+i, y+y_inc))
				next_row_level.sort(key=lambda p: int(p[0])) # ordenando pelo x
			row_level = next_row_level
			
			if not(len(row_level)):
				break

	return horn_pixels

def find_corners(image, rows, cols):
	corner_pixels = []
	for y in range(rows):
		for x in range(cols):
			if is_corner(image, x, y):
				corner_pixels.append((x,y))
	return corner_pixels

def find_horn_pixels(image, rows, cols, corner_pixels):
	horn_pixels = []
	for y in range(rows):
		for x in range(cols):
			if (x,y) in corner_pixels:
				corner_pixels_grouped = [(x,y)]
				if (x,y+1) in corner_pixels:
					corner_pixels_grouped.append((x,y+1))
				horn_pixels += rooting_horn(image, corner_pixels_grouped)
	return horn_pixels

def paint_pixels(image, rows, cols, pixels, color_array=[255,0,0]):
	result_img = image
	# result_img = np.zeros([rows, cols,3],dtype=np.uint8)
	# result_img[:] = np.array([0,0,0])

	for (x,y) in pixels:
		result_img[y][x] = np.array(color_array)

	# for y in range(rows):
	# 	for x in range(cols):
	# 		if (x,y) in pixels:
	# 			result_img[y][x] = np.array(color_array)
	# 		else:
	# 			result_img[y][x] = image[y][x]

	return result_img

def orientation(image, x, y, exit=True):
	up_white_pixels_x = []
	down_white_pixels_x = []

	max_x_up = -1
	max_x_down = -1

	pixel_val = 0
	if exit:
		pixel_val = 255

	for i in [-1,0,1]:
		if pixel_value(image, x+i, y-1)==pixel_val:
			up_white_pixels_x.append(x+i)

	for i in [-1,0,1]:
		if pixel_value(image, x+i, y+1)==pixel_val:
			down_white_pixels_x.append(x+i)

	if up_white_pixels_x:
		max_x_up = max(up_white_pixels_x)

	if down_white_pixels_x:
		max_x_down = max(down_white_pixels_x)

	max_x = max(max_x_up, max_x_down)

	orientation = 0
	up_orientation = 0
	down_orientation = 0

	if max_x_up == max_x:
		up_orientation = 1
	if max_x_down == max_x:
		down_orientation = -1

	orientation = up_orientation + down_orientation
	return orientation


def is_corner_old(image, x, y):
	value = False

	count_pixel_in_line_y = 1

	if pixel_value(image, x-1, y):
		count_pixel_in_line_y += 1

	if pixel_value(image, x-2, y):
		count_pixel_in_line_y += 1

	y_2_line = pixel_value(image, x-1, y-2)==255 or pixel_value(image, x, y-2)==255 or pixel_value(image, x+1, y-2)==255
	y_1_line = pixel_value(image, x-1, y-1)==255 or pixel_value(image, x, y-1)==255 or pixel_value(image, x+1, y-1)==255
	yp1_line = pixel_value(image, x-1, y+1)==255 or pixel_value(image, x, y+1)==255 or pixel_value(image, x+1, y+1)==255
	yp2_line = pixel_value(image, x-1, y+2)==255 or pixel_value(image, x, y+2)==255 or pixel_value(image, x+1, y+2)==255

	if yp1_line and y_1_line:
		return False

	bottom_part = yp1_line and yp2_line
	up_part = y_1_line and y_2_line

	if bottom_part^up_part:
		if count_pixel_in_line_y <= 2:
			value = True

	return value

def raster(image, rows, cols, horn_pixels):
	fill_status = False
	result_img = np.zeros([rows, cols,3],dtype=np.uint8)
	result_img[:] = np.array([0,0,0])
	draw = False
	
	for y in range(rows):
		#print("----------------------------ROW:", y, "-----------------------------------------")
		sentido_inicial = 0
		sentido_final = 0
		is_tangent = False
		is_tangent_or_inflexion = False
		for x in range(cols):
			if fill_status:
				#result_img[y][x] = image[y][x]
				#pass
				result_img[y][x] = np.array([255,255,255])
			else:
				#if pixel_value(image, x, y) == 255:
				result_img[y][x] = image[y][x]#np.array([255,255,0])

			top_white_pixel    = (pixel_value(image,x-1,y-1)==255) or (pixel_value(image,x,y-1)==255) or (pixel_value(image,x+1,y-1)==255)
			bottom_white_pixel = (pixel_value(image,x-1,y+1)==255) or (pixel_value(image,x,y+1)==255) or (pixel_value(image,x+1,y+1)==255)

			if pixel_value(image, x, y) == 255 and not((x,y) in horn_pixels): # o atual pixel eh branco

				if pixel_value(image, x+1, y) == 0: #eh branco e o proximo eh preto (saindo)
					if top_white_pixel or bottom_white_pixel: #tem branco em cima ou embaixo

						# ------------------------
						if top_white_pixel:
							sentido_final = 1
						if bottom_white_pixel:
							sentido_final =-1
			
						if top_white_pixel and bottom_white_pixel:
							sentido_final = 0
						# ------------------------

						fill_status = not(fill_status)

					sentido_inicial = 0
					sentido_final = 0
					is_tangent_or_inflexion = False

				elif pixel_value(image, x-1, y) == 0:
					if (top_white_pixel or bottom_white_pixel):#pixel_value(image, x+1, y) == 255:# and (top_white_pixel^bottom_white_pixel): #anterior eh preto E o proximo eh branco E xor(cima, baixo)


						# ------------------------
						if top_white_pixel:
							sentido_inicial = 1 
						if bottom_white_pixel:
							sentido_inicial = -1

						if top_white_pixel and bottom_white_pixel:
							sentido_inicial = 0
						# ------------------------
						
						if sentido_inicial!=0:
							is_tangent_or_inflexion = True 


		fill_status = False
	return result_img

def detection_hole(image, rows, cols):
	hole_pixels = []

	for x in range(0, cols):
		for y in range(0, rows):
			hole_pixels_temp = have_hole(image, rows, cols, x, y)
			if hole_pixels_temp:
				hole_pixels += hole_pixels_temp
	return hole_pixels

def have_hole(image, rows, cols, x, y):
	hole_pixels = []
	is_closed = False
	if is_initial_hole_corner(image, rows, cols, x, y):
		xi = x+1
		while is_mid_hole(image, xi, y):
			hole_pixels.append((xi,y))
			if pixel_value(image,xi+1,y) == 255:#is_final_hole_corner(image, rows, cols, x, y):
				is_closed = True	
				break
			xi += 1

	if not(is_closed):
		hole_pixels = []

	return hole_pixels

# hole detectors
def is_mid_hole(image, x, y):
	blank_pixel_up   = (pixel_value(image,x,y-1) == 255)
	blank_pixel_down = (pixel_value(image,x,y+1) == 255)
	black_pixel = (pixel_value(image,x,y) == 0)

	return (black_pixel and blank_pixel_down and blank_pixel_up)


def is_final_hole_corner(image, rows, cols, x, y):
	blank_pixel_up   = (pixel_value(image,x,y-1) == 255)
	blank_pixel_down = (pixel_value(image,x,y+1) == 255)
	next_pixel_blank = (pixel_value(image, x+1, y) == 255)
	return blank_pixel_up and blank_pixel_down and next_pixel_blank

def is_initial_hole_corner(image, rows, cols, x, y):
	diagonal_up   = (pixel_value(image,x+1,y-1) == 255)
	diagonal_down = (pixel_value(image,x+1,y+1) == 255)
	next_pixel_black = (pixel_value(image, x+1, y) == 0)
	white_pixel = (pixel_value(image, x, y) == 255)

	return diagonal_up and diagonal_down and next_pixel_black and white_pixel



# ----- Tangent detectors
def is_initial_tangent_corner(image, x, y, reverse_direction=False):
	#diagonal_up_black   = (pixel_value(image,x+1,y-1) == 0)
	y_inc = 1
	if reverse_direction:
		y_inc = -1

	up_black_side = (pixel_value(image,x-1,y-y_inc) == 0) and (pixel_value(image,x,y-y_inc) == 0) and (pixel_value(image,x+1,y-y_inc) == 0)


	prev_black = (pixel_value(image, x-1, y) == 0)
	next_white = (pixel_value(image, x+1, y) == 255)
	white_pixel = (pixel_value(image, x, y) == 255)	#return (diagonal_down_white or down_white) and up_black_side and white_pixel and prev_pixel_black
	#return (diagonal_down_white) and up_black_side and white_pixel and prev_pixel_black
	return up_black_side and white_pixel and prev_black and next_white

def is_final_tangents_corner(image, x, y, reverse_direction=False):
	#print("is_final_tangents_corner")
	y_inc = 1
	if reverse_direction:
		y_inc = -1

	up_black_side = (pixel_value(image,x-1,y-y_inc) == 0) and (pixel_value(image,x,y-y_inc) == 0) and (pixel_value(image,x+1,y-y_inc) == 0)

	next_black = (pixel_value(image, x+1, y) == 0)
	prev_white = (pixel_value(image, x-1, y) == 255)
	white_pixel = (pixel_value(image, x, y) == 255)
	
	#return (diagonal_down_white or down_white) and up_black_side and white_pixel and next_pixel_black
	return up_black_side and white_pixel and prev_white and next_black


def is_mid_tangent(image, x, y, reverse_direction=False):
	y_inc = 1
	if reverse_direction:
		y_inc = -1

	#print("is_mid_tangent")
	up_black   = (pixel_value(image,x,y-y_inc) == 0)
	down_black = (pixel_value(image,x,y+y_inc) == 0)

	down_white = (pixel_value(image,x,y+y_inc) == 255)
	white_pixel = (pixel_value(image, x, y) == 255)
	#print(up_black)
	#print(down_white)
	#print(white_pixel)

	return up_black and white_pixel
	#return (up_black and down_white and white_pixel) or (up_black and down_black and white_pixel)

def is_external_tangent_1px(image, x, y, reverse_direction=False):

	y_inc = 1
	if reverse_direction:
		y_inc = -1

	up_black_side = (pixel_value(image,x-1,y-y_inc) == 0) and (pixel_value(image,x,y-y_inc) == 0) and (pixel_value(image,x+1,y-y_inc) == 0)

	
	white_pixel = (pixel_value(image, x, y) == 255)
	prev_black = (pixel_value(image, x-1, y) == 0)
	next_black = (pixel_value(image, x+1, y) == 0)

	return up_black_side and white_pixel and prev_black and next_black

def continuity_test(image, row_pixels):
	values_list = []

	for (x, y) in row_pixels:
		values_list.append(pixel_value(image, x, y))

	values_list = [0] + values_list + [0]

	count_diff = 0
	for i in range(0, len(values_list)-1):
		if not(values_list[i] == values_list[i+1]):
			count_diff += 1

	return count_diff == 2 or count_diff == 0

def is_external_tangent(image, rows, cols, x, y, reverse_direction=False, external_flag=True):
	external_tangent_pixels = []

	y_inc = 1
	if reverse_direction:
		y_inc = -1

	is_external_tg = False
	if is_external_tangent_1px(image, x, y, reverse_direction=reverse_direction):
		external_tangent_pixels.append((x,y))
		is_external_tg = True
	elif is_initial_tangent_corner(image, x, y, reverse_direction=reverse_direction):
		#print("is_initial_tangent_corner")
		external_tangent_pixels.append((x,y))
		xi = x+1
		while is_mid_tangent(image, xi, y, reverse_direction=reverse_direction) or is_final_tangents_corner(image, xi, y, reverse_direction=reverse_direction):
			#print("is_mid_tangent ")
			external_tangent_pixels.append((xi,y))
			if is_final_tangents_corner(image, xi, y, reverse_direction=reverse_direction):#is_final_hole_corner(image, rows, cols, x, y):
				#print("is_final_tangents_corner!")
				is_external_tg = True	
				break
			xi += 1

	if not(is_external_tg):
		external_tangent_pixels = []


	if external_tangent_pixels and external_flag:
		# testando continuidade da linha de baixo
		external_tangent_pixels = list(set(external_tangent_pixels))
		external_tangent_pixels.sort(key=lambda p: int(p[0]))

		below_row_pixels = []

		(x,y) = external_tangent_pixels[0]
		below_row_pixels.append((x-1,y+y_inc))

		for (x,y) in external_tangent_pixels:
			below_row_pixels.append((x,y+y_inc))

		(x,y) = external_tangent_pixels[-1]
		below_row_pixels.append((x+1,y+y_inc))

		if not(continuity_test(image, below_row_pixels)):
			external_tangent_pixels = []

	return external_tangent_pixels

def is_small_gap(image, x, y):
	#up_black_side = (pixel_value(image,x-1,y-y_inc) == 0) and (pixel_value(image,x,y-y_inc) == 0) and (pixel_value(image,x+1,y-y_inc) == 0)
	prev_white  = (pixel_value(image,x-1,y) == 255)
	next_white  = (pixel_value(image,x+1,y) == 255)
	black_pixel = (pixel_value(image, x, y) == 0)
	mid_pixels = prev_white and black_pixel and next_white
	 
	topside_white = (pixel_value(image,x,y-1) == 255) and (pixel_value(image,x,y-1) == 255)

	down_pixel_white = (pixel_value(image,x,y+1) == 255)

	# case1 = top_pixel_white and down_pixel_black
	# case2 = top_pixel_white and down_pixel_black

	return mid_pixels and (topside_white or down_pixel_white) # errado

	#return prev_white and black_pixel and next_white


def fill_gaps(image, rows, cols):
	gaps_pixels = []

	for y in range(0, rows):
		for x in range(0, cols):
			if is_small_gap(image, x, y):
				#print("fill_gaps")
				gaps_pixels.append((x, y))

	return gaps_pixels


def simple_rooting(tg_border_pixels_pair):
	tg_border_pixels_pair.sort(key=lambda p: int(p[0]))
	(x0,y0) = tg_border_pixels_pair[0] # corner left
	(x1,y1) = tg_border_pixels_pair[1] # corner right
	root_pixels = [(x,y0) for x in range(x0, x1+1)] # expandindo linha tangente inicial e armazenando na lista de root_pixels
	tangent_pixels_temp = root_pixels

	return tangent_pixels_temp

def remove_apendices(image, rows, cols, reverse_direction=False):
	# CONTINUAR AQUI! adaptar para que possa ser usado TOP-DOWN e BOTTOM-UP
	external_tangents = []
	y_values = list(range(0, rows))

	if reverse_direction:
		y_values.reverse()

	for y in y_values:
		for x in range(0, cols):
			
			#external_tg_pixels = is_external_tangent(image, rows, cols, x, y, reverse_direction=reverse_direction)
			if tg_border_pixels_pairs:
				#print("tg_border_pixels_pair = ", tg_border_pixels_paira)
				for pair in tg_border_pixels_pairs:
					external_tg_pixels = simple_rooting(pair)

				y_inc = 1
				if reverse_direction:
					y_inc = -1
				(x0, _) = external_tg_pixels[0]
				(xf, _) = external_tg_pixels[-1]
				below_row = [(x,y+y_inc) for x in range(x0, xf+1)]
				#print("removendo external tangents")

				if is_continuos_row_image(image, below_row):
					external_tangents += external_tg_pixels
					image = paint_pixels(image, rows, cols, external_tg_pixels, color_array=[0,0,0])

	return image, external_tangents


# def remove_external_tangents(image, rows, cols, reverse_direction=False):
# 	# CONTINUAR AQUI! adaptar para que possa ser usado TOP-DOWN e BOTTOM-UP
# 	external_tangents = []
# 	y_values = list(range(0, rows))

# 	if reverse_direction:
# 		y_values.reverse()

# 	for y in y_values:
# 		for x in range(0, cols):

# 			tg_border_pixels_pairs = find_tangent_borders_in_row(image, rows, cols, y)
			
# 			#external_tg_pixels = is_external_tangent(image, rows, cols, x, y, reverse_direction=reverse_direction)
# 			if tg_border_pixels_pairs:
# 				#print("tg_border_pixels_pair = ", tg_border_pixels_paira)
# 				for pair in tg_border_pixels_pairs:
# 					external_tg_pixels = simple_rooting(pair)

# 				y_inc = 1
# 				if reverse_direction:
# 					y_inc = -1
# 				(x0, _) = external_tg_pixels[0]
# 				(xf, _) = external_tg_pixels[-1]
# 				below_row = [(x,y+y_inc) for x in range(x0, xf+1)]
# 				#print("removendo external tangents")

# 				if is_continuos_row_image(image, below_row):
# 					external_tangents += external_tg_pixels
# 					image = paint_pixels(image, rows, cols, external_tg_pixels, color_array=[0,0,0])

# 	return image, external_tangents

def find_internal_tangents(image, rows, cols, reverse_direction=False):
	# CONTINUAR AQUI! adaptar para que possa ser usado TOP-DOWN e BOTTON-UP (verificar o kirby 39)
	internal_tangents = []
	y_values = list(range(0, rows))

	if reverse_direction:
		y_values.reverse()

	internal_tangents = []
	for y in y_values:
		for x in range(0, cols):
			external_tg_pixels = is_external_tangent(image, rows, cols, x, y, reverse_direction=reverse_direction, external_flag=False)

			if external_tg_pixels:
				#print("removendo external tangents")
				#image = paint_pixels(image, rows, cols, external_tg_pixels, color_array=[0,0,0])
				internal_tangents += external_tg_pixels
	return internal_tangents	

def remove_external_tangents(image, rows, cols, reverse_direction=False):
	# CONTINUAR AQUI! adaptar para que possa ser usado TOP-DOWN e BOTTON-UP (verificar o kirby 39)

	external_tangents = []
	y_values = list(range(0, rows))

	if reverse_direction:
		y_values.reverse()

	external_tangents = []
	for y in y_values:
		for x in range(0, cols):
			external_tg_pixels = is_external_tangent(image, rows, cols, x, y, reverse_direction=reverse_direction)

			if external_tg_pixels:
				#print("removendo external tangents")
				image = paint_pixels(image, rows, cols, external_tg_pixels, color_array=[0,0,0])
				external_tangents += external_tg_pixels
	return image, external_tangents




def is_noise(image, x, y):
	#if pixel_value(image, x, y) == 255:
	top_side = (pixel_value(image,x-1,y-1) == 255) and (pixel_value(image,x,y-1) == 255) and (pixel_value(image,x+1,y-1) == 255)
	bot_side = (pixel_value(image,x-1,y+1) == 255) and (pixel_value(image,x,y+1) == 255) and (pixel_value(image,x+1,y+1) == 255)
	left_side = (pixel_value(image,x-1,y+1) == 255) and (pixel_value(image,x-1,y) == 255) and (pixel_value(image,x-1,y-1) == 255)
	right_side = (pixel_value(image,x+1,y+1) == 255) and (pixel_value(image,x+1,y) == 255) and (pixel_value(image,x+1,y-1) == 255)

	top_pixel = (pixel_value(image,x,y-1) == 255)
	bot_pixel = (pixel_value(image,x,y+1) == 255)
	left_pixel = (pixel_value(image,x-1,y) == 255)
	right_pixel = (pixel_value(image,x+1,y) == 255)

	diagonal_1 = (pixel_value(image,x-1,y-1) == 255)
	diagonal_2 = (pixel_value(image,x+1,y-1) == 255)
	diagonal_3 = (pixel_value(image,x-1,y+1) == 255)
	diagonal_4 = (pixel_value(image,x+1,y+1) == 255)

	side_list = [top_side, bot_side, left_side, right_side]
	n_sides = sum(side_list)

	pixel_list = [top_pixel, bot_pixel, left_pixel, right_pixel]
	n_pixels = sum(pixel_list)

	diagonal_list = [diagonal_1, diagonal_2, diagonal_3, diagonal_4]
	n_diagonal = sum(diagonal_list)

	return (pixel_value(image, x, y)==255) and (n_sides==1) and (n_pixels==1) and (n_diagonal==2)

def is_double_noise(image, x, y):
	#if pixel_value(image, x, y) == 255:
	mid_pixels = (pixel_value(image,x-1,y) == 0) and (pixel_value(image,x,y) == 255) and (pixel_value(image,x+1,y) == 255) and (pixel_value(image,x+2,y) == 0)

	top_side_black = (pixel_value(image,x-1,y-1) == 0) and (pixel_value(image,x,y-1) == 0) and (pixel_value(image,x+1,y-1) == 0) and (pixel_value(image,x+2,y-1) == 0)
	top_side_white = (pixel_value(image,x-1,y-1) == 255) and (pixel_value(image,x,y-1) == 255) and (pixel_value(image,x+1,y-1) == 255) and (pixel_value(image,x+2,y-1) == 255)
	
	bottom_side_black = (pixel_value(image,x-1,y+1) == 0) and (pixel_value(image,x,y+1) == 0) and (pixel_value(image,x+1,y+1) == 0) and (pixel_value(image,x+2,y+1) == 0)
	bottom_side_white = (pixel_value(image,x-1,y+1) == 255) and (pixel_value(image,x,y+1) == 255) and (pixel_value(image,x+1,y+1) == 255) and (pixel_value(image,x+2,y+1) == 255)

	return (mid_pixels and top_side_black and bottom_side_white) or (mid_pixels and top_side_white and bottom_side_black)


def find_inactive_pixels(image, rows, cols):
	inactive_pixels = []

	for x in range(0, cols):
		for y in range(0, rows):
			if pixel_value(image, x, y) == 255:
				top_white_pixel    = (pixel_value(image,x-1,y-1)==255) or (pixel_value(image,x,y-1)==255) or (pixel_value(image,x+1,y-1)==255)
				bottom_white_pixel = (pixel_value(image,x-1,y+1)==255) or (pixel_value(image,x,y+1)==255) or (pixel_value(image,x+1,y+1)==255)

				if not(top_white_pixel or bottom_white_pixel): #nao tem branco em cima e nem embaixo
					if pixel_value(image, x-1, y) == 0 or pixel_value(image, x+1, y) == 0: # nao pixel branco nem em cima e nem embaixo e eh entrando ou saindo
						inactive_pixels.append((x, y))
				elif is_noise(image, x, y):
					inactive_pixels.append((x, y))
				elif is_double_noise(image, x, y):
					inactive_pixels.append((x, y))
					inactive_pixels.append((x+1, y))

	return inactive_pixels

def add_border(image):
	print("type img = ", type(image))
	#print("len shape: ", len(image.shape))

	(rows, cols, n_channels) = (0,0,0)
	if len(image.shape) == 3:
		(rows, cols, n_channels) = image.shape
	else:
		(rows, cols) = image.shape
		n_channels = 1

	#print("n_channels = ", n_channels)
	temp_image = None

	if n_channels == 1:
		temp_image = np.zeros([rows+2, cols+2],dtype=np.uint8)
		temp_image[:] = 0
	elif n_channels == 3:
		temp_image = np.zeros([rows+2, cols+2, 3],dtype=np.uint8)
		temp_image[:] = [0,0,0]

	for y in range(0, rows):
		for x in range(0, cols):
			if n_channels == 1: # Black and White
				temp_image[y+1][x+1] = image[y][x]
			elif n_channels == 3: #RGB
				temp_image[y+1][x+1][0] = image[y][x][0]
				temp_image[y+1][x+1][1] = image[y][x][1]
				temp_image[y+1][x+1][2] = image[y][x][2]

	return temp_image, rows+2, cols+2

def main_raster(original_image, rows, cols):
	hole_pixels = detection_hole(original_image, rows, cols)
	original_image = paint_pixels(original_image, rows, cols,  hole_pixels, color_array=[255,255,255])
	
	temp_image1 = original_image
	temp_image2 = original_image

	temp_image1, external_tg_pixels1 = remove_external_tangents(temp_image1, rows, cols, reverse_direction=False)
	temp_image2, external_tg_pixels2 = remove_external_tangents(temp_image2, rows, cols, reverse_direction=True)

	external_tg_pixels = external_tg_pixels1 + external_tg_pixels2
	original_image = paint_pixels(original_image, rows, cols, external_tg_pixels, color_array=[0,0,0])

	tangent_pixels = find_internal_tangents(original_image, rows, cols, reverse_direction=False)
	tangent_pixels += find_internal_tangents(original_image, rows, cols, reverse_direction=True)

	raster_img = raster(original_image, rows, cols, tangent_pixels)                
	raster_img = paint_pixels(raster_img, rows, cols, tangent_pixels, color_array=[255,255,255]) # tangent pixels (verde)
	raster_img = paint_pixels(raster_img, rows, cols,  external_tg_pixels, color_array=[255,255,255]) # external tangents (ciano)

	return raster_img


