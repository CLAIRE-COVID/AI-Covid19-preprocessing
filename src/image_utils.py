import cv2
import numpy as np

#assumption: all the images to process are grey-scale
def imgtonumpy(source):
	img = cv2.imread(source)
	if len(img.shape) > 1:
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	return img

def numpytoimg(numpyarray, destination):
	cv2.imwrite(destination, numpyarray)

def equalize_image(source):
	return cv2.equalizeHist(source)

#intended as contrast enhancement
def normalization(source):
	return cv2.normalize(source,None,0,255,cv2.NORM_MINMAX)

#to binarize the image
def threshold(source, T = 0.5):
	T_int = int(np.round(255 * T))
	ret, equ = cv2.threshold(img, T_int, 255,cv2.THRESH_BINARY)
	return equ

def shape_as(source, target_dims=(1024, 1024)):
	max_height = 1024
	max_width = 1024
	height,width = source.shape

	# only shrink if img is bigger than required
	if target_dims[0] < height or target_dims[1] < width:
		# get scaling factor
		scaling_factor = target_dims[0] / float(height)
		if target_dims[1]/float(width) < scaling_factor:
			scaling_factor = target_dims[1] / float(width)
		# resize image
		source = cv2.resize(source, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
	height,width = source.shape
	if target_dims[0] > height or target_dims[1] > width:
		delta_x = int((target_dims[0] - source.shape[0])/2)
		delta_y = int((target_dims[1] - source.shape[1])/2)
		if (delta_x > 0) or (delta_y > 0):
			source = cv2.copyMakeBorder(source,delta_x,delta_x,delta_y,delta_y,cv2.BORDER_CONSTANT,value=[0,0,0])
	return source

def equalize_image(path):
	img = cv2.imread(path)
	#some CXR are saved in RGB....
	if len(img.shape) > 1:
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	#equ = cv2.equalizeHist(img)
	ret, equ = cv2.threshold(img, 127, 255,cv2.THRESH_BINARY_INV)
	#equ = cv2.clip_hist_percent(img)
	return equ


def clip_hist(path, clip_hist_percent=10):
	img = cv2.imread(path)
	#some CXR are saved in RGB....
	if len(img.shape) > 1:
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# Calculate grayscale histogram
	hist = cv2.calcHist([img],[0],None,[256],[0,256])
	hist_size = len(hist)

	# Calculate cumulative distribution from the histogram
	accumulator = []
	accumulator.append(float(hist[0]))
	for index in range(1, hist_size):
	    accumulator.append(accumulator[index -1] + float(hist[index]))

	# Locate points to clip
	maximum = accumulator[-1]
	clip_hist_percent *= (maximum/100.0)
	clip_hist_percent /= 2.0

	# Locate left cut
	minimum_gray = 0
	while accumulator[minimum_gray] < clip_hist_percent:
	    minimum_gray += 1

	# Locate right cut
	maximum_gray = hist_size -1
	while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
	    maximum_gray -= 1

	# Calculate alpha and beta values
	alpha = 255 / (maximum_gray - minimum_gray)
	beta = -minimum_gray * alpha
	auto_result = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
	ret, equ = cv2.threshold(auto_result, 127, 255,cv2.THRESH_BINARY_INV)
	return equ

"""
Equalize
"""
def equalize(source_path, target_path):
	print('entro')
	for this_image_path in glob.glob(source_path+'*.png'):
		print(this_image_path)
		#equalized_image = clip_hist(this_image_path)
		equalized_image = cv2.imread(this_image_path)
		if len(equalized_image.shape) > 1:
			equalized_image = ~(cv2.cvtColor(equalized_image, cv2.COLOR_BGR2GRAY))
		equalized_image = equalized_image - equalized_image.min()
		equalized_image = equalized_image/equalized_image.max()*255
		equalized_image = equalized_image - equalized_image.min()#cv2.equalizeHist(equalized_image)
		#equalized_image = cv2.Sobel(equalized_image,cv2.CV_64F,1,0,ksize=5)#cv2.Laplacian(equalized_image,cv2.CV_64F)
		#thr = int(equalized_image.max()/4)
		#ret, equalized_image = cv2.threshold(equalized_image, 25, 255,cv2.THRESH_BINARY)

		if (equalized_image.shape[0] > 128) and (equalized_image.shape[1] > 128):
			equalized_image = cv2.copyMakeBorder(equalized_image.copy(),112,112,112,112,cv2.BORDER_CONSTANT,value=[0,0,0])
			begin_x = int(np.floor((equalized_image.shape[0] - 224)/2))
			begin_y = int(np.floor((equalized_image.shape[1] - 224)/2))
			cropped_image = equalized_image[begin_x:begin_x+224, begin_y:begin_y+224]
			#freq = im2freq(cropped_image)
			#print(cropped_image.shape)
			#error()
			#cropped_image = cv2.resize(cropped_image, (56,56), interpolation = cv2.INTER_CUBIC)
			cv2.imwrite(target_path + this_image_path.split('/')[-1] ,cropped_image)
