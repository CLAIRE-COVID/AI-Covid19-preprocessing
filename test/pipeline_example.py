import os
import cv2
import glob
from PIL import Image, ImageFilter
import numpy as np
import argparse
import scipy.fftpack as fp

im2freq = lambda data: fp.rfft(fp.rfft(data, axis=0), axis=1)

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

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
	parser.add_argument('--input', type=str, default="")
	parser.add_argument('--output', type=str, default="")
	args = parser.parse_args()
	main_dir = args.output
	os.makedirs(main_dir, exist_ok=True)
	for str in ['train', 'test']:
		this_dir = main_dir + '/' + str
		os.makedirs(this_dir, exist_ok=True)
		for str2 in ['TA.HG', 'TA.LG']:
			this_dir_2 = this_dir + '/' + 'TA'
			os.makedirs(this_dir_2, exist_ok=True)
			equalize(args.input + '/' + str + '/' + str2 + '/', this_dir_2 + '/')
		for str2 in ['TVA.HG', 'TVA.LG']:
			this_dir_2 = this_dir + '/' + 'TVA'
			os.makedirs(this_dir_2, exist_ok=True)
			equalize(args.input + '/' + str + '/' + str2 + '/', this_dir_2 + '/')
