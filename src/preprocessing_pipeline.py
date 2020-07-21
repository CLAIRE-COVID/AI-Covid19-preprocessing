import os
import glob
from image_utils import imgtonumpy, numpytoimg, normalization
from registration import registration

import numpy
import cv2
#from matplotlib import pyplot as plt

root = '/home/tarta/dataset_claire/V1.0/bimcv_covid19_posi_subjects_1/'

ref_numpy = imgtonumpy("reference.png").astype(numpy.float)
patients = glob.glob(os.path.join(root,"*"))
for patient_path in patients:
	files = glob.glob(patient_path+"/**/*.png", recursive=True)
	print(files)
	if len(files)>0:
			print(files)
			for file in files:
				x = imgtonumpy(file).astype(numpy.float)
				x = normalization(x)
				x = cv2.GaussianBlur(x,(5,5),cv2.BORDER_DEFAULT)
				x = registration(x, ref_numpy)
				numpytoimg(x, file)
