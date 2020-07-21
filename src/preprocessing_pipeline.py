import os
import glob
from image_utils import imgtonumpy, numpytoimg, normalization, shape_as
from registration import registration

import numpy
import cv2
#from matplotlib import pyplot as plt

root = '/home/ubuntu/dataset_claire_1/'

ref_numpy = imgtonumpy("reference.png").astype(numpy.float)
patients = glob.glob(os.path.join(root,"*"))
for patient_path in patients:
	files = glob.glob(patient_path+"/**/*.png", recursive=True)
	if len(files)>0:
			for filex in files:
				print(filex)
				x = imgtonumpy(filex).astype(numpy.float)
				f = open("cxr_path.csv", "a")
				f.write("{}\t{}\t{}\t{}\n".format(filex, x.shape, x.min(), x.max()))
				f.close()
				x = shape_as(x)
				x = normalization(x)
				x = cv2.GaussianBlur(x,(5,5),cv2.BORDER_DEFAULT)
				#if "lateral" not in filex:
				#	x = registration(x, ref_numpy)
				numpytoimg(x, filex)
