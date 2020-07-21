import os
import glob
from image_utils import imgtonumpy, numpytoimg, normalization, shape_as
from registration import registration

import numpy
import cv2
import json
import pydicom
from os import path
from matplotlib import pyplot as plt

root = '/home/ubuntu/dataset_claire_1/'
#root = '/home/tarta/dataset_claire/V1.0/bimcv_covid19_posi_subjects_1'

os.makedirs("preproc", exist_ok=True)
os.makedirs("excluded", exist_ok=True)

ref_numpy = imgtonumpy("reference.png").astype(numpy.float)
patients = glob.glob(os.path.join(root,"*"))
for patient_path in patients:
	files = glob.glob(patient_path+"/**/*.png", recursive=True)
	if len(files)>0:
			for filex in files:
				if path.exists(filex.replace("png", "json")):
					flag_excluded = False
					if ("ll" in filex) or ("lateral" in filex):
						target_folder = "excluded/"
						flag_excluded = True
					else:
						target_folder = "preprocessed/"
					print(filex)
					x = imgtonumpy(filex)#.astype(numpy.float)
					datax = json.load(open(filex.replace("png", "json")))

					#fix some errors in the JSON exports
					if '00281054' in datax.keys():
						if datax['00281054']['vr'] != ['CS']:
							datax['00281054']['vr'] = ['CS']
					if '20500020' in datax.keys():
						if datax['20500020']['vr'] != ['CS']:
							datax['20500020']['vr'] = ['CS']

					ds1 = pydicom.dataset.Dataset.from_json(datax)
					lut = pydicom.pixel_data_handlers.util.apply_modality_lut(x, ds1).astype(numpy.uint16)
					print(x.max(), lut.max(), lut.shape)
					lut = pydicom.pixel_data_handlers.util.apply_voi_lut(lut, ds1).astype(numpy.float)
					f = open("cxr_path.csv", "a")
					f.write("{}\t{}\t{}\t{}\t{}\t{}".format(filex, lut.shape[0], lut.shape[1], lut.min(), lut.max(), ds1[0x0028,0x0004].value))
					if flag_excluded:
						f.write("\t EXCLUDED\n")
					else:
						f.write("\t  \n")
					f.close()
					lut = normalization(lut)
					if ds1[0x0028,0x0004].value == "MONOCHROME1":
						lut = -1*(lut-255)
					lut = shape_as(lut)
					#x = cv2.GaussianBlur(x,(5,5),cv2.BORDER_DEFAULT)
					#if "lateral" not in filex:
					#	x = registration(x, ref_numpy)
					target_file_name = filex.replace(".png", "_preproc.png")
					#numpytoimg(lut, target_file_name)
					numpytoimg(lut, target_folder + target_file_name.split("/")[-1])
