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
import nibabel as nib

root = '/home/ubuntu/dataset_claire_1/'
#root = '/home/tarta/dataset_claire/V1.0/bimcv_covid19_posi_subjects_1'

os.makedirs("preproc_ct", exist_ok=True)

ref_numpy = imgtonumpy("reference.png").astype(numpy.float)
patients = glob.glob(os.path.join(root,"*"))
for patient_path in patients:
	files = glob.glob(patient_path+"/**/*.nii.gz", recursive=True)
	if len(files)>0:
			for filex in files:
				if path.exists(filex.replace("nii.gz", "json")):
					print(filex)
					img = nib.load(filex)
					x_tot = numpy.array(img.dataobj)
					datax = json.load(open(filex.replace("nii.gz", "json")))
					#fix some errors in the JSON exports
					'''
					if '00281054' in datax.keys():
						if datax['00281054']['vr'] != ['CS']:
							datax['00281054']['vr'] = ['CS']
					if '20500020' in datax.keys():
						if datax['20500020']['vr'] != ['CS']:
							datax['20500020']['vr'] = ['CS']
					'''
					ds1 = pydicom.dataset.Dataset.from_json(datax)
					x_tot = x_tot.reshape(x_tot.shape[0], x_tot.shape[1], -1)
					f = open("ct_path.csv", "a")
					f.write("{}\t{}\n".format(filex, x_tot.shape[2]))
					f.close()
					for img_idx in range(x_tot.shape[2]):
						x = x_tot[:,:,img_idx]
						lut = pydicom.pixel_data_handlers.util.apply_modality_lut(x, ds1)#.astype(numpy.int)
						#print(x.max(), lut.max(), lut.shape)
						lut = pydicom.pixel_data_handlers.util.apply_voi_lut(x, ds1).astype(numpy.float)
						lut = normalization(lut)
						if ds1[0x0028,0x0004].value == "MONOCHROME1":
							lut = -1*(lut-255)
						#x = cv2.GaussianBlur(x,(5,5),cv2.BORDER_DEFAULT)
						target_file_name = filex.replace(".nii.gz", "_"+str(img_idx)+".png")
						numpytoimg(lut, "preproc_ct/" + target_file_name.split("/")[-1])
