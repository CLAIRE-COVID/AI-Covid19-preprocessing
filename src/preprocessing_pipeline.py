import argparse
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

parser = argparse.ArgumentParser(description='Preprocess CT scans.')
parser.add_argument('--patient', help="Path to a patient's folder")
args = parser.parse_args()

os.makedirs("preproc", exist_ok=True)
os.makedirs("excluded", exist_ok=True)
os.makedirs("registered", exist_ok=True)

ref_numpy = imgtonumpy(os.path.join(os.path.dirname(__file__), "reference.png")).astype(numpy.float)
files = glob.glob(args.patient + "/**/*.png", recursive=True)
if len(files) > 0:
	for filex in files:
		target_file_name = filex.replace(".png", "_preproc.png")
		if ((path.exists(filex.replace("png", "json"))) and (not path.exists("registered/"+target_file_name.split('/')[-1]))):
			flag_excluded = False
			if ("ll" in filex) or ("lateral" in filex):
				target_folder = "excluded/"
				flag_excluded = True
			else:
				target_folder = "preproc/"
			print(filex)
			x = imgtonumpy(filex)  # .astype(numpy.float)
			datax = json.load(open(filex.replace("png", "json")))

			# fix some errors in the JSON exports
			if '00281054' in datax.keys():
				if datax['00281054']['vr'] != ['CS']:
					datax['00281054']['vr'] = ['CS']
			if '20500020' in datax.keys():
				if datax['20500020']['vr'] != ['CS']:
					datax['20500020']['vr'] = ['CS']

			ds1 = pydicom.dataset.Dataset.from_json(datax)
			lut = pydicom.pixel_data_handlers.util.apply_modality_lut(x, ds1).astype(numpy.uint16)
			# print(x.max(), lut.max(), lut.shape)
			lut = pydicom.pixel_data_handlers.util.apply_voi_lut(lut, ds1).astype(numpy.float)
			lut = normalization(lut)
			if ds1[0x0028,0x0004].value == "MONOCHROME1":
				lut = -1*(lut-255)
			lut = shape_as(lut)
			# x = cv2.GaussianBlur(x,(5,5),cv2.BORDER_DEFAULT)
			target_file_name = filex.replace(".png", "_preproc.png")
			# numpytoimg(lut, target_folder + target_file_name.split("/")[-1])
			if ((not flag_excluded) and (not path.exists("registered/"+target_file_name.split('/')[-1]))):
				# print("entro {}".format("registered/"+filex.split('/')[-1]))
				lut = registration(lut, ref_numpy)
				numpytoimg(lut, "registered/" + target_file_name.split("/")[-1])
