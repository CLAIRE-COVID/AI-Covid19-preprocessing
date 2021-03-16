import copy
import glob
import json
import os
import shutil
from os import path

import cv2
import nibabel as nib
import numpy
import pandas as pd
import pydicom
import SimpleITK as sitk
from matplotlib import pyplot as plt
from PIL import Image
from tqdm import tqdm

from image_utils import imgtonumpy, normalization, numpytoimg, shape_as
from mask import apply

root = 'F:\Bimcv\V1.0'

dest_folder = 'final3'
dest_masks_folder = 'masks3'
dest_masked_folder = 'final3_masked'
dest_bb_folder = 'final3_BB'

patients = glob.glob(os.path.join(root,"*"))

for patient_num, patient_path in tqdm(enumerate(patients)):
    files = glob.glob(patient_path+"/**/*.nii.gz", recursive=True)
    if len(files)>0:
            for filex in files:
                if path.exists(filex.replace("nii.gz", "json")):
                    img = nib.load(filex)
                    x_tot = numpy.array(img.dataobj)
                    datax = json.load(open(filex.replace("nii.gz", "json")))
                    ds1 = pydicom.dataset.Dataset.from_json(datax)

                    x_tot = x_tot.reshape(x_tot.shape[0], x_tot.shape[1], -1)

                    # check if CT is in AXIAL plane
                    if all(x in ds1.ImageType for x in ['ORIGINAL', 'PRIMARY', 'AXIAL']):

                        for img_idx in range(x_tot.shape[2]):
                            x = x_tot[:,:,img_idx]

                            # generate lung masks
                            x_rotated = copy.deepcopy(x)
                            x_rotated = numpy.rot90(x_rotated)
                            img_temp = sitk.GetImageFromArray(x_rotated)
                            lung_mask = apply(img_temp)
                            lung_mask[lung_mask==2] = 1
                            lung_mask = numpy.rot90(lung_mask.squeeze(),k=3)

                            # check if lung is not present or is too small
                            bb = Image.fromarray(lung_mask).getbbox()
                            if bb is None:
                                continue
                            area = (bb[3]-bb[1]) * (bb[2]-bb[0])
                            if area <10000:
                                continue
                            perc = (lung_mask==1).sum() / (lung_mask==0).sum()
                            if perc < 0.10:
                                continue
                            # Fixed windowing
                            ds1.WindowCenter = -500 #center
                            ds1.WindowWidth = 1500 #width
                            ds1[0x0028,0x1050].value = -500 #center
                            ds1[0x0028,0x1051].value = 1500 #width

                            lut = pydicom.pixel_data_handlers.util.apply_modality_lut(x, ds1)
                            lut = pydicom.pixel_data_handlers.util.apply_voi_lut(x, ds1).astype(numpy.float)
                            lut = normalization(lut)
                            if ds1[0x0028,0x0004].value == "MONOCHROME1":
                                lut = -1*(lut-255)
                            
                            target_file_name = filex.replace(".nii.gz", "_"+str(img_idx)+".png")
                            # save preprocessed
                            dest = os.path.join(dest_folder,target_file_name.split(os.sep)[-4],target_file_name.split(os.sep)[-3],target_file_name.split(os.sep)[-1])
                            if not os.path.exists(os.path.dirname(dest)):
                                os.makedirs(os.path.dirname(dest))
                            numpytoimg(lut, dest)
                            # save mask
                            dest = os.path.join(dest_masks_folder,target_file_name.split(os.sep)[-4],target_file_name.split(os.sep)[-3],target_file_name.split(os.sep)[-1])
                            if not os.path.exists(os.path.dirname(dest)):
                                os.makedirs(os.path.dirname(dest))
                            numpytoimg(lung_mask*255,dest)

                            # save masked
                            res = cv2.bitwise_and(lut,lut,mask = lung_mask)
                            dest = os.path.join(dest_masked_folder,target_file_name.split(os.sep)[-4],target_file_name.split(os.sep)[-3],target_file_name.split(os.sep)[-1])
                            if not os.path.exists(os.path.dirname(dest)):
                                os.makedirs(os.path.dirname(dest))
                            numpytoimg(res,dest)

                            # save BB
                            bb_image = lut[bb[1]:bb[3],bb[0]:bb[2]]
                            dest = os.path.join(dest_bb_folder,target_file_name.split(os.sep)[-4],target_file_name.split(os.sep)[-3],target_file_name.split(os.sep)[-1])
                            if not os.path.exists(os.path.dirname(dest)):
                                os.makedirs(os.path.dirname(dest))
                            numpytoimg(bb_image,dest)



# remove patients with imgs < 50
imgs_path = dest_folder
patients = glob.glob(os.path.join(imgs_path,"*"))

patient_dict = {}

for patient_path in tqdm(patients):
    patient_dict[patient_path] = 0
    sessions = glob.glob(os.path.join(patient_path,"*"))
    for session_path in sessions:
        images = glob.glob(os.path.join(session_path,"*"))
        patient_dict[patient_path] = patient_dict[patient_path] + len(images)
    
    if patient_dict[patient_path] <50:
        shutil.rmtree(patient_path)
        shutil.rmtree(patient_path.replace(dest_folder,dest_bb_folder))
        shutil.rmtree(patient_path.replace(dest_folder,dest_masked_folder))
    

# save dataset info
df = pd.DataFrame.from_dict(patient_dict,orient='index')
df.sort_values(by=0).to_csv(imgs_path+'.csv')

