import argparse
import copy
import glob
import json
import os
import shutil
from os import path

import SimpleITK as sitk
import cv2
import nibabel as nib
import numpy
import pydicom
from PIL import Image

from image_utils import normalization, numpytoimg
from segmentation.mask import apply

parser = argparse.ArgumentParser(description='Preprocess CT scans.')
parser.add_argument('--patient', help="Path to a patient's folder")
args = parser.parse_args()

dest_folder = 'final3'
dest_masks_folder = 'masks3'
dest_masked_folder = 'final3_masked'
dest_bb_folder = 'final3_BB'

files = glob.glob(args.patient+"/**/*.nii.gz", recursive=True)
if len(files) > 0:
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
                    x = x_tot[:, :, img_idx]

                    # generate lung masks
                    x_rotated = copy.deepcopy(x)
                    x_rotated = numpy.rot90(x_rotated)
                    img_temp = sitk.GetImageFromArray(x_rotated)
                    lung_mask = apply(img_temp)
                    lung_mask[lung_mask == 2] = 1
                    lung_mask = numpy.rot90(lung_mask.squeeze(), k=3)

                    # check if lung is not present or is too small
                    bb = Image.fromarray(lung_mask).getbbox()
                    if bb is None:
                        continue
                    area = (bb[3]-bb[1]) * (bb[2]-bb[0])
                    if area < 10000:
                        continue
                    perc = (lung_mask == 1).sum() / (lung_mask == 0).sum()
                    if perc < 0.10:
                        continue
                    # Fixed windowing
                    ds1.WindowCenter = -500  # center
                    ds1.WindowWidth = 1500  # width
                    ds1[0x0028, 0x1050].value = -500  # center
                    ds1[0x0028, 0x1051].value = 1500  # width

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
                    numpytoimg(lung_mask*255, dest)

                    # save masked
                    res = cv2.bitwise_and(lut,lut,mask = lung_mask)
                    dest = os.path.join(dest_masked_folder,target_file_name.split(os.sep)[-4],target_file_name.split(os.sep)[-3],target_file_name.split(os.sep)[-1])
                    if not os.path.exists(os.path.dirname(dest)):
                        os.makedirs(os.path.dirname(dest))
                    numpytoimg(res, dest)

                    # save BB
                    bb_image = lut[bb[1]:bb[3],bb[0]:bb[2]]
                    dest = os.path.join(dest_bb_folder,target_file_name.split(os.sep)[-4],target_file_name.split(os.sep)[-3],target_file_name.split(os.sep)[-1])
                    if not os.path.exists(os.path.dirname(dest)):
                        os.makedirs(os.path.dirname(dest))
                    numpytoimg(bb_image, dest)


num_images = 0
sessions = glob.glob(dest_folder+"/**/ses-*")
for session_path in sessions:
    images = glob.glob(os.path.join(session_path, "*"))
    num_images += len(images)

if num_images < 50:
    if os.path.exists(dest_folder):
        shutil.rmtree(dest_folder)
    if os.path.exists(dest_bb_folder):
        shutil.rmtree(dest_bb_folder)
    if os.path.exists(dest_masked_folder):
        shutil.rmtree(dest_masked_folder)
