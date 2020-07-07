import argparse
import os

import numpy as np
import torch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import torchvision
import lung_segmentation.importAndProcess as iap
from models import model
from models.unet_models import unet11, unet16


parser = argparse.ArgumentParser()
#parser.add_argument('img_path')
parser.add_argument('-m', '--model', choices=['unet11', 'unet16', 'resnet'], default='unet16')
#parser.add_argument('-r', '--resume-from', help='resume from a specific savepoint', required=True)
parser.add_argument('-t', '--input-type', choices=['dicom', 'png'], default='dicom')
parser.add_argument('--non-montgomery', action='store_true', help='toggle this flag if you are working on a non-montgomery dataset')
parser.add_argument('--no-normalize', action='store_true')
args = parser.parse_args()

normalize = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

if args.model == 'resnet':
    model = model.segmentNetwork().cuda()
    resize_dim = (400, 400)
    convert_to = 'L'
elif args.model == 'unet11':
    model = unet11(out_filters=3).cuda()
    resize_dim = (224, 224)
    convert_to = 'RGB'
elif args.model == 'unet16':
    model = unet16(out_filters=3).cuda()
    resize_dim = (224, 224)
    convert_to = 'RGB'

model = torch.nn.DataParallel(model)
model.load_state_dict(torch.load("unet16_100.pth"))

if args.no_normalize:
    transforms = Compose([Resize(resize_dim),ToTensor()])
else:
    transforms = Compose([Resize(resize_dim),ToTensor(),normalize])
convert_to = 'RGB'

source_path = '/home/enzo/data/Cohen/'#CORDA-dataset-v3/'
#mask_path = '/home/enzo/data/CORDA-dataset-v4-masks/'
target_path = '/home/enzo/data/Cohen-masks/'#'/home/enzo/data/CORDA-dataset-v3-masked/'

for this_folder in ["data"]:#["RX--COVID-", "RX--COVID+", "RX+-COVID-", "RX+-COVID+"]:
	dataset = iap.LungTest(source_path+this_folder, transforms, convert_to)
	dataloader = torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=False)

	with torch.no_grad():
	    for i, sample in enumerate(dataloader):
	        img = sample['image'].cuda()
	        mask = (model(img)>0)#.type(torch.float)
	        mask = (~(mask[:,0,:,:])).type(torch.float).cpu()##0 ritaglia fuiri dai polmoni
	        #mask = mask.cpu() * sample['image'].squeeze(dim=0)#.numpy()
	        mask=torchvision.transforms.ToPILImage(mode='L')(mask)
	        #to_save = transforms.ToPILImage(mode='RGB')(img)
	        #im = Image.fromarray(mask)
	        mask.save(target_path + this_folder +'/'+ sample['filename'][0])#to_save.save('prova.png')
