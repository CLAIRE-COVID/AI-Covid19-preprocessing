import pydicom
import numpy as np

def get_raw_array(data):
	rows = int(data[(0x0028,0x0010)].value)
	cols = int(data[(0x0028,0x0011)].value)
	print(rows*cols*2)
	base = list(np.arange(0,rows*cols*2, 2))
	adv = list(np.arange(1,rows*cols*2, 2))
	raw_full = data.PixelData
	raw = [raw_full[x] for x in base]
	converted = ((np.asarray(raw)).reshape(rows, cols))
	adv_raw = [raw_full[x] for x in adv]
	converted += 256*((np.asarray(adv_raw)).reshape(rows,cols))
	return converted

def get_windowed_image(data):
	data_new = get_raw_array(data)
	lut = pydicom.pixel_data_handlers.util.apply_voi_lut(data_new, data)
	lut = lut-lut.min()
	lut = lut / lut.max() * 255.0
	if data[0x0028,0x0004].value == "MONOCHROME1":
		lut = -1.0*(lut-255.0)
	return lut

def dicomtonumpy(map_name):
	map_data = pydicom.dcmread(map_name)
	my_array_1 = get_windowed_image(map_data)
