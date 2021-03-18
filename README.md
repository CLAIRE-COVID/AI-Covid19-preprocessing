# AI-Covid19-preprocessing

Currently, the main core for the code is ready.
The main pipeline defined includes:
-image windowing
-histogram equalization
-image resizing (most likely: 1280x720)
-rigid registration for CXR

Both extended documentation and a python notebook example will be provided soon.

Statistics on the dataset to be used (https://bimcv.cipf.es/bimcv-projects/bimcv-covid19/) will be as well provided.

An example pipeline (to be customized soon according to the dataset) is provided in examples/pipeline_example.py 

"lung_segmentation" contains experimental deeplearning-based lung segmentation code. It requires a UNet pre-trained model, to be decided whether to include it or not in the pre-processing pipeline.

## CT Preprocessing v2

- generate lung masks (model weights are too large to upload on github, downlaod link will be available soon)
- fixed windowing (-500, 1500)
- removed non axial plane slices
- removed slices with no lung or small lung (computing lung masks)
- remove patients with a low number of slices (lower than 50)
- generate a masked and a BB version
