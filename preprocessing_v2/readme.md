# Preprocessing for CT
- generate lung masks (model weights are too large to upload on github, downlaod link will be available soon)
- fixed windowing (-500, 1500)
- removed non axial plane slices
- removed slices with no lung or small lung (computing lung masks)
- remove patients with a low number of slices (lower than 50)
- generate a masked and a BB version